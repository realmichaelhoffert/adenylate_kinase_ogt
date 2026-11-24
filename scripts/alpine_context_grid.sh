#!/bin/bash
# Context depth and seed grid for ROSEATE (revision steps 4-5) on CURC Alpine.
#
# Run from a login node; every phase is its own Slurm job that calls this script again:
#   bash scripts/alpine_context_grid.sh test     # setup, then ONE small grid run on CPU (cpu-testing, < 1 h)
#   bash scripts/alpine_context_grid.sh submit   # setup -> GPU embedding array -> CPU fitting array -> summary
#
# Phases (submitted for you; listed for troubleshooting):
#   setup      clone or update the repo, create the conda environment (CUDA build of PyTorch) from
#              envs/roseate.yaml, pip install roseate, download the MSA Transformer weights, write the run list
#   embed      GPU array task: align + embed the training rows for one (depth, seed)
#   fit        CPU array task, after its embed task: fitting-workflow (skips the finished align and embed),
#              then prediction-workflow on the test sequences with that model's config
#   summarize  scripts/context_grid.py summarize over all runs
#
# Why two arrays: embedding needs a GPU for minutes, fitting needs CPUs for about an hour; one GPU job per
# run would leave the GPU idle most of the time. Run folders match scripts/context_grid.py:
# $RUNS/grid/dDD_sS/{fit,predict}, with runtime_*.txt files from GNU time.
set -eo pipefail          # no -u: CURC's module and conda scripts reference unset variables

# ---------------------------------------------------------------- settings (edit these)
ACCOUNT=""                                         # e.g. ucbNNN_asc1; empty = your default account
REPO_URL=https://github.com/realmichaelhoffert/adenylate_kinase_ogt.git
BRANCH=main
WORK=/projects/$USER/roseate                       # repo clone, conda environment, model weights
INPUTS=$WORK/inputs                                # copy the five input files here
RUNS=/scratch/alpine/$USER/roseate_context_grid    # outputs (about 0.8 GB of embeddings per run)

BACKBONE=$INPUTS/backbone.afa                      # training-only backbone from step 2
TRAIN_IDS=$INPUTS/even_sequence_ids.txt
TRAITS=$INPUTS/temp_data.tsv
SEQUENCES=$INPUTS/20240122_all_adks_filtered.faa   # unaligned; contains the test sequences
TEST_IDS=$INPUTS/corkrey_sequence_ids.txt

DEPTHS="0 2 4 8 16 32 64 128 256"
SEEDS=10                                            # seeds per depth ...
SEEDS_AT_DEPTH_8=10                                 # ... and at depth 8 (used for component stability)
GPU_TYPE=a100-40gb
MAX_GPUS=6                                         # gpu-normal: at most 6 a100-40gb per user at once
EMBED_CPUS=8
FIT_CPUS=16

# test mode: one small run (TEST_TRAIN training rows, TEST_TEST test sequences, depth 2, seed 1)
TEST_TRAIN=60
TEST_TEST=20

# ---------------------------------------------------------------- derived paths
REPO=$WORK/adenylate_kinase_ogt
ENV=$WORK/env
CONFIG=$REPO/configs/adk.yaml
if [[ "${TEST:-0}" == 1 ]]; then RUNS=$RUNS/test; fi
LOGS=$RUNS/logs
RUN_LIST=$RUNS/runs.tsv
TEST_FASTA=$RUNS/test_sequences.faa

# ---------------------------------------------------------------- helpers
account() { [[ -n "$ACCOUNT" ]] && echo "--account=$ACCOUNT" || true; }

activate() {
    module purge
    module load miniforge
    eval "$(conda shell.bash hook)"
    conda activate "$ENV"
    export TORCH_HOME=$WORK/torch                  # model weights outside the small home quota
    export XDG_CACHE_HOME=$WORK/cache PIP_CACHE_DIR=$WORK/cache/pip
    export PYTHONNOUSERSITE=1                      # ignore packages in ~/.local
}

timed() {  # timed <output file> <command...>: GNU time writes "seconds peak_KB" to the file
    local out=$1; shift
    mkdir -p "$(dirname "$out")"
    if [[ -x /usr/bin/time ]]; then /usr/bin/time -f "%e %M" -o "$out" "$@"; else "$@"; fi
}

grid_runs() {  # one "depth seed" line per run
    for depth in $DEPTHS; do
        local n=$SEEDS
        [[ "$depth" == 8 ]] && n=$SEEDS_AT_DEPTH_8
        for seed in $(seq 1 "$n"); do echo "$depth $seed"; done
    done
}

run_of() {  # run_of <array index>: sets DEPTH, SEED and RUN from the run list
    read -r DEPTH SEED < <(sed -n "$(( $1 + 1 ))p" "$RUN_LIST")
    RUN=$RUNS/grid/$(printf 'd%02d_s%d' "$DEPTH" "$SEED")
}

inputs() {  # training IDs and test sequences for this mode
    if [[ "${TEST:-0}" == 1 ]]; then echo "$RUNS/train_ids.txt"; else echo "$TRAIN_IDS"; fi
}

# ---------------------------------------------------------------- phases
setup() {
    for f in "$BACKBONE" "$TRAIN_IDS" "$TRAITS" "$SEQUENCES" "$TEST_IDS"; do
        [[ -s "$f" ]] || { echo "missing input: $f" >&2; exit 1; }
    done
    module purge
    module load miniforge
    eval "$(conda shell.bash hook)"

    if [[ -d "$REPO/.git" ]]; then
        git -C "$REPO" fetch --quiet origin && git -C "$REPO" checkout --quiet "$BRANCH" && git -C "$REPO" pull --quiet --ff-only
    else
        git clone --quiet --branch "$BRANCH" "$REPO_URL" "$REPO"
    fi
    echo "repo at $(git -C "$REPO" rev-parse --short HEAD)"

    if [[ ! -d "$ENV" ]]; then
        # setup runs on a CPU node: tell conda to pick PyTorch's CUDA build anyway
        CONDA_OVERRIDE_CUDA=12.0 mamba env create --quiet -p "$ENV" -f "$REPO/envs/roseate.yaml"
    fi
    conda activate "$ENV"
    pip install --quiet --no-deps -e "$REPO"
    export TORCH_HOME=$WORK/torch PYTHONNOUSERSITE=1
    export XDG_CACHE_HOME=$WORK/cache PIP_CACHE_DIR=$WORK/cache/pip
    python -c "import torch, roseate; print('torch', torch.__version__, '| CUDA build:', torch.version.cuda)"
    python -c "import esm; esm.pretrained.esm_msa1b_t12_100M_UR50S()" >/dev/null   # download weights once

    mkdir -p "$RUNS"
    if [[ "${TEST:-0}" == 1 ]]; then
        echo "2 1" > "$RUN_LIST"
        head -n "$TEST_TRAIN" "$TRAIN_IDS" > "$RUNS/train_ids.txt"
        head -n "$TEST_TEST" "$TEST_IDS" > "$RUNS/test_ids.txt"
        local test_ids=$RUNS/test_ids.txt
    else
        grid_runs > "$RUN_LIST"
        local test_ids=$TEST_IDS
    fi
    python - "$SEQUENCES" "$test_ids" "$TEST_FASTA" <<'EOF'
import sys
from Bio import SeqIO
sequences, ids, out = sys.argv[1:]
wanted = set(open(ids).read().split())
records = [r for r in SeqIO.parse(sequences, 'fasta') if r.id in wanted]
missing = wanted - {r.id for r in records}
assert not missing, f'{len(missing)} test IDs are not in {sequences}'
SeqIO.write(records, out, 'fasta')
print(f'{len(records)} test sequences written to {out}')
EOF
    echo "$(wc -l < "$RUN_LIST") runs listed in $RUN_LIST"
}

embed_task() {  # embed_task <array index>: align and embed the training rows (GPU if available)
    run_of "$1"
    activate
    python -c "import torch; print('device:', 'cuda' if torch.cuda.is_available() else 'cpu')"
    echo "run $RUN: depth $DEPTH, seed $SEED"
    timed "$RUN/runtime_embed.txt" bash -c "
        roseate align -o '$RUN/fit' --config '$CONFIG' --backbone '$BACKBONE' --train-ids '$(inputs)' -v &&
        roseate embed -o '$RUN/fit' --config '$CONFIG' --context-dataset '$RUN/fit/2_align/protein_to_backbone.pickle' \
            --exclude-self --context-size $DEPTH --context-seed $SEED --threads ${SLURM_CPUS_PER_TASK:-4} -v"
    timed "$RUN/runtime_embed_test.txt" bash -c "
        roseate extract -o '$RUN/predict' --config '$CONFIG' --sequences '$TEST_FASTA' -v &&
        roseate align -o '$RUN/predict' --config '$CONFIG' --backbone '$RUN/fit/2_align/protein_to_backbone.afa' -v &&
        roseate embed -o '$RUN/predict' --config '$CONFIG' --context-dataset '$RUN/fit/2_align/protein_to_backbone.pickle' \
            --context-size $DEPTH --context-seed $SEED --threads ${SLURM_CPUS_PER_TASK:-4} -v"
}

fit_task() {  # fit_task <array index>: fit (align and embed are reused), then predict the test set
    run_of "$1"
    activate
    local threads=${SLURM_CPUS_PER_TASK:-4}
    echo "run $RUN: depth $DEPTH, seed $SEED"
    timed "$RUN/runtime_fit.txt" roseate fitting-workflow --alignment "$BACKBONE" --train-ids "$(inputs)" \
        --traits "$TRAITS" --set embed.context_size="$DEPTH" --set embed.context_seed="$SEED" \
        --threads "$threads" -v -o "$RUN/fit"
    timed "$RUN/runtime_predict.txt" roseate prediction-workflow --config "$RUN/fit/model_config.yaml" \
        --sequences "$TEST_FASTA" --threads "$threads" -v -o "$RUN/predict"
}

summarize() {
    activate
    python "$REPO/scripts/context_grid.py" summarize --runs "$RUNS" --traits "$TRAITS" --results "$RUNS/results"
}

# ---------------------------------------------------------------- submission
submit() {
    mkdir -p "$LOGS"
    local n common setup_id embed_id fit_id
    n=$(grid_runs | wc -l)
    common="$(account) --parsable --nodes=1 --ntasks=1"
    setup_id=$(sbatch $common --job-name=cg_setup --output=$LOGS/%x_%j.out --partition=acpu --qos=cpu-normal \
        --cpus-per-task=4 --mem=15G --time=02:00:00 "$0" setup)
    embed_id=$(sbatch $common --job-name=cg_embed --output=$LOGS/%x_%A_%a.out --dependency=afterok:$setup_id \
        --array=0-$((n - 1))%$MAX_GPUS \
        --partition=aa100 --qos=gpu-normal --gres=gpu:$GPU_TYPE:1 --cpus-per-task=$EMBED_CPUS --mem=30G \
        --time=06:00:00 "$0" embed)
    fit_id=$(sbatch $common --job-name=cg_fit --output=$LOGS/%x_%A_%a.out --dependency=aftercorr:$embed_id \
        --array=0-$((n - 1)) \
        --partition=acpu --qos=cpu-normal --cpus-per-task=$FIT_CPUS --mem=60G --time=12:00:00 "$0" fit)
    sbatch $common --job-name=cg_summary --output=$LOGS/%x_%j.out --dependency=afterany:$fit_id \
        --partition=acpu --qos=cpu-normal \
        --cpus-per-task=4 --mem=15G --time=01:00:00 "$0" summarize >/dev/null
    echo "submitted $n runs: setup $setup_id, embed array $embed_id, fit array $fit_id; logs in $LOGS"
}

submit_test() {
    export TEST=1
    RUNS=$RUNS/test
    LOGS=$RUNS/logs
    mkdir -p "$LOGS"
    local common setup_id
    common="$(account) --parsable --nodes=1 --ntasks=1 --export=ALL,TEST=1 --output=$LOGS/%x_%j.out"
    setup_id=$(sbatch $common --job-name=cg_test_setup --partition=acpu --qos=cpu-normal --cpus-per-task=4 \
        --mem=15G --time=02:00:00 "$0" setup)
    sbatch $common --job-name=cg_test_run --dependency=afterok:$setup_id --partition=acpu --qos=cpu-testing \
        --cpus-per-task=16 --time=01:00:00 "$0" test_run >/dev/null
    echo "submitted test: setup $setup_id, then one CPU run; logs in $LOGS"
}

case "${1:-}" in
    submit)    submit ;;
    test)      submit_test ;;
    setup)     setup ;;
    embed)     embed_task "$SLURM_ARRAY_TASK_ID" ;;
    fit)       fit_task "$SLURM_ARRAY_TASK_ID" ;;
    summarize) summarize ;;
    test_run)  embed_task 0 && fit_task 0 && summarize ;;
    *) sed -n '2,8p' "$0"; exit 1 ;;
esac