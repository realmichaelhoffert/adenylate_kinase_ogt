# adenylate_kinase_ogt

This is a repository for the code an analyses associated with the project "Predicting bacterial optimal growth temperatures using learned structural information from a single protein." This project develops a model, ROSEATE, which performs **R**egression **O**n **S**ite-specific **E**mbeddings for **A**denylate **K**inase-Based OG**T** **E**stimation to predict the optimal growth temperature (OGT) of bacteria. Its name is based on our use of the Rosetta (the macromolecular modeling suite), and my love of birds and contrived backronyms.

## Table of Contents
* [Installation](#installation)
* [Usage](#usage)
* [Repo](#repo)
* [Architecture](#architecture)

## Installation

Environments are managed with [mamba](https://mamba.readthedocs.io). All commands
below assume the environment is active, so `pip` is the environment's own pip
(check with `which pip`, which should print a path under `$CONDA_PREFIX`).

### 1: Only run ROSEATE to predict OGT

Install `envs/roseate.yaml` environment. This environment contains packages required to run the ROSEATE module functions.

```bash
mamba env create -f envs/roseate.yml
mamba activate roseate
pip install --no-deps -e .
```

Check the installation with `roseate --help` and `pip check`. Tests are for ensuring code matches published figures during refactoring, and require an archive file.

### 2: Perform additional analyses

Install `envs/analysis.yaml` environment.

```bash
mamba env create -f envs/roseate.yml
mamba activate roseate
pip install --no-deps -e .
```

Enables rerunning notebooks.

### 3: Protein folding, running TOME, and other tasks

Other specific tools have specific environments. Details below.  The environment to fold proteins was constructed using the instructions here:
[https://git.embl.de/grp-kosinski/alphafold_howto/-/tree/main](https://git.embl.de/grp-kosinski/alphafold_howto/-/tree/main)
Also install `ambertools` and `nglview` using the conda-forge channel. You can also install seaborn. If importing doesn't work, downgrading numpy to 1.23.5 fixed this problem for me.

#### Notes

`envs/*.lock.txt` lists the exact versions installed (created with
`mamba list --explicit`). `adenylate_kinase_ogt.yaml` is the original single
environment used for the preprint and is kept for reference.

  

## Usage

### Run ROSEATE on new proteins using the ADK model

On genomes:

```bash
roseate prediction-workflow --genomes [dir/*.fa] --verbose --threads [n_threads] -o out_dir
```

On metagenomic assemblies:
```bash
roseate prediction-workflow --metagenomes [dir/*.fa] --verbose --threads [n_threads] -o out_dir
```

On pre-annotated proteins (ORF amino acid fastas) from genomes:
```bash
roseate prediction-workflow --genome-proteins [dir/*.faa] --verbose --threads [n_threads] -o out_dir
```

On pre-annotated proteins (ORF amino acid fastas) from metagenomes:
```bash
roseate prediction-workflow --metagenome-proteins [dir/*.faa] --verbose --threads [n_threads] -o out_dir
```

On a pre-identified set of ADK proteins:
```bash
roseate prediction-workflow --sequences [dir/*.faa] --verbose --threads [n_threads] -o out_dir
```

### Notes
Files can be provided as list (`genome1.fasta` `genome1.fasta`), folder (`genomes/`), or folder with wildcard extension (`genome/*.fasta`). When providing a folder, all files with common extensions (.fa, .fasta, .fna, .faa) will be used. `--genomes` and `--metagenomes` must be nucleotide fastas.

### Outputs
Saved under `out_dir`:
* `1_extract`: `genomes`, folder containing files generated during annotation
* `2_align`: proteins aligned to backbone, the alignment used for generating embeddings during training
* `3_embed`: embedding results as .npy
* `4_fit`: Dummy folder containing model copied from inputs, not refit in `prediction-workflow`
* `5_predict`: predictions as `predictions.tsv`

### Notes

Prediction workflow uses a pre-specified config file `configs/adk.yaml` to load pre-fitted model components and parameters. This model **does not** work for proteins other than adenylate kinase. Proteins need to be embedded and aligned, which can take time. Speed up runs by selecting fewer proteins per genome or metagenome.

`metagenome` mode sets Prodigal gene caller mode to `-p meta` and allows up to 1000 proteins to be annotated per file: adjust protein count (for example, to 10) with `--set extract.n_proteins=10`.

## Repo
* `roseate/` : ROSEATE model source code
* `tests/` : test, mostly written for refactoring
* `reference/` and `models/`: storage for pretrained model and data objects
* `reproduce/` : go here if you want to quickly reproduce figures - refactored with AI from `notebooks/` and `roseate/`
* `src/` : Legacy code storage to ensure reproducibility for notebooks
* `notebooks/` : analysis code
* `scripts/` : helper scripts
* `data/` : project-relevant data
  * `metric_tables/` : structural metrics, genome to temperature table, etc.
  * `figures` : miscellaneous figures
  * `outputs`: The raw outputs of AlphaFold for the OGT dataset, compiled using script in Section E: Folding With ColabFold.
 
## Architecture

Each stage is a command, roseate <stage> -o OUT. It reads the previous stage's files from OUT unless a path is given, skips itself if its outputs exist, and appends a record to OUT/run.jsonl. Settings come from configs/<protein>.yaml: the HMM for extract, the backbone for align, the context dataset for embed and the model for fit. Each can be overridden with --set stage.option=value.


```
  prediction-workflow                                        fitting-workflow
  genomes | metagenomes | proteins            backbone .afa + train IDs + traits
      │                                                               │
┌─────▼───────────────────────────────────────────────────┐           |
│ 1_extract  extract.py                                   │           |
│ prodigal (single / meta) → hmmsearch → best hits,       │           |
│ or given sequences scored against the HMM               │           |
│ → proteins.faa                                          │           |
└─────┬───────────────────────────────────────────────────┘           |
      │                                                               │
┌─────▼───────────────────────────────────────────────────────────────▼───┐
│ 2_align  align.py, io.py                                                │
│ predict: MAFFT --add the proteins to the backbone                       │
│ fit:     select the training rows of the backbone                       │
│ → protein_to_backbone.afa + .pickle (MSADataset)                        │
└─────┬───────────────────────────────────────────────────────────────────┘
      │
┌─────▼───────────────────────────────────────────────────────────────────┐
│ 3_embed  context.py, embed.py                                           │
│ contexts (sampled with size + seed, or loaded) → MSA Transformer        │
│ → embeddings.npy   N × (L + 1) × 768                                    │
└─────┬───────────────────────────────────────────────────────────────────┘
      │
┌─────▼───────────────────────────────────────────────────────────────────┐
│ 4_fit  fit.py, io.py                                                    │
│ predict: copy the model and scaler                                      │
│ fit:     scale traits → StandardScaler + ElasticNetCV per site          │
│ → model.pickle, scaler.pickle                                           │
└─────┬───────────────────────────────────────────────────────────────┬───┘
      │                                                               │
┌─────▼───────────────────────────────────────────────────┐           │
│ 5_predict  predict.py, aggregate.py                     │           ▼
│ per-site predictions → mean over sites → °C             │   model_config.yaml
│ → predictions.tsv                                       │     for --config
└─────────────────────────────────────────────────────────┘
```

## Notebook outline
1. `notebooks/00_adenylate_kinase.ipynb`
   * Assess OGT dataset (Gosha a.k.a Melnikov)
   * Find and save sequences for ADKs from each genome in Melnikov
   * Old code to classify PDB blast hits for ADKs into open / closed conformations
2. `notebooks/01_compile_data.ipynb`
   * Code for assembling metric tables / running Rosetta and Biopython-based structural feature calculation
3. `notebooks/02_filtering.ipynb`
   * Filtering structures
4. `notebooks/03_plots_and_PL.ipynb`
   * Plots of structural features vs temperature
   * Miscellaneous attempts to predict temperature from structural
   <!-- * Canonical (current, best, manuscript-ready (????) ) structural predictive model -->
5. `notebooks/05_esm2_model_v2.ipynb`
   * Model architecture and embedding-based regression development
6. `notebooks/figures_final.ipynb`
   * Notebook containing the majority of preprint figures.
7. `notebooks/dev_deltaSASA.ipynb`
   * Code for calculating structural characteristics


# Compilation of structural features of ADK

## F. Folding with ColabFold

### Folding on HPC
Log in
```
ssh [username]@login.rc.colorado.edu
```
Launch compile node
```
acompile
```
Navigate to working directory `/projects/miho1832/adenylate_kinase_ogt/data/`  
Directory structure:
```
/outputs/[protein].tgz # compressed colabfold outputs
/adks/ # adk proteins
```
Run slurm script `colabfold_script.sh`
```
#!/bin/bash

#A typical runs takes couple of hours but may be much longer
#SBATCH --time=03:00:00
#SBATCH --job-name=colabfold_adks

#log files:
#SBATCH -e AF_%x_%j_err.txt
#SBATCH -o AF_%x_%j_out.txt

#SBATCH --qos=normal

#SBATCH --partition=aa100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=16000
# current array at 101

FILES=(adks/*.faa)
FILE=${FILES[$SLURM_ARRAY_TASK_ID]}

echo "At task: ${SLURM_ARRAY_TASK_ID}"
echo ${FILE}
OUTFILE=$(basename ${FILE} _adk.faa)
echo ${OUTFILE}

module purge
module load cuda/11.8
module load cudnn/8.6
module load mambaforge/23.1.0-1

mamba activate /projects/miho1832/miniforge3/envs/adenylate_kinase_ogt

cd /projects/miho1832/adenylate_kinase_ogt/data/

mkdir ./outputs/${OUTFILE}_temp
cp ./folded/closed/* ./outputs/${OUTFILE}_temp/
srun colabfold_batch ./adks/${OUTFILE}_adk.faa ./outputs/${OUTFILE}_closed \
--templates --custom-template-path ./outputs/${OUTFILE}_temp/ \
--overwrite-existing-results --amber --num-relax 5 --model-order 1,2,3,4,5 \
--random-seed 1040 --num-recycle 7 --save-recycles --data ./outputs/${OUTFILE}_temp --use-gpu-relax
rm -r ./outputs/${OUTFILE}_temp/
cd outputs/
tar -czf ${OUTFILE}_closed.tgz ${OUTFILE}_closed
rm -r ${OUTFILE}_closed
cd ..
```

### Moving to analysis server

Move outputs to microbe (analysis server) from alpine (HPC server) (run on microbe)
```
rsync -r miho1832@login.rc.colorado.edu:/projects/miho1832/adenylate_kinase_ogt/data/outputs/ ./
```
Unpack outputs
```
ls outputs/ | sed 's/_closed.*//' > inds_unpack.txt
rm file_names.txt
touch file_names.txt
mkdir test_structures2
# from parent directory of outputs
parallel -j 12 './unpack_structures.sh {}' :::: inds_unpack.txt
```

### Computing structural metrics

Run Biopython code
```
python src/protein_utils/mp_metrics.py -s [structures to process] -n [number to test code, -1 for all] -t [threads, structures to process at once] -p True --save-processed [dir to save processed structures] -o [file to save complete table]
```

This code both creates trimmed, processed versions of the ColabFold structures and a table containing BioPython-based calculation of two metrics: Shrake-Rupley-based SASA and contacts at 4.5 Angstroms.  
File locations:
* `processed_structures`: Processed, cleaned outputs of ColabFold
* `metric_tables/20240524_full_metrics.tsv.gz`: BioPython-based structural metrics
Command run:
```
python src/protein_utils/mp_metrics.py -s  adenylate_kinase_ogt/data/test_structures/ \
                    -o  adenylate_kinase_ogt/data/test_structures/\
                    -t 16 \
                    --write-processed True \
                    --save-processed adenylate_kinase_ogt/data/processed_structures/
```  

Additionally, Rosetta was used to calculate more structural metrics. 

Run Rosetta code
```
ls processed_structures/ | sed 's/\.pdb//' > new_rosetta_inputs.txt
# from data/
parallel -j 20 './../scripts/run_rosetta.sh -i {} -x ./rosetta/ogt_metrics.xml -o ./rosetta_out/ -d ./processed_structures/' :::: ./new_rosetta_inputs.txt
```

```
parallel -j 12 './run_rosetta.sh {}' :::: inds_rosetta.txt
```

`inds_rosetta.txt` and `inds_unpack.txt` are lists of the genome ids, one per line, in a text file. Unfortunately the code is crappy and runs with hard coded file paths. So it must be run from `data/`

### Other structural information included:  

1. H-bond data (2026, from Dru)
2. Lid type (2026, v4)

## H. Current workflow 
1. Get GTDB genomes with OGTs from Melnikov dataset.
2. Get highest quality match to PF00406, "adenylate kinase" from each genome using existing mapping of PFam to GTDB r207.
3. Save genome ADK proteins (GAPs) to files
4. Match GAPs to PDB proteins.
5. *Only once* Manually examine and classify ADKs as "open" or closed using Ipython widgets.
6. (5) identified that core-lid distance > 25 for "open" looking proteins.
7. Make folders containing matching closed / open templates for each protein
8. Run ColabFold
9. Pick the best open / closed template for each genome
10. Manually relaxed unrelaxed open templates
11. Dru performs analysis

## G. Improved workflow
1. Get GTDB genomes with OGTs from Melnikov + Enqvist + Corkeys
2. Get highest quality match to PFO0406 (Adk) + Core + lid + NMP for GTDBr214.1 genomes in (1)
3. Save Adks with annotations of each domain location.
4. Match GAPs to PDB proteins.
5. In an iterative process, build database of templates simultaneously: Captures steps 7-10 above.
```
unfolded_proteins = [p1, p2, ... pn]
folded_proteins = [f1, f2, ..., fn]
while len(unfolded_proteins) > 0:
    for u in unfolded_proteins:
        templates = find_templates(u, folded_proteins)
        if len(templates) > 0:
            folded = fold_protein(u, templates)
            if check_conformation(folded) == 'open':
                unfolded_proteins.pop(u)
                folded_proteins.append(folded)
```
6. Analysis

## I. Need to figure out
1. Annotating domains
2. Efficient programmatic relaxation on server
3. Streamlining "best protein" selection code.
4. Iterative folding + database generation
5. Snakemake?
