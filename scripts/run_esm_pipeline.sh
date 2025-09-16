#!/bin/bash

set -euo pipefail

# Resolve the directory where this script lives
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# Assume the repo root is the parent directory of the script
REPO_ROOT="$(realpath "$SCRIPT_DIR/..")"

echo "Running from: $SCRIPT_DIR"
echo "Repo root: $REPO_ROOT"

# Default values
GENOME_LIST=""
HMM_MODEL=""

JOBS=4  # Default parallel jobs

# Print help
usage() {
    echo "Usage: $0 --genome-list FILE --hmm-model FILE [--jobs N]"
    echo ""
    echo "  --genome-list FILE   File with one genome FASTA path per line (.fa, .fasta, .gz allowed)"
    echo "  --hmm-model FILE   File of hmm model to run against genomes"
    echo "  --jobs N             Number of parallel jobs (default: 4)"
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --genome-list)
            GENOME_LIST="$2"
            shift 2
            ;;
        --jobs)
            JOBS="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        --hmm-model)
            HMM_MODEL="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            usage
            ;;
    esac
done

if [[ -z "$GENOME_LIST" ]]; then
    echo "Error: --genome-list is required."
    usage
fi

if [[ ! -f "$GENOME_LIST" ]]; then
    echo "Error: genome list file '$GENOME_LIST' not found."
    exit 1
fi

if [[ -n "$HMM_MODEL" && ! -f "$HMM_MODEL" ]]; then
    echo "Error: HMM model '$HMM_MODEL' not found."
    exit 1
fi

# Function to run prodigal on one genome
run_prodigal() {
    genome="$1"

    # Strip directory and extensions: .fasta, .fa, .fasta.gz, .fa.gz
    # filename=$(basename "$genome")
    base="${genome%.fasta}"
    base="${base%.fa}"
    base="${base%.fasta.gz}"
    base="${base%.fa.gz}"

    # Support gzipped files transparently
    if [[ "$genome" == *.gz ]]; then
        zcat "$genome" > "/tmp/${base}_temp.fa"
        genome="/tmp/${base}_temp.fa"
    fi

    seqtk seq -A -L 300 "$genome" > "${base}_filtered.fasta"

    prodigal -i "${base}_filtered.fasta" -c \
             -a "${base}_proteins.faa" \
             -d "${base}_proteins.fna" \
             -o "${base}_prodigal_output.gbk" \
             -p meta

    # Optionally run hmmsearch
    if [[ -n "$HMM_MODEL" ]]; then
        hmmsearch --tblout "${base}_hmmsearch.tbl" "$HMM_MODEL" "${base}_proteins.faa" > "${base}_hmmsearch.log"
    fi

    # Cleanup
    [[ -f "/tmp/${base}_temp.fa" ]] && rm "/tmp/${base}_temp.fa"
}

export -f run_prodigal
export HMM_MODEL

# Run in parallel
parallel -j "$JOBS" run_prodigal :::: "$GENOME_LIST"
