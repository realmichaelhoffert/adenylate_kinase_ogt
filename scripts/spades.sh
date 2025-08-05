#!/bin/bash

# # re-add isolate flag if running spades
# while read sample; do
#   echo "Sample: $sample"
#   echo "writing to ${sample}_spades"
#   spades.py -1 "${sample}_1.fastq" -2 "${sample}_2.fastq" -o "${sample}_spades"
# done < $1

set -euo pipefail

# Default values
THREADS=1
ISOLATE_FLAG=""
POSITIONAL_ARGS=()

show_help() {
  echo "Usage: $0 [--threads N] [--isolate] sample_list.txt"
  echo
  echo "Options:"
  echo "  --threads N    Number of samples to run in parallel (default: 1)"
  echo "  --isolate      Add --isolate flag to SPAdes command"
  echo "  -h, --help     Show this help message and exit"
}


# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --threads)
      THREADS="$2"
      shift 2
      ;;
    --isolate)
      ISOLATE_FLAG="--isolate"
      shift
      ;;
      -h|--help)
        show_help
        exit 0
        ;;
      -*|--*)
        echo "Unknown option $1"
        show_help
        exit 1
        ;;
    *)
      POSITIONAL_ARGS+=("$1")  # Save positional arg
      shift
      ;;
  esac
done

# Restore positional parameters
set -- "${POSITIONAL_ARGS[@]}"

# Check input
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 [--threads N] [--isolate] sample_list.txt"
  exit 1
fi

SAMPLE_LIST="$1"

# Function to run spades for a single sample
run_spades() {
  local sample="$1"
  echo "Sample: $sample"
  echo "writing to ${sample}_spades"
  # Locate forward and reverse reads using globbing
  fwd_file=$(ls ${sample}_*1*.f*q* 2>/dev/null | grep -E '_.*1[^/]*\.f.*q.*' | head -n 1)
  echo "fwd: $fwd_file"
  rev_file=$(ls ${sample}_*2*.f*q* 2>/dev/null | grep -E '_.*2[^/]*\.f.*q.*' | head -n 1)
  echo "rev: $rev_file"
    
  # Validate that exactly one of each was found
  if [[ -z "$fwd_file" || -z "$rev_file" ]]; then
    echo "❌ Error: Missing forward or reverse read for sample $sample" >&2
    exit 1
  fi
    
  if [[ $(ls ${sample}_*1*.f*q* 2>/dev/null | wc -l) -ne 1 ]] || [[ $(ls ${sample}_*2*.f*q* 2>/dev/null | wc -l) -ne 1 ]]; then
    echo "❌ Error: Multiple forward or reverse reads found for sample $sample" >&2
    exit 1
  fi
    
  # Run SPAdes with matched files
  echo "Running SPAdes for $sample"
  spades.py -1 "$fwd_file" -2 "$rev_file" -o "${sample}_spades" $ISOLATE_FLAG -t 4
}

export -f run_spades
export ISOLATE_FLAG  # for access in subprocess

# Run in parallel
cat "$SAMPLE_LIST" | parallel -j "$THREADS" run_spades