#!/bin/bash
#SBATCH --job-name=fasterq_array
#SBATCH --output=logs/fasterq_%A_%a.out
#SBATCH --error=logs/fasterq_%A_%a.err
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --time=01:00:00
#SBATCH --array=0-<MAX>    # REPLACE <MAX> below with the number of accessions minus 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4  # adjust if needed

module load sra-toolkit

ACCESSION_FILE=$1

# Get the accession corresponding to this array task
ACCESSION=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$ACCESSION_FILE")

# Run fasterq-dump
fasterq-dump "$ACCESSION" -O ./ -S
