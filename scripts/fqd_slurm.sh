#!/bin/bash
#SBATCH --job-name=fasterq_array
#SBATCH --output=logs/fasterq_%A_%a.out
#SBATCH --error=logs/fasterq_%A_%a.err
#SBATCH --partition=amilan
#SBATCH --array=0-<MAX>    # REPLACE <MAX> below with the number of accessions minus 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4  # adjust if needed

ACCESSION_FILE=$1

# Get the accession corresponding to this array task
ACCESSION=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$ACCESSION_FILE")

# Run fasterq-dump
~/tools/sratoolkit.2.11.3-centos_linux64/bin/fasterq-dump.2.11.3 "$ACCESSION" -O ./ -S
