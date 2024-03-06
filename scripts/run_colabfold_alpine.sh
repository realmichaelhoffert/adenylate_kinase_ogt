#!/bin/bash

#A typical runs takes couple of hours but may be much longer
#SBATCH --time=2-00:00:00
#SBATCH --job-name=colabfold

#log files:
#SBATCH -e AF_%x_%j_err.txt
#SBATCH -o AF_%x_%j_out.txt

#qos sets priority, you can set to high or highest but there is a limit of high priority jobs per user: https://wiki.embl.de/cluster/Slurm#QoS
#SBATCH --qos=normal

# The full list of nodes: https://wiki.embl.de/cluster/Hardware. 
#SBATCH -p gpu-el8

#Reserve the entire GPU so no-one else slows you down
#SBATCH --gres=gpu:1

#Limit the run to a single node
#SBATCH -N 1

#SBATCH --ntasks=8
#SBATCH --mem=32000

module load cuda/11.8
module load cudnn/8.6
module load mambaforge/23.1.0-1
mamba activate /projects/miho1832/miniforge3/envs/adenylate_kinase_ogt

# If you use --cpus-per-task=X and --ntasks=1 your script should contain:
# export ALPHAFOLD_JACKHMMER_N_CPU=$SLURM_CPUS_PER_TASK
# export ALPHAFOLD_HHBLITS_N_CPU=$SLURM_CPUS_PER_TASK

# TF_FORCE_UNIFIED_MEMORY='1' XLA_PYTHON_CLIENT_MEM_FRACTION are optional but may be necessary for bigger sequences.
# export TF_FORCE_UNIFIED_MEMORY='1'
# MAXRAM=$(echo `ulimit -m` '/ 1024.0'|bc)
# GPUMEM=`nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits|tail -1`
# export XLA_PYTHON_CLIENT_MEM_FRACTION=`echo "scale=3;$MAXRAM / $GPUMEM"|bc`

# echo 'MAXRAM:' $MAXRAM
# echo 'GPUMEM:' $GPUMEM
# echo 'XLA_PYTHON_CLIENT_MEM_FRACTION:' $XLA_PYTHON_CLIENT_MEM_FRACTION

# Add "--model-type AlphaFold2-ptm" option to run the old ColabFold for complexes, 
# equivalent to the original https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/beta/AlphaFold2_advanced.ipynb
colabfold_batch ./adk_sequences/${IND}.faa ./out_run2/${IND}_closed --templates --custom-template-path ./closed_templates/${IND} --overwrite-existing-results --amber --num-relax 3 --model-order 1,2 --random-seed 1040 --num-recycle 7 --save-recycles

colabfold_batch ./adk_sequences/${IND}.faa ./out_run2/${IND}_open --templates --custom-template-path ./open_templates/${IND} --overwrite-existing-results --amber --num-relax 3 --model-order 1,2 --random-seed 1040 --num-recycle 7 --save-recycles
