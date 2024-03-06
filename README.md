# adenylate_kinase_ogt

## Installation
Follow the instructions here:
https://git.embl.de/grp-kosinski/alphafold_howto/-/tree/main
Also install `ambertools` and `nglview` using the conda-forge channel. You can also install seaborn. If importing doesn't work, downgrading numpy to 1.23.5 fixed this problem for me.
## Repo
* `notebooks/` : current analysis code
* `scripts/` : helper scripts

## Server connection and data running
Log in
```
ssh miho1832@login.rc.colorado.edu
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
Move outputs to microbe from alpine (run on microbe)
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
Run code in python notebooks


## Current workflow 
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

## Improved workflow
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

## Need to figure out
1. Annotating domains
2. Efficient programmatic relaxation on server
3. Streamlining "best protein" selection code.
4. Iterative folding + database generation
5. Snakemake?
