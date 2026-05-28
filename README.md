# adenylate_kinase_ogt

This is a repository for the code an analyses associated with the project "Predicting bacterial optimal growth temperatures using protein structural information." This project develops a model, ROSEATE, which performs **R**egression **O**n **S**ite-specific **E**mbeddings for **A**denylate **K**inase-Based **O**GT **E**stimation.

## A. Installation
Follow the instructions here:
https://git.embl.de/grp-kosinski/alphafold_howto/-/tree/main
Also install `ambertools` and `nglview` using the conda-forge channel. You can also install seaborn. If importing doesn't work, downgrading numpy to 1.23.5 fixed this problem for me.
## B. Repo
* `notebooks/` : current analysis code
* `scripts/` : helper scripts
* `data/` : project-relevant data
  * `metric_tables/` : structural metrics, genome to temperature table, etc.
  * `figures` : miscellaneous figures
  * `outputs`: The raw outputs of AlphaFold for the OGT dataset, compiled using script in Section E: Folding With ColabFold.
 
## C. Data flow

Diagram is a work in progress

## D. Notebook outline
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
   * Canonical (current, best, manuscript-ready (????) ) structural predictive model

# Compilation of structural features of ADK

## E. Folding with ColabFold

### Folding on HPC
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

## F. Current workflow 
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

## H. Need to figure out
1. Annotating domains
2. Efficient programmatic relaxation on server
3. Streamlining "best protein" selection code.
4. Iterative folding + database generation
5. Snakemake?
