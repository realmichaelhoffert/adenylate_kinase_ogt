# adenylate_kinase_ogt

This is a repository for the code an analyses associated with the project "Predicting bacterial optimal growth temperatures using learned structural information from a single protein." This project develops a model, ROSEATE, which performs **R**egression **O**n **S**ite-specific **E**mbeddings for **A**denylate **K**inase-Based OG**T** **E**stimation to predict the optimal growth temperature (OGT) of bacteria. It's name is based on our use of the Rosetta, the macromolecular modeling suite, my love of birds, and because acronyms are fun.

## A. Installation

### Option 1: Only run ROSEATE

Install `adenylate_kinase_ogt.yaml` environment. This environment contains *all* packages required for the analyses from the manuscript, as well as those required to predict for new proteins using ROSEATE.

```bash
mamba env create -f adenylate_kinase_ogt.yaml
mamba activate adenylate_kinase_ogt
```

### Option 2: Do protein folding.

The environment to fold proteins was constructed using the instructions here:
https://git.embl.de/grp-kosinski/alphafold_howto/-/tree/main
Also install `ambertools` and `nglview` using the conda-forge channel. You can also install seaborn. If importing doesn't work, downgrading numpy to 1.23.5 fixed this problem for me.

## V. Run ROSEATE on new proteins

### Annotate your proteins
I have a helper script, `scripts/run_esm_pipeline.sh`, which can take a list of fastas (nucleotide assemblies or genomes) and extract ADKs from them using PFAM00406 (ADK).  

```bash
# from repo root, will run 8 jobs simultaneously
# if genomes, change line 93 to isolate mode, currently meta mode
# annotates into same dir
bash scripts/run_esm_pipeline.sh --genome-list files.txt --hmm-model data/PF00406.hmm --jobs 8

# list of genome should be absolute paths
> head -5 genome_list.txt
SRR32180441.fasta
SRR32180442.fasta
SRR32180443.fasta
SRR32180444.fasta
SRR32180445.fasta
```
  
Then, use another helper script to compile a file of the adks in each sample / genome:  
```bash
# from repo root
# extract up to 1000 adks per sample / genome, minimum bitscore of 100
python src/protein_utils/extract_adks.py --loc [dir where the assemblies were] --n-proteins 1000 --min-score 100
```
I recommend using a bitscore of at least 100, because PF00406 can match other nucleoside kinases. This script will write an ADKs file for each sample / genome, and then you can combine:  

```bash
cat *_adk.faa > my_samples_all_adks.faa
```  
### Prep to run ROSEATE

Insert the ADKs into ROSEATE's backbone alignment:
```bash
mafft --keeplength --add my_samples_all_adks.faa data/high_qual_alignments/backbone.filtered_only.afa > backbone_plus_my_adks.afa
```

### Embed
Note: This code is based on `Generate embeddings for each dataset` heading in `notebooks/05_esm_model_v2.ipynb` - I'm working on scripting it as fast as I can!

```python

# imports
import sys
sys.path.append('/data/mhoffert/fiererlab/adenylate_kinase_ogt/')
from src.models.utils import construct_msa_dataset, scale_ogts, unscale_ogts, reload_model, compute_gaussian_weights
from src.models.utils import plot_performance
import esm
import pickle

# load ESM model
esm_msa_model, alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
conv = alphabet.get_batch_converter()

# turn off gradients
for name, param in esm_msa_model.named_parameters():
    param.requires_grad = False

# load training dataset
# get dataset
full_msa_dataset = construct_msa_dataset('/data/high_qual_alignments/backbone.filtered_only.afa')
# scale ogts
full_msa_dataset, scaler = scale_ogts(full_msa_dataset)
# should be 8532: number of ADKs we used
print('Number of observations:', len(full_msa_dataset))
# load "Even" dataset: data actually used to train ROSEATE
with open('data/predictive_models/supporting_data/even_msa_dataset.pickle', 'rb') as handle:
    even_msa_dataset = pickle.load(handle)

test_data_afa = 'path/to/backbone_plus_my_adks.afa'
test_dataset = construct_msa_dataset(test_data_afa, ogt_parse_fn=lambda _str: _str)
print('Before removing training:',len(test_dataset))
# remove backbone sequences
full_msa_keys = set([i[0] for i in full_msa_dataset])
test_dataset = [i for i in test_dataset if not i[0] in full_msa_keys]
print('After removing training:',len(test_dataset))


dataset_name = 'your_dataset_name'
# eliminate nans
test_dataset = [i for i in test_dataset if not np.isnan(i[2])]
# scale ogts
test_dataset, _ = scale_ogts(test_dataset, scaler)

emb_base = f'/path/to/save/embeddings/'
os.makedirs(emb_base, exist_ok=True)
    
with open(f'{emb_base}{dataset_name}_msa_dataset.pickle', 'wb') as handle:
    pickle.dump(test_dataset, handle)

# because the "context" - sequences used to embed a query - can effect
# the embeddings, this code is used to generate a trackable context
# use the same random context for every embedding
random_context = get_random_context(msa_dataset=even_msa_dataset,
                                    n_contexts=len(test_dataset),
                                    context_size=8, 
                                    self=False)
            
test_emb, test_temps, _ = get_embeddings(esm_msa_model, 
                                      conv, 
                                      test_dataset, 
                                      even_msa_dataset,
                                      random_context,
                                      return_mean=False)
            
with open(f'{emb_base}{dataset_name}_{label}_even.pickle', 'wb') as handle:
    pickle.dump(test_emb, handle)
with open(f'{emb_base}{dataset_name}_temps.pickle', 'wb') as handle:
    pickle.dump(test_temps, handle)
                    
with open(f'{emb_base}{dataset_name}_even_context_values.pickle', 'wb') as handle:
        pickle.dump(random_context, handle)
```

### Run predictions
Finally, embed your alignment and run ROSEATE:

```python

# additional imports
from src.models.predict import run_model, per_site_predict
from src.models.utils import assess_model

scaler = reload_model('predictive_models/supporting_data/ogt_scaler.pickle')
model = f'data/predictive_models/even_enetcv_ind.pickle'
model_label = 'even_enetcv_ind'


preds = run_model(embedding_loc=f'{emb_base}{dataset_name}_{label}_even.pickle',
         model_loc=model,
         msa_loc=f'{base}{dataset}/embeddings/{dataset}_msa_dataset.pickle',
         predict_fn=per_site_predict)

msa_dataset = reload_model(f'{base}{dataset}/embeddings/{dataset}_msa_dataset.pickle')
y_pred = pd.Series(data=unscale_ogts(preds.mean(axis=0), scaler),
                   index=[i[0] for i in msa_dataset])
```


## C. Repo
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
