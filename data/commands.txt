# files
PF05191 - hits for GTDB genomes to ADK lid domain Pfam

python ../protein_utils/mp_metrics.py ~/fiererlab/adenylate_kinase_ogt/data/processed_structures/ ./20240307_python_metrics.tsv
# afas
# make protein fasta of all adks:
cat adks/*.faa > 20240122_all_adks.faa
20240122_all_adks_filtered.faa: produced in 02_filtering by subsetting original to structurally-filtered adks

# outgroup ADKs from interproscan:
high_qual_alignments/protein-matching-IPR023477.fasta
high_qual_alignments/protein-matching-TIGR01360.fasta

# initial mafft-based alignment to filter out highly gapped sequences:
cd high_qual_alignments
cp ./../20240122_all_adks.faa .

# new:
cat 20240122_all_adks.faa protein-matching-TIGR01351.fasta > full_adk_dataset.faa
grep "^>" full_adk_dataset.faa | wc -l
>>> 10052
# which is 9351 (adks) + 701 (outgroup)

# old: included protein-matching-IPR023477.fasta and protein-matching-TIGR01360.fasta
# but NOT protein-matching-TIGR01351.fasta
cat *.f*a > full_adk_dataset.faa

# dereplicating with CD-HIT
cd-hit -i full_adk_dataset.faa -o full_adk_reps.faa -c 0.95 -n 5 -M 0 -T 0
grep "^>" full_adk_reps.faa | wc -l
>>> 7539

# initial clustering in mafft
mafft --localpair --thread 24 --maxiterate 1000 full_adk_reps.faa > full_adk_reps_initial.afa

# did some filtering in 02_filtering notebook

# second / third alignment
mafft --thread 24 --localpair --maxiterate 1000 full_adk_reps.filtered.faa > full_adk_reps.filtered.afa

# trimming
trimal -in full_adk_reps.filtered.afa -out full_adk_reps.filtered.trimmed.afa -gappyout

# ACTUAL BACKBONE, FILTERED (N = 8532):
# every sequence has a match in the Topt dataset
mafft --keeplength --add extra_adk_seqs.faa full_adk_reps.filtered.trimmed.afa > backbone.afa
grep "^>" backbone.filtered_only.afa | wc -l
>>> 8532

# file descriptions
# march 2026: need to update
20240122_all_adks.faa: copied from ./.., all adks from genomes in temp dataset.
full_adk_dataset.faa: 20240122_all_adks.faa + euk / archaea adks
full_adk_reps.faa: full_adk_dataset.faa dereplicated with CD HIT (didn't make much of a difference)
full_adk_reps_initial.afa: initial mafft alignment. Data from this alignment collected in 02_filtering was used to filter sequences.
full_adk_reps.filtered.faa: Filtered set of representatives based on initial alignment + filtering.
full_adk_reps.filtered.afa: "Backbone" alignment: used to insert original sequences; untrimmed
full_adk_reps.filtered.trimmed.afa: "Backbone" alignment: used to insert original sequences; trimmed with trimal
high_qual_alignments/extra_adk_seqs.faa: see end of 02_filtering for what this file means

### Other workflows

## Isolates
# When adding isolates to alignment, use "adks"

## Metagenomics
# dereplicate metagenomic/genomic protein fasta for ADKs from a large dataset:
mmseqs easy-linclust input.fasta res tmp --min-seq-id 1.0 -c 1.0 --cov-mode 1

# filtering bacteria from other ADKs:
# align
mafft --localpair --thread 12 --maxiterate 1000 protein-matching-TIGR01351.fasta > protein-matching-TIGR01351.afa
# trim
clipkit protein-matching-TIGR01351.afa -m smart-gap
# make tree
iqtree2 -s protein-matching-TIGR01351.afa.clipkit -mset LG,WAG,JTT,Blosum62 -bb 1000 -nt AUTO

## PFAM
PF00406.hmm was retrieved from a copy of the PFam database from 2023

## Running TOMES:
cp *_protein.faa proteomes/
for i in *.faa; do mv "$i" "${i/faa/fasta}"; done
tome predOGT --indir ./proteomes/ --out tome_predictions.tsv -p 16

# Retraining TOME:
tome predOGT --train
# uses Tome/tome/data/train.csv

# when retrained on the 1227 even genomes:
A new model has beed successfully trained.
Model performance:
        RMSE: 2.532157280011231
          r2: 0.9675105740554271
  Pearson r:PearsonRResult(statistic=0.9839501188400385, pvalue=0.0)
  Spearman r:SignificanceResult(statistic=0.9714500713004794, pvalue=0.0)

Saving the new model to replace the original one...
Done!


# run for corkrey data, trained with original genomes:
tome predOGT --indir ./corkrey_genomes/ --out corkrey_tome_predictions_OGtraining.tsv -p 16
# run for corkrey data, trained with even genomes:
tome predOGT --indir ./corkrey_genomes/ --out corkrey_tome_predictions_eventraining.tsv -p 16

## Old Code

Not sure how the filtered file was made (whoops!) but that's the one that I'm using for proclam / ESM2 analyses
Alignment with mafft, trimmed with clipkit:
`clipkit 20240122_all_adks_filtered.afa -m gappy -o 20240122_all_adks_filtered_nogap.afa`
Includes only adks in the filtered set

# ACTUAL BACKBONE:
# mafft --keeplength --add extra_adk_seqs.faa full_adk_reps.filtered.trimmed.afa > backbone.afa
# grep "^>" backbone.afa | wc -l
# >>> 8842
