python ../protein_utils/mp_metrics.py ~/fiererlab/adenylate_kinase_ogt/data/processed_structures/ ./20240307_python_metrics.tsv
# afas
Not sure how the filtered file was made (whoops!) but that's the one that I'm using for proclam / ESM2 analyses
Alignment with mafft, trimmed with clipkit:
`clipkit 20240122_all_adks_filtered.afa -m gappy -o 20240122_all_adks_filtered_nogap.afa`
Includes only adks in the filtered set

# PFAM
PF00406.hmm was retrieved from a copy of the PFam database from 2023

# 
