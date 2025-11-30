# files
PF05191 - hits for GTDB genomes to ADK lid domain Pfam




python ../protein_utils/mp_metrics.py ~/fiererlab/adenylate_kinase_ogt/data/processed_structures/ ./20240307_python_metrics.tsv
# afas
# make protein fasta of all adks:
cat adks/*.faa > 20240122_all_adks.faa
# outgroup ADKs from interproscan:
high_qual_alignments/protein-matching-IPR023477.fasta
high_qual_alignments/protein-matching-TIGR01360.fasta

# initial mafft-based alignment to filter out highly gapped sequences:
cd high_qual_alignments
cp ./../20240122_all_adks.faa .
cat *.f*a > full_adk_dataset.faa

Not sure how the filtered file was made (whoops!) but that's the one that I'm using for proclam / ESM2 analyses
Alignment with mafft, trimmed with clipkit:
`clipkit 20240122_all_adks_filtered.afa -m gappy -o 20240122_all_adks_filtered_nogap.afa`
Includes only adks in the filtered set

# PFAM
PF00406.hmm was retrieved from a copy of the PFam database from 2023

# 
