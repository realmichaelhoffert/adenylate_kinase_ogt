# adenylate_kinase_ogt

## Installation
Follow the instructions here:
https://git.embl.de/grp-kosinski/alphafold_howto/-/tree/main
Also install `ambertools` and `nglview` using the conda-forge channel. You can also install seaborn. If importing doesn't work, downgrading numpy to 1.23.5 fixed this problem for me.
## Repo
* `notebooks/` : current analysis code
* `scripts/` : helper scripts

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
