# Extract adenylate kinase (ADK) proteins from genomes or assemblies
# Written by Michael Hoffert, refactored with Claude Opus 5.5
# October 2026
#
# Workflow (iso: "Step 1. annotate genomes"):
#   run_annotation  -> {genome}_proteins.faa and {genome}_hmmsearch.tbl next to each genome
#   process_hmms    -> {genome}_adk.faa with the best ADK hit(s) per genome
#   concat_adks     -> adks.faa, the input to align.add_to_backbone

import glob
import os
import subprocess
from pathlib import Path

import pandas as pd
from Bio import SeqIO

ANNOTATION_SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'run_esm_pipeline.sh'

pfam_table_columns = ['target_name', 'target_accession', 'query_name', 'query_accession',
                      'full_e_value', 'full_score', 'full_bias',
                      'best_domain_e_value', 'best_domain_score', 'best_domain_bias',
                      'exp', 'reg', 'clu', 'ov', 'env', 'dom', 'rep', 'inc', 'description_of_target']


def run_annotation(genome_list, hmm_model, jobs=4, script=ANNOTATION_SCRIPT):
    '''
    Gene calling (prodigal -p meta) and hmmsearch for every genome in genome_list,
    one genome FASTA path per line (.fa, .fasta, optionally .gz).
    Runs scripts/run_esm_pipeline.sh unchanged; needs seqtk, prodigal, hmmer and GNU parallel.
    Outputs are written next to each genome. Returns the hmmsearch tables.
    Success is judged by the outputs, not the exit code.
    '''
    result = subprocess.run(['bash', str(script), '--genome-list', str(genome_list),
                             '--hmm-model', str(hmm_model), '--jobs', str(jobs)],
                            capture_output=True, text=True)
    tables = []
    for genome in Path(genome_list).read_text().split():
        base = genome
        for ext in ['.fasta', '.fa', '.fasta.gz', '.fa.gz']:   # same order as the script
            base = base[:-len(ext)] if base.endswith(ext) else base
        expected = [f'{base}_proteins.faa', f'{base}_hmmsearch.tbl']
        if not all(os.path.exists(f) for f in expected):
            raise RuntimeError(f'annotation failed for {genome}:\n{result.stderr[-2000:]}')
        tables.append(expected[1])
    return tables


def read_pfam_table_pandas(file_path):
    '''
    Read an hmmsearch --tblout table. Raises ValueError (pandas EmptyDataError) if it has no hits.
    '''
    table = pd.read_table(file_path, skiprows=3, skipfooter=10, sep=r'\s+', header=None, engine='python')
    # the free-text description is split on whitespace into columns 18+; join it back
    description = table.iloc[:, 18:].apply(lambda row: ''.join([str(v).lstrip() for v in row.values]), axis=1)
    table = table.iloc[:, :18].assign(description_of_target=description.values)
    table.columns = pfam_table_columns
    return table


def process_hmms(loc, protein_name='adk', n_proteins=1, min_score=100, write_file=True):
    '''
    For every *.tbl under loc, keep the n_proteins best hits with full_score >= min_score and
    write them from {genome}_proteins.faa to {genome}_adk.faa (existing files are not overwritten).
    Headers are 'protein|genome|XXXtemp=XXX|bitscore=score'. Kept from v1:
      - genome is the .tbl file name up to '_PF', i.e. usually '{genome}_hmmsearch.tbl'
      - 'XXXtemp=XXX' (a missing comma in v1, quirk 7)
    Returns the list of _adk.faa files written.
    '''
    hmm_files = sorted(glob.glob(os.path.join(loc, '**/*.tbl'), recursive=True))
    print(f'Found {len(hmm_files)} tables to parse')

    written = []
    for count, file in enumerate(hmm_files):
        print(f'{count+1} tables processed of {len(hmm_files)}', end='\r')
        try:
            df = read_pfam_table_pandas(file)
        except ValueError:
            print(f'Table contained no records:\n{file}')
            continue
            
        genome = os.path.basename(file).removesuffix('_hmmsearch.tbl').split('_PF')[0]
        assert '|' not in genome, "Genome ids cannot contain (|) (and really shouldn't contain special chars other than _ and .)"
        top = df[df['full_score'] >= min_score].sort_values('full_score', ascending=False).iloc[:n_proteins]
        gene2score = dict(zip(top['target_name'], top['full_score']))
        if not write_file:
            continue

        with open(file.replace('_hmmsearch.tbl', '_proteins.faa')) as handle:
            records = [r for r in SeqIO.parse(handle, 'fasta') if r.id in gene2score]
        assert len(records) == len(gene2score), f'{file}: hits missing from the protein file'

        for r in records:
            r.seq = r.seq.replace('*', '')
            new_id = '|'.join([r.id.split(' # ')[0], genome, protein_name, f'bitscore={gene2score[r.id]}'])
            r.id = r.description = new_id
        outfile = file.replace('_hmmsearch.tbl', f'_{protein_name}.faa')
        if not os.path.exists(outfile):
            with open(outfile, 'w') as out_handle:
                SeqIO.write(records, out_handle, 'fasta')
            written.append(outfile)
    return written


def concat_adks(loc, out='adks.faa', protein_name='adk'):
    '''
    Concatenate {genome}_adk.faa files in loc into one FASTA (iso: 'cat ./genomes/*_adk.faa > adks.faa').
    Files are joined in sorted name order.
    '''
    with open(out, 'wb') as out_handle:
        print('files found', glob.glob(os.path.join(loc, f'*_{protein_name}.faa')))
        for f in sorted(glob.glob(os.path.join(loc, f'*_{protein_name}.faa'))):
            with open(f, 'rb') as handle:
                out_handle.write(handle.read())
    return out
