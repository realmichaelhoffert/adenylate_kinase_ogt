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

FASTA_SUFFIXES = ('.fa', '.fna', '.fasta', '.fa.gz', '.fna.gz', '.fasta.gz', '.faa', '.faa.gz')


def genome_name(path):
    '''Sample name of a genome file: its name without .gz and .fasta/.fna/.fa, as in the annotation script.'''
    name = Path(path).name.removesuffix('.gz')
    for ext in ('.fasta', '.fna', '.fa', '.faa'):
        name = name.removesuffix(ext)
    return name


def find_genomes(inputs):
    '''
    Genome FASTA files from files and folders (a folder means every FASTA file directly in it).
    Checks that every input exists and that sample names (genome_name) are unique.
    Returns absolute paths.
    '''
    genomes = []
    for item in map(Path, inputs):
        if item.is_dir():
            genomes += sorted(p for p in item.iterdir() if p.name.endswith(FASTA_SUFFIXES))
        elif item.is_file():
            genomes.append(item)
        else:
            raise FileNotFoundError(f'{item} not found')
    if not genomes:
        raise FileNotFoundError(f'no FASTA files ({", ".join(FASTA_SUFFIXES)}) found in {", ".join(map(str, inputs))}')
    names = [genome_name(g) for g in genomes]
    repeated = sorted({n for n in names if names.count(n) > 1})
    if repeated:
        raise ValueError(f'genome file names must be unique after removing extensions; repeated: {repeated[:5]}')
    return [g.resolve() for g in genomes]

def search_proteins(fastas, hmm_model, out_dir, jobs=4):
    '''
    hmmsearch on protein FASTA files (genes already called), one file per genome or sample.
    Each file is linked (or uncompressed) to out_dir as {genome_name}_proteins.faa (decompressed if .gz) and searched,
    giving the same files as run_annotation, so process_hmms and concat_adks work unchanged.
    Returns: a list of the hmmsearch tables.
    '''
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = []
    for fasta in fastas:
        base = out_dir / genome_name(fasta)
        proteins, table = Path(f'{base}_proteins.faa'), Path(f'{base}_hmmsearch.tbl')
        if proteins.exists() or proteins.is_symlink():      # from an earlier run, e.g. with --overwrite
            proteins.unlink()
        if str(fasta).endswith('.gz'):                      # gzipped: decompress into the run folder
            with gzip.open(fasta, 'rt') as source, open(proteins, 'w') as copy:
                shutil.copyfileobj(source, copy)
        else:                                               # plain: link to the original, no copy
            proteins.symlink_to(Path(fasta).resolve())
        subprocess.run(['hmmsearch', '--cpu', str(jobs), '--tblout', str(table), str(hmm_model), str(proteins)],
                       stdout=subprocess.DEVNULL, check=True)
        tables.append(table)
    return tables

# Todo: add metagenome / isolate flag
def run_annotation(genomes, hmm_model, out_dir, mode='meta', jobs=4, script=ANNOTATION_SCRIPT):
    '''
    Gene calling (prodigal -p {mode}) and hmmsearch for every genome in genome_list,
    one genome FASTA path per line (.fa, .fasta, optionally .gz).
    Runs scripts/run_esm_pipeline.sh unchanged; needs seqtk, prodigal, hmmer and GNU parallel.
    Outputs are written next to each genome. 
    Returns: a list of the hmmsearch tables.
    '''
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    genome_list = out_dir / 'genome_list.txt'
    genome_list.write_text(''.join(f'{g}\n' for g in genomes))
    result = subprocess.run(['bash', str(script), '--genome-list', str(genome_list),
                             '--hmm-model', str(hmm_model), '--jobs', str(jobs), 
                             '--mode', str(mode), '--outdir', str(out_dir)],
                            capture_output=True, text=True)
    tables = []
    for genome in genomes:
        base = out_dir / genome_name(genome)
        expected = [f'{base}_proteins.faa', f'{base}_hmmsearch.tbl']
        if not all(os.path.exists(f) for f in expected):
            raise RuntimeError(f'annotation failed for {genome}:\n{result.stderr[-2000:]}')
        tables.append(expected[1])
    return tables

def check_proteins(faa, hmm_model, min_score=100, out_dir='.'):
    '''
    Checks a set of input proteins by running them against a provided HMM model
    avoids incorrectly annotated proteins
    faa: str or path? path to protein fasta
    hmm_model: str or path to hmm model
    min_score: minimum score for check (recommend something like PFAM trusted cutoff)
    '''

    table = Path(out_dir) / 'hmmsearch.tbl'
    subprocess.run(['hmmsearch', '--tblout', str(table), str(hmm_model), str(faa)],
                   stdout=subprocess.DEVNULL, check=True)
    ids = [r.id for r in SeqIO.parse(faa, 'fasta')]
    try:
        scores = read_pfam_table_pandas(table).set_index('target_name')['full_score']
    except ValueError:                       # no hits at all
        scores = pd.Series(dtype=float)
    report = pd.DataFrame({'id': ids, 'bitscore': scores.reindex(ids).values})
    report['passed'] = report['bitscore'] >= min_score
    return report

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


def process_hmms(loc, protein_name='adk', n_proteins=1, min_score=100, write_file=True, overwrite=False):
    '''
    For every *.tbl under loc, keep the n_proteins best hits with full_score >= min_score and
    write them from {genome}_proteins.faa to {genome}_adk.faa (existing files are kept unless overwrite=True).
    Headers are 'protein|genome|bitscore=score'.
    Returns the list of _{protein_name}.faa files written.
    '''
    hmm_files = sorted(glob.glob(os.path.join(loc, '**/*.tbl'), recursive=True))
    print(f'Found {len(hmm_files)} tables to parse')

    written = []
    for count, file in enumerate(hmm_files):
        # print(f'{count+1} tables processed of {len(hmm_files)}', end='\r')
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
        if overwrite or not os.path.exists(outfile):
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
        for f in sorted(glob.glob(os.path.join(loc, f'*_{protein_name}.faa'))):
            with open(f, 'rb') as handle:
                data = handle.read()
                out_handle.write(data)
    return out
