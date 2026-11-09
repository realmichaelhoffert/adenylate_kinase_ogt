"""
Tests for roseate.extract.

Unit tests use small fixtures in tests/fixtures/extract (two real ADKs from the backbone and a decoy,
with real hmmsearch output). The end-to-end test needs seqtk, prodigal, hmmer and GNU parallel.
The archive test checks the header format of ADKs added to the archived alignments.
"""
import re
import shutil
from pathlib import Path

import pytest
from Bio import SeqIO

from roseate import extract
from tests.test_initial_predict import path

DATA = Path(__file__).parent / 'fixtures' / 'extract'
HMM = Path(__file__).parents[1] / 'data' / 'PF00406.hmm'
# HEADER = re.compile(r'^[^|]+\|[^|]+\|XXXtemp=XXX\|bitscore=[0-9.]+$')   # v1 format, quirk 7
V1_HEADER = re.compile(r'^[^|]+\|[^|]+\|XXXtemp=XXX\|bitscore=[0-9.]+$')
HEADER = re.compile(r'^[^|]+\|[^|]+\|adk\|bitscore=[0-9.]+$')


@pytest.fixture
def genome_dir(tmp_path):
    '''Copy of the fixtures, so tests can write outputs.'''
    for f in ['genomeA_proteins.faa', 'genomeA_hmmsearch.tbl', 'empty_hmmsearch.tbl']:
        shutil.copy(DATA / f, tmp_path / f)
    return tmp_path


def read(faa):
    return list(SeqIO.parse(faa, 'fasta'))


# ---------------------------------------------------------------- tables
def test_read_pfam_table():
    table = extract.read_pfam_table_pandas(DATA / 'genomeA_hmmsearch.tbl')
    assert list(table.columns) == extract.pfam_table_columns
    assert list(table['target_name']) == ['contig_2_1', 'contig_1_1']   # hmmsearch sorts by score
    assert table['full_score'].dtype.kind == 'f'

def test_empty_table_raises_value_error():
    with pytest.raises(ValueError):
        extract.read_pfam_table_pandas(DATA / 'empty_hmmsearch.tbl')


# ---------------------------------------------------------------- process_hmms
def test_best_hit_written_with_header(genome_dir):
    written = extract.process_hmms(genome_dir, n_proteins=1, min_score=100)
    assert [Path(w).name for w in written] == ['genomeA_adk.faa']
    records = read(genome_dir / 'genomeA_adk.faa')
    assert [r.id for r in records] == ['contig_2_1|genomeA_hmmsearch.tbl|bitscore=192.6']
    assert '*' not in str(records[0].seq)

def test_n_proteins_keeps_hits_in_protein_file_order(genome_dir):
    extract.process_hmms(genome_dir, n_proteins=2, min_score=100)
    ids = [r.id.split('|')[0] for r in read(genome_dir / 'genomeA_adk.faa')]
    assert ids == ['contig_1_1', 'contig_2_1']

def test_min_score_filters_hits(genome_dir):
    extract.process_hmms(genome_dir, n_proteins=2, min_score=180)
    assert [r.id.split('|')[0] for r in read(genome_dir / 'genomeA_adk.faa')] == ['contig_2_1']

def test_existing_output_is_not_overwritten(genome_dir):
    (genome_dir / 'genomeA_adk.faa').write_text('>keep\nM\n')
    assert extract.process_hmms(genome_dir) == []
    assert [r.id for r in read(genome_dir / 'genomeA_adk.faa')] == ['keep']

def test_write_file_false_writes_nothing(genome_dir):
    extract.process_hmms(genome_dir, write_file=False)
    assert not (genome_dir / 'genomeA_adk.faa').exists()


# ---------------------------------------------------------------- concat_adks
def test_concat_adks_sorted_order(tmp_path):
    (tmp_path / 'b_adk.faa').write_text('>b\nMB\n')
    (tmp_path / 'a_adk.faa').write_text('>a\nMA\n')
    (tmp_path / 'ignored.faa').write_text('>x\nMX\n')
    out = extract.concat_adks(tmp_path, tmp_path / 'adks.faa')
    assert [r.id for r in read(out)] == ['a', 'b']


# ---------------------------------------------------------------- end to end
@pytest.mark.skipif(not all(shutil.which(t) for t in ['seqtk', 'prodigal', 'hmmsearch', 'parallel']),
                    reason='needs seqtk, prodigal, hmmer and GNU parallel')
def test_annotation_to_adks_faa(tmp_path):
    genome = tmp_path / 'genomeB.fasta'
    shutil.copy(DATA / 'genomeB.fasta', genome)
    (tmp_path / 'genome_list.txt').write_text(f'{genome}\n')

    tables = extract.run_annotation(tmp_path / 'genome_list.txt', HMM, jobs=1)
    print("TABLES:", tables)
    written = extract.process_hmms(tmp_path)
    print('WRITTEN:', written)
    adks = read(extract.concat_adks(tmp_path, tmp_path / 'adks.faa'))
    assert len(adks) == 1 and HEADER.match(adks[0].id)
    assert 'genomeB' == adks[0].id.split('|')[1]
    assert str(adks[0].seq) == str(read(DATA / 'genomeA_proteins.faa')[0].seq).rstrip('*')


# ---------------------------------------------------------------- archive
def test_archived_new_data_headers_match_v1_format():
    aligned = path('inputs/high_qual_alignments/alignments_to_backbone/Arthrobacter_adk_to_backbone.afa')
    backbone_ids = {r.id for r in SeqIO.parse(path('inputs/high_qual_alignments/backbone.afa'), 'fasta')}
    added = [r.id for r in SeqIO.parse(aligned, 'fasta') if r.id not in backbone_ids]
    assert added and all(V1_HEADER.match(i) for i in added), added[:3]
