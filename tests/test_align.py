"""
Tests for roseate.align.

Synthetic tests need mafft. Archive tests check that the archived alignments contain backbone.afa
unchanged (quirk 12); the full re-alignment test also needs the original adks.faa files:
    ROSEATE_DATA=/path/to/repo/data pytest tests/test_align.py -m slow
"""
import os
import shutil
from pathlib import Path

import pytest
from Bio import SeqIO

from roseate import align
from tests.test_initial_predict import REF, path

ALIGNMENTS = sorted(Path(p).name.replace('_adk_to_backbone.afa', '') for p in
                    (REF / 'inputs/high_qual_alignments/alignments_to_backbone').glob('*_adk_to_backbone.afa'))
needs_mafft = pytest.mark.skipif(not shutil.which('mafft'), reason='needs mafft')


def records(fasta):
    return {r.id: str(r.seq) for r in SeqIO.parse(fasta, 'fasta')}


@pytest.fixture
def toy(tmp_path):
    '''Three-sequence backbone and two new sequences, one with an insertion.'''
    backbone = tmp_path / 'backbone.afa'
    backbone.write_text('>b1\nMKIILLG-PGAGKGT\n>b2\nMRIILLGAPGAGKGT\n>b3\nMKLVLLG-PGSGKGT\n')
    adks = tmp_path / 'adks.faa'
    adks.write_text('>n1|g1|XXXtemp=XXX|bitscore=150.0\nMKIVLLGPPGAGKGT\n'
                    '>n2|g2|XXXtemp=XXX|bitscore=140.0\nMKIILLGWWWPGAGKGT\n')
    return backbone, adks


# ---------------------------------------------------------------- synthetic
@needs_mafft
def test_add_to_backbone_keeps_backbone_and_length(toy, tmp_path):
    backbone, adks = toy
    out = align.add_to_backbone(adks, backbone, tmp_path / 'out.afa')
    aligned, original = records(out), records(backbone)
    assert {k: aligned[k] for k in original} == original          # backbone rows unchanged
    assert {len(s) for s in aligned.values()} == {15}             # --keeplength: no new columns
    assert align.added_ids(out, backbone) == ['n1|g1|XXXtemp=XXX|bitscore=150.0',
                                              'n2|g2|XXXtemp=XXX|bitscore=140.0']

@needs_mafft
def test_insertion_is_dropped(toy, tmp_path):
    backbone, adks = toy
    out = align.add_to_backbone(adks, backbone, tmp_path / 'out.afa')
    assert 'WWW' not in records(out)['n2|g2|XXXtemp=XXX|bitscore=140.0']

def test_missing_mafft_raises(toy, tmp_path):
    backbone, adks = toy
    with pytest.raises(FileNotFoundError):
        align.add_to_backbone(adks, backbone, tmp_path / 'out.afa', mafft='mafft-not-installed')


# ---------------------------------------------------------------- archive
@pytest.mark.parametrize('name', ALIGNMENTS)
def test_archived_alignment_contains_full_backbone(name):
    aligned = records(path(f'inputs/high_qual_alignments/alignments_to_backbone/{name}_adk_to_backbone.afa'))
    backbone = records(path('inputs/high_qual_alignments/backbone.filtered_only.afa'))
    assert {k: aligned.get(k) for k in backbone} == backbone

@pytest.mark.slow
@needs_mafft
@pytest.mark.parametrize('name', ALIGNMENTS[:1])
def test_realignment_reproduces_archive(name, tmp_path):
    adks = Path(os.environ.get('ROSEATE_DATA', 'data')) / 'validation_datasets' / name / 'adks.faa'
    if not adks.exists():
        pytest.skip(f'{adks} not found; set ROSEATE_DATA')
    out = align.add_to_backbone(adks, path('inputs/high_qual_alignments/backbone.filtered_only.afa'), tmp_path / 'out.afa')
    archived = path(f'inputs/high_qual_alignments/alignments_to_backbone/{name}_adk_to_backbone.afa')
    assert records(out) == records(archived)
