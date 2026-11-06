# Align new ADK sequences to the ROSEATE backbone alignment
# Written by Michael Hoffert, refactored with Claude Opus 5.5
# October 2026

import subprocess
from pathlib import Path

from Bio import SeqIO


def add_to_backbone(adks_faa, backbone, out, mafft='mafft'):
    '''
    Add unaligned ADKs to the backbone without changing its columns (mg: --keeplength):
        mafft --keeplength --add {adks_faa} {backbone} > {out}
    Residues that would need new columns are dropped, so every output row has the backbone length.
    The output holds the backbone rows followed by the new ones.
    v1 used backbone.afa for new data, not backbone.filtered_only.afa (quirk 12).
    '''
    with open(out, 'w') as handle:
        subprocess.run([mafft, '--keeplength', '--add', str(adks_faa), str(backbone)],
                       stdout=handle, stderr=subprocess.PIPE, check=True)
    return Path(out)


def added_ids(aligned, backbone):
    '''IDs in an add_to_backbone output that are not in the backbone, in output order.'''
    backbone_ids = {r.id for r in SeqIO.parse(backbone, 'fasta')}
    return [r.id for r in SeqIO.parse(aligned, 'fasta') if r.id not in backbone_ids]


def dereplicate(adks_faa, out_prefix, tmp_dir='tmp', mmseqs='mmseqs'):
    '''
    Cluster identical sequences (mg: easy-linclust, 100% identity and coverage).
    Writes {out_prefix}_rep_seq.fasta and related files. v1 output does not appear to be used downstream.
    '''
    subprocess.run([mmseqs, 'easy-linclust', str(adks_faa), str(out_prefix), str(tmp_dir),
                    '--min-seq-id', '1.0', '-c', '1.0', '--cov-mode', '1'],
                   capture_output=True, check=True)
    return Path(f'{out_prefix}_rep_seq.fasta')
