"""
Keep only proteins whose closest reference ADK is bacterial.

Each input protein is searched against a reference of ADKs labelled by domain (MMseqs2 easy-search); it is
kept if its best hit (highest bit score) is bacterial with at least --min-identity % identity, and removed
if its best hit is archaeal or eukaryotic, too distant, or absent.

    python scripts/screen_domains.py proteins.faa bacterial_proteins.faa --threads 16

The reference (default data/adk_domain_reference.faa) names each sequence <domain>|<id>, with domain one
of bac, arc or euk. A report (one row per input protein: best hit, identity, bit score, domain, kept) is
written next to the output as <output>.domains.tsv. Needs Biopython, pandas and MMseqs2.

designed for screen proteins in metagenomes
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
from Bio import SeqIO

DEFAULT_REFERENCE = Path(__file__).resolve().parents[1] / 'data' / 'domain_screen' / 'adk_reference.faa'


def best_hits(proteins, reference, threads):
    '''Best reference hit of each protein: target, % identity, bit score and domain.'''
    with tempfile.TemporaryDirectory() as tmp:
        hits_file = Path(tmp) / 'hits.tsv'
        subprocess.run(['mmseqs', 'easy-search', str(proteins), str(reference), str(hits_file), str(Path(tmp) / 'work'),
                        '--threads', str(threads), '--format-output', 'query,target,pident,bits'],
                       check=True, capture_output=True)
        hits = pd.read_csv(hits_file, sep='\t', names=['protein', 'target', 'pident', 'bits'])
    best = hits.sort_values('bits', ascending=False).drop_duplicates('protein').set_index('protein')
    best['domain'] = best['target'].str.split('|').str[0]
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('proteins', type=Path, help='protein FASTA to screen, e.g. an extract stage output')
    ap.add_argument('output', type=Path, help='protein FASTA of the kept (bacterial) proteins')
    ap.add_argument('--reference', type=Path, default=DEFAULT_REFERENCE,
                    help='reference ADKs named <domain>|<id> (default: data/adk_domain_reference.faa)')
    ap.add_argument('--min-identity', type=float, default=40,
                    help='minimum %% identity of the best (bacterial) hit to keep a protein (default: 40)')
    ap.add_argument('--threads', type=int, default=4)
    args = ap.parse_args()

    if not args.reference.exists():
        sys.exit(f'reference not found: {args.reference}')
    records = list(SeqIO.parse(args.proteins, 'fasta'))
    if len({r.id for r in records}) < len(records):
        sys.exit(f'{args.proteins}: sequence IDs (first word of each header) must be unique')

    report = best_hits(args.proteins, args.reference, args.threads).reindex([r.id for r in records])
    report['domain'] = report['domain'].fillna('no hit')
    report['kept'] = (report['domain'] == 'bac') & (report['pident'] >= args.min_identity)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    SeqIO.write([r for r in records if report.loc[r.id, 'kept']], args.output, 'fasta')
    report.to_csv(args.output.with_suffix(args.output.suffix + '.domains.tsv'), sep='\t', index_label='protein')
    summary = report.groupby('domain')['kept'].agg(['size', 'sum'])
    print(f"kept {int(report['kept'].sum())} of {len(report)} proteins; by best-hit domain (proteins, kept): "
          + ', '.join(f'{d} {int(s)}/{int(k)}' for d, (s, k) in summary.iterrows()))


if __name__ == '__main__':
    main()