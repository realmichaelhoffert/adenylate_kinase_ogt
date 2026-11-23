# Code written by Michael Hoffert, refactored into a single script with Claude Opus 5.5
# November 2026
'''
Code drawn from commands.txt and notebooks/02_filtering.ipynb, cells 75–106:

Cells 75–83: gap-percentage filtering of the first alignment, full_adk_reps_initial.afa.
Cells 87–97: find_alignment_bloaters and the rounds of gappy-sequence filtering, giving full_adk_reps.filtered.afa.
Cell 98: a summary of the sequence counts per round: 7,539 → 6,901 → 6,842 → 6,641.
Cells 100–106: trimming, and splitting off extra_adk_seqs.faa, giving full_adk_reps.filtered.trimmed.afa and full_adk_reps.filtered_only.trimmed.afa.


Purpose: implementation of feedback from revisions and facilitate alignment for new proteins

Pipeline (preprint lines 539-551, with the manual filtering automated):

1. drop excluded sequences (e.g. the Corkrey test set), so they never influence the alignment
  2. length filter: keep sequences within --length-range x the median length (outgroup included), which
     removes fragments and proteins with long extensions before they open hundreds of columns
  3. CD-HIT at --identity: representatives, plus the length-filtered outgroup
  4. MAFFT once; drop fragments (missing more than --max-core-gap of the core columns) and sequences that
     put many residues into rare columns (occupied by fewer than --rare-occupancy of sequences);
     realign the rest once
  5. trimAl -gappyout: the core alignment
  6. add every other non-excluded sequence with mafft --keeplength --add: backbone.afa
     (sequences removed in steps 2-4 are kept in the backbone, with their extra residues dropped)

Reliability:
  heads-or-tails  the step-4 sequences are also aligned in reverse (Landan & Graur 2007); a residue is
                  consistent if it lands in the corresponding column in both alignments. Reported
                  overall, per sequence and per core column (core column k is model site k + 1;
                  site 0 is the CLS token).
  --compare-to B  the same score between backbone.afa and an existing backbone, e.g. the preprint's

    python revision/scripts/s02_build_backbone.py --sequences adk_sequences.faa \
        --exclude reproduce/corkrey_sequence_ids.txt --outgroup tigr01351_outgroup.faa \
        --compare-to data/high_qual_alignments/backbone.filtered_only.afa --threads 16 -v \
        --out runs/revision/s02_backbone

Outputs in --out: backbone.afa (the model's alignment), core.afa, removed.tsv (sequences left out of the
core and why), heads_or_tails.tsv (overall), heads_or_tails_sequences.tsv, heads_or_tails_columns.tsv,
compare_to.tsv and compare_to_sequences.tsv, and build.json (settings, counts, tool versions).

External tools change some header characters (trimAl cuts names at ':'), so every sequence gets a plain
internal name (seq000001, out000001 for the outgroup) for the run; outputs use the original IDs again.
Rerunning the same command resumes; a different command in the same --out stops with an error (use a
new --out, or --overwrite). -v/--verbose shows each step as it runs.
'''

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm

from roseate.align import add_to_backbone
from roseate.decorators import spinner, verbose_print

MAFFT_STRATEGIES = {
    'fftns': ['--retree', '2', '--maxiterate', '0'],       # FFT-NS-2: minutes for thousands of sequences
    'fftnsi': ['--retree', '2', '--maxiterate', '2'],      # FFT-NS-i: slower, iterative refinement
    'linsi': ['--localpair', '--maxiterate', '1000'],      # the preprint's L-INS-i: slowest
}


# ---------------------------------------------------------------- sequences and tools
def read_fasta(path):
    '''{id: sequence} in file order; IDs must be unique.'''
    records = {}
    for r in SeqIO.parse(path, 'fasta'):
        assert r.id not in records, f'{path}: duplicate ID {r.id}'
        records[r.id] = str(r.seq).replace('-', '').replace('*', '').upper()
    return records


def write_fasta(records, path):
    with open(path, 'w') as f:
        f.writelines(f'>{i}\n{s}\n' for i, s in records.items())
    return Path(path)


def read_alignment(path):
    return {r.id: str(r.seq).upper() for r in SeqIO.parse(path, 'fasta')}


def dereplicate(records, folder, identity, verbose):
    '''CD-HIT representatives, as {id: sequence}.'''
    raw, reps = write_fasta(records, folder / 'all.faa'), folder / 'representatives.faa'
    with spinner(f'Dereplicating {len(records)} sequences with CD-HIT at {identity:.0%}', enabled=verbose):
        subprocess.run(['cd-hit', '-i', str(raw), '-o', str(reps), '-c', str(identity), '-n', '5' if identity >= 0.7 else '4',
                        '-M', '0', '-T', '0'], check=True, capture_output=True)
    return read_fasta(reps)


def mafft(records, out, args, label):
    '''Align records with MAFFT, writing out; returns {id: aligned row}.'''
    with spinner(f'  {label}: aligning {len(records)} sequences with MAFFT ({args.mafft})', enabled=args.verbose):
        with open(out, 'w') as handle:
            subprocess.run(['mafft', *MAFFT_STRATEGIES[args.mafft], '--thread', str(args.threads), '--quiet',
                            str(write_fasta(records, Path(out).with_suffix('.faa')))],
                           check=True, stdout=handle, stderr=subprocess.PIPE)
    return read_alignment(out)


def trim(untrimmed, core, verbose):
    '''trimAl -gappyout; returns the untrimmed column index of each core column.'''
    with spinner('  Trimming with trimAl -gappyout', enabled=verbose):
        result = subprocess.run(['trimal', '-in', str(untrimmed), '-out', str(core), '-gappyout', '-fasta',
                                 '-colnumbering'], check=True, capture_output=True, text=True)
    line = next(l for l in result.stdout.splitlines() if l.startswith('#ColumnsMap'))
    return [int(c) for c in line.split('\t', 1)[1].split(',')]


# ---------------------------------------------------------------- filtering
def length_filter(records, median, low, high):
    '''Sequences within [low, high] x median length, and those outside.'''
    keep = {i: s for i, s in records.items() if low * median <= len(s) <= high * median}
    return keep, [i for i in records if i not in keep]


def sequence_stats(aligned, core_occupancy, rare_occupancy):
    '''Per sequence: fraction of core columns left empty, and residues in rare columns.'''
    ids = list(aligned)
    present = np.array([[c != '-' for c in aligned[i]] for i in ids])
    occupancy = present.mean(axis=0)
    core, rare = occupancy >= core_occupancy, occupancy < rare_occupancy
    return pd.DataFrame({'core_gap_fraction': 1 - present[:, core].mean(axis=1),
                         'rare_residues': present[:, rare].sum(axis=1)}, index=ids)


# ---------------------------------------------------------------- comparing alignments
def residue_columns(aligned, originals, verbose=False, label=''):
    '''
    (id, residue position in the original sequence, column) for every residue in an alignment, and the
    IDs whose row does not match their original sequence (skipped).
    --keeplength deletes residues, so each row is matched back to its original sequence in order.
    '''
    rows, skipped = [], []
    for seq_id, row in tqdm(aligned.items(), total=len(aligned), desc=f'  Mapping residues{label}',
                            disable=not verbose, leave=False):
        original, j, mapped = originals[seq_id], 0, []
        for col, c in enumerate(row):
            if c == '-':
                continue
            j = original.find(c, j)                    # skips residues deleted by --keeplength
            if j < 0:
                break
            mapped.append((seq_id, j, col))
            j += 1
        if j < 0:
            skipped.append(seq_id)                     # row is not this sequence (e.g. another version)
        else:
            rows += mapped
    return pd.DataFrame(rows, columns=['id', 'pos', 'col']), skipped


def consistency(main, other, originals, verbose=False):
    '''
    Residue-level agreement of other with main, on the sequences and residues both contain.
    Each main column corresponds to the other column most of its residues landed in.
    Returns (summary, score per sequence, score per main column).
    '''
    shared = sorted(set(main) & set(other))
    main_cols, skipped_main = residue_columns({i: main[i] for i in shared}, originals, verbose, ' (main)')
    other_cols, skipped_other = residue_columns({i: other[i] for i in shared}, originals, verbose, ' (other)')
    skipped = set(skipped_main) | set(skipped_other)
    pairs = main_cols[~main_cols['id'].isin(skipped)].merge(
        other_cols[~other_cols['id'].isin(skipped)], on=['id', 'pos'], suffixes=('_main', '_other'))
    corresponding = pairs.groupby('col_main')['col_other'].agg(lambda c: c.mode().iloc[0])
    pairs['consistent'] = pairs['col_other'].values == corresponding.loc[pairs['col_main']].values
    summary = {'shared_sequences': len(shared) - len(skipped), 'skipped_sequences': len(skipped),
               'residues_compared': len(pairs),
               'consistent_fraction': pairs['consistent'].mean()}
    return summary, pairs.groupby('id')['consistent'].mean(), pairs.groupby('col_main')['consistent'].mean()


def tool_versions():
    def first_line(cmd):
        out = subprocess.run(cmd, capture_output=True, text=True)
        text = (out.stdout or out.stderr).strip()
        return text.splitlines()[0] if text else '?'
    return {'mafft': first_line(['mafft', '--version']), 'cd-hit': first_line(['cd-hit', '-h']),
            'trimal': first_line(['trimal', '--version'])}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sequences', type=Path, required=True, help='unaligned sequences (gaps are removed)')
    ap.add_argument('--exclude', type=Path, help='IDs to leave out entirely, one per line (e.g. a test set)')
    ap.add_argument('--outgroup', type=Path, help='sequences that guide the alignment but are not in the backbone')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--length-range', type=float, nargs=2, default=[0.85, 1.15], metavar=('LOW', 'HIGH'),
                    help='keep sequences within LOW-HIGH x the median length for the core (default 0.7 1.3)')
    ap.add_argument('--identity', type=float, default=0.9, help='CD-HIT identity (default 0.9)')
    ap.add_argument('--mafft', choices=list(MAFFT_STRATEGIES), default='fftns', help='default fftns (FFT-NS-2)')
    ap.add_argument('--core-occupancy', type=float, default=0.5, help='columns with at least this occupancy are core')
    ap.add_argument('--max-core-gap', type=float, default=0.15, help='drop sequences missing more core columns')
    ap.add_argument('--rare-occupancy', type=float, default=0.01, help='columns below this occupancy are rare')
    ap.add_argument('--max-rare-residues', type=int, default=10, help='drop sequences with more rare-column residues')
    ap.add_argument('--compare-to', type=Path, help='existing backbone to compare backbone.afa with')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--overwrite', action='store_true', help='delete --out and start again')
    ap.add_argument('-v', '--verbose', action='store_true', help='show each step and progress bars')
    build_all(ap.parse_args())


@verbose_print
def build_all(args, vprint):
    out = args.out
    if args.overwrite and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    settings = {k: str(v) for k, v in vars(args).items() if k not in ('threads', 'overwrite', 'verbose')}
    settings_file = out / 'settings.json'
    if settings_file.exists() and json.load(open(settings_file)) != settings:
        raise SystemExit(f'{out} was made with different settings; use a new --out, or --overwrite')
    json.dump(settings, open(settings_file, 'w'), indent=2)

    # inputs, under plain internal names for the external tools
    records = read_fasta(args.sequences)
    n_input = len(records)
    excluded = set(args.exclude.read_text().split()) if args.exclude else set()
    if excluded - set(records):
        print(f'note: {len(excluded - set(records))} excluded IDs are not in {args.sequences}')
    targets = [i for i in records if i not in excluded]
    outgroup_in = read_fasta(args.outgroup) if args.outgroup else {}
    names = {f'seq{n:06d}': i for n, i in enumerate(targets)}
    names.update({f'out{n:06d}': i for n, i in enumerate(outgroup_in)})
    internal = {v: k for k, v in names.items() if k.startswith('seq')}
    records = {internal[i]: records[i] for i in targets}
    outgroup = {f'out{n:06d}': s for n, s in enumerate(outgroup_in.values())}
    originals = {**records, **outgroup}

    backbone_file, removed_file = out / 'backbone.afa', out / 'removed.tsv'
    if backbone_file.exists():
        vprint(f'Reusing the finished backbone in {out}')
        backbone = {internal[i]: s for i, s in read_alignment(backbone_file).items()}
        kept = read_alignment(out / 'filtered.afa')
        removed = pd.read_csv(removed_file, sep='\t', index_col=0)
    else:
        # 2. length filter, from the median target length
        median = float(np.median([len(s) for s in records.values()]))
        low, high = args.length_range
        targets_ok, too_long_or_short = length_filter(records, median, low, high)
        outgroup_ok, outgroup_dropped = length_filter(outgroup, median, low, high)
        vprint(f'Length filter {low * median:.0f}-{high * median:.0f} residues (median {median:.0f}): '
               f'{len(too_long_or_short)} targets and {len(outgroup_dropped)} outgroup sequences left out of the core')

        # 3. representatives
        reps_file = out / 'representatives.faa'
        representatives = read_fasta(reps_file) if reps_file.exists() else dereplicate(targets_ok, out, args.identity, args.verbose)

        # 4. align once, filter once, realign once
        aligned = mafft({**representatives, **outgroup_ok}, out / 'first_pass.afa', args, 'First pass')
        stats = sequence_stats(aligned, args.core_occupancy, args.rare_occupancy)
        fragment = stats['core_gap_fraction'] > args.max_core_gap
        insertion = stats['rare_residues'] > args.max_rare_residues
        drop = stats[(fragment | insertion)]
        vprint(f'  First pass: {len(next(iter(aligned.values())))} columns; dropping {len(drop)} sequences '
               f'({fragment.sum()} fragments, {(insertion & ~fragment).sum()} with rare insertions)')
        kept_records = {i: s for i, s in {**representatives, **outgroup_ok}.items() if i not in drop.index}
        kept = mafft(kept_records, out / 'filtered.afa', args, 'Second pass')
        vprint(f'  Second pass: {len(next(iter(kept.values())))} columns')

        # 5. core, 6. backbone
        core_columns = trim(out / 'filtered.afa', out / 'core.afa', args.verbose)
        to_add = {i: s for i, s in records.items() if i not in kept}
        full = out / 'backbone_with_outgroup.afa'
        with spinner(f'  Adding {len(to_add)} sequences with mafft --keeplength --add', enabled=args.verbose):
            add_to_backbone(write_fasta(to_add, out / 'to_add.faa'), out / 'core.afa', full)
        backbone = {i: s for i, s in read_alignment(full).items() if i in records}

        removed = pd.concat([
            pd.DataFrame({'reason': 'length'}, index=too_long_or_short + outgroup_dropped),
            drop.assign(reason=np.where(fragment[drop.index], 'fragment', 'rare insertions'))])
        removed['length'] = [len(originals[i]) for i in removed.index]
        removed.rename(index=names).to_csv(removed_file, sep='\t', index_label='id')
        json.dump(core_columns, open(out / 'core_columns.json', 'w'))
        write_fasta({names[i]: s for i, s in backbone.items()}, backbone_file)   # written last: marks a finished build
    width = len(next(iter(backbone.values())))
    print(f'backbone: {len(backbone)} sequences x {width} columns; {len(removed)} sequences left out of the core')

    # heads-or-tails: the core's sequences aligned in reverse
    hot_file = out / 'heads_or_tails.tsv'
    if not hot_file.exists():
        reversed_records = {i: originals[i][::-1] for i in kept}
        tails = mafft(reversed_records, out / 'tails.afa', args, 'Heads-or-tails')
        tails = {i: row[::-1] for i, row in tails.items()}                     # back to forward order
        vprint('Scoring heads-or-tails consistency')
        summary, per_seq, per_col = consistency(kept, tails, originals, args.verbose)
        core_columns = json.load(open(out / 'core_columns.json'))
        columns = pd.DataFrame({'core_column': range(len(core_columns)), 'model_site': range(1, len(core_columns) + 1),
                                'aligned_column': core_columns,
                                'heads_or_tails': per_col.reindex(core_columns).values})
        columns.to_csv(out / 'heads_or_tails_columns.tsv', sep='\t', index=False)
        per_seq.rename(index=names).rename('heads_or_tails').sort_values() \
               .to_csv(out / 'heads_or_tails_sequences.tsv', sep='\t', index_label='id')
        pd.DataFrame([summary]).to_csv(hot_file, sep='\t', index=False)
    hot = pd.read_csv(hot_file, sep='\t').iloc[0]
    print(f"heads-or-tails: {hot['consistent_fraction']:.4f} of residues in corresponding columns "
          f"({int(hot['shared_sequences'])} sequences)")

    # comparison with an existing backbone
    comparison = None
    if args.compare_to:
        reference = {internal[i]: s for i, s in read_alignment(args.compare_to).items() if i in internal}
        vprint(f'Comparing with {args.compare_to.name}')
        comparison, per_seq, _ = consistency(backbone, reference, originals, args.verbose)
        pd.DataFrame([comparison]).to_csv(out / 'compare_to.tsv', sep='\t', index=False)
        per_seq.rename(index=names).rename('consistent_fraction').sort_values() \
               .to_csv(out / 'compare_to_sequences.tsv', sep='\t', index_label='id')
        print(f"{args.compare_to.name}: {comparison['consistent_fraction']:.4f} of residues in corresponding columns "
              f"({comparison['shared_sequences']} shared sequences)")

    json.dump({'settings': settings, 'tools': tool_versions(),
               'counts': {'input': n_input, 'excluded': n_input - len(records), 'outgroup': len(outgroup),
                          'left_out_of_core': len(removed), 'core_sequences': len(kept),
                          'backbone_sequences': len(backbone), 'backbone_columns': width},
               'heads_or_tails': hot.to_dict(), 'compare_to': comparison},
              open(out / 'build.json', 'w'), indent=2, default=float)


if __name__ == '__main__':
    main()
