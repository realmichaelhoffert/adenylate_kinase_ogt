#!/usr/bin/env python
"""
Step 10: how much does TOME's accuracy depend on the number of residues it sees?

TOME (Li et al. 2019) is reimplemented here exactly as in its predOGT.py: 441 dipeptide frequencies
(21 letters incl. X, counted within each protein, divided by the total count), standardized with the
training means and SDs, then an RBF SVR with C = 64, epsilon = 1 (gamma = 'auto', TOME's sklearn default).
This avoids TOME's own retraining, which overwrites the model inside its installed package.

Each test proteome is downsampled by drawing whole proteins in random order until N residues are reached
(the last protein is cut at N). Two designs:
  full     train on full Even proteomes, test on downsampled proteomes (a model applied to partial data)
  matched  train and test both downsampled to N (the information available at N residues)

    python revision/scripts/s10_tome_downsample.py \
        --train-dataset reproduce/even_sequence_ids.txt --test-dataset reproduce/corkrey_sequence_ids.txt \
        --traits data/high_qual_alignments/temp_data.tsv \
        --train-proteomes data/validation_datasets/tomes/even_genomes \
        --test-proteomes data/validation_datasets/tomes/corkrey_genomes \
        --out runs/revision/s10_tome_downsample --threads 16

Writes predictions.tsv (one row per genome, design, level and replicate) and metrics.tsv (R2 and RMSE per
design, level and replicate). Proteome files are matched to genomes by GTDB accession in the file name.
Each proteome is read once, by one worker, which computes its features at every level and replicate.
"""
import argparse
import gzip
from collections import Counter
from itertools import product
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from Bio import SeqIO
from joblib import Parallel, delayed
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR

from tqdm import tqdm

import roseate.decorators

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWXY'
DIMERS = [a + b for a in AMINO_ACIDS for b in AMINO_ACIDS]          # TOME's 441 features, in its order
DIMER_INDEX = {d: i for i, d in enumerate(DIMERS)}
LEVELS = [215, 1_000, 5_000, 20_000, 50_000, 100_000, 1_000_000, None]                 # None = the full proteome


# ---------------------------------------------------------------- inputs
def read_ids(path):
    '''Sequence IDs, one per line (or an MSADataset pickle); GTDB accession = second "|" field.'''
    path = Path(path)
    if path.suffix == '.pickle':
        import pickle
        return [i[0] for i in pickle.load(open(path, 'rb'))]
    return path.read_text().split()


def genome_traits(ids, traits_file):
    '''{GTDB accession: OGT} for the given sequence IDs.'''
    traits = pd.read_csv(traits_file, sep=None, engine='python', index_col=0).iloc[:, 0]
    missing = [i for i in ids if i not in traits.index]
    assert not missing, f'{len(missing)} IDs have no trait in {traits_file}, e.g. {missing[:3]}'
    return {i: float(traits[i]) for i in ids}


def find_proteome(folder, accession):
    hits = sorted(Path(folder).glob(f'{accession}*'))
    assert hits, f'no proteome for {accession} in {folder}'
    return hits[0]


def read_proteins(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt') as handle:
        return [str(r.seq).upper() for r in SeqIO.parse(handle, 'fasta')]


# ---------------------------------------------------------------- TOME features and model
def dimer_frequencies(proteins):
    '''TOME features: dipeptide counts within each protein over all proteins, divided by the total.'''
    counts = Counter()
    for seq in proteins:
        counts.update(seq[i:i + 2] for i in range(len(seq) - 1))
    x = np.array([counts.get(d, 0.0) for d in DIMERS])
    return x / x.sum()


def downsample(proteins, n_residues, rng):
    '''Whole proteins in random order until n_residues; the last one is cut. None = all proteins.'''
    if n_residues is None:
        return proteins
    sample, total = [], 0
    for i in rng.permutation(len(proteins)):
        seq = proteins[i][:n_residues - total]
        sample.append(seq)
        total += len(seq)
        if total >= n_residues:
            break
    return sample


def fit_tome(X, y):
    '''TOME's standardization and SVR; returns (model, mean, sd).'''
    mean, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1.0                                                  # dimers absent from all training genomes
    return SVR(kernel='rbf', C=64.0, epsilon=1.0, gamma='auto').fit((X - mean) / sd, y), mean, sd


def genome_features(path, levels, reps, seed):
    '''
    One genome's features at every level and replicate: {(level, rep): vector}, plus (None, 0) for the
    full proteome. Each proteome is read once; each genome and replicate has its own random stream.
    '''
    proteins = read_proteins(path)
    genome_key = int.from_bytes(Path(path).name.encode()[:8], 'little')
    out = {(None, 0): dimer_frequencies(proteins)}
    for level, rep in product(levels, range(reps)):
        out[(level, rep)] = dimer_frequencies(downsample(proteins, level, np.random.default_rng([seed, rep, genome_key])))
    return out, sum(map(len, proteins))


def all_features(paths, levels, reps, seed, threads):
    '''Features for every genome, one worker per proteome file: {(level, rep): genomes x 441 matrix}.'''
    results = Parallel(n_jobs=threads)(delayed(genome_features)(p, levels, reps, seed) for p in paths)
    return {k: np.array([r[0][k] for r in results]) for k in results[0][0]}, [r[1] for r in results]

def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--train-dataset', type=Path, required=True, help='training sequence IDs (txt or MSADataset pickle)')
    ap.add_argument('--test-dataset', type=Path, required=True, help='test sequence IDs (txt or MSADataset pickle)')
    ap.add_argument('--traits', type=Path, required=True, help='trait table keyed by sequence ID')
    ap.add_argument('--train-proteomes', type=Path, required=True, help='folder of training proteomes')
    ap.add_argument('--test-proteomes', type=Path, required=True, help='folder of test proteomes')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--levels', type=int, nargs='+', help='residue counts (default: 215 1000 5000 20000 100000)')
    ap.add_argument('--reps', type=int, default=5, help='replicate draws per level (default: 5)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--no-verbose', action='store_false', help='Turn off verbosity')
    return ap


@roseate.decorators.verbose_print
def main(args, vprint):


    levels = args.levels or LEVELS[:-1]
    data = {}
    with roseate.decorators.spinner('Loading input data', enabled=args.no_verbose):
        for split, ids_file, folder, seed in [('train', args.train_dataset, args.train_proteomes, args.seed),
                                            ('test', args.test_dataset, args.test_proteomes, args.seed + 1)]:
            acc2ogt = genome_traits(read_ids(ids_file), args.traits)
            accessions = sorted(acc2ogt)
            X, sizes = all_features([find_proteome(folder, a) for a in accessions], levels, args.reps, seed, args.threads)
            data[split] = (accessions, X, np.array([acc2ogt[a] for a in accessions]))
            vprint(f'{split}: {len(accessions)} proteomes, median {int(np.median(sizes)):,} residues')

    # one TOME fit per design, level and replicate: full = trained on full proteomes, matched = same level
    vprint('Retraining TOME models ... ')
    (_, X_train, y_train), (test_acc, X_test, y_test) = data['train'], data['test']
    rows = []
    for (level, rep), X_t in tqdm(X_test.items()):
        designs = {'full': X_train[(None, 0)]}
        if level is not None:
            designs['matched'] = X_train[(level, rep)]
        for design, X in designs.items():
            model, mean, sd = fit_tome(X, y_train)
            pred = model.predict((X_t - mean) / sd)
            rows += [{'design': design, 'residues': level or 'full', 'rep': rep, 'accession': a,
                      'measured': t, 'predicted': pr} for a, t, pr in zip(test_acc, y_test, pred)]
    
    with roseate.decorators.spinner('Finishing up', enabled=args.no_verbose):
        predictions = pd.DataFrame(rows)

        args.out.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(args.out / 'predictions.tsv', sep='\t', index=False)
        metrics = (predictions.groupby(['design', 'residues', 'rep'], sort=False)
                .apply(lambda g: pd.Series({'r2': r2_score(g['measured'], g['predicted']),
                                            'rmse': np.sqrt(mean_squared_error(g['measured'], g['predicted']))}))
                .reset_index())
        metrics.to_csv(args.out / 'metrics.tsv', sep='\t', index=False)
    print(metrics.groupby(['design', 'residues'], sort=False)[['r2', 'rmse']].agg(['mean', 'std']).round(3))


if __name__ == '__main__':
    ap = build_parser()
    args = ap.parse_args()
    sys.exit(main(args))