#!/usr/bin/env python

# WRITTEN FOR REVISIONS BY CLAUDE OPUS 5.5

# Steps 4 and 5: how much do ROSEATE's fits and predictions depend on the context sequences?

#   commands   write one shell command per run (for a SLURM array or GNU parallel), in two groups:
#              grid:     for each context depth and seed, fit on the training rows of the backbone
#                        (fitting-workflow), then predict the test sequences with that model's config,
#                        which embeds them with the same depth and seed (prediction-workflow --sequences)
#              seeds:    the main model fixed, the test sequences re-embedded with different context seeds
#   summarize  read the runs and write, to --results:
#              grid_runs.tsv               R2 and RMSE per run
#              grid_by_depth.tsv           mean and SD of R2 and RMSE per context depth
#              prediction_seed_sd.tsv      each test protein's prediction SD across context seeds
#              component_stability.tsv     step 5: per site, overlap of the largest coefficients across
#                                          the seeds of one depth (pairwise Jaccard; chance = k / (2E - k))
#              context_grid.png

#     python revision/scripts/s04_context_grid.py commands --alignment runs/revision/s02_backbone/backbone.afa \
#         --train-ids reproduce/even_sequence_ids.txt --traits data/high_qual_alignments/temp_data.tsv \
#         --sequences data/20240122_all_adks_filtered.faa --test-ids reproduce/corkrey_sequence_ids.txt \
#         --main-config models/v2_main/model_config.yaml --runs runs/revision/s04_context_grid > s04_commands.txt
#     parallel -j 4 < s04_commands.txt          # or one SLURM array task per line
#     python revision/scripts/s04_context_grid.py summarize --runs runs/revision/s04_context_grid \
#         --traits data/high_qual_alignments/temp_data.tsv --results revision/results/s04_context_grid

# Run folders: <runs>/grid/d08_s3/{fit,predict} (context depth 8, seed 3) and <runs>/seeds/s07/.


import argparse
import itertools
import shlex
from pathlib import Path

import numpy as np
import pandas as pd

DEPTHS = [0, 2, 4, 8, 16, 32]
SEEDS = 3
SEEDS_AT_MAIN_DEPTH = 5
MAIN_DEPTH = 8
PREDICTION_SEEDS = 10


# ---------------------------------------------------------------- commands
def test_fasta(sequences, test_ids, out):
    '''Unaligned test sequences, taken from the full sequence file by ID.'''
    from Bio import SeqIO
    wanted = set(Path(test_ids).read_text().split())
    records = [r for r in SeqIO.parse(sequences, 'fasta') if r.id in wanted]
    missing = wanted - {r.id for r in records}
    assert not missing, f'{len(missing)} test IDs are not in {sequences}'
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    SeqIO.write(records, out, 'fasta')
    return Path(out)


def grid_runs(depths, seeds, seeds_at_main):
    for depth in depths:
        for seed in range(1, (seeds_at_main if depth == MAIN_DEPTH else seeds) + 1):
            yield depth, seed

def timed(cmd, out):
    '''Run cmd under GNU time, writing wall-clock seconds and peak memory (KB) to out.'''
    return f'mkdir -p {shlex.quote(str(out.parent))} && /usr/bin/time -f "%e %M" -o {shlex.quote(str(out))} {cmd}'

def commands(args):
    runs = args.runs.resolve()
    test = test_fasta(args.sequences, args.test_ids, runs / 'test_sequences.faa')
    q = lambda p: shlex.quote(str(p))
    threads = f'--threads {args.threads}'
    for depth, seed in grid_runs(args.depths, args.seeds, args.seeds_at_main_depth):
        run = runs / 'grid' / f'd{depth:02d}_s{seed}'
        settings = f'--set embed.context_size={depth} --set embed.context_seed={seed}'
        fit = (f'roseate fitting-workflow --alignment {q(args.alignment)} --train-ids {q(args.train_ids)} '
              f'--traits {q(args.traits)} {settings} {threads} -v -o {q(run / "fit")}')
        predict = (f'roseate prediction-workflow --config {q(run / "fit" / "model_config.yaml")} '
              f'--sequences {q(test)} {threads} -o {q(run / "predict")} '
              f'--verbose --overwrite')
        print(f'{timed(fit, run / "runtime_fit.txt")} && {timed(predict, run / "runtime_predict.txt")}')
    if args.main_config:
        for seed in range(1, args.prediction_seeds + 1):
            predict = (f'roseate prediction-workflow --config {q(args.main_config)} --sequences {q(test)} '
                  f'--set embed.context_seed={seed} {threads} -o {q(runs / "seeds" / f"s{seed:02d}")}')
            print(f'{timed(cmd, runs / "seeds" / f"s{seed:02d}" / "runtime.txt")}')


# ---------------------------------------------------------------- summary
def predictions(folder):
    path = Path(folder) / '5_predict' / 'predictions.tsv'
    return pd.read_csv(path, sep='\t', index_col=0).iloc[:, 0] if path.exists() else None


def score(pred, traits):
    measured = traits.reindex(pred.index)
    assert measured.notna().all(), f'{measured.isna().sum()} test IDs have no trait'
    resid = pred - measured
    return {'n': len(pred), 'r2': 1 - (resid ** 2).sum() / ((measured - measured.mean()) ** 2).sum(),
            'rmse': float(np.sqrt((resid ** 2).mean()))}


def component_stability(model_files, k):
    '''Per site: mean pairwise Jaccard overlap of the k largest |coefficients| across fits.'''
    import pickle
    tops = []
    for f in model_files:
        model = pickle.load(open(f, 'rb'))['best_model']
        tops.append({site: set(np.argsort(-np.abs(m.coef_))[:k]) for site, m in model.items()})
    sites = sorted(tops[0])
    n_dims = len(next(iter(pickle.load(open(model_files[0], 'rb'))['best_model'].values())).coef_)
    rows = []
    for site in sites:
        pairs = [len(a[site] & b[site]) / len(a[site] | b[site]) for a, b in itertools.combinations(tops, 2)]
        rows.append({'site': site, 'jaccard': np.mean(pairs)})
    return pd.DataFrame(rows), k / (2 * n_dims - k)


def runtime(path):
    '''(seconds, peak memory in GB) from a GNU time file, or (NaN, NaN) if missing.'''
    try:
        seconds, kb = Path(path).read_text().split()[-2:]
        return float(seconds), int(kb) / 1e6
    except (FileNotFoundError, ValueError):
        return float('nan'), float('nan')


def summarize(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    traits = pd.read_csv(args.traits, sep=None, engine='python', index_col=0).iloc[:, 0]
    out = args.results
    out.mkdir(parents=True, exist_ok=True)

    # grid: one score per finished run
    rows = []
    for run in sorted((args.runs).glob('d*_s*')):
        pred = predictions(run / 'predict')
        if pred is None:
            print(f'not finished: {run.name}')
            continue
        depth, seed = (int(x) for x in run.name[1:].split('_s'))
        fit_s, fit_gb = runtime(run / 'runtime_fit.txt')
        predict_s, predict_gb = runtime(run / 'runtime_predict.txt')
        rows.append({'run': run.name, 'depth': depth, 'seed': seed, **score(pred, traits),
                     'fit_minutes': fit_s / 60, 'predict_minutes': predict_s / 60,
                     'peak_memory_gb': max(fit_gb, predict_gb)})

    grid = pd.DataFrame(rows)
    if not grid.empty:
        grid.to_csv(out / 'grid_runs.tsv', sep='\t', index=False)
        by_depth = grid.groupby('depth')[['r2', 'rmse']].agg(['mean', 'std', 'count'])
        by_depth.columns = ['_'.join(c) for c in by_depth.columns]
        by_depth.to_csv(out / 'grid_by_depth.tsv', sep='\t')
        print(by_depth.round(3).to_string())

    # prediction-time variation: the main model, different context seeds
    seed_preds = {run.name: predictions(run) for run in sorted((args.runs / 'seeds').glob('s*'))}
    seed_preds = pd.DataFrame({k: v for k, v in seed_preds.items() if v is not None})
    if not seed_preds.empty:
        spread = pd.DataFrame({'mean': seed_preds.mean(axis=1), 'sd': seed_preds.std(axis=1),
                               'range': seed_preds.max(axis=1) - seed_preds.min(axis=1)})
        spread.to_csv(out / 'prediction_seed_sd.tsv', sep='\t', index_label='protein')
        scores = pd.DataFrame([score(seed_preds[c], traits) for c in seed_preds])
        print(f'prediction seeds ({seed_preds.shape[1]}): per-protein SD median {spread["sd"].median():.2f} °C '
              f'(95th percentile {spread["sd"].quantile(0.95):.2f}); R2 {scores["r2"].mean():.3f} ± {scores["r2"].std():.3f}, '
              f'RMSE {scores["rmse"].mean():.2f} ± {scores["rmse"].std():.2f}')

    # step 5: component stability across seeds, for every depth with at least two finished fits
    rows = []
    for depth in sorted({int(p.name[1:3]) for p in (args.runs / 'grid').glob('d*_s*')}):
        fits = sorted((args.runs / 'grid').glob(f'd{depth:02d}_s*/fit/4_fit/model.pickle'))
        if len(fits) >= 2:
            stability, chance = component_stability(fits, args.top_k)
            rows.append(stability.assign(depth=depth, fits=len(fits), chance=chance))
            print(f'component stability, depth {depth} ({len(fits)} fits): mean Jaccard of the top '
                  f'{args.top_k} coefficients per site {stability["jaccard"].mean():.3f} (chance {chance:.3f})')
    if rows:
        pd.concat(rows).to_csv(out / 'component_stability.tsv', sep='\t', index=False)

    # figure
    panels = [p for p in [not grid.empty, not seed_preds.empty] if p]
    if panels:
        fig, axes = plt.subplots(1, 3 if not seed_preds.empty and not grid.empty else 2, figsize=(12, 3.4))
        axes = list(np.atleast_1d(axes))
        if not grid.empty:
            for ax, metric, label in [(axes.pop(0), 'r2', 'Test R²'), (axes.pop(0), 'rmse', 'Test RMSE (°C)')]:
                ax.scatter(grid['depth'], grid[metric], s=14, color='0.6', zorder=2)
                means = grid.groupby('depth')[metric].agg(['mean', 'std'])
                ax.errorbar(means.index, means['mean'], yerr=means['std'], color='k', marker='o', capsize=3, zorder=3)
                ax.set(xscale='symlog', xticks=sorted(grid['depth'].unique()), xlabel='Context depth', ylabel=label)
                ax.set_xticklabels(sorted(grid['depth'].unique()))
        if not seed_preds.empty:
            ax = axes.pop(0)
            ax.hist(spread['sd'], bins=30, color='0.5')
            ax.set(xlabel='Prediction SD across context seeds (°C)', ylabel='Test proteins')
        for ax in fig.axes:
            ax.spines[['top', 'right']].set_visible(False)
        fig.tight_layout()
        fig.savefig(out / 'context_grid.png', dpi=300)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
    c = sub.add_parser('commands', help='print one shell command per run')
    c.add_argument('--alignment', type=Path, required=True, help='training-only backbone (step 2)')
    c.add_argument('--train-ids', type=Path, required=True)
    c.add_argument('--traits', type=Path, required=True)
    c.add_argument('--sequences', type=Path, required=True, help='unaligned sequences containing the test set')
    c.add_argument('--test-ids', type=Path, required=True)
    c.add_argument('--main-config', type=Path, help='model_config.yaml of the main model, for the prediction seeds')
    c.add_argument('--runs', type=Path, required=True)
    c.add_argument('--depths', type=int, nargs='+', default=DEPTHS)
    c.add_argument('--seeds', type=int, default=SEEDS)
    c.add_argument('--seeds-at-main-depth', type=int, default=SEEDS_AT_MAIN_DEPTH)
    c.add_argument('--prediction-seeds', type=int, default=PREDICTION_SEEDS)
    c.add_argument('--threads', type=int, default=16)
    s = sub.add_parser('summarize', help='score the runs and write tables and a figure')
    s.add_argument('--runs', type=Path, required=True, help='the output folder from context_grid.py command --out')
    s.add_argument('--traits', type=Path, required=True, help='trait table keyed by sequence ID, for scoring')
    s.add_argument('--results', type=Path, required=True, help='output folder for results')
    s.add_argument('--top-k', type=int, default=20, help='number of largest |coefficients| per site compared across seeds (default: 20)')
    args = ap.parse_args()
    commands(args) if args.command == 'commands' else summarize(args)


if __name__ == '__main__':
    main()