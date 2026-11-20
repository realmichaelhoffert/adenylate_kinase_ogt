# Written by Claude Opus 5.5
# Script for performing reproduction of figures
#!/usr/bin/env python
"""
Preprint figures that show the ROSEATE model itself, made with the roseate package:
  fig2a  predicted vs measured OGT on the test set (Corkrey)
  fig3a  accuracy of each site's model along the alignment
  figS3  predictions for the Arthrobacter and Mycobacteria isolates
  fig5   community OGT distributions in metagenomes

Inputs are the preprint's archived embeddings (reference/v1_preprint), so differences between
models come from the models, not from new embedding draws. Any folder holding model.pickle and
scaler.pickle can be plotted: the preprint model, or a refit from fitting-workflow (its 4_fit folder).
    python reproduce/preprint.py --model-dir models/v1_preprint --out figures/preprint
    python reproduce/preprint.py --model-dir sweeps/c16_s1/4_fit --out figures/c16_s1 --figures fig2a fig3a
Writes one PNG per figure and metrics.tsv with the numbers shown in them.
"""
import argparse
import glob
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from roseate import aggregate, evaluate, io
from roseate.predict import per_site_predict
from roseate.cli import RUN_FILES

# archive locations (reference/v1_preprint)
TEST_DATASET = 'inputs/predictive_models/supporting_data/corkrey_msa_dataset.pickle'
TEST_EMBEDDINGS = 'intermediates/esm_msa/c_w_e_full_embeddings.pickle'
TRAITS = 'inputs/high_qual_alignments/temp_data.tsv'
SPEARMANS = 'outputs/predictive_models/supporting_data/even_enetcv_ind_spearmans.tsv*'
ISOLATES = ['Arthrobacter', 'Mycobacteria']
HABITATS = {'Antarctic': 'Antarctic soil', 'RoyoLlonch2021': 'Arctic ocean (TARA)',
            'Salazar2019': 'Arctic ocean (TARA)', 'Shah2024': 'European freshwater lakes',
            'Australia': 'Australian soil', 'Conteville2024': 'Rumen', 'Youngblut2020': 'Avian gut',
            'Mara2023': 'Geothermal sediments', 'Colman2024': 'Hot springs'}
EXAMPLE_HABITATS = ['Arctic ocean (TARA)', 'Avian gut', 'Australian soil', 'Hot springs']   # Fig 5A


# ---------------------------------------------------------------- data and predictions
def archive(ref, rel):
    '''Path of an archived file (rel may be a glob).'''
    hits = sorted(glob.glob(str(Path(ref) / rel)))
    if not hits:
        raise FileNotFoundError(f'not in the archive: {Path(ref) / rel}')
    return Path(hits[0])


def predict_ogt(model, scaler, emb, sites=None):
    '''Per-site predictions (sites x proteins) and their mean over sites, in degrees C.'''
    per_site = np.array(per_site_predict(model['best_model'], emb, model['scalers']))
    return per_site, io.unscale_traits(aggregate.mean_sites(per_site, sites), scaler)


def validation_dataset(ref, name, model, scaler, sites):
    '''Predictions for one archived validation dataset, joined to its metadata by protein ID.'''
    folder = f'intermediates/validation_datasets/{name}/embeddings'
    msa = io.reload_pickle(archive(ref, f'{folder}/{name}_msa_dataset.pickle'))
    emb = io.load_embeddings(archive(ref, f'{folder}/{name}_full_even.pickle'))
    _, pred = predict_ogt(model, scaler, emb, sites)
    md = pd.read_csv(archive(ref, f'inputs/validation_datasets/{name}/md/{name}_env_temps.tsv'), sep='\t')
    md = md.drop(columns=[c for c in md.columns if c.startswith('Unnamed')]).set_index('protein')
    table = pd.DataFrame({'pred': pred}, index=[i[0] for i in msa]).join(md, how='inner')
    return table.rename(columns={'env temp': 'measured'})

def test_set(ref, test_run=None):
    '''
    Test sequences with measured OGT (°C, from the trait table) and their embeddings, in the same order.
    Default: the archived test set. With test_run: the dataset and embeddings of a roseate run folder.
    '''
    if test_run:
        dataset = io.reload_pickle(Path(test_run) / RUN_FILES['align']['msa_dataset'])
        emb = io.load_embeddings(Path(test_run) / RUN_FILES['embed']['embeddings'])
    else:
        dataset = io.reload_pickle(archive(ref, TEST_DATASET))
        emb = io.load_embeddings(archive(ref, TEST_EMBEDDINGS))
    if len(dataset) != emb.shape[0]:
        raise SystemExit(f'test dataset has {len(dataset)} sequences but {emb.shape[0]} embeddings')
    return io.attach_traits(dataset, archive(ref, TRAITS)), emb

def metrics(figure, dataset, true, pred):
    _, (r2, rmse, spearman) = evaluate.assess_model(np.asarray(true), np.asarray(pred))
    return {'figure': figure, 'dataset': dataset, 'n': len(true), 'r2': r2, 'rmse': rmse, 'spearman': spearman}


def scatter_panel(ax, true, pred, title):
    '''Predicted vs measured, with the x=y line and a 5 degree band, as in Fig 2.'''
    lo, hi = np.floor(min(np.min(true), np.min(pred))) - 2, np.ceil(max(np.max(true), np.max(pred))) + 2
    ax.fill_between([lo, hi], [lo - 5, hi - 5], [lo + 5, hi + 5], color='0.88', label='± 5 °C')
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1, label='x = y')
    ax.scatter(true, pred, s=18, alpha=0.7, edgecolors='w', linewidths=0.4)
    m = metrics('', '', true, pred)
    ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel='Measured OGT (°C)', ylabel='Predicted OGT (°C)', aspect='equal',
           title=f"{title}\n$R^2$ = {m['r2']:.2f}, RMSE = {m['rmse']:.2f} °C")
    ax.legend(loc='lower right', frameon=False, fontsize=8)
    sns.despine(ax=ax)


# ---------------------------------------------------------------- figures
def fig2a(ref, model, scaler, sites, out, test_run=None):
    test, emb = test_set(ref, test_run)
    _, pred = predict_ogt(model, scaler, emb, sites)
    true = np.array([i[2] for i in test])
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    scatter_panel(ax, true, pred, 'ROSEATE, test set')
    fig.savefig(out / 'fig2a_test_set.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    return [metrics('fig2a', 'test set', true, pred)]


def fig3a(ref, model, scaler, sites, out, test_run=None):
    test, emb = test_set(ref, test_run)
    # emb = io.load_embeddings(archive(ref, TEST_EMBEDDINGS))
    rs = aggregate.site_spearmans(model['best_model'], model['scalers'], emb, None, [i[2] for i in test])
    positions = rs.drop(index=0)                  # site 0 is the CLS token, not an alignment column
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.scatter(positions.index, positions.values, c=positions.values, cmap='viridis', s=16)
    ax.plot(positions.index, positions.rolling(5, center=True, min_periods=1).mean(), 'k-', lw=1.5)
    ax.set(xlabel='Alignment position', ylabel='Spearman R',
           title=f'Per-site model accuracy on the test set (CLS token: {rs.loc[0]:.3f})')
    sns.despine(ax=ax)
    fig.savefig(out / 'fig3a_site_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    rs.rename('spearman').to_csv(out / 'fig3a_site_spearmans.tsv', sep='\t', index_label='site')
    return [{'figure': 'fig3a', 'dataset': 'test set', 'n': len(rs), 'spearman': rs.median()}]


def figS3(ref, model, scaler, sites, out, test_run=None):
    fig, axes = plt.subplots(1, len(ISOLATES), figsize=(4.5 * len(ISOLATES), 4.5))
    rows = []
    for ax, name in zip(np.atleast_1d(axes), ISOLATES):
        table = validation_dataset(ref, name, model, scaler, sites)
        scatter_panel(ax, table['measured'], table['pred'], f'{name} isolates')
        rows.append(metrics('figS3', name, table['measured'], table['pred']))
    fig.savefig(out / 'figS3_isolates.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    return rows


def fig5(ref, model, scaler, sites, out, test_run=None):
    archived = {n: label for n, label in HABITATS.items()
                if glob.glob(str(Path(ref) / f'intermediates/validation_datasets/{n}/embeddings/*_full_even.pickle'))}
    if len(archived) < len(HABITATS):
        print(f'  not in the archive, skipped: {", ".join(sorted(set(HABITATS) - set(archived)))}')
    proteins = pd.concat([validation_dataset(ref, name, model, scaler, sites).assign(habitat=label)
                          for name, label in archived.items()])
    samples = proteins.groupby(['habitat', 'sample_or_genome'])['pred'].mean().reset_index()
    order = samples.groupby('habitat')['pred'].median().sort_values().index

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw={'width_ratios': [1, 1.8]})
    for habitat in [h for h in EXAMPLE_HABITATS if h in order]:     # largest sample of each example habitat
        in_habitat = proteins[proteins['habitat'] == habitat]
        example = in_habitat['sample_or_genome'].value_counts().index[0]
        sns.kdeplot(in_habitat.loc[in_habitat['sample_or_genome'] == example, 'pred'], ax=ax_a, fill=True, label=habitat)
    ax_a.set(xlabel='Estimated OGT (°C)', title='OGT distributions in single samples')
    ax_a.legend(frameon=False, fontsize=8)
    sns.violinplot(data=samples, x='habitat', y='pred', order=order, ax=ax_b, cut=0, color='0.8')
    for i, habitat in enumerate(order):
        ax_b.text(i, ax_b.get_ylim()[0], f"n = {(samples['habitat'] == habitat).sum()}", ha='center', va='bottom', fontsize=7)
    ax_b.set(xlabel='', ylabel='Mean estimated OGT per sample (°C)', title='OGT distributions in environments')
    ax_b.tick_params(axis='x', rotation=40)
    plt.setp(ax_b.get_xticklabels(), ha='right')
    sns.despine(fig)
    fig.savefig(out / 'fig5_metagenomes.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    summary = samples.groupby('habitat')['pred'].agg(n='size', median='median')
    return [{'figure': 'fig5', 'dataset': h, 'n': int(r['n']), 'median': r['median']} for h, r in summary.iterrows()]


FIGURES = {'fig2a': fig2a, 'fig3a': fig3a, 'figS3': figS3, 'fig5': fig5}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--reference', type=Path, default=Path('reference/v1_preprint'), help='preprint archive')
    ap.add_argument('--model-dir', type=Path, default=Path('models/v1_preprint'),
                    help='folder with model.pickle and scaler.pickle')
    ap.add_argument('--out', type=Path, required=True, help='folder for figures and metrics.tsv')
    ap.add_argument('--sites', choices=['all', 'legacy_fig2a'], default='all',
                    help='sites averaged in predictions; legacy_fig2a is the published test-set selection')
    ap.add_argument('--figures', nargs='+', choices=list(FIGURES), default=list(FIGURES))
    ap.add_argument('--test-run', type=Path,
                    help='roseate run folder holding the test set (2_align dataset, 3_embed embeddings) '
                         'for fig2a and fig3a; default: the archived test set')
    args = ap.parse_args()

    model = io.reload_pickle(args.model_dir / 'model.pickle')
    scaler = io.reload_pickle(args.model_dir / 'scaler.pickle')
    sites = None
    if args.sites == 'legacy_fig2a':
        sites = aggregate.good_cols(aggregate.load_site_spearmans(archive(args.reference, SPEARMANS)), 0.84)
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for name in args.figures:
        print(f'{name}...')
        rows += FIGURES[name](args.reference, model, scaler, sites, args.out, args.test_run)
    table = pd.DataFrame(rows)
    table.to_csv(args.out / 'metrics.tsv', sep='\t', index=False)
    print(table.to_string(index=False, float_format=lambda x: f'{x:.3f}'))


if __name__ == '__main__':
    main()