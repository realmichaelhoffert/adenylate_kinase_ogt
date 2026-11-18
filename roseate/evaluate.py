# Model evaluation: metrics, performance plots, GTDB taxonomy and phylogenetic cross-validation
# Written by Michael Hoffert, refactored with Claude Opus 5.5
# October 2026

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

TAXONOMY_LEVELS = ['domain', 'phylum', 'class', 'order', 'family', 'genus', 'species']


# ---------------------------------------------------------------- metrics and plots
def assess_model(y_true, y_pred):
    '''
    R2, RMSE and Spearman R of predictions.
    Returns (label string for plot titles, (r2, rmse, spearman)).
    Note: pairs values by position, not by index (quirk 10); align pandas inputs first.
    '''
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    _spearmanr = spearmanr(y_true, y_pred)[0]
    return '$R^{2}$=' + f'{r2:.2f}|RMSE={rmse:.2f}|SpR={_spearmanr:.2f}', (r2, rmse, _spearmanr)


def plot_performance(y_true, y_pred, title, show=False, save_loc=None):
    '''
    Scatter of predicted vs. true values with an x=y line, R2 and RMSE in the title.
    '''
    import matplotlib.pyplot as plt   # imported here so the module loads without plotting libraries
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(y_true, y_pred, alpha=0.7, edgecolors='w')

    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    ax.set_title(f"{title}\n$R^{2}$: {r2:.2f} | RMSE: {rmse:.2f}")
    ax.tick_params(labelsize=14, axis='both')
    ax.set_xlabel("True", fontsize=16)
    ax.set_ylabel("Predicted", fontsize=16)
    sns.despine()
    ax.set_aspect('equal')

    # square axes padded by 10% of the shared range
    _min = np.floor(min(np.min(y_true), np.min(y_pred)))
    _max = np.floor(max(np.max(y_true), np.max(y_pred)))
    _range = np.abs(_max - _min)
    ax.set_xlim(_min - _range * .1, _max + _range * .1)
    ax.set_ylim(_min - _range * .1, _max + _range * .1)
    ax.plot((_min + 0.5, _max - 0.5), (_min + 0.5, _max - 0.5), 'k--', label='x=y')

    if save_loc is not None:
        plt.savefig(save_loc, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------- taxonomy
def load_gtdb_taxonomy(metadata_files, accession_col='accession', taxonomy_col='gtdb_taxonomy'):
    '''
    GTDB taxonomy split into one column per rank, indexed by genome accession.
    metadata_files: path or list of paths, e.g. bac120_metadata.tsv.gz and ar53_metadata.tsv.gz
        from any GTDB release, or the archive's inputs/genome2tax.tsv (same two columns).
    Column names are arguments so other releases or taxonomy tables can be used.
    '''
    if isinstance(metadata_files, (str, bytes)) or not hasattr(metadata_files, '__iter__'):
        metadata_files = [metadata_files]
    md = pd.concat([pd.read_csv(f, sep='\t', usecols=[accession_col, taxonomy_col], low_memory=False)
                    for f in metadata_files])
    ranks = md[taxonomy_col].str.split(';', expand=True)
    ranks = ranks.apply(lambda col: col.str.split('__').str[-1])   # 'p__Bacillota' -> 'Bacillota'
    ranks.columns = TAXONOMY_LEVELS[:ranks.shape[1]]
    ranks.index = md[accession_col].values
    return ranks


def taxonomy_for_dataset(dataset, taxonomy):
    '''
    Taxonomy rows for each sequence in an MSADataset, indexed by sequence ID (05: bac120_metadata).
    Sequence IDs are 'protein|genome accession|...'; every genome must be in the taxonomy.
    '''
    ids = [i[0] for i in dataset]
    rows = taxonomy.loc[[i.split('|')[1] for i in ids]]
    rows.index = ids
    return rows


# ---------------------------------------------------------------- phylogenetic cross-validation
def process_taxon(level, taxon, X, y, taxa, models):
    '''
    Leave one taxon out: fit each model on all other taxa and score it on this one.
    level: rank name, recorded in the output
    taxa: rank label of every row of X
    models: {name: function(X_train, y_train) -> fitted object with .predict}
    Returns one {'level', 'taxon', 'error', 'model'} dict per model; error is RMSE in units of y.
    '''
    is_test = (np.asarray(taxa) == taxon)
    if not is_test.any():
        return []
    X_train, X_test = X[~is_test], X[is_test]
    y_train, y_test = y[~is_test], y[is_test]

    results = []
    for name, fit_fn in models.items():
        pred = np.ravel(fit_fn(X_train, y_train).predict(X_test))
        results.append({'level': level, 'taxon': taxon, 'model': name,
                        'error': np.sqrt(mean_squared_error(y_test, pred))})
    return results


def taxonomic_cv(X, y, taxonomy, models, levels=None, n_jobs=3):
    '''
    Leave-one-taxon-out cross-validation at each taxonomic level (05: def process_taxon).
    X, y: features and targets, rows in the same order as taxonomy
    taxonomy: DataFrame of rank labels, e.g. from taxonomy_for_dataset
    levels: ranks to test; default every rank except the first (v1 skipped domain)
    Returns a DataFrame with columns level, taxon, model, error.
    '''
    X, y = np.asarray(X), np.asarray(y)
    levels = list(taxonomy.columns[1:]) if levels is None else levels
    results = []
    for level in levels:
        taxa = taxonomy[level].values
        per_taxon = Parallel(n_jobs=n_jobs)(
            delayed(process_taxon)(level, t, X, y, taxa, models) for t in tqdm(pd.unique(taxa), desc=level))
        results += [r for res in per_taxon for r in res]
    return pd.DataFrame(results, columns=['level', 'taxon', 'model', 'error'])


# ---------------------------------------------------------------- PCV models (Fig S4)
def legacy_pcv_features(train_emb):
    '''
    PCV features: CLS embedding (site 0) scaled over the whole training set,
    test taxa included (quirk 5). float64 avoids scikit-learn's float32 Gram-matrix check failing.
    '''
    return StandardScaler().fit_transform(np.asarray(train_emb[:, 0, :], dtype=np.float64))


# one fit function per model; each imports its own dependencies, so statsmodels and torch
# are needed only if LM or NN is run
def _fit_lm(X, y):
    import statsmodels.api as sm
    return sm.WLS(y, X).fit()


def _fit_gpr(X, y):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
    kernel = C(1.0) * RBF(1.0) + WhiteKernel(noise_level=1e-1)
    return GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, random_state=1234).fit(X, y)


def _fit_nn(X, y):
    from roseate.nn_models import MLPRegressor, train_model
    return train_model(X, y, MLPRegressor, epochs=200, batch_size=32, lr=1e-4)


def legacy_pcv_models(svr_model, rf_model, enet_params):
    '''
    The PCV models (Fig S4) as {name: fit function} for taxonomic_cv.
    svr_model, rf_model: fitted SVR and RF from the Table S2 pickles
        (even_svr_means.pickle, even_rf_means.pickle: ['best_model']); their parameters are reused.
    enet_params: {'alpha': ..., 'l1_ratio': ...} for the fixed-penalty scikit-learn ElasticNet,
        e.g. ROSEATE's CLS model: model['best_model'][0].alpha_ and .l1_ratio_
    ElasticNetCV uses ROSEATE's settings (fit.ENETCV_PARAMS) and reselects the penalty in each fold.
    v1 used glmnet's ElasticNet here; both ElasticNets are now scikit-learn's and seeded.
    NN and RF remain unseeded, as in v1.
    '''
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import ElasticNet, ElasticNetCV
    from sklearn.svm import SVR
    from roseate.fit import ENETCV_PARAMS

    svr_params = {k: getattr(svr_model, k) for k in ['C', 'degree', 'epsilon', 'gamma', 'kernel']}
    rf_params = {k: getattr(rf_model, k) for k in ['criterion', 'max_depth', 'min_samples_split', 'n_estimators']}
    rf_params['n_jobs'] = 12
    enet_params = {**ENETCV_PARAMS, **enet_params}       # same solver settings as ElasticNetCV
    enet_params = {k: enet_params[k] for k in ['alpha', 'l1_ratio', 'max_iter', 'selection', 'tol',
                                               'precompute', 'random_state']}
    enetcv_params = {**ENETCV_PARAMS, 'n_jobs': 1}       # taxa already run in parallel

    return {
        'ElasticNet': lambda X, y: ElasticNet(**enet_params).fit(X, y),
        'ElasticNetCV': lambda X, y: ElasticNetCV(**enetcv_params).fit(X, y),
        'SVR': lambda X, y: SVR(**svr_params).fit(X, y),
        'RF': lambda X, y: RandomForestRegressor(**rf_params).fit(X, y),
        'LM': _fit_lm,
        'GPR': _fit_gpr,
        'NN': _fit_nn,
    }
