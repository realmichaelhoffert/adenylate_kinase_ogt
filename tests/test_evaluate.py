"""
Tests for roseate.evaluate.

Unit tests use toy data. Archive tests check the GTDB taxonomy mapping and that the phylogenetic CV
visits the same taxa as the v1 run (taxonomic_cv_results.tsv). PCV errors are not compared with v1:
its ElasticNet came from glmnet and has been replaced by scikit-learn's ElasticNet and ElasticNetCV.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.svm import SVR

from roseate import evaluate, fit, io
from tests.archive import *

MEAN_MODEL = {'mean': lambda X, y: DummyRegressor().fit(X, y)}   # predicts the training mean
PCV_RESULTS = 'outputs/predictive_models/supporting_data/taxonomic_cv_results.tsv*'
GENOME2TAX = 'inputs/genome2tax.tsv'


@pytest.fixture
def toy():
    taxonomy = pd.DataFrame({'domain': ['Bacteria'] * 6,
                             'phylum': ['P1', 'P1', 'P1', 'P2', 'P2', 'P2'],
                             'genus': ['g1', 'g1', 'g2', 'g3', 'g3', 'g3']}, index=list('abcdef'))
    X = np.arange(12.).reshape(6, 2)
    y = np.array([1., 2., 3., 10., 11., 12.])
    return X, y, taxonomy


def write_metadata(file, rows):
    pd.DataFrame(rows, columns=['accession', 'gtdb_taxonomy']).assign(other=0).to_csv(file, sep='\t', index=False)


# ---------------------------------------------------------------- metrics and plots
def test_assess_model_values():
    label, (r2, rmse, spr) = evaluate.assess_model([1, 2, 3, 4], [1, 2, 3, 5])
    assert (r2, rmse, spr) == pytest.approx((0.8, 0.5, 1.0))
    assert label == '$R^{2}$=0.80|RMSE=0.50|SpR=1.00'

def test_assess_model_pairs_by_position():
    '''Quirk 10: pandas indexes are ignored.'''
    y_true = pd.Series([1., 2., 3.], index=['a', 'b', 'c'])
    y_pred = pd.Series([3., 2., 1.], index=['c', 'b', 'a'])      # same values by label
    assert evaluate.assess_model(y_true, y_pred)[1][2] == pytest.approx(-1.0)

def test_plot_performance_saves(tmp_path):
    evaluate.plot_performance([1, 2, 3], [1.1, 2.2, 2.9], 'test', save_loc=tmp_path / 'p.png')
    assert (tmp_path / 'p.png').stat().st_size > 0


# ---------------------------------------------------------------- taxonomy
def test_load_gtdb_taxonomy_splits_ranks(tmp_path):
    bac, arc = tmp_path / 'bac120_metadata.tsv', tmp_path / 'ar53_metadata.tsv'
    write_metadata(bac, [['RS_GCF_1', 'd__Bacteria;p__Bacillota;c__Bacilli;o__O;f__F;g__Bacillus;s__Bacillus subtilis']])
    write_metadata(arc, [['GB_GCA_2', 'd__Archaea;p__Thermoproteota;c__C;o__O;f__F;g__G;s__G sp1']])
    tax = evaluate.load_gtdb_taxonomy([bac, arc])
    assert list(tax.columns) == evaluate.TAXONOMY_LEVELS
    assert tax.loc['RS_GCF_1', 'genus'] == 'Bacillus'
    assert tax.loc['GB_GCA_2', 'species'] == 'G sp1'

def test_load_gtdb_taxonomy_custom_columns(tmp_path):
    f = tmp_path / 'md.tsv'
    pd.DataFrame({'genome': ['X'], 'lineage': ['d__Bacteria;p__P']}).to_csv(f, sep='\t', index=False)
    tax = evaluate.load_gtdb_taxonomy(f, accession_col='genome', taxonomy_col='lineage')
    assert list(tax.columns) == ['domain', 'phylum'] and tax.loc['X', 'phylum'] == 'P'

def test_taxonomy_for_dataset_indexes_by_sequence():
    tax = pd.DataFrame({'phylum': ['P1', 'P2']}, index=['RS_GCF_1', 'RS_GCF_2'])
    dataset = [('p9|RS_GCF_2|adk|temp=30.0', 'M-K', 0.1), ('p1|RS_GCF_1|adk|temp=40.0', 'MKK', 0.5)]
    rows = evaluate.taxonomy_for_dataset(dataset, tax)
    assert list(rows.index) == [d[0] for d in dataset] and list(rows['phylum']) == ['P2', 'P1']


# ---------------------------------------------------------------- phylogenetic CV
def test_process_taxon_holds_out_one_taxon(toy):
    X, y, taxonomy = toy
    [row] = evaluate.process_taxon('genus', 'g2', X, y, taxonomy['genus'], MEAN_MODEL)
    expected = np.sqrt(np.mean((y[2] - np.mean(np.delete(y, 2))) ** 2))
    assert row == {'level': 'genus', 'taxon': 'g2', 'model': 'mean', 'error': pytest.approx(expected)}

def test_taxonomic_cv_one_row_per_taxon_and_model(toy):
    X, y, taxonomy = toy
    models = {**MEAN_MODEL, 'mean2': MEAN_MODEL['mean']}
    results = evaluate.taxonomic_cv(X, y, taxonomy, models, n_jobs=1)
    assert list(results.columns) == ['level', 'taxon', 'model', 'error']
    assert len(results) == (2 + 3) * 2                     # domain skipped, as in v1
    assert set(results['level']) == {'phylum', 'genus'}


# ---------------------------------------------------------------- archive
@pytest.fixture(scope='module')
def even():
    return io.reload_pickle(path(f'{SUPPORT}/even_msa_dataset.pickle'))

@pytest.fixture(scope='module')
def even_taxonomy(even):
    return evaluate.taxonomy_for_dataset(even, evaluate.load_gtdb_taxonomy(path(GENOME2TAX)))

@pytest.fixture(scope='module')
def archived_pcv():
    return pd.read_csv(path(PCV_RESULTS), sep='\t', index_col=0)


def test_archived_taxonomy_covers_training_and_test_sets(even_taxonomy):
    corkrey = io.reload_pickle(path(f'{SUPPORT}/corkrey_msa_dataset.pickle'))
    corkrey_taxonomy = evaluate.taxonomy_for_dataset(corkrey, evaluate.load_gtdb_taxonomy(path(GENOME2TAX)))
    for tax in [even_taxonomy, corkrey_taxonomy]:
        assert list(tax.columns) == evaluate.TAXONOMY_LEVELS and not tax.isna().any().any()

def test_pcv_visits_the_same_taxa_as_v1(even, even_taxonomy, archived_pcv):
    y = np.array([i[2] for i in even])
    results = evaluate.taxonomic_cv(X=np.zeros((len(even), 1)), y=y, 
                                    taxonomy=even_taxonomy, 
                                    models=MEAN_MODEL, 
                                    levels=['phylum', 'class', 'order', 'family', 'genus'],
                                    n_jobs=1)
    v1 = archived_pcv[archived_pcv['model'] == 'ElasticNet']
    as_pairs = lambda df: sorted(zip(df['level'], df['taxon'].astype(str)))
    assert as_pairs(results) == as_pairs(v1)

# ---------------------------------------------------------------- PCV models
@pytest.fixture
def pcv_models():
    return evaluate.legacy_pcv_models(SVR(), RandomForestRegressor(n_estimators=10),
                                      {'alpha': 0.05, 'l1_ratio': 0.5})

def test_pcv_models_are_the_paper_architectures(pcv_models):
    assert list(pcv_models) == ['ElasticNet', 'ElasticNetCV', 'SVR', 'RF', 'LM', 'GPR', 'NN']

def test_pcv_elasticnets_are_sklearn_with_roseate_settings(pcv_models, toy):
    X, y, _ = toy
    enet, enetcv = pcv_models['ElasticNet'](X, y), pcv_models['ElasticNetCV'](X, y)
    assert type(enet) is ElasticNet and (enet.alpha, enet.l1_ratio) == (0.05, 0.5)
    assert enet.random_state == fit.ENETCV_PARAMS['random_state']
    assert type(enetcv) is ElasticNetCV
    expected = {**fit.ENETCV_PARAMS, 'n_jobs': 1}
    assert {k: enetcv.get_params()[k] for k in expected} == expected

def test_elasticnet_pcv_is_deterministic(pcv_models):
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(40, 5)), rng.normal(size=40)
    taxonomy = pd.DataFrame({'domain': ['B'] * 40, 'phylum': np.repeat(['P1', 'P2', 'P3', 'P4'], 10)})
    models = {k: pcv_models[k] for k in ['ElasticNet', 'ElasticNetCV']}
    first = evaluate.taxonomic_cv(X, y, taxonomy, models, n_jobs=1)
    second = evaluate.taxonomic_cv(X, y, taxonomy, models, n_jobs=2)
    pd.testing.assert_frame_equal(first, second)

def test_pcv_features_are_scaled_cls_float64():
    emb = np.random.default_rng(0).normal(size=(20, 3, 4)).astype(np.float32)
    X = evaluate.legacy_pcv_features(emb)
    assert X.dtype == np.float64 and X.shape == (20, 4)
    np.testing.assert_allclose(X.mean(axis=0), 0, atol=1e-7)
