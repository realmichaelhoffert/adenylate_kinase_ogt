# Authors: Michael Hoffert and Claude Opus 5.5
# Code design informed by Claude Opus 5.5

"""
Level 1 reproduction tests: roseate.io, roseate.predict and roseate.aggregate
against the frozen v1 reference archive.

Run from the repo root, in the original environment:
    pytest tests/test_level1_predict.py                 # everything
    pytest tests/test_level1_predict.py -m "not slow"   # skip validation-dataset embeddings
Override the archive location with ROSEATE_REFERENCE=/path/to/v1_preprint.
"""
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import r2_score

from reference.verify_reference import Reference
from roseate import aggregate, io, predict
from tests.archive import *
from tests.conftest import even


def metrics(y_true, y_pred):
    return {"r2": float(r2_score(y_true, y_pred)),
            "rmse": float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))}


# ---------------------------------------------------------------- io.py
def test_full_dataset_has_8532_records(full):
    assert len(full) == 8532

@pytest.mark.parametrize("split", ["even", "corkrey"])
def test_split_ids_and_sequences_come_from_backbone(split, full, request):
    '''
    check all datasets contain sequences in backbone alignment
    '''
    lookup = {i[0]: i[1] for i in full}
    for seq_id, seq, _ in request.getfixturevalue(split):
        assert lookup[seq_id] == seq

@pytest.mark.parametrize("split", ["even", "corkrey"])
def test_split_ogts_match_cell15_scaling(split, full_scaled_cell15, request):
    '''
    check that scaled dataset passes
    '''
    scaled, _ = full_scaled_cell15
    lookup = {i[0]: i[2] for i in scaled}
    archived = request.getfixturevalue(split)
    np.testing.assert_allclose([i[2] for i in archived], [lookup[i[0]] for i in archived], atol=1e-12, rtol=0)

# @pytest.mark.xfail(reason="quirk 3: archived scaler was fit on 8,831 records, not this alignment", strict=False)
def test_archived_scaler_equals_cell15_scaler(full_scaled_cell15, scaler):
    _, cell15 = full_scaled_cell15
    np.testing.assert_allclose([cell15.mean_[0], cell15.scale_[0]], [scaler.mean_[0], scaler.scale_[0]])

def test_subset_corkrey_reproduces_ids(full_scaled_cell15, corkrey):
    '''
    test that corkrey can be reproduced
    '''
    scaled, _ = full_scaled_cell15
    ds, _ = io.subset_corkrey(scaled, path("inputs/extra/corkey_gtdb_matching.tsv"),
                              path("inputs/extra/Corkey_temperatures.csv"))
    assert [i[0] for i in ds] == [i[0] for i in corkrey]

def test_sample_even_reproduces_ids(full_scaled_cell15, even):
    '''
    test that 
    '''
    scaled, _ = full_scaled_cell15
    corkrey_ds, corkrey_ids = io.subset_corkrey(scaled, 
                                       path("inputs/extra/corkey_gtdb_matching.tsv"),
                                       path("inputs/extra/Corkey_temperatures.csv"))
    ds, ds_ids = io.sample_even(scaled, corkrey_ids)

    # make sure even is completely unique and all in DS
    len_even = len([i[0] for i in even])
    assert len_even == len(set([i[0] for i in even]))
    assert len_even == len(set([i[0] for i in even]).intersection([i[0] for i in ds]))
    # check none in corkrey
    assert len(set([i[0] for i in even]).intersection([i[0] for i in corkrey_ds])) == 0

def test_unscale_round_trip(scaler, corkrey):
    values = np.array([i[2] for i in corkrey])
    back = (io.unscale_traits(values, scaler) - scaler.mean_[0]) / scaler.scale_[0]
    np.testing.assert_allclose(back, values, atol=1e-12, rtol=0)

@pytest.mark.slow
@pytest.mark.parametrize("name", VALIDATION)
def test_load_new_dataset_reproduces_archived(name, full, scaler):
    # construct data
    # for this test, trait_data goes in
    ds = io.load_new_dataset(
        path(f"inputs/high_qual_alignments/alignments_to_backbone/{name}_adk_to_backbone.afa"),
        trait_data=path(f"inputs/validation_datasets/{name}/md/{name}_temp_data.tsv"),
        scaler=scaler, 
        exclude_ids={i[0] for i in full})
    # load archieved version
    archived = io.reload_pickle(path(f"intermediates/validation_datasets/{name}/embeddings/{name}_msa_dataset.pickle"))
    # drop nas from ds

    # check lengths
    assert len(archived) == len(ds)
    ds_sort = sorted(ds, key=lambda x: x[0])
    archived_sort = sorted(archived, key=lambda x: x[0])
    # make sure scaled values are identical
    assert [i[:2] for i in ds_sort] == [i[:2] for i in archived_sort]
    np.testing.assert_allclose([i[2] for i in ds_sort], [i[2] for i in archived_sort], atol=1e-12, rtol=0)


# ---------------------------------------------------------------- predict.py
@pytest.fixture(scope="module")
def c_w_e_preds():
    return predict.run_model(path(C_W_E), path(MODEL),
                             path(f"{SUPPORT}/corkrey_msa_dataset.pickle"), predict.per_site_predict)

def test_run_model_site_means_match_stored_predictions(ref, c_w_e_preds):
    '''
    make sure predictions are the same
    '''
    ref.assert_array_close(c_w_e_preds.mean(axis=0), "portable/even_enetcv_ind.npz",
                           key="predictions", atol=1e-5)

def test_run_model_rows_follow_dict_order(model, c_w_e_preds):
    """Quirk 9: run_model rows are in model-key order; full_predictions is in range order."""
    '''
    FIXED
    '''
    keys = list(sorted(model["best_model"].keys()))
    np.testing.assert_allclose(c_w_e_preds, np.asarray(model["full_predictions"])[keys], atol=1e-5, rtol=0)

@pytest.mark.slow
@pytest.mark.parametrize("name", VALIDATION)
def test_validation_predictions_match_archive(name, spearmans, scaler, archived_scaler):
    '''
    make sure all validation dataset predictions are the same
    '''
    emb_dir = f"intermediates/validation_datasets/{name}/embeddings"
    msa = path(f"{emb_dir}/{name}_msa_dataset.pickle")
    preds = predict.run_model(path(f"{emb_dir}/{name}_full_even.pickle"), path(MODEL), msa, predict.per_site_predict)
    _spearmans = spearmans
    # cols = _spearmans[_spearmans > 0.84].index
    # pred_c = io.unscale_traits(aggregate.mean_sites(preds, cols), archived_scaler)
    pred_c = io.unscale_traits(aggregate.mean_sites(preds), archived_scaler)

    archived = pd.read_csv(path(f"portable/predictions/{name}_roseate_all_sites.tsv"),
                           sep="\t", index_col="protein")["pred_temp"]
    assert list(archived.index) == [i[0] for i in io.reload_pickle(msa)]
    np.testing.assert_allclose(pred_c, archived.values, atol=1e-5, rtol=1e-6)


# ---------------------------------------------------------------- aggregate.py
def test_site_spearmans_recomputed(ref, model, c_w_e):
    s = aggregate.site_spearmans(model["best_model"], 
                                 model["scalers"], c_w_e, 
                                 model['true_labels'],
                                 model["true_values"])
    ref.assert_matches("sites.spearman_test", s.tolist())

def test_load_site_spearmans_matches_expected(ref, spearmans):
    ref.assert_matches("sites.spearman_test", spearmans.tolist())

def test_good_cols(ref, spearmans):
    ref.assert_matches("fig2a.legacy_good_cols",
                       {"sites": list(aggregate.good_cols(spearmans, cutoff=0.84))}, subset=True)

@pytest.mark.parametrize("key, use_good_cols", [("fig2a.all_sites", False), ("fig2a.legacy_good_cols", True)])
def test_fig2a(ref, model, scaler, archived_scaler, spearmans, key, use_good_cols):
    fp = np.asarray(model["full_predictions"])          # range order, as in figures_final
    sites = aggregate.good_cols(spearmans, 0.84) if use_good_cols else None
    # print('LENGTH OF SITES', len(sites))
    y_true = io.unscale_traits(model["true_values"], archived_scaler)
    # aggregate
    y_pred = io.unscale_traits(aggregate.mean_sites(fp, sites), archived_scaler)
    ref.assert_matches(key, metrics(y_true, y_pred), subset=True)





    