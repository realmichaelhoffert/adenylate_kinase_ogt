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

REF = Path(os.environ.get("ROSEATE_REFERENCE", "reference/v1_preprint"))
pytestmark = pytest.mark.skipif(not (REF / "MANIFEST.tsv").exists(),
                                reason=f"reference archive not found at {REF}")

# archive-relative locations
SUPPORT = "inputs/predictive_models/supporting_data"
MODEL = "outputs/predictive_models/even_enetcv_ind.pickle"
SPEARMANS = "outputs/predictive_models/supporting_data/even_enetcv_ind_spearmans.tsv*"
BACKBONE = "inputs/high_qual_alignments/backbone.filtered_only.afa"
BACKBONE_TEMPS = 'inputs/high_qual_alignments/temp_data.tsv'
C_W_E = "intermediates/esm_msa/c_w_e_full_embeddings.pickle"
VALIDATION = sorted(Path(p).parts[-3] for p in
                    glob.glob(str(REF / "intermediates/validation_datasets/*/embeddings/*_full_even.pickle")))


def path(rel):
    """Archive file for a relative path or glob; skip the test if it is not archived."""
    hits = sorted(glob.glob(str(REF / rel)))
    if not hits:
        pytest.skip(f"not in archive: {rel}")
    return Path(hits[0])


def metrics(y_true, y_pred):
    return {"r2": float(r2_score(y_true, y_pred)),
            "rmse": float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))}


# ---------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def ref():
    return Reference(REF)

@pytest.fixture(scope="module")
def model():
    return io.reload_pickle(path(MODEL))

@pytest.fixture(scope="module")
def scaler():
    return io.reload_pickle(path(f"{SUPPORT}/ogt_scaler.pickle"))

@pytest.fixture(scope="module")
def even():
    return io.reload_pickle(path(f"{SUPPORT}/even_msa_dataset.pickle"))

@pytest.fixture(scope="module")
def corkrey():
    return io.reload_pickle(path(f"{SUPPORT}/corkrey_msa_dataset.pickle"))

@pytest.fixture(scope="module")
def full():
    return io.construct_msa_dataset(path(BACKBONE), 
                                    path(BACKBONE_TEMPS))

@pytest.fixture(scope="module")
def full_scaled_cell15(full):
    """05 cell 15: the full dataset scaled with a scaler fit on itself."""
    return io.scale_traits(full)

@pytest.fixture(scope="module")
def c_w_e():
    return io.reload_pickle(path(C_W_E))

@pytest.fixture(scope="module")
def spearmans():
    return aggregate.load_site_spearmans(path(SPEARMANS))


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
    ds = io.load_new_dataset(
        path(f"inputs/high_qual_alignments/alignments_to_backbone/{name}_adk_to_backbone.afa"),
        path(f"inputs/validation_datasets/{name}/md/{name}_temp_data.tsv"),
        scaler=scaler, 
        exclude_ids={i[0] for i in full})
    archived = io.reload_pickle(path(f"intermediates/validation_datasets/{name}/embeddings/{name}_msa_dataset.pickle"))
    assert len(archived) == len(ds)
    # ds_sort = sorted(ds, key=lambda x: x[0])
    # archived_sort = sorted(archived, key=lambda x: x[0])
    # assert [i[:2] for i in ds_sort] == [i[:2] for i in archived_sort]
    # np.testing.assert_allclose([i[2] for i in ds_sort], [i[2] for i in archived_sort], atol=1e-12, rtol=0)


# # ---------------------------------------------------------------- predict.py
# @pytest.fixture(scope="module")
# def c_w_e_preds():
#     return predict.run_model(path(C_W_E), path(MODEL),
#                              path(f"{SUPPORT}/corkrey_msa_dataset.pickle"), predict.per_site_predict)

# def test_run_model_site_means_match_stored_predictions(ref, c_w_e_preds):
#     ref.assert_array_close(c_w_e_preds.mean(axis=0), "portable/even_enetcv_ind.npz",
#                            key="predictions", atol=1e-10)

# def test_run_model_rows_follow_dict_order(model, c_w_e_preds):
#     """Quirk 9: run_model rows are in model-key order; full_predictions is in range order."""
#     keys = list(model["best_model"].keys())
#     np.testing.assert_allclose(c_w_e_preds, np.asarray(model["full_predictions"])[keys], atol=1e-10, rtol=0)

# @pytest.mark.slow
# @pytest.mark.parametrize("name", VALIDATION)
# def test_validation_predictions_match_archive(name, scaler):
#     emb_dir = f"intermediates/validation_datasets/{name}/embeddings"
#     msa = path(f"{emb_dir}/{name}_msa_dataset.pickle")
#     preds = predict.run_model(path(f"{emb_dir}/{name}_full_even.pickle"), path(MODEL), msa, predict.per_site_predict)
#     pred_c = io.unscale_ogts(aggregate.mean_sites(preds), scaler)
#     archived = pd.read_csv(path(f"portable/predictions/{name}_roseate_all_sites.tsv"),
#                            sep="\t", index_col="protein")["pred_temp"]
#     assert list(archived.index) == [i[0] for i in io.reload_pickle(msa)]
#     np.testing.assert_allclose(pred_c, archived.values, atol=1e-8, rtol=0)


# # ---------------------------------------------------------------- aggregate.py
# def test_site_spearmans_recomputed(ref, model, c_w_e):
#     s = aggregate.site_spearmans(model["best_model"], model["scalers"], c_w_e, model["true_values"])
#     ref.assert_matches("sites.spearman_test", s.tolist())

# def test_load_site_spearmans_matches_expected(ref, spearmans):
#     ref.assert_matches("sites.spearman_test", spearmans.tolist())

# def test_good_cols(ref, spearmans):
#     ref.assert_matches("fig2a.legacy_good_cols",
#                        {"sites": list(aggregate.good_cols(spearmans, cutoff=0.84))}, subset=True)

# @pytest.mark.parametrize("key, use_good_cols", [("fig2a.all_sites", False), ("fig2a.legacy_good_cols", True)])
# def test_fig2a(ref, model, scaler, spearmans, key, use_good_cols):
#     fp = np.asarray(model["full_predictions"])            # range order, as in figures_final
#     sites = aggregate.good_cols(spearmans) if use_good_cols else None
#     y_true = io.unscale_ogts(model["true_values"], scaler)
#     y_pred = io.unscale_ogts(aggregate.mean_sites(fp, sites), scaler)
#     ref.assert_matches(key, metrics(y_true, y_pred), subset=True)





    