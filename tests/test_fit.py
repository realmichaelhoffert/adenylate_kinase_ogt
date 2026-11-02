"""
Level 2 reproduction tests: roseate.fit against the archived ROSEATE model.

Fast by default (a few sites); full refit with:
    ROSEATE_FULL_REFIT=1 pytest tests/test_level2_fit.py
Training targets are already scaled, so these tests do not depend on which OGT scaler is used.
"""
import os

import numpy as np
import pytest

from roseate import fit, io
from tests.test_initial_predict import C_W_E, MODEL, REF, SUPPORT, path

pytestmark = pytest.mark.skipif(not (REF / "MANIFEST.tsv").exists(),
                                reason=f"reference archive not found at {REF}")

# CLS, early core, middle, and the last-31 block where dict order differs (quirk 9)
SITES = list(range(215)) if os.environ.get("ROSEATE_FULL_REFIT") else [0, 1, 107, 184, 214]
ENETCV_KEYS = ["l1_ratio", "cv", "n_alphas", "max_iter", "selection", "tol", "precompute", "random_state"]


# ---------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def model():
    return io.reload_pickle(path(MODEL))

@pytest.fixture(scope="module")
def even_emb():
    return io.reload_pickle(path("intermediates/esm_msa/even_full_embeddings.pickle"))

@pytest.fixture(scope="module")
def even_y():
    return np.asarray(io.reload_pickle(path("intermediates/esm_msa/even_temps.pickle")))

@pytest.fixture(scope="module")
def corkrey():
    return io.reload_pickle(path(f"{SUPPORT}/corkrey_msa_dataset.pickle"))

@pytest.fixture(scope="module")
def refit(even_emb, even_y):
    """Fit once per module; every comparison below reuses it."""
    return fit.train_per_site(even_emb, even_y, sites=SITES, verbose=True)


# ---------------------------------------------------------------- parameters
def test_enetcv_params_match_archived(model):
    archived = model["best_model"][0].get_params()
    for key in ENETCV_KEYS:
        if key != 'precompute':
            assert fit.ENETCV_PARAMS[key] == archived[key], key

def test_training_targets_match_dataset(even_y):
    even = io.reload_pickle(path(f"{SUPPORT}/even_msa_dataset.pickle"))
    np.testing.assert_array_equal(even_y, [i[2] for i in even])


# ---------------------------------------------------------------- refit
print("SITES", SITES)
@pytest.mark.slow
@pytest.mark.parametrize("site", SITES)
def test_refit_scalers_match(site, refit, model):
    _, scalers = refit
    print(scalers.keys())
    np.testing.assert_array_equal(scalers[site].mean_, model["scalers"][site].mean_)
    np.testing.assert_array_equal(scalers[site].scale_, model["scalers"][site].scale_)

@pytest.mark.slow
@pytest.mark.parametrize("site", SITES)
def test_refit_hyperparameters_match(site, refit, model):
    enets, _ = refit
    assert enets[site].alpha_ == model["best_model"][site].alpha_
    assert enets[site].l1_ratio_ == model["best_model"][site].l1_ratio_

@pytest.mark.slow
@pytest.mark.parametrize("site", SITES)
def test_refit_coefficients_match(site, refit, model):
    enets, _ = refit
    np.testing.assert_allclose(enets[site].coef_, model["best_model"][site].coef_, atol=1e-8, rtol=0)
    np.testing.assert_allclose(enets[site].intercept_, model["best_model"][site].intercept_, atol=1e-8, rtol=0)


# ---------------------------------------------------------------- prediction and packaging
def test_predict_test_reproduces_full_predictions(model):
    """Archived models + archived test embeddings -> stored full_predictions, range order."""
    c_w_e = io.reload_pickle(path(C_W_E))
    preds = fit.test_per_site(model["best_model"], model["scalers"], c_w_e)
    np.testing.assert_allclose(preds, np.asarray(model["full_predictions"]), atol=1e-10, rtol=0)

def test_package_model_reproduces_archived_dict(model, corkrey):
    packaged = fit.package_model(model_label='test_model',
        enets=model["best_model"], scalers=model["scalers"],
        test_preds=model["full_predictions"], test_dataset=corkrey,
        input_shape=model["metadata"]["input_shape"], model_type=model["metadata"]["type"],
        extra_metadata={k: v for k, v in model["metadata"].items() if k not in ("input_shape", "type")},
        extra_fields={k: model[k] for k in model if k not in
                      ("best_model", "scalers", "full_predictions", "predictions",
                       "true_values", "true_labels", "metadata")})
    assert set(packaged) == set(model)
    assert packaged["metadata"] == model["metadata"]
    assert list(packaged["true_labels"]) == list(model["true_labels"])
    np.testing.assert_array_equal(packaged["true_values"], model["true_values"])
    np.testing.assert_allclose(packaged["predictions"], model["predictions"], atol=1e-12, rtol=0)

def test_save_model_round_trip_and_no_overwrite(tmp_path, model):
    target = tmp_path / "model.pickle"
    fit.save_model({"metadata": model["metadata"]}, target)
    assert io.reload_pickle(target)["metadata"] == model["metadata"]
    with pytest.raises(FileExistsError):
        fit.save_model({}, target)
    fit.save_model({}, target, overwrite=True)