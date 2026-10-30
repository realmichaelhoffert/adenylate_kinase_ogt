# Tests for command line interface
# Written by Michael Hoffert and Claude Opus 5.5
"""
CLI wiring tests: roseate.cli calls the module functions and writes their
results unchanged. Logic is tested elsewhere; these only check plumbing.
"""

import json
import numpy as np
import pandas as pd
import pytest
from roseate import aggregate, cli, io, predict
from tests.test_initial_predict import MODEL, REF, SUPPORT, VALIDATION, path

pytestmark = pytest.mark.skipif(not VALIDATION, reason="no validation datasets in archive")
NAME = VALIDATION[0] if VALIDATION else None

def emb(name):
    return path(f"intermediates/validation_datasets/{name}/embeddings/{name}_full_even.pickle")

def msa(name):
    return path(f"intermediates/validation_datasets/{name}/embeddings/{name}_msa_dataset.pickle")

def scaler_path():
    return path(f"{SUPPORT}/ogt_scaler.pickle")      # same file for CLI and module calls

def predict_args(out, **extra):
    args = ["predict", "--model", str(path(MODEL)), "--emb", str(emb(NAME)),
            "--msa-dataset", str(msa(NAME)), "--scaler", str(scaler_path()), "--out", str(out)]
    for k, v in extra.items():
        args += [f"--{k.replace('_', '-')}", str(v)]
    return args


@pytest.mark.slow
def test_predict_matches_module_calls(tmp_path):
    out = tmp_path / "preds.tsv"
    cli.main(predict_args(out))

    preds = predict.run_model(emb(NAME), path(MODEL), msa(NAME), predict.per_site_predict)
    expected = io.unscale_traits(aggregate.mean_sites(preds), io.reload_pickle(scaler_path()))
    ids = [i[0] for i in io.reload_pickle(msa(NAME))]

    got = pd.read_csv(out, sep="\t", index_col="protein")["pred_ogt_C"]
    assert list(got.index) == ids
    np.testing.assert_allclose(got.values, expected, atol=1e-6, rtol=1e-7)


@pytest.mark.slow
def test_predict_writes_run_record(tmp_path):
    out = tmp_path / "preds.tsv"
    cli.main(predict_args(out))
    record = json.loads((tmp_path / "run.json").read_text())
    assert record["args"]["sites"] == "all"
    assert str(path(MODEL)) in record["inputs"]


@pytest.mark.slow
def test_per_site_output(tmp_path):
    out, per_site = tmp_path / "preds.tsv", tmp_path / "per_site.npy"
    cli.main(predict_args(out, per_site=per_site))
    preds = predict.run_model(emb(NAME), path(MODEL), msa(NAME), predict.per_site_predict)
    np.testing.assert_array_equal(np.load(per_site), preds)


@pytest.mark.slow
def test_config_supplies_defaults_and_flags_override(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(f"predict:\n  model: {path(MODEL)}\n  scaler: {scaler_path()}\n  cutoff: 0.5\n")
    out = tmp_path / "preds.tsv"
    cli.main(["predict", "--config", str(config), "--emb", str(emb(NAME)),
              "--msa-dataset", str(msa(NAME)), "--out", str(out), "--cutoff", "0.9"])
    record = json.loads((tmp_path / "run.json").read_text())
    assert record["args"]["model"] == str(path(MODEL))      # from config
    assert record["args"]["cutoff"] == "0.9"                # flag wins


def test_missing_required_argument_exits(tmp_path):
    with pytest.raises(SystemExit):
        cli.main(["predict", "--out", str(tmp_path / "preds.tsv")])