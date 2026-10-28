#!/usr/bin/env python
"""
Build the frozen v1 (preprint) reference archive for ROSEATE.

Run ONCE, in the environment that produced the preprint (python 3.10,
scikit-learn 1.4.1.post1, numpy 1.23.5, pandas 1.5.3), from anywhere:

    python reference/build_reference.py \
        --repo /data/mhoffert/fiererlab/adenylate_kinase_ogt \
        --out  reference/v1_preprint \
        --gtdb-metadata /data/mhoffert/genomes/GTDB_r214.1/bac120_metadata.tsv.gz \
        --extra-input /data/mhoffert/fiererlab/ogt/results/db/corkey_gtdb_matching.tsv \
        --extra-input /data/mhoffert/fiererlab/ogt/source_data/Corkey_temperatures.csv \
        --html --freeze

What it does (nothing in the repo is modified):
  1. records the environment (conda, pip, git, system, ESM weight hashes)
  2. copies inputs / intermediates / outputs / notebooks into the archive
  3. writes portable copies (.npy/.tsv/.json/.npz) readable without this env
  4. runs consistency checks showing the archived files belong together
  5. computes every reported number into expected.json (never typed by hand)
  6. writes MANIFEST.tsv (sha256 of every file) and BUILD_INFO.json
  7. optionally makes the archive read-only (--freeze)

The analysis functions below (per_site_predict, unscale_ogts, assess_model,
and the figure logic) are COPIED from the v1 code, quirks included, so the
archive does not depend on future versions of src/. Do not "fix" them here.
"""
import argparse
import datetime
import glob
import hashlib
import json
import os
import pickle
import platform
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ARCHIVE_VERSION = "v1_preprint"
N_SITES = 215                 # CLS + 214 alignment columns (quirk 2)
GOOD_COLS_CUTOFF = 0.84       # figures_final cell 74 (quirk 1)
ISOLATE_DATASETS = ("Arthrobacter", "Mycobacteria")
# figures_final cell 394
DATASET2LABEL = {
    "Antarctic": "Antarctic soil",
    "Australia": "Australian soil",
    "Colman2024": "Hot springs",
    "RoyoLlonch2021": "Arctic ocean (TARA)",
    "Shah2024": "European freshwater lakes",
    "Youngblut2020": "Avian gut",
    "Conteville2024": "Rumen",
    "Mara2023": "Geothermal sediments",
    "Salazar2019": "Arctic ocean (TARA)",
}
# which dataset each saved (id, seq) context was drawn from
CONTEXT_SOURCE = {
    "even_full_context": "even",      # 05_esm2_model_v2 cell 49, self=True
    "corkrey_full_context": "corkrey",  # cell 46, self=True
    "c_w_e_context": "even",          # cell 53, self=False
    "e_w_c_context": "corkrey",       # cell 53 (save was commented out)
}

# (role, glob relative to --data-root, provenance, required)
SPEC = [
    # ---- inputs
    ("inputs", "high_qual_alignments/backbone.filtered_only.afa", "backbone alignment (README / data/README.txt)", True),
    ("inputs", "high_qual_alignments/backbone.afa", "backbone incl. outgroup", True),
    ("inputs", "high_qual_alignments/alignments_to_backbone/*_adk_to_backbone.afa", "mafft --keeplength --add into backbone", True),
    ("inputs", "validation_datasets/*/md/*_env_temps.tsv", "per-protein env/measured temps", True),
    ("inputs", "validation_datasets/adk_afa_hashes.tsv", "05_esm2_model_v2 cell 274", False),
    ("inputs", "predictive_models/supporting_data/even_msa_dataset.pickle", "05_esm2_model_v2 cell 38 (quirk 6)", True),
    ("inputs", "predictive_models/supporting_data/corkrey_msa_dataset.pickle", "05_esm2_model_v2 cell 38", True),
    ("inputs", "predictive_models/supporting_data/even_genome_ids.txt", "05_esm2_model_v2 cell 37", True),
    ("inputs", "predictive_models/supporting_data/corkrey_genome_ids.txt", "05_esm2_model_v2 cell 37", True),
    ("inputs", "predictive_models/supporting_data/ogt_scaler.pickle", "fit on 8,831 records; not regenerable (quirk 3)", True),
    ("inputs", "PF00406.hmm", "Pfam 2023 ADK HMM", True),
    # ---- intermediates
    ("intermediates", "esm_msa/even_full_embeddings.pickle", "05_esm2_model_v2 cell 49-50", True),
    ("intermediates", "esm_msa/even_temps.pickle", "05_esm2_model_v2 cell 50", True),
    ("intermediates", "esm_msa/even_full_context.pickle", "05_esm2_model_v2 cell 50 (quirk 4)", True),
    ("intermediates", "esm_msa/c_w_e_full_embeddings.pickle", "Corkrey test set embedded with Even context, cell 53/58", True),
    ("intermediates", "esm_msa/c_w_e_temps.pickle", "cell 58", True),
    ("intermediates", "esm_msa/c_w_e_context.pickle", "cell 58 (quirk 4)", True),
    ("intermediates", "esm_msa/corkrey_full_embeddings.pickle", "cell 46-47 (Corkrey-trained models, Table S2)", False),
    ("intermediates", "esm_msa/corkrey_temps.pickle", "cell 47", False),
    ("intermediates", "esm_msa/corkrey_full_context.pickle", "cell 47", False),
    ("intermediates", "esm_msa/e_w_c_full_embeddings.pickle", "cell 53/59 (Corkrey-trained models, Table S2)", False),
    ("intermediates", "esm_msa/e_w_c_temps.pickle", "cell 59", False),
    ("intermediates", "validation_datasets/*/embeddings/*_full_even.pickle", "05_esm2_model_v2 cell 276", True),
    ("intermediates", "validation_datasets/*/embeddings/*_msa_dataset.pickle", "05_esm2_model_v2 cell 276", True),
    ("intermediates", "validation_datasets/*/embeddings/*_temps.pickle", "05_esm2_model_v2 cell 276", True),
    ("intermediates", "validation_datasets/*/embeddings/*_even_context_values.pickle", "05_esm2_model_v2 cell 276 (quirk 4)", True),
    # ---- outputs
    ("outputs", "predictive_models/even_enetcv_ind.pickle", "ROSEATE, 05_esm2_model_v2 cell 131-132", True),
    ("outputs", "predictive_models/supporting_data/even_enetcv_ind_spearmans.tsv*", "05_esm2_model_v2 cell 138; read by figures_final cell 74", True),
    ("outputs", "predictive_models/supporting_data/even_enet_ind_spearmans.tsv*", "05_esm2_model_v2 cell 110", False),
    ("outputs", "predictive_models/supporting_data/even_lm_ind_spearmans.tsv*", "05_esm2_model_v2 cell 84", False),
    ("outputs", "predictive_models/supporting_data/taxonomic_cv_results.tsv.gz", "Fig S4 PCV, 05_esm2_model_v2 cell 177 (quirk 5)", True),
    ("outputs", "predictive_models/supporting_data/enet_col0_boostrap_coefs.tsv", "05_esm2_model_v2 cell 144", False),
    ("outputs", "validation_datasets/prediction_results/*", "05_esm2_model_v2 cell 279", False),
    ("outputs", "validation_datasets/**/*tome*.tsv", "TOME predictions (Fig 2B, S3)", False),
    ("outputs", "phylogenetic_model/*", "phylogenetic nearest-neighbour model (Fig 2C)", False),
]
TABLE_S2_SPEC = [
    ("outputs", "predictive_models/*.pickle", "Table S2 model set", False),
    ("outputs", "predictive_models/*_state_dict.pt", "Table S2 NN weights", False),
]
NOTEBOOKS = ["05_esm2_model_v2.ipynb", "figures_final.ipynb", "dev_validation_models.ipynb",
             "04_environmental_data.ipynb", "06_isolate_validation.ipynb", "phylogenetic_model.ipynb",
             "run_tomes.ipynb"]


# --------------------------------------------------------------------------
# v1 functions, copied verbatim in behaviour (src/models/predict.py, utils.py)
# --------------------------------------------------------------------------
def per_site_predict(model, emb, scalers):
    """Copy of src/models/predict.py::per_site_predict (v1)."""
    preds = []
    for col in sorted(model.keys()):
        if scalers is not None:
            site_emb = scalers[col].transform(emb[:, col, :])
        else:
            site_emb = emb[:, col, :]
        preds.append(model[col].predict(site_emb))
    return preds


def unscale_ogts(ogt_list, scaler):
    """Copy of src/models/utils.py::unscale_ogts (v1), including reshape(1, -1)."""
    return scaler.inverse_transform(np.array(ogt_list).reshape(1, -1))[0]


def assess_model(y_true, y_pred):
    """Copy of src/models/utils.py::assess_model (v1). Positional: no index alignment."""
    from scipy.stats import spearmanr
    from sklearn.metrics import mean_squared_error, r2_score
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    _spearmanr = spearmanr(y_true, y_pred)[0]
    return float(r2), float(rmse), float(_spearmanr)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def load_pickle(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def run(cmd):
    try:
        out = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
        return out.stdout + (("\n# stderr:\n" + out.stderr) if out.returncode else "")
    except Exception as e:  # noqa: BLE001
        return f"# failed: {cmd}\n# {e!r}\n"


class Recorder:
    """Collects consistency checks and expected values."""

    def __init__(self):
        self.checks = []
        self.expected = {}
        self.provenance = {}   # archive relpath -> (source, provenance)

    def check(self, name, passed, **detail):
        status = "PASS" if passed else "FAIL"
        self.checks.append({"name": name, "status": status, **detail})
        print(f"  [{status}] {name}" + (f"  {detail}" if detail and not passed else ""))

    def skip(self, name, reason):
        self.checks.append({"name": name, "status": "SKIP", "reason": reason})
        print(f"  [SKIP] {name}: {reason}")

    def put(self, key, value, computed_by, atol=1e-8, note=None):
        entry = {"value": value, "atol": atol, "rtol": 0.0, "computed_by": computed_by}
        if note:
            entry["note"] = note
        self.expected[key] = entry


def to_builtin(x):
    if isinstance(x, dict):
        return {str(k): to_builtin(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_builtin(v) for v in x]
    if isinstance(x, np.ndarray):
        return to_builtin(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    return x


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(to_builtin(obj), f, indent=2)


# --------------------------------------------------------------------------
# 1. environment
# --------------------------------------------------------------------------
def capture_environment(out, repo):
    env = out / "env"
    env.mkdir(parents=True, exist_ok=True)
    print("[1/7] environment")
    (env / "conda-explicit.txt").write_text(run("conda list --explicit"))
    (env / "environment.lock.yml").write_text(run("conda env export --no-builds"))
    (env / "conda-list.txt").write_text(run("conda list"))
    (env / "pip-freeze.txt").write_text(run(f"{sys.executable} -m pip freeze"))
    (env / "git.txt").write_text(
        "# git rev-parse HEAD\n" + run(f"git -C {repo} rev-parse HEAD")
        + "\n# git status --porcelain\n" + run(f"git -C {repo} status --porcelain")
        + "\n# git diff\n" + run(f"git -C {repo} diff")
    )
    info = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "machine": platform.machine(), "processor": platform.processor(),
            "hostname": platform.node(), "cpu_count": os.cpu_count()}
    for mod in ["numpy", "pandas", "scipy", "sklearn", "statsmodels", "glmnet", "torch", "esm", "Bio"]:
        try:
            m = __import__(mod)
            info[f"{mod}_version"] = getattr(m, "__version__", "unknown")
        except Exception as e:  # noqa: BLE001
            info[f"{mod}_version"] = f"not importable: {e!r}"
    weights = {}
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        info["torch_cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        ckpt = Path(torch.hub.get_dir()) / "checkpoints"
        for name in ["esm_msa1b_t12_100M_UR50S.pt", "esm_msa1b_t12_100M_UR50S-contact-regression.pt"]:
            p = ckpt / name
            weights[name] = {"path": str(p), "sha256": sha256(p), "bytes": p.stat().st_size} if p.exists() \
                else {"path": str(p), "missing": True}
    except Exception as e:  # noqa: BLE001
        weights["error"] = repr(e)
    write_json(env / "system.json", info)
    write_json(env / "esm_weights.json", weights)


# --------------------------------------------------------------------------
# 2. copy files
# --------------------------------------------------------------------------
def copy_files(out, data_root, repo, rec, spec, extra_inputs, html):
    print("[2/7] copying files")
    missing_required = []
    for role, pattern, prov, required in spec:
        matches = sorted(p for p in glob.glob(str(data_root / pattern), recursive=True) if os.path.isfile(p))
        if not matches:
            msg = f"{'MISSING (required)' if required else 'missing (optional)'}: {pattern}"
            print("  " + msg)
            if required:
                missing_required.append(pattern)
            continue
        for src in matches:
            rel = Path(role) / Path(src).relative_to(data_root)
            dst = out / rel
            if dst.exists():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            rec.provenance[str(rel)] = (str(Path(src).resolve()), prov)
    for src in extra_inputs:
        rel = Path("inputs") / "extra" / Path(src).name
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out / rel)
        rec.provenance[str(rel)] = (str(Path(src).resolve()), "extra input (--extra-input)")
    nb_dir = out / "notebooks"
    nb_dir.mkdir(parents=True, exist_ok=True)
    for nb in NOTEBOOKS:
        src = repo / "notebooks" / nb
        if not src.exists():
            print(f"  missing notebook: {nb}")
            continue
        shutil.copy2(src, nb_dir / nb)
        rec.provenance[f"notebooks/{nb}"] = (str(src.resolve()), "executed notebook with outputs")
        if html:
            print(run(f"jupyter nbconvert --to html --output-dir {nb_dir} {nb_dir / nb}").strip()[:200])
    return missing_required


def write_genome2tax(out, gtdb_metadata, rec):
    """Small taxonomy table for the Even + Corkrey genomes (needed for PCV), instead of all of GTDB."""
    ids = []
    for f in ["even_genome_ids.txt", "corkrey_genome_ids.txt"]:
        p = out / "inputs/predictive_models/supporting_data" / f
        if p.exists():
            ids += [line.strip() for line in p.read_text().splitlines() if line.strip()]
    md = pd.read_csv(gtdb_metadata, sep="\t", usecols=["accession", "gtdb_taxonomy"], low_memory=False)
    md = md[md["accession"].isin(set(ids))]
    rel = Path("inputs/genome2tax.tsv")
    md.to_csv(out / rel, sep="\t", index=False)
    rec.provenance[str(rel)] = (str(Path(gtdb_metadata).resolve()), "GTDB r214.1 bac120_metadata subset")
    rec.check("genome2tax covers all Even + Corkrey genomes", len(set(ids) - set(md["accession"])) == 0,
              n_ids=len(set(ids)), n_found=int(md["accession"].nunique()))


# --------------------------------------------------------------------------
# 3. portable copies
# --------------------------------------------------------------------------
def msa_dataset_to_tsv(ds, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ds, columns=["id", "aligned_seq", "ogt_scaled"]).to_csv(path, sep="\t", index=False)


def context_to_indices(ctx, dataset):
    """Saved get_embeddings contexts are lists of (id, seq); convert to integer indices."""
    id2i = {}
    for i, item in enumerate(dataset):
        assert item[0] not in id2i, f"duplicate id in context dataset: {item[0]}"
        id2i[item[0]] = i
    rows = []
    for row in ctx:
        idx = []
        for cid, cseq in row:
            i = id2i[cid]
            assert dataset[i][1] == cseq, f"sequence mismatch for context id {cid}"
            idx.append(i)
        rows.append(idx)
    return np.array(rows, dtype=np.int64)


def make_portable(out, rec):
    print("[3/7] portable copies")
    port = out / "portable"
    datasets = {}
    for name in ["even", "corkrey"]:
        p = out / f"inputs/predictive_models/supporting_data/{name}_msa_dataset.pickle"
        if p.exists():
            datasets[name] = load_pickle(p)
            msa_dataset_to_tsv(datasets[name], port / f"{name}_msa_dataset.tsv")

    scaler_p = out / "inputs/predictive_models/supporting_data/ogt_scaler.pickle"
    scaler = load_pickle(scaler_p)
    write_json(port / "ogt_scaler.json", {"mean_": scaler.mean_, "scale_": scaler.scale_,
                                          "var_": scaler.var_, "n_samples_seen_": scaler.n_samples_seen_})

    # embeddings, temps, contexts from esm_msa/
    for p in sorted((out / "intermediates/esm_msa").glob("*.pickle")) if (out / "intermediates/esm_msa").exists() else []:
        stem = p.stem
        obj = load_pickle(p)
        if stem.endswith("_context"):
            write_json(port / "esm_msa" / f"{stem}.json", [[list(t) for t in row] for row in obj])
            src = CONTEXT_SOURCE.get(stem)
            if src in datasets:
                try:
                    idx = context_to_indices(obj, datasets[src])
                    np.save(port / "esm_msa" / f"{stem}_indices.npy", idx)
                    rec.check(f"{stem}: contexts map to {src} dataset", True, shape=list(idx.shape))
                except Exception as e:  # noqa: BLE001
                    rec.check(f"{stem}: contexts map to {src} dataset", False, error=repr(e))
        else:
            arr = np.asarray(obj)
            (port / "esm_msa").mkdir(parents=True, exist_ok=True)
            np.save(port / "esm_msa" / f"{stem}.npy", arr)
        del obj

    # validation datasets
    for p in sorted(glob.glob(str(out / "intermediates/validation_datasets/*/embeddings/*.pickle"))):
        p = Path(p)
        rel = p.relative_to(out / "intermediates")
        dst = port / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        obj = load_pickle(p)
        if p.stem.endswith("_msa_dataset"):
            msa_dataset_to_tsv(obj, dst.with_suffix(".tsv"))
        else:
            np.save(dst.with_suffix(".npy"), np.asarray(obj))
        del obj

    # model
    model_p = out / "outputs/predictive_models/even_enetcv_ind.pickle"
    if model_p.exists():
        m = load_pickle(model_p)
        keys = list(m["best_model"].keys())
        np.savez(port / "even_enetcv_ind.npz",
                 site_keys=np.array(keys),
                 mu=np.stack([m["scalers"][k].mean_ for k in keys]),
                 sd=np.stack([m["scalers"][k].scale_ for k in keys]),
                 coef=np.stack([m["best_model"][k].coef_ for k in keys]),
                 intercept=np.array([m["best_model"][k].intercept_ for k in keys]),
                 alpha=np.array([m["best_model"][k].alpha_ for k in keys]),
                 l1_ratio=np.array([m["best_model"][k].l1_ratio_ for k in keys]),
                 full_predictions=np.asarray(m["full_predictions"]),
                 predictions=np.asarray(m["predictions"]),
                 true_values=np.asarray(m["true_values"]))
        first = m["best_model"][keys[0]]
        write_json(port / "even_enetcv_ind.meta.json", {
            "metadata": {k: str(v) for k, v in m["metadata"].items()},
            "true_labels": list(m["true_labels"]),
            "statsmodels_version_recorded": m.get("statsmodels_version"),
            "enetcv_params": {k: (str(v) if not isinstance(v, (int, float, str, bool, type(None), list)) else v)
                              for k, v in first.get_params().items()},
        })
    return datasets, scaler


# --------------------------------------------------------------------------
# 4 + 5. consistency checks and expected values
# --------------------------------------------------------------------------
def checks_and_expected(out, rec, datasets, scaler):
    from scipy.stats import spearmanr
    from sklearn.metrics import r2_score
    print("[4-5/7] consistency checks and expected values")
    port = out / "portable"
    esm = out / "intermediates/esm_msa"
    rec.put("scaler", {"mean": float(scaler.mean_[0]), "scale": float(scaler.scale_[0]),
                       "n_samples_seen": int(scaler.n_samples_seen_)}, "ogt_scaler.pickle", atol=0.0)

    def metrics(y_true, y_pred):
        return {"r2": float(r2_score(y_true, y_pred)),
                "rmse": float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))}

    model_p = out / "outputs/predictive_models/even_enetcv_ind.pickle"
    if not model_p.exists():
        rec.skip("model checks", "even_enetcv_ind.pickle missing")
        return
    m = load_pickle(model_p)
    fp = np.asarray(m["full_predictions"])
    tv = np.asarray(m["true_values"])
    rec.check("model has 215 sites and 215 scalers",
              len(m["best_model"]) == N_SITES and len(m["scalers"]) == N_SITES,
              n_models=len(m["best_model"]), input_shape=str(m["metadata"]["input_shape"]))
    rec.check("model['predictions'] == mean of full_predictions over all sites",
              bool(np.allclose(fp.mean(0), np.asarray(m["predictions"]), atol=1e-12, rtol=0)))

    # check 2: true values / labels vs Corkrey dataset
    if "corkrey" in datasets:
        ck = datasets["corkrey"]
        rec.check("true_labels == corkrey_msa_dataset ids (same order)",
                  list(m["true_labels"]) == [i[0] for i in ck])
        rec.check("true_values == corkrey_msa_dataset scaled OGTs",
                  bool(np.allclose(tv, [i[2] for i in ck], atol=0, rtol=0)))

    # check 3: temps arrays vs datasets
    for stem, ds in [("even_temps", "even"), ("corkrey_temps", "corkrey"),
                     ("c_w_e_temps", "corkrey"), ("e_w_c_temps", "even")]:
        p = esm / f"{stem}.pickle"
        if p.exists() and ds in datasets:
            t = np.asarray(load_pickle(p))
            rec.check(f"{stem} == {ds}_msa_dataset OGTs",
                      t.shape[0] == len(datasets[ds]) and bool(np.array_equal(t, [i[2] for i in datasets[ds]])))
        elif not p.exists():
            rec.skip(f"{stem} == dataset OGTs", "file not archived")

    # check 4: self-exclusion and depth of contexts
    for stem, self_excl in [("even_full_context", True), ("corkrey_full_context", True), ("c_w_e_context", False)]:
        p = port / "esm_msa" / f"{stem}_indices.npy"
        if p.exists():
            idx = np.load(p)
            ok = idx.shape[1] == 8
            if self_excl:
                ok = ok and not np.any(idx == np.arange(idx.shape[0])[:, None])
            rec.check(f"{stem}: depth 8{' and excludes self' if self_excl else ''}", bool(ok), shape=list(idx.shape))

    # embedding shapes
    for stem, ds in [("even_full_embeddings", "even"), ("c_w_e_full_embeddings", "corkrey"),
                     ("corkrey_full_embeddings", "corkrey"), ("e_w_c_full_embeddings", "even")]:
        p = port / "esm_msa" / f"{stem}.npy"
        if p.exists() and ds in datasets:
            a = np.load(p, mmap_mode="r")
            rec.check(f"{stem} shape == ({len(datasets[ds])}, 215, 768)",
                      tuple(a.shape) == (len(datasets[ds]), N_SITES, 768), shape=list(a.shape), dtype=str(a.dtype))

    # check 1: model on archived c_w_e embeddings reproduces full_predictions
    p = esm / "c_w_e_full_embeddings.pickle"
    if p.exists():
        emb = load_pickle(p)
        preds = np.array(per_site_predict(m["best_model"], emb, m["scalers"]))
        diff = float(np.abs(preds - fp).max())
        rec.check("model(c_w_e embeddings) reproduces stored full_predictions", diff <= 1e-10, max_abs_diff=diff)
        del emb
    else:
        rec.skip("model(c_w_e embeddings) reproduces stored full_predictions", "c_w_e embeddings not archived")

    # check 5: per-site Spearmans and good_cols
    spr_files = sorted((out / "outputs/predictive_models/supporting_data").glob("even_enetcv_ind_spearmans.tsv*"))
    recomputed = np.array([spearmanr(fp[i], tv)[0] for i in range(fp.shape[0])])
    rec.put("sites.spearman_test", recomputed.tolist(),
            "spearmanr(full_predictions[i], true_values) per site", atol=1e-10,
            note="computed on the Corkrey TEST set (quirk 1)")
    if spr_files:
        col_rs = pd.read_csv(spr_files[0], sep="\t", index_col=0)["0"]   # figures_final cell 74
        rec.check("recomputed per-site Spearmans match saved TSV",
                  bool(np.allclose(col_rs.values, recomputed, atol=1e-10)),
                  max_abs_diff=float(np.abs(col_rs.values - recomputed).max()))
        good_cols = col_rs[col_rs > GOOD_COLS_CUTOFF].index.to_numpy()
        rec.check("good_cols cutoff 0.84 selects 73 sites", len(good_cols) == 73, n=int(len(good_cols)))
    else:
        good_cols = np.where(recomputed > GOOD_COLS_CUTOFF)[0]
        rec.skip("recomputed per-site Spearmans match saved TSV", "TSV not archived; good_cols from recomputation")

    # expected: Fig 2A
    y_true = unscale_ogts(tv, scaler)
    rec.put("fig2a.all_sites", metrics(y_true, unscale_ogts(fp.mean(axis=0), scaler)),
            "unscale(full_predictions.mean(0)) vs unscale(true_values)")
    rec.put("fig2a.legacy_good_cols", {**metrics(y_true, unscale_ogts(fp[good_cols, :].mean(axis=0), scaler)),
                                       "cutoff": GOOD_COLS_CUTOFF, "sites": good_cols.tolist()},
            "figures_final cells 74 + 109: full_predictions[good_cols].mean(0)",
            note="published Fig 2A; sites selected on test set (quirk 1)")

    # validation datasets: isolates (Fig S3) and metagenomes (Fig 5)
    val_root = out / "intermediates/validation_datasets"
    names = sorted(Path(p).parts[-3] for p in glob.glob(str(val_root / "*/embeddings/*_full_even.pickle")))
    pred_dir = port / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    mg_frames = []
    n_even = len(datasets.get("even", []))
    for name in names:
        emb_dir = val_root / name / "embeddings"
        msa = load_pickle(emb_dir / f"{name}_msa_dataset.pickle")
        emb = load_pickle(emb_dir / f"{name}_full_even.pickle")
        ok_shape = tuple(emb.shape) == (len(msa), N_SITES, 768)
        rec.check(f"{name}: embeddings shape == (n_msa, 215, 768)", ok_shape, shape=list(emb.shape), n_msa=len(msa))
        tp = emb_dir / f"{name}_temps.pickle"
        if tp.exists():
            rec.check(f"{name}: temps == msa_dataset OGTs",
                      bool(np.array_equal(np.asarray(load_pickle(tp)), [i[2] for i in msa])))
        cp = emb_dir / f"{name}_even_context_values.pickle"
        if cp.exists():
            ctx = np.asarray(load_pickle(cp))
            rec.check(f"{name}: context indices valid (n x 8, < {n_even})",
                      ctx.shape == (len(msa), 8) and (n_even == 0 or int(ctx.max()) < n_even), shape=list(ctx.shape))
        if not ok_shape:
            continue
        preds = np.array(per_site_predict(m["best_model"], emb, m["scalers"]))   # run_model, all sites
        del emb
        y_pred = pd.Series(data=unscale_ogts(preds.mean(axis=0), scaler), index=[i[0] for i in msa])
        y_pred.rename("pred_temp").to_frame().assign(actual=unscale_ogts([i[2] for i in msa], scaler)) \
            .to_csv(pred_dir / f"{name}_roseate_all_sites.tsv", sep="\t", index_label="protein")
        md_files = list((out / "inputs/validation_datasets" / name / "md").glob(f"{name}_env_temps.tsv"))
        if not md_files:
            rec.skip(f"{name}: metrics", "env_temps.tsv not archived")
            continue
        md_path = md_files[0]
        if name in ISOLATE_DATASETS:
            # figures_final cells 140-143 / 157-161: assess_model(y_true, y_pred), POSITIONAL
            true_temps = pd.read_csv(md_path, sep="\t")
            y_true = true_temps.set_index("protein")["env temp"]
            same_order = list(y_true.index) == list(y_pred.index)
            rec.check(f"{name}: md and msa_dataset in same protein order (positional metrics valid)",
                      same_order, n_md=len(y_true), n_pred=len(y_pred))
            entry = {"n": int(len(y_pred))}
            try:
                r2, rmse, spr = assess_model(y_true, y_pred)
                entry["as_published"] = {"r2": r2, "rmse": rmse, "spearman": spr}
            except Exception as e:  # noqa: BLE001
                entry["as_published"] = {"error": repr(e)}
            r2, rmse, spr = assess_model(y_true.loc[y_pred.index].values, y_pred.values)
            entry["aligned_by_protein"] = {"r2": r2, "rmse": rmse, "spearman": spr}
            rec.put(f"figS3.{name}", entry, "figures_final assess_model on all-site ROSEATE predictions",
                    note="as_published is positional, as in the notebook; aligned_by_protein matches by id")
        elif name in DATASET2LABEL:
            # figures_final cell 396 + 401
            true_temps = pd.read_csv(md_path, sep="\t", index_col=0)
            dp = pd.merge(true_temps.rename(columns={"env temp": "env_temp"}), y_pred.rename("pred_temp"),
                          left_on="protein", right_index=True).assign(dataset=name)
            mg_frames.append(dp)
    if mg_frames:
        mg = pd.concat(mg_frames)
        mg["dataset_label"] = mg["dataset"].map(DATASET2LABEL)
        violin = mg.groupby(["dataset_label", "sample_or_genome"])[["pred_temp", "env_temp"]].mean().reset_index()
        violin.to_csv(port / "fig5_per_sample_means.tsv", sep="\t", index=False)
        for label, g in violin.groupby("dataset_label"):
            q = g["pred_temp"].quantile([0.25, 0.5, 0.75])
            rec.put(f"fig5.{label}", {"n_samples": int(g["sample_or_genome"].nunique()),
                                      "q1": float(q[0.25]), "median": float(q[0.5]), "q3": float(q[0.75]),
                                      "datasets": sorted(mg.loc[mg.dataset_label == label, "dataset"].unique())},
                    "figures_final cells 396, 401: per-sample mean of all-site ROSEATE predictions")

    # Fig S4 (legacy PCV, quirk 5)
    tcv = out / "outputs/predictive_models/supporting_data/taxonomic_cv_results.tsv"
    if tcv.exists():
        e = pd.read_csv(tcv, sep="\t", index_col=0)
        e["error_C"] = e["error"] * scaler.scale_[0]
        summary = e.groupby(["model", "level"])["error_C"].agg(["mean", "median", "count"])
        rec.put("figS4.legacy_pcv", {f"{mdl}.{lvl}": row.to_dict() for (mdl, lvl), row in summary.iterrows()},
                "taxonomic_cv_results.tsv error x scaler.scale_",
                note="glmnet ElasticNet(alpha=0.29) on CLS only; not per-site ROSEATE (quirk 5)")
    else:
        rec.skip("figS4.legacy_pcv", "taxonomic_cv_results.tsv not archived")


# --------------------------------------------------------------------------
# 6. manifest, 7. freeze
# --------------------------------------------------------------------------
def write_manifest(out, rec):
    print("[6/7] manifest")
    rows = []
    for p in sorted(out.rglob("*")):
        if not p.is_file() or p.name == "MANIFEST.tsv":
            continue
        rel = str(p.relative_to(out))
        dtype = shape = ""
        if p.suffix == ".npy":
            a = np.load(p, mmap_mode="r")
            dtype, shape = str(a.dtype), "x".join(map(str, a.shape))
        source, prov = rec.provenance.get(rel, ("", ""))
        if not prov and rel.startswith("portable/"):
            prov = "derived by build_reference.py"
        rows.append({"path": rel, "role": rel.split("/")[0] if "/" in rel else "meta", "sha256": sha256(p), "bytes": p.stat().st_size,
                     "dtype": dtype, "shape": shape, "source": source, "provenance": prov})
    pd.DataFrame(rows).to_csv(out / "MANIFEST.tsv", sep="\t", index=False)
    print(f"  {len(rows)} files, {sum(r['bytes'] for r in rows) / 1e9:.2f} GB")


def freeze(out):
    print("[7/7] freezing (read-only)")
    for p in sorted(out.rglob("*"), reverse=True):
        p.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH |
                ((stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH) if p.is_dir() else 0))
    out.chmod(0o555)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, type=Path, help="repo root (contains notebooks/ and data/)")
    ap.add_argument("--data-root", type=Path, default=None, help="default: <repo>/data")
    ap.add_argument("--out", type=Path, default=Path("reference") / ARCHIVE_VERSION)
    ap.add_argument("--gtdb-metadata", type=Path, default=None, help="bac120_metadata.tsv(.gz) for genome2tax")
    ap.add_argument("--extra-input", type=Path, action="append", default=[], help="extra input file (repeatable)")
    ap.add_argument("--include-table-s2-models", action="store_true", help="also archive all predictive_models/*")
    ap.add_argument("--html", action="store_true", help="also export notebooks to HTML (needs jupyter)")
    ap.add_argument("--freeze", action="store_true", help="make the archive read-only when done")
    ap.add_argument("--allow-failed-checks", action="store_true", help="freeze / exit 0 even if checks fail")
    ap.add_argument("--allow-missing", action="store_true", help="continue if required files are missing")
    args = ap.parse_args()

    repo = args.repo.resolve()
    data_root = (args.data_root or repo / "data").resolve()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} exists and is not empty; refusing to overwrite a reference archive.")
    out.mkdir(parents=True, exist_ok=True)
    started = datetime.datetime.now().isoformat(timespec="seconds")
    rec = Recorder()

    capture_environment(out, repo)
    spec = SPEC + (TABLE_S2_SPEC if args.include_table_s2_models else [])
    missing = copy_files(out, data_root, repo, rec, spec, args.extra_input, args.html)
    if missing and not args.allow_missing:
        sys.exit(f"Required files missing: {missing}. Fix paths or pass --allow-missing.")
    if args.gtdb_metadata:
        write_genome2tax(out, args.gtdb_metadata, rec)
    datasets, scaler = make_portable(out, rec)
    checks_and_expected(out, rec, datasets, scaler)

    write_json(out / "expected.json", rec.expected)
    write_json(out / "checks.json", rec.checks)
    n_fail = sum(c["status"] == "FAIL" for c in rec.checks)
    write_json(out / "BUILD_INFO.json", {
        "archive_version": ARCHIVE_VERSION, "started": started,
        "finished": datetime.datetime.now().isoformat(timespec="seconds"),
        "argv": sys.argv, "build_script_sha256": sha256(__file__),
        "repo": str(repo), "data_root": str(data_root), "missing_required": missing,
        "checks": {s: sum(c["status"] == s for c in rec.checks) for s in ("PASS", "FAIL", "SKIP")},
    })
    write_manifest(out, rec)
    print(f"\nchecks: {sum(c['status'] == 'PASS' for c in rec.checks)} pass, {n_fail} fail, "
          f"{sum(c['status'] == 'SKIP' for c in rec.checks)} skipped (details in checks.json)")
    if n_fail and not args.allow_failed_checks:
        print("Not freezing: resolve or document failed checks, or rerun with --allow-failed-checks.")
        sys.exit(1)
    if args.freeze:
        freeze(out)


if __name__ == "__main__":
    main()
