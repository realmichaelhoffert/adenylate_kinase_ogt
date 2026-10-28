import hashlib, json, pickle, shutil
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

REF = Path("reference/v1_preprint")

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def metrics(y_true, y_pred):
    return {"r2": float(r2_score(y_true, y_pred)),
            "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2)))}

model = pickle.load(open(REF / "outputs/even_enetcv_ind.pickle", "rb"))
scaler = pickle.load(open(REF / "inputs/ogt_scaler.pickle", "rb"))
unscale = lambda x: scaler.inverse_transform(np.asarray(x).reshape(-1, 1)).ravel()

# portable model
np.savez(REF / "portable/even_enetcv_ind.npz",
         mu=np.stack([model["scalers"][i].mean_ for i in range(215)]),
         sd=np.stack([model["scalers"][i].scale_ for i in range(215)]),
         coef=np.stack([model["best_model"][i].coef_ for i in range(215)]),
         intercept=np.array([model["best_model"][i].intercept_ for i in range(215)]),
         full_predictions=np.asarray(model["full_predictions"]),
         true_values=np.asarray(model["true_values"]))

# expected numbers
fp, yt = np.asarray(model["full_predictions"]), unscale(model["true_values"])
spr = pd.read_csv(REF / "outputs/even_enetcv_ind_spearmans.tsv", sep="\t", index_col=0)["0"]
good = spr[spr > 0.84].index.to_numpy()
expected = {
  "fig2a.all_sites": metrics(yt, unscale(fp.mean(0))),
  "fig2a.legacy_good_cols": {**metrics(yt, unscale(fp[good].mean(0))),
                             "sites": good.tolist()},
}
json.dump(expected, open(REF / "expected.json", "w"), indent=2)
