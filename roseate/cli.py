# ROSEATE: command line interface module
# Written by Michael Hoffert, with planning by Claude Opus 5.5
# October 2026

import argparse
import hashlib
import json
import subprocess
import os
import sys
from importlib.metadata import version
from pathlib import Path

# arguments that must come from the command line or --config
REQUIRED = {"predict": ["model", "emb", "msa_dataset", "scaler", "out"]}

def _sha256(path):
    '''
    Function for hashing a file specified at path
    '''
    assert os.path.exists(path), f'{path} does not exist'
    
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def write_run_record(args, out_dir, inputs):
    '''
    Create a file that describes what commands were run and with what outputs
    '''
    # make record dict
    record = {
        "command": sys.argv,
        "args": {k: str(v) for k, v in vars(args).items() if k != "func"},
        "versions": {p: version(p) for p in ("numpy", "pandas", "scikit-learn", "fair-esm")},
        "git": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "inputs": {str(p): _sha256(p) for p in inputs if p and Path(p).is_file()},
    }
    # make dir if not made
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    # write json dataload to dir
    (Path(out_dir) / "run.json").write_text(json.dumps(record, indent=2))


def cmd_predict(args):
    '''
    Run PREDICT command
    USAGE:
    roseate predict 
    '''
    
    import pandas as pd
    from roseate import aggregate, io, predict

    # load trait scaler
    scaler = io.reload_pickle(args.scaler)
    # construct predictions
    preds = predict.run_model(args.emb, 
                              args.model, 
                              args.msa_dataset, predict.per_site_predict)
    # which sites to aggregate in preds?
    sites = None if args.sites == "all" else aggregate.good_cols(
        pd.read_csv(args.spearmans, sep="\t", index_col=0)["0"], cutoff=args.cutoff)
    # get protein ids
    ids = [i[0] for i in io.reload_pickle(args.msa_dataset)]
    # perform prediction
    pred_c = io.unscale_traits(aggregate.mean_sites(preds, sites), scaler)
    # save preds
    pd.Series(pred_c, index=ids, name="pred_ogt_C").to_csv(args.out, sep="\t", index_label="protein")
    # if user wants per-site saved too,
    if args.per_site:
        import numpy as np
        np.save(args.per_site, preds)
    # record what happened
    write_run_record(args, Path(args.out).parent, [args.model, args.emb, args.msa_dataset, args.scaler])


def build_parser():
    p = argparse.ArgumentParser(prog="roseate")
    sub = p.add_subparsers(dest="command", required=True)
    subs = {}

    s = subs["predict"] = sub.add_parser("predict", help="per-protein OGT predictions from embeddings")
    s.add_argument("--model", type=Path)
    s.add_argument("--emb", type=Path)
    s.add_argument("--msa-dataset", type=Path)
    s.add_argument("--scaler", type=Path)
    s.add_argument("--sites", choices=["all", "legacy_fig2a"], default="all")
    s.add_argument("--spearmans", type=Path, help="site Spearman TSV, for --sites legacy_fig2a")
    s.add_argument("--cutoff", type=float, default=0.84)
    s.add_argument("--per-site", type=Path, help="also save the (215, n) per-site predictions")
    s.add_argument("--out", type=Path)
    s.set_defaults(func=cmd_predict)

    # extract, align, embed, fit, run, reproduce: same pattern, one cmd_* each
    for s in subs.values():
        s.add_argument("--config", type=Path, help="YAML defaults; command-line flags override")
    return p, subs


def main(argv=None):
    parser, subs = build_parser()
    args = parser.parse_args(argv)
    # allow some args to be computed from config
    if args.config:
        import yaml
        with open(args.config) as f:
            section = (yaml.safe_load(f) or {}).get(args.command, {})
        subs[args.command].set_defaults(**{k.replace("-", "_"): v for k, v in section.items()})
        args = parser.parse_args(argv)   # second pass: config values are defaults, flags override
    # catch missed args
    for name in REQUIRED.get(args.command, []):
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required (on the command line or in --config)")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())