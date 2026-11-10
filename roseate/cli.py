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
from roseate.decorators import *

# arguments that must come from the command line or --config
REQUIRED = {"predict": ["model", "emb", "msa_dataset", "scaler", "out"], 
            'embed': ['embed_dataset', 'context_dataset', 'out']}

class UsageError(Exception):
    '''Invalid argument combination, reported like an argparse error.'''

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
    
@verbose_print
def cmd_embed(args, vprint):
    '''
    run embed command
    Usage:
    roseate embed
    '''
    import warnings
    import numpy as np
    from pathlib import Path
    from roseate import io, context, embed
    
    # load msa dataset
    with spinner('Loading MSA dataset', enabled=args.verbose):
        embed_dataset = io.reload_pickle(args.embed_dataset)
    vprint(f"{len(embed_dataset)} sequences loaded for embedding.")

    with spinner('Loading MSA dataset', enabled=args.verbose):
        context_dataset = io.reload_pickle(args.context_dataset)
    vprint(f"{len(context_dataset)} sequences loaded for context.")
    
    # check requested context size
    assert args.context_size < len(context_dataset) - 1, "Context size can't be larger than context MSA"
    # check self_exclude
    if (_sha256(args.embed_dataset) == _sha256(args.context_dataset)) and not args.exclude_self:
        warnings.warn("embedding and context dataset sha256 match - are you sure you don't want --exclude-self?")

    # get context
    ctx = context.get_random_context(msa_dataset=context_dataset,
                        n_contexts=len(embed_dataset), 
                        context_size=args.context_size,
                        self_exclude=args.exclude_self,
                        seed = args.context_seed)
    optional = ' excluding self from context' if args.exclude_self else ''
    vprint(f'Generated contexts: shape {np.array(ctx).shape}' + optional)
    # check self arg
    if args.exclude_self:
        assert not any([i in ctx[i] for i in range(len(ctx))]), 'found own index in context'
    # save
    if args.save:
        outpath = Path(args.out) / 'context.pickle'
        io.save_pickle(ctx, outpath)

        info = {'context_seed':args.context_seed,
               'context_size':args.context_size,
               'context_dataset':args.context_dataset,
               'context_dataset_sha256':_sha256(args.context_dataset)}
        outpath = Path(args.out) / 'context_info.pickle'
        io.save_pickle(info, outpath)
        
        vprint(f'Saved context and info to {Path(args.out)}.')

    # load transformer
    with spinner('Loading MSA Transformer', enabled=args.verbose):
        model, alphabet, conv = embed.load_plm()
    layer = model.num_layers if args.layer == -1 else args.layer
        
    # generate embeddings
    print('Embedding:')
    emb = embed.get_embeddings(model=model, 
                              batch_converter=conv, 
                              data_to_embed = embed_dataset,
                              context_dataset = context_dataset,
                              context_inds = ctx,
                              layer = layer,
                               return_mean = False)    
    vprint(f'Embeddings generated: shape {emb.shape}')
    outpath = Path(args.out) / 'embeddings.pickle'
    io.save_pickle(emb, outpath)
    np.save(outpath.replace('.pickle', '.npy'), emb)
    vprint(f'Embeddings saved to {outpath}')
    write_run_record(args, Path(args.out).parent, [args.embed_dataset, args.context_dataset, args.context_seed, args.context_size, args.exclude_self])
    
@verbose_print
def cmd_predict(args, vprint):
    '''
    Run PREDICT command
    USAGE:
    roseate predict 
    '''
    
    import pandas as pd
    from roseate import aggregate, io, predict

    # load trait scaler
    scaler = io.reload_pickle(args.scaler)
    # vprint()
    # construct predictions
    preds = predict.run_model(args.emb, 
                              args.model, 
                              args.msa_dataset, predict.per_site_predict)
    # which sites to aggregate in preds?
    sites = None if args.sites == "all" else aggregate.good_cols(
        pd.read_csv(args.spearmans, sep="\t", index_col=0)["0"], cutoff=args.cutoff)
    # get protein ids
    ids = [i[0] for i in io.reload_pickle(args.msa_dataset)]
    
    # check sites and spearmans
    if args.sites == 'all':
        sites = None
    elif args.sites == 'cv_top':
        if args.n_sites is None:
            raise UsageError('--n-sites is required with --sites cv_top')
        sites = aggregate.cv_top_sites(model['best_model'], args.n_sites)
    elif args.sites == 'legacy_fig2a':
        if args.spearmans is None:
            raise UsageError("A pre-calculated file of spearmans is required")
        sites = aggregate.good_cols(aggregate.load_site_spearmans(args.spearmans))
        
    pred = aggregate.mean_sites(preds, sites)
    # perform prediction
    pred_c = io.unscale_traits(pred, scaler)
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

    # embed
    sub_embed = subs["embed"] = sub.add_parser("embed", help="Generate embeddings from MSA dataset")
    input_args = sub_embed.add_argument_group("Input arguments")
    input_args.add_argument("--embed-dataset", type=Path, help="Path to pickled MSADataset to embed")
    input_args.add_argument("--context-dataset", type=Path, help="Path to pickled MSADataset for context")
    input_args.add_argument("--out", type=Path)
    embedding_args = sub_embed.add_argument_group("Embedding arguments")
    embedding_args.add_argument("--layer", type=int, default=-1, 
                                help="Layer of model, default -1 (last)")
    context_args = sub_embed.add_argument_group("Embedding arguments")
    context_args.add_argument("--exclude-self", action='store_true', 
                              help='Flag if dataset uses itself for embedding context')
    context_args.add_argument("--context-size", type=int, default=8)
    context_args.add_argument('--context-seed', type=int, default=1234)
    
    sub_embed.set_defaults(func=cmd_embed)
    
    # predict 
    sub_pred = subs["predict"] = sub.add_parser("predict", help="per-protein OGT predictions from embeddings")
    sub_pred.add_argument("--model", type=Path)
    sub_pred.add_argument("--emb", type=Path)
    sub_pred.add_argument("--msa-dataset", type=Path)
    sub_pred.add_argument("--scaler", type=Path)
    sub_pred.add_argument("--sites", choices=["all", "cv_top", "legacy_fig2a"], default="all")
    sub_pred.add_argument("--n-sites", type=int, default=10, help="with --sites=cv_top: # of sites to use")
    sub_pred.add_argument("--spearmans", type=Path, help="site Spearman TSV, for --sites legacy_fig2a")
    sub_pred.add_argument("--cutoff", type=float, default=0.84, help='sites to use, spearman R cutoff')
    sub_pred.add_argument("--per-site", type=Path, help="also save the (protein_len, n) per-site predictions")
    sub_pred.add_argument("--out", type=Path)
    sub_pred.set_defaults(func=cmd_predict)

    # extract, align, embed, fit, run, reproduce: same pattern, one cmd_* each
    for s in subs.values():
        s.add_argument("--config", type=Path, help="YAML defaults; command-line flags override")
        s.add_argument("-v", "--verbose", action="store_true", help="print progress")
        s.add_argument("-s", "--save", action="store_true", help="Whether to save intermediate files")
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
    try:
        return args.func(args)
    except UsageError as e:
        subs[args.command].error(str(e))

if __name__ == "__main__":
    sys.exit(main())