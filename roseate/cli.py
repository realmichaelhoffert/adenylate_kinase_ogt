# ROSEATE: command line interface module
# Written by Michael Hoffert, with planning by Claude Opus 5.5
# October 2026

import argparse
import hashlib
import json
import subprocess
import os
import sys
import shutil
import warnings
from importlib.metadata import version
from pathlib import Path
import numpy as np
import roseate.decorators

## HELPING FUNCTIONS -----
# ------------------------------------

# arguments that must come from the command line or --config
REQUIRED = {"align":["out", "backbone"],
            "extract":["out", "genome_list"],
            "embed": ["context_dataset", "out"],
            "fit":["out"],
            "predict": ["out"], 
            
            }

# Layout of an output folder (--out): one folder per stage
#   <out>/1_extract/ 2_align/ 3_embed/ 4_fit/ 5_predict/
# 4_fit always holds the model used by predict, whether fitted here or copied in.
RUN_FILES = {
    'extract': {'proteins': '1_extract/proteins.faa'},
    'align':   {'alignment': '2_align/protein_to_backbone.afa',
                'msa_dataset': '2_align/protein_to_backbone.pickle'},
    'embed':   {'contexts': '3_embed/contexts.npy',
                'context_info': '3_embed/context_info.pickle',
                'embeddings': '3_embed/embeddings.npy'},
    'fit':     {'model': '4_fit/model.pickle',
                'scaler': '4_fit/scaler.pickle'},
    'predict': {'predictions': '5_predict/predictions.tsv',
                'per_site': '5_predict/per_site.npy'},
}


def stage_file(out, stage, key):
    '''
    Path of one stage output, creating its folder: stage_file(out, 'embed', 'embeddings').
    '''
    path = Path(out) / RUN_FILES[stage][key]
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

def input_file(given, out, stage, key, flag):
    '''The path given with flag, else the earlier stage's output in the output folder; it must exist.'''
    path = Path(given) if given else Path(out) / RUN_FILES[stage][key]
    if not path.exists():
        raise UsageError(f'{path} not found: pass {flag} or run `roseate {stage}` first')
    return path

def done(paths, overwrite):
    '''
    True if every output already exists and --overwrite was not given, so the stage can be skipped.
    '''
    return not overwrite and all(Path(p).exists() for p in paths)

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

def require_tools(*names):
    '''
    Helper function to check for required tools
    '''
    missing = [n for n in names if shutil.which(n) is None]
    if missing:
        raise UsageError(f"missing external tools: {', '.join(missing)} (see roseate.yml)")

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

## MAIN MODULES -----
# ------------------------------------


@roseate.decorators.verbose_print
def cmd_extract(args, vprint):

    import roseate.extract
    
    require_tools('prodigal', 'seqtk', 'hmmsearch', 'parallel')
    out = stage_file(args.out, 'extract', 'proteins')
    if done([out], args.overwrite):
        warnings.warn(f"extract: specify --overwrite to overwrite existing file {out}")
        print("Nothing to do.")
        return

    # To do: reformat this to better handle errors and avoid intermediates
    # call proteins and run hmm on proteins
    with roseate.decorators.spinner('Annotating', enabled=args.verbose):
        tables = roseate.extract.run_annotation(genome_list=args.genome_list, 
                                                hmm_model=args.hmm, 
                                                jobs=args.threads)

    # process hmm table to make per-genome protein file
    with roseate.decorators.spinner('Processing tables',   enabled=args.verbose):
        written = roseate.extract.process_hmms(loc=Path(tables[0]).parent, 
                                               protein_name=args.protein_name,
                                               n_proteins=args.n_proteins,
                                               min_score=args.bitscore,
                                               write_file=True,
                                               overwrite=True)
    vprint(f'Processed {len(written)} tables.')
    # currently assumes all files are in the same directory,
    # adk name is hard-coded
    file = roseate.extract.concat_adks(loc=Path(tables[0]).parent, 
                                       out=out, 
                                       protein_name=args.protein_name)
    with open(file, "rb") as f:
        count = sum(1 for line in f if line.startswith(b">"))
    vprint(f'Wrote {count} proteins to {out}')

    # record args
    write_run_record(args, Path(args.out), [args.genome_list, args.hmm])



@roseate.decorators.verbose_print
def cmd_align(args, vprint):

    from roseate import align, io
    
    require_tools('mafft')
    protein_file = input_file(args.proteins, args.out, 'extract', 'proteins', '--proteins')
    backbone = args.backbone
    out = stage_file(args.out, 'align', 'alignment')
    if done([out], args.overwrite):
        warnings.warn(f"align: specify --overwrite to overwrite existing file {out}")
        print("Nothing to do.")
        return
    
    # insert sequences into alignment
    with roseate.decorators.spinner('Aligning to backbone', enabled=args.verbose):
        aligned = align.add_to_backbone(protein_file, backbone, out)
        added_ids = align.added_ids(aligned, backbone)
        vprint(f"Added {len(added_ids)} sequences.")
    
    # save MSA dataset
    msa_dataset = [i for i in io.construct_msa_dataset(aligned) if i[0] in added_ids]
    msa_out = stage_file(args.out, 'align', 'msa_dataset')
    io.save_pickle(msa_dataset, msa_out)

    # run record
    write_run_record(args, Path(args.out), [protein_file, args.backbone])

    
@roseate.decorators.verbose_print
def cmd_embed(args, vprint):
    '''
    run embed command
    Usage:
    roseate embed
    '''
    import warnings
    import numpy as np
    from pathlib import Path
    from roseate import io, context, embed, decorators

    embed_out = stage_file(args.out, 'embed', 'embeddings')
    if done([embed_out], args.overwrite):
        warnings.warn(f"embed: specify --overwrite to overwrite existing file {embed_out}")
        print("Nothing to do.")
        return
    
    # load msa dataset
    msa = input_file(args.embed_dataset, args.out, 'align', 'msa_dataset', '--embed-dataset')
    with roseate.decorators.spinner('Loading MSA dataset', enabled=args.verbose):
        embed_dataset = io.reload_pickle(msa)
    vprint(f"{len(embed_dataset)} sequences loaded for embedding.")

    with roseate.decorators.spinner('Loading MSA dataset', enabled=args.verbose):
        context_dataset = io.reload_pickle(args.context_dataset)
    vprint(f"{len(context_dataset)} sequences loaded for context.")
    
    # check requested context size
    assert args.context_size < len(context_dataset) - 1, "Context size can't be larger than context MSA"
    # check self_exclude
    if (_sha256(msa) == _sha256(args.context_dataset)) and not args.exclude_self:
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
    # if args.save:
    outpath = stage_file(args.out, 'embed', 'contexts')
    np.save(outpath, np.array(ctx))
    
    info = {'context_seed':args.context_seed,
            'context_size':args.context_size,
            'context_dataset':args.context_dataset,
            'context_dataset_sha256':_sha256(args.context_dataset)}
    outpath = stage_file(args.out, 'embed', 'context_info')
    io.save_pickle(info, outpath)
    
    vprint(f'Saved context and info to {Path(args.out)}.')

    # load transformer
    with decorators.spinner('Loading MSA Transformer', enabled=args.verbose):
        model, _, conv = embed.load_plm()
    layer = model.num_layers if args.layer == -1 else args.layer
        
    # generate embeddings
    print('Embedding:')
    emb, _, _ = embed.get_embeddings(model=model, 
                              batch_converter=conv, 
                              data_to_embed = embed_dataset,
                              context_dataset = context_dataset,
                              context_inds = ctx,
                              layer = layer,
                              return_mean = False)    
    vprint(f'Embeddings generated: shape {emb.shape}')
    outpath = stage_file(args.out, 'embed', 'embeddings')
    np.save(outpath, emb)
    vprint(f'Embeddings saved to {outpath}')

    # save info
    write_run_record(args, Path(args.out), [msa, args.context_dataset, str(args.context_seed), str(args.context_size), args.exclude_self])
    
@roseate.decorators.verbose_print
def cmd_fit(args, vprint):
    '''
    Run FIT command
    Usage:
    roseate fit
    '''

    from roseate import fit, io

    # check outputs
    model_out = stage_file(args.out, 'fit', 'model')
    scaler_out = stage_file(args.out, 'fit', 'scaler')
    if done([model_out, scaler_out], args.overwrite):
        warnings.warn(f'fit: specify --overwrite to overwrite existing file {model_out}')
        print('Nothing to do.')
        return

    # dummy code for using existing model
    if args.model:
        if args.scaler is None:
            raise UsageError('Scaler is required')
        model = io.reload_pickle(args.model)
        if not all(i in model.keys() for i in ['best_model', 'metadata']):
            raise UsageError('Is this a roseate model?')

        shutil.copy(args.model, model_out)
        shutil.copy(args.scaler, scaler_out)
        
        vprint(f'Copied existing model and scaler to {args.out}.')
        write_run_record(args, Path(args.out), [args.model, args.scaler])
        return

    # new model
    if args.traits is None:
        raise UsageError('Traits (--traits) must be specified to fit a model or use --model')
    
    # load model
    with roseate.decorators.spinner('Loading model', enabled=args.verbose):
        train_emb_path = input_file(args.train_emb, args.out, 'embed', 'embeddings', '--train-emb')
        train_ds_path = input_file(args.train_dataset, args.out, 'align', 'msa_dataset', '--train-dataset')

        train_emb = io.load_embeddings(train_emb_path)
        train_ds = io.attach_traits(io.reload_pickle(train_ds_path), args.traits)
        # check match on embeddings and dataset
        assert train_emb.shape[0] == len(train_ds), f"Lengths of embedding and msa should match: {train_emb.shape[0]} != {train_ds}"
        
        # scaler for scaling traits, used if provided, otherwise fitted from training
        scaler = io.reload_pickle(args.scaler) if args.scaler else None
        train_ds, scaler = io.scale_traits(train_ds, scaler)
        vprint('Loaded.')

    vprint(f'Fitting {train_emb.shape[1]} sites on {len(train_ds)} sequences.')
    enets, scalers = fit.train_per_site(train_emb, np.array([i[2] for i in train_ds]), verbose=args.verbose)

    # optional test set
    inputs = [train_emb_path, train_ds_path, args.traits, args.scaler]
    test_traits = args.test_traits or args.traits
    if args.test_emb:
        if args.test_dataset is None:
            raise UsageError("--test-dataset is required with --test-emb")
        test_ds, _ = io.scale_traits(io.attach_traits(io.reload_pickle(args.test_dataset), test_traits), scaler)
        test_preds = fit.test_per_site(enets, scalers, io.load_embeddings(args.test_emb))
        inputs += [args.test_emb, args.test_dataset, test_traits]
    else:
        test_ds, test_preds = [], np.empty((len(enets), 0))

    model = fit.package_model(model_label = args.model_label or Path(args.out).name,
                             enets = enets, scalers=scalers,
                             test_preds=test_preds, test_dataset=test_ds,
                             input_shape=train_emb.shape, model_type='per-site ElasticNetCV',
                             extra_metadata={'enetcv_params': fit.ENETCV_PARAMS,
                                              'train_dataset': str(train_ds_path),
                                              'traits': str(args.traits)})
    fit.save_model(model, model_out, overwrite=args.overwrite)
    io.save_pickle(scaler, scaler_out)
    vprint(f"Saved model and scaler to {model_out.parent}")
    write_run_record(args, Path(args.out), inputs)
    

@roseate.decorators.verbose_print
def cmd_predict(args, vprint):
    '''
    Run PREDICT command
    USAGE:
    roseate predict 
    Uses various inputs
    '''
    
    import pandas as pd
    from roseate import aggregate, io, predict

    out = stage_file(args.out, 'predict', 'predictions')
    if done([out], args.overwrite):
        warnings.warn(f"predict: specify --overwire to overwrite existing file {out}")
        print("Nothing to do.")
        return
    
    # Loading inputs, single syntax to allow default to specified pipeline
    with roseate.decorators.spinner('Loading inputs', enabled=args.verbose):
        model_path = input_file(args.model, args.out, 'fit', 'model', '--model')
        scaler_path = input_file(args.scaler, args.out, 'fit', 'scaler', '--scaler')
        emb_path = input_file(args.emb, args.out, 'embed', 'embeddings', '--train-emb')
        msa_path = input_file(args.msa_dataset, args.out, 'align', 'msa_dataset', '--train-dataset')

    with roseate.decorators.spinner('Predicting', enabled=args.verbose):
        # construct predictions
        preds = predict.run_model(emb_path, 
                                model_path, 
                                msa_path, predict.per_site_predict)
        ids = [i[0] for i in io.reload_pickle(msa_path)]
    with roseate.decorators.spinner('Aggregating across sites', enabled=args.verbose):        
        # check sites and spearmans
        if args.sites == 'all':
            sites = None
        elif args.sites == 'cv_top':
            if args.n_sites is None:
                raise UsageError('--n-sites is required with --sites cv_top')
            sites = aggregate.cv_top_sites(io.reload_pickle(model_path)['best_model'], args.n_sites)
        elif args.sites == 'legacy_fig2a':
            if args.spearmans is None:
                raise UsageError("--spearmans is required with --sites legacy-fig2a")
            sites = aggregate.good_cols(aggregate.load_site_spearmans(args.spearmans), cutoff=args.cutoff)

        # mean over sites, back to trait scale
        pred_c = io.unscale_traits(aggregate.mean_sites(preds, sites), io.reload_pickle(scaler_path))
        # save preds
        pd.Series(pred_c, index=ids, name="pred_ogt_C").to_csv(out, sep="\t", index_label="protein")

    # if user wants per-site saved too,
    if args.per_site:
        np.save(stage_file(args.out, 'predict', 'per_site'), preds)

    # record what happened
    write_run_record(args, Path(args.out), [model_path, scaler_path, emb_path, msa_path])


def build_parser():
    p = argparse.ArgumentParser(prog="roseate")
    sub = p.add_subparsers(dest="command", required=True)
    subs = {}

    # extract
    sub_extract = subs['extract'] = sub.add_parser('extract', help='Extract proteins')
    input_args = sub_extract.add_argument_group("Input arguments")
    input_args.add_argument('--genome-list', type=Path, help="Newline delimited file or location of genomes or samples for prediction")
    input_args.add_argument('--hmm', type=Path, help='HMM for annotating proteins',
                            default=Path(__file__).parents[1] / 'data' / 'PF00406.hmm')
    input_args.add_argument('--n-proteins', type=int, default=1, help='max proteins per genome')
    input_args.add_argument('--bitscore', type=float, default=100, help='minimum bitscore')
    sub_extract.set_defaults(func=cmd_extract)


    # align
    sub_align = subs["align"] = sub.add_parser("align", help="Align protein to backbone to make MSA Dataset")
    input_args = sub_align.add_argument_group("Input arguments")
    input_args.add_argument('--proteins',
                            type=Path, help='Path to file of unaligned proteins')
    input_args.add_argument('--backbone',
                            type=Path, help='Path to MSA file of proteins')
    sub_align.set_defaults(func=cmd_align)

    # embed
    sub_embed = subs["embed"] = sub.add_parser("embed", help="Generate embeddings from MSA dataset")
    input_args = sub_embed.add_argument_group("Input arguments")
    input_args.add_argument("--embed-dataset", type=Path, help="Path to pickled MSADataset to embed")
    input_args.add_argument("--context-dataset", type=Path, help="Path to pickled MSADataset for context")
    embedding_args = sub_embed.add_argument_group("Embedding arguments")
    embedding_args.add_argument("--layer", type=int, default=-1, 
                                help="Layer of model, default -1 (last)")
    context_args = sub_embed.add_argument_group("Context arguments")
    context_args.add_argument("--exclude-self", action='store_true', 
                              help='Flag if dataset uses itself for embedding context')
    context_args.add_argument("--context-size", type=int, default=8)
    context_args.add_argument('--context-seed', type=int, default=1234)
    sub_embed.set_defaults(func=cmd_embed)

    # fit
    sub_fit = subs["fit"] = sub.add_parser("fit", help="Fit or copy predictive model")
    existing_args = sub_fit.add_argument_group("Use an existing model (copied to 4_fit)")
    existing_args.add_argument("--model", type=Path, help="Pickled roseate model")
    existing_args.add_argument("--scaler", type=Path, help="Pickled roseate trait data scaler")
    train_args = sub_fit.add_argument_group("Fit a new model")
    train_args.add_argument("--traits", type=Path, help="Exact matching table of trait values for seq ids in --train-dataset")
    train_args.add_argument("--train-emb", type=Path, help="Embeddings of training dataset")
    train_args.add_argument("--train-dataset", type=Path, help="Pickled training MSA dataset")
    train_args.add_argument("--test-traits", type=Path,
                        help="Trait table for --test-dataset; default: --traits")
    train_args.add_argument("--test-emb", type=Path, help="Optional: testing dataset embeddings")
    train_args.add_argument("--test-dataset", type=Path, help="Optional: Testing MSA Dataset")
    train_args.add_argument("--model-label", type=str, help="label stored in model metadata")
    sub_fit.set_defaults(func=cmd_fit)
    
    # predict 
    sub_pred = subs["predict"] = sub.add_parser("predict", help="per-protein OGT predictions from embeddings")
    sub_pred.add_argument("--model", type=Path, help='Model pickle; defaults to 4_fit if not specified')
    sub_pred.add_argument("--emb", type=Path, help='Embeddings; defaults to 3_embed if not specified')
    sub_pred.add_argument("--msa-dataset", type=Path, help='MSA Dataset pickle; defaults to 2_align if not specified')
    sub_pred.add_argument("--scaler", type=Path, help='Trait scaler, defaults to scaler in 4_fit if not specified')
    sub_pred.add_argument("--sites", choices=["all", "cv_top", "legacy_fig2a"], default="all")
    sub_pred.add_argument("--n-sites", type=int, default=10, help="with --sites=cv_top: # of sites to use")
    sub_pred.add_argument("--spearmans", type=Path, help="site Spearman TSV, for --sites legacy_fig2a")
    sub_pred.add_argument("--cutoff", type=float, default=0.84, help='sites to use, spearman R cutoff')
    sub_pred.add_argument("--per-site", action="store_true", help="Save the (protein_len, n) per-site predictions")
    sub_pred.set_defaults(func=cmd_predict)

    # extract, align, embed, fit, run, reproduce: same pattern, one cmd_* each
    for s in subs.values():
        s.add_argument('-o', '--out', type=Path, help='Output location')
        s.add_argument('-p', '--protein-name', required=True, type=str, help='protein prefix (for multiple proteins)')
        s.add_argument("--config", type=Path, help="YAML defaults; command-line flags override")
        s.add_argument("-v", "--verbose", action="store_true", help="Enable vebosity")
        s.add_argument("-s", "--save", action="store_true", help="Save intermediate files")
        s.add_argument('--overwrite', action="store_true", help="Overwrite existing outputs")
        s.add_argument('--threads', type=int, help='Number of threads', default=4)
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