"""
Tests for roseate.cli: each command reads and writes the stage folders of an output directory,
and calls the same module functions the other tests check.

Uses small synthetic data, so no reference archive is needed. The extract -> align test needs
the external tools; the embed test needs torch and the ESM weights and is marked slow.
"""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from Bio import SeqIO

from roseate import aggregate, cli, fit, io, predict

FIXTURES = Path(__file__).parent / 'fixtures' / 'extract'
BACKBONE = Path(__file__).parents[1] / 'data' / 'high_qual_alignments' / 'backbone.filtered_only.afa'
N_TRAIN, N_TEST, N_SITES, N_DIM = 40, 12, 3, 4
ATOL = 1e-5   # predictions are float32 (from float32 embeddings) and written to text


def run(*argv):
    '''Call the CLI as `roseate <argv>`.'''
    return cli.main([str(a) for a in argv])


def records(out):
    '''Run records of an output folder, one row per command.'''
    return pd.read_json(Path(out) / 'run.jsonl', lines=True)


def stage(out, stage_name, key):
    return Path(out) / cli.RUN_FILES[stage_name][key]


def predictions(out):
    return pd.read_csv(stage(out, 'predict', 'predictions'), sep='\t', index_col='protein')['pred_ogt_C']


def write_stage_inputs(out, dataset, emb):
    '''Place a dataset and its embeddings where align and embed would have written them.'''
    stage(out, 'align', 'msa_dataset').parent.mkdir(parents=True, exist_ok=True)
    stage(out, 'embed', 'embeddings').parent.mkdir(parents=True, exist_ok=True)
    io.save_pickle(dataset, stage(out, 'align', 'msa_dataset'))
    np.save(stage(out, 'embed', 'embeddings'), emb)


@pytest.fixture(scope='module')
def data(tmp_path_factory):
    '''Synthetic training and test sets: the trait depends on the first embedding dimension.'''
    root = tmp_path_factory.mktemp('cli_data')
    rng = np.random.default_rng(0)
    n = N_TRAIN + N_TEST
    emb = rng.normal(size=(n, N_SITES, N_DIM)).astype(np.float32)
    y = 30 + 8 * emb[:, :, 0].mean(axis=1) + rng.normal(scale=0.5, size=n)
    ds = [(f'p{i}|G{i}|adk|bitscore=150.0', 'M' * N_SITES, np.nan) for i in range(n)]
    pd.Series(y, index=[d[0] for d in ds], name='ogt').to_csv(root / 'traits.tsv', sep='\t')
    io.save_pickle(ds[N_TRAIN:], root / 'test_ds.pickle')
    np.save(root / 'test_emb.npy', emb[N_TRAIN:])
    return {'root': root, 'traits': root / 'traits.tsv', 'test_ds': root / 'test_ds.pickle',
            'test_emb': root / 'test_emb.npy', 'train': (ds[:N_TRAIN], emb[:N_TRAIN]),
            'test': (ds[N_TRAIN:], emb[N_TRAIN:])}


@pytest.fixture(scope='module')
def trained(data):
    '''An output folder after `roseate fit` on the training set, with a test set.'''
    out = data['root'] / 'trained'
    write_stage_inputs(out, *data['train'])
    run('fit', '-o', out, '--traits', data['traits'],
        '--test-emb', data['test_emb'], '--test-dataset', data['test_ds'])
    return out


@pytest.fixture
def copy_of(tmp_path):
    '''A fresh copy of an output folder, so tests can add stages without affecting each other.'''
    def _copy(out):
        new = tmp_path / out.name
        shutil.copytree(out, new)
        return new
    return _copy


# ---------------------------------------------------------------- fit
def test_fit_writes_model_scaler_and_record(trained):
    model = io.reload_pickle(stage(trained, 'fit', 'model'))
    assert stage(trained, 'fit', 'scaler').exists()
    assert len(model['best_model']) == N_SITES and len(model['true_labels']) == N_TEST
    assert model['metadata']['label'] == 'trained'
    assert list(records(trained)['stage']) == ['fit']

def test_fit_matches_module_calls(trained, data):
    '''The CLI fit equals train_per_site on the same scaled traits (ElasticNetCV is seeded).'''
    ds, emb = data['train']
    scaled, _ = io.scale_traits(io.attach_traits(ds, data['traits']))
    enets, _ = fit.train_per_site(emb, np.array([i[2] for i in scaled]))
    model = io.reload_pickle(stage(trained, 'fit', 'model'))
    for site in enets:
        np.testing.assert_array_equal(model['best_model'][site].coef_, enets[site].coef_)

def test_fit_copies_existing_model(trained, tmp_path):
    out = tmp_path / 'copied'
    run('fit', '-o', out, '--model', stage(trained, 'fit', 'model'),
        '--scaler', stage(trained, 'fit', 'scaler'))
    for key in ['model', 'scaler']:
        assert stage(out, 'fit', key).read_bytes() == stage(trained, 'fit', key).read_bytes()


# ---------------------------------------------------------------- predict
def test_predict_uses_pipeline_defaults(trained, data, copy_of):
    out = copy_of(trained)
    run('predict', '-o', out)
    preds = predict.run_model(stage(out, 'embed', 'embeddings'), stage(out, 'fit', 'model'),
                              stage(out, 'align', 'msa_dataset'), predict.per_site_predict)
    expected = io.unscale_traits(aggregate.mean_sites(preds), io.reload_pickle(stage(out, 'fit', 'scaler')))
    got = predictions(out)
    assert list(got.index) == [i[0] for i in data['train'][0]]
    np.testing.assert_allclose(got.values, expected, atol=ATOL, rtol=0)

def test_copied_model_reproduces_stored_test_predictions(trained, data, tmp_path):
    '''Fit in one folder, copy the model into another, and predict the test set there.'''
    out = tmp_path / 'new_data'
    write_stage_inputs(out, *data['test'])
    run('fit', '-o', out, '--model', stage(trained, 'fit', 'model'), '--scaler', stage(trained, 'fit', 'scaler'))
    run('predict', '-o', out)
    model = io.reload_pickle(stage(trained, 'fit', 'model'))
    scaler = io.reload_pickle(stage(trained, 'fit', 'scaler'))
    np.testing.assert_allclose(predictions(out).values, io.unscale_traits(model['predictions'], scaler),
                               atol=ATOL, rtol=0)

def test_per_site_and_cv_top(trained, copy_of):
    out = copy_of(trained)
    run('predict', '-o', out, '--sites', 'cv_top', '--n-sites', 2, '--per-site')
    per_site = np.load(stage(out, 'predict', 'per_site'))
    assert per_site.shape == (N_SITES, N_TRAIN)
    sites = aggregate.cv_top_sites(io.reload_pickle(stage(out, 'fit', 'model'))['best_model'], 2)
    expected = io.unscale_traits(aggregate.mean_sites(per_site, sites), io.reload_pickle(stage(out, 'fit', 'scaler')))
    np.testing.assert_allclose(predictions(out).values, expected, atol=ATOL, rtol=0)


# ---------------------------------------------------------------- shared behaviour
def test_existing_outputs_are_skipped_unless_overwrite(trained, copy_of, capsys):
    out = copy_of(trained)
    run('predict', '-o', out)
    file = stage(out, 'predict', 'predictions')
    file.write_text('marker')
    with pytest.warns(UserWarning, match='--overwrite'):
        run('predict', '-o', out)
    assert 'Nothing to do.' in capsys.readouterr().out
    assert file.read_text() == 'marker'
    run('predict', '-o', out, '--overwrite')
    assert file.read_text() != 'marker'

def test_run_records_accumulate(trained, copy_of):
    out = copy_of(trained)
    run('predict', '-o', out)
    run('predict', '-o', out, '--overwrite', '--sites', 'cv_top', '--n-sites', 2)
    rec = records(out)
    assert list(rec['stage']) == ['fit', 'predict', 'predict']
    assert rec['args'].iloc[-1]['sites'] == 'cv_top'
    assert str(stage(out, 'fit', 'model')) in rec['inputs'].iloc[-1]

def test_config_supplies_defaults_and_flags_override(trained, copy_of, tmp_path):
    out = copy_of(trained)
    config = tmp_path / 'config.yaml'
    config.write_text('predict:\n  sites: cv_top\n  n_sites: 2\n')
    run('predict', '-o', out, '--config', config, '--n-sites', 1)
    args = records(out)['args'].iloc[-1]
    assert (args['sites'], args['n_sites']) == ('cv_top', '1')     # sites from config, flag wins for n_sites


# ---------------------------------------------------------------- errors
@pytest.mark.parametrize('argv, message', [
    (['predict'], 'run `roseate fit` first'),              # an earlier stage's output is missing
    (['align'], '--backbone is required'),                 # REQUIRED entry
    (['fit', '--model', 'model.pickle'], '--scaler'),      # copying a model needs its scaler
    (['fit'], '--traits'),                                 # nothing to fit on
])
def test_usage_errors_exit_with_message(argv, message, tmp_path, capsys):
    with pytest.raises(SystemExit) as exit_info:
        run(*argv, '-o', tmp_path / 'empty')
    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err


# ---------------------------------------------------------------- extract -> align
@pytest.mark.skipif(not all(shutil.which(t) for t in ['seqtk', 'prodigal', 'hmmsearch', 'parallel', 'mafft']),
                    reason='needs seqtk, prodigal, hmmer, GNU parallel and mafft')
def test_extract_then_align(tmp_path):
    genome = tmp_path / 'genomes' / 'genomeB.fasta'
    genome.parent.mkdir()
    shutil.copy(FIXTURES / 'genomeB.fasta', genome)
    (tmp_path / 'genomes.txt').write_text(f'{genome}\n')
    backbone = tmp_path / 'backbone.afa'
    SeqIO.write(list(SeqIO.parse(BACKBONE, 'fasta'))[:50], backbone, 'fasta')   # small, so mafft is fast
    out = tmp_path / 'run'

    run('extract', '-o', out, '--genome-list', tmp_path / 'genomes.txt', '--threads', 1)
    run('align', '-o', out, '--backbone', backbone)

    [protein] = SeqIO.parse(stage(out, 'extract', 'proteins'), 'fasta')
    assert protein.id.split('|')[1:3] == ['genomeB', 'adk']
    dataset = io.reload_pickle(stage(out, 'align', 'msa_dataset'))
    assert [i[0] for i in dataset] == [protein.id]
    assert len(dataset[0][1]) == len(next(SeqIO.parse(backbone, 'fasta')).seq)
    assert list(records(out)['stage']) == ['extract', 'align']


# ---------------------------------------------------------------- embed
@pytest.mark.slow
def test_embed_writes_embeddings_and_contexts(tmp_path):
    pytest.importorskip('esm')
    rows = list(SeqIO.parse(BACKBONE, 'fasta'))[:12]
    dataset = [(r.id, str(r.seq), np.nan) for r in rows]
    io.save_pickle(dataset[:3], tmp_path / 'embed_ds.pickle')
    io.save_pickle(dataset[3:], tmp_path / 'context_ds.pickle')
    out = tmp_path / 'run'

    run('embed', '-o', out, '--embed-dataset', tmp_path / 'embed_ds.pickle',
        '--context-dataset', tmp_path / 'context_ds.pickle', '--context-size', 4, '--context-seed', 0)

    emb = np.load(stage(out, 'embed', 'embeddings'))
    assert emb.shape == (3, len(rows[0].seq) + 1, 768)          # +1 for the CLS position
    assert np.load(stage(out, 'embed', 'contexts')).shape == (3, 4)
    assert list(records(out)['stage']) == ['embed']
