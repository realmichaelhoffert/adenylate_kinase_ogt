"""
Tests for the prediction and fitting workflows.

Most tests record the stage commands a workflow would run, so they need nothing installed.
A resumed workflow test runs fit (copy) and predict for real on synthetic data. The slow test
runs both workflows end to end on 30 backbone rows and needs the external tools, torch and fair-esm.
"""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from Bio import SeqIO

from roseate import cli, io

REPO = Path(__file__).parents[1]
BACKBONE = REPO / 'data' / 'high_qual_alignments' / 'backbone.filtered_only.afa'
FIXTURES = Path(__file__).parent / 'fixtures' / 'extract'
ADK_CONFIG = cli.CONFIG_DIR / 'adk.yaml'


def run(*argv):
    return cli.main([str(a) for a in argv])


def stage_commands(monkeypatch, *argv):
    '''The stage commands a workflow would run, recorded instead of run.'''
    parser, _ = cli.build_parser()
    args = parser.parse_args([str(a) for a in argv])
    calls = []
    monkeypatch.setattr(cli, 'main', lambda stage_argv: calls.append(stage_argv))
    monkeypatch.setattr(cli, 'write_model_config', lambda *a: calls.append(['write_model_config']))
    cli.cmd_workflow(args)
    return calls


def options(call):
    '''{option: value} for the options in one recorded stage command.'''
    pairs = {}
    for i, a in enumerate(call):
        if a.startswith('--'):
            nxt = call[i + 1] if i + 1 < len(call) and not call[i + 1].startswith('--') else True
            pairs.setdefault(a, []).append(nxt)
    return pairs


# ---------------------------------------------------------------- stage commands
def test_prediction_workflow_runs_all_stages_with_the_protein_config(monkeypatch):
    calls = stage_commands(monkeypatch, 'prediction-workflow', '--genomes', 'g1.fna', 'g2.fna', '-o', 'run')
    assert [c[0] for c in calls] == ['extract', 'align', 'embed', 'fit', 'predict']
    assert all(options(c)['--config'] == [str(ADK_CONFIG)] for c in calls)
    assert calls[0][calls[0].index('--genomes') + 1:][:2] == ['g1.fna', 'g2.fna']
    assert options(calls[0])['--mode'] == ['single']

@pytest.mark.parametrize('inputs, extract_options', [
    (['--genomes', 'g.fna'], {'--mode': ['single']}),
    (['--metagenomes', 'm.fa'], {'--mode': ['meta'], '--n-proteins': ['1000']}),
    (['--proteins', 'p.faa'], {'--protein-fasta': ['p.faa']}),
])
def test_prediction_inputs_become_extract_options(monkeypatch, inputs, extract_options):
    extract = options(stage_commands(monkeypatch, 'prediction-workflow', *inputs, '-o', 'run')[0])
    for option, value in extract_options.items():
        assert extract[option] == value

def test_set_options_go_to_their_stage_and_win(monkeypatch):
    calls = stage_commands(monkeypatch, 'prediction-workflow', '--metagenomes', 'm.fa', '-o', 'run',
                           '--set', 'extract.n_proteins=50', '--set', 'predict.sites=cv_top', '--set', 'predict.n-sites=20')
    extract, predict = options(calls[0]), options(calls[-1])
    assert extract['--n-proteins'] == ['1000', '50']          # last value wins in argparse
    assert predict['--sites'] == ['cv_top'] and predict['--n-sites'] == ['20']
    assert '--sites' not in options(calls[1])

def test_shared_options_reach_every_stage(monkeypatch):
    calls = stage_commands(monkeypatch, 'prediction-workflow', '--proteins', 'p.faa', '-o', 'run',
                           '--threads', 8, '--verbose', '--overwrite')
    for call in calls:
        o = options(call)
        assert o['--threads'] == ['8'] and o['--verbose'] == [True] and o['--overwrite'] == [True]

def test_fitting_workflow_trains_instead_of_copying(monkeypatch):
    calls = stage_commands(monkeypatch, 'fitting-workflow', '--alignment', 'B.afa', '--train-ids', 'ids.txt',
                           '--traits', 't.tsv', '--scaler', 's.pickle', '-o', 'fitrun')
    assert [c[0] for c in calls] == ['align', 'embed', 'fit', 'write_model_config']
    align, embed, fit = (options(c) for c in calls[:3])
    assert align['--backbone'] == ['B.afa'] and align['--train-ids'] == ['ids.txt']
    assert embed['--context-dataset'] == [str(Path('fitrun') / cli.RUN_FILES['align']['msa_dataset'])]
    assert embed['--exclude-self'] == [True]
    assert '--config' not in fit                                # the config's fit section names a model to copy
    assert fit['--traits'] == ['t.tsv'] and fit['--scaler'] == ['s.pickle']


# ---------------------------------------------------------------- model config and errors
def test_model_config_points_to_the_fitted_run(tmp_path):
    parser, _ = cli.build_parser()
    args = parser.parse_args(['fitting-workflow', '--alignment', 'B.afa', '--train-ids', 'i', '--traits', 't',
                              '-o', str(tmp_path)])
    cli.write_model_config(ADK_CONFIG, args, [cli.parse_setting('embed.context_size=16')])
    config = yaml.safe_load((tmp_path / 'model_config.yaml').read_text())
    out = tmp_path.resolve()
    assert config['align']['backbone'] == str(out / cli.RUN_FILES['align']['alignment'])
    assert config['embed']['context_dataset'] == str(out / cli.RUN_FILES['align']['msa_dataset'])
    assert config['embed']['context_size'] == 16                 # the training setting, as a number
    assert config['fit'] == {'model': str(out / cli.RUN_FILES['fit']['model']),
                             'scaler': str(out / cli.RUN_FILES['fit']['scaler'])}
    assert Path(config['extract']['hmm']).is_absolute() and Path(config['extract']['hmm']).exists()

@pytest.mark.parametrize('argv, message', [
    (['prediction-workflow', '--proteins', 'p.faa', '--set', 'predict.sites'], 'STAGE.OPTION=VALUE'),
    (['prediction-workflow', '--proteins', 'p.faa', '--set', 'nope.x=1'], 'STAGE.OPTION=VALUE'),
    (['prediction-workflow', 'not_a_protein', '--proteins', 'p.faa'], 'config'),
    (['prediction-workflow'], 'one of the arguments'),
    (['fitting-workflow', '--alignment', 'B.afa'], '--train-ids is required'),
])
def test_workflow_errors_exit_before_any_stage_runs(argv, message, tmp_path, capsys):
    with pytest.raises(SystemExit):
        run(*argv, '-o', tmp_path / 'run')
    assert message in capsys.readouterr().err
    assert not (tmp_path / 'run').exists()


# ---------------------------------------------------------------- resumed workflow (no torch needed)
def test_prediction_workflow_resumes_from_existing_stages(tmp_path):
    '''Stages 1-3 already done: the workflow skips them, copies the configured model and predicts.'''
    rng = np.random.default_rng(0)
    n, sites, dim = 30, 3, 4
    emb = rng.normal(size=(n, sites, dim)).astype(np.float32)
    ds = [(f'p{i}', 'M' * sites, np.nan) for i in range(n)]

    # a small model, fitted with the fit command in its own folder
    model_run = tmp_path / 'model_run'
    io.save_pickle(ds, cli.stage_file(model_run, 'align', 'msa_dataset'))
    np.save(cli.stage_file(model_run, 'embed', 'embeddings'), emb)
    pd.Series(30 + 8 * emb[:, :, 0].mean(1), index=[d[0] for d in ds]).to_csv(tmp_path / 'traits.tsv', sep='\t')
    run('fit', '-o', model_run, '--traits', tmp_path / 'traits.tsv')
    # the adk config with its paths made absolute, pointing fit at the small model
    _, subs = cli.build_parser()
    config_dict = {s: {k: str(v) if isinstance(v, Path) else v
                       for k, v in cli.load_config_section(ADK_CONFIG, s, subs[s]).items()}
                   for s in ['extract', 'align', 'embed', 'predict']}
    config_dict['fit'] = {'model': str(cli.stage_file(model_run, 'fit', 'model')),
                          'scaler': str(cli.stage_file(model_run, 'fit', 'scaler'))}
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump(config_dict))

    # a prediction run whose first three stages are already done
    out = tmp_path / 'run'
    cli.stage_file(out, 'extract', 'proteins').write_text('>p0\nMMM\n')
    io.save_pickle(ds[:5], cli.stage_file(out, 'align', 'msa_dataset'))
    cli.stage_file(out, 'align', 'alignment').write_text('')
    np.save(cli.stage_file(out, 'embed', 'embeddings'), emb[:5])

    with pytest.warns(UserWarning, match='--overwrite'):
        run('prediction-workflow', '--proteins', FIXTURES / 'genomeA_proteins.faa', '-o', out, '--config', config)
    predictions = pd.read_csv(cli.stage_file(out, 'predict', 'predictions'), sep='\t', index_col=0)
    assert list(predictions.index) == [d[0] for d in ds[:5]]
    assert list(pd.read_json(out / 'run.jsonl', lines=True)['stage']) == ['fit', 'predict']


# ---------------------------------------------------------------- end to end
@pytest.mark.slow
@pytest.mark.skipif(not all(shutil.which(t) for t in ['hmmsearch', 'mafft']), reason='needs hmmer and mafft')
def test_fit_then_predict_end_to_end(tmp_path):
    '''Fit on 25 backbone rows, then predict three proteins with the new model's config.'''
    pytest.importorskip('esm')
    rows = list(SeqIO.parse(BACKBONE, 'fasta'))[:30]
    SeqIO.write(rows, tmp_path / 'backbone.afa', 'fasta')
    (tmp_path / 'train_ids.txt').write_text('\n'.join(r.id for r in rows[:25]) + '\n')
    temps = {r.id: float(r.id.split('temp=')[1].split('|')[0]) for r in rows}   # temperatures from the headers
    pd.Series(temps, name='ogt').to_csv(tmp_path / 'traits.tsv', sep='\t')

    fitted = tmp_path / 'fitted'
    run('fitting-workflow', '--alignment', tmp_path / 'backbone.afa', '--train-ids', tmp_path / 'train_ids.txt',
        '--traits', tmp_path / 'traits.tsv', '-o', fitted, '--set', 'embed.context_size=4', '--threads', 2)
    model_config = fitted / 'model_config.yaml'
    assert yaml.safe_load(model_config.read_text())['embed']['context_size'] == 4

    out = tmp_path / 'predicted'
    with pytest.warns(UserWarning, match='1 of 3'):                               # the fixture's decoy protein
        run('prediction-workflow', '--config', model_config, '--proteins', FIXTURES / 'genomeA_proteins.faa',
            '-o', out, '--threads', 2)
    predictions = pd.read_csv(cli.stage_file(out, 'predict', 'predictions'), sep='\t', index_col=0)
    assert list(predictions.index) == ['contig_1_1', 'contig_1_2', 'contig_2_1']
    assert predictions.iloc[:, 0].notna().all()
    assert list(pd.read_json(out / 'run.jsonl', lines=True)['stage']) == ['extract', 'align', 'embed', 'fit', 'predict']