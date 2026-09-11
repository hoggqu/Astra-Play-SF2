"""Bounded native-time batched PPO and natural-coin clears with frozen executable sources."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import sys
import time

from astra_play_sf2.config import atomic_json
from astra_play_sf2.runner import sha256
from .dataset import load_dataset


# Reuse only subprocess lifecycle/result readers; the old campaign is unchanged.
from .campaign import run_stage, read_result


def source_paths():
    """Fixed executing/staging dependencies, not every file in the experiment folder."""
    import astra_play_sf2
    here = Path(__file__).resolve().parent
    package = Path(astra_play_sf2.__file__).resolve().parent
    experiment = (
        '__init__.py', 'native_campaign.py', 'campaign.py', 'batch_train.py',
        'batch_env.py', 'batch_runtime.lua', 'env.py', 'runtime.lua', 'dataset.py',
        'vector.py', 'export.py', 'nn.lua', 'actions.lua', 'continuous.py',
        'continuous_core.lua', 'native_continuous.py', 'native_continuous_core.lua',
    )
    production = ('__init__.py', 'config.py', 'runner.py', 'opening.py', 'transport.py')
    paths = {'experiments/rl/'+name: here/name for name in experiment}
    paths.update({'astra_play_sf2/'+name: package/name for name in production})
    # Freeze the existing asset list once. Adding unrelated experiment files is
    # allowed; replacing a file this campaign reads is not.
    paths.update({'astra_play_sf2/assets/'+path.name: path for path in sorted((package/'assets').iterdir())
                  if path.suffix in ('.lua', '.json')})
    return paths


def freeze_sources(paths):
    return {name: sha256(path) for name, path in paths.items()}


def require_sources(paths, expected):
    changed = [name for name, path in paths.items()
               if not path.is_file() or sha256(path) != expected[name]]
    if changed:
        raise RuntimeError('Frozen executable source changed: '+', '.join(changed))


def campaign(dataset, output, init_model=None, workers=8, cycles=5,
             steps_per_cycle=102400, block=64, seed=42,
             stage_timeout=3600, stage_runner=run_stage):
    if type(workers) is not int or workers not in range(1, 9):
        raise ValueError('workers must be 1..8')
    if any(type(n) is not int or n < 1 for n in (cycles, steps_per_cycle, stage_timeout)):
        raise ValueError('Cycle, step and timeout budgets must be positive integers')
    if steps_per_cycle % (workers*256) or block not in (1, 16, 32, 64, 128, 256):
        raise ValueError('Steps must be workers*256 multiples; block must divide 256')
    dataset = Path(dataset).resolve()
    _, difficulty = load_dataset(dataset)
    if difficulty != 3:
        raise ValueError('This campaign targets Normal difficulty 3')
    initial = Path(init_model).resolve() if init_model else None
    initial_hash = sha256(initial) if initial else None
    paths = source_paths()
    frozen_sources = freeze_sources(paths)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = {'schema': 'astra.rl-native-campaign.v1', 'status': 'running',
              'goal': 'one neural Normal native eleven-match clear; no continue, state load or in-match pause',
              'difficulty': difficulty, 'seed': seed, 'workers': workers,
              'cycles_requested': cycles, 'steps_per_cycle': steps_per_cycle,
              'maximum_training_steps': cycles*steps_per_cycle, 'block': block,
              'selection': 'last model; cumulative optimizer/weights carried forward, no dev selection',
              'holdout_opened': False, 'native_timing_required': True,
              'verification_attempts_per_cycle': 3, 'stage_timeout_seconds': stage_timeout,
              'dataset_sha256': sha256(dataset), 'initial_model_sha256': initial_hash,
              'frozen_v4_certification': False, 'cycles': [],
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'sources': frozen_sources}
    def save():
        atomic_json(output/'result.json', result)
    def stage(row, name, module, arguments, folder):
        require_sources(paths, frozen_sources)
        if sha256(dataset) != result['dataset_sha256']:
            raise RuntimeError('Dataset manifest changed during campaign')
        model_flag = '--init-model' if name == 'train' else '--model'
        input_model = Path(arguments[arguments.index(model_flag)+1]) if model_flag in arguments else None
        expected_model = row['initial_model_sha256'] if name == 'train' else row['model_sha256']
        if input_model and sha256(input_model) != expected_model:
            raise RuntimeError('Stage input model changed before launch')
        row['stage'] = name
        row.setdefault('commands', {})[name] = [sys.executable, '-m', module, *map(str, arguments)]
        save()
        code = stage_runner(module, arguments, folder/f'{name}.log', stage_timeout)
        row.setdefault('exit_codes', {})[name] = code
        save()
        require_sources(paths, frozen_sources)
        if sha256(dataset) != result['dataset_sha256']:
            raise RuntimeError('Dataset manifest changed during stage')
        if input_model and sha256(input_model) != expected_model:
            raise RuntimeError('Stage input model changed during execution')
        return code
    model = initial
    try:
        save()
        for ordinal in range(1, cycles+1):
            folder = output/f'cycle-{ordinal:03d}'
            folder.mkdir()
            row = {'ordinal': ordinal, 'status': 'running',
                   'initial_model_sha256': sha256(model) if model else None}
            result['cycles'].append(row)
            train_dir = folder/'train'
            arguments = ['--dataset', dataset, '--output', train_dir, '--workers', workers,
                         '--steps', steps_per_cycle, '--block', block, '--seed', seed+ordinal-1]
            if model:
                arguments += ['--init-model', model]
            code = stage(row, 'train', 'experiments.rl.batch_train', arguments, folder)
            trained = read_result(train_dir/'result.json', 'astra.rl-batch-prototype.v1')
            if code != 0 or trained.get('actual_steps') != steps_per_cycle or trained.get('difficulty') != difficulty:
                raise RuntimeError('Training exit, step budget or difficulty mismatch')
            if trained.get('benchmark') is not False or trained.get('parity') is not False:
                raise RuntimeError('Expected real batched PPO training, not benchmark/parity')
            if trained.get('dataset_sha256') != result['dataset_sha256']:
                raise RuntimeError('Training used another dataset')
            model = train_dir/'ppo-batch.zip'
            if sha256(model) != trained.get('model_sha256'):
                raise RuntimeError('Selected model checksum mismatch')
            row.update(training_steps=trained['actual_steps'],
                       model_sha256=trained['model_sha256'], training_result=str((train_dir/'result.json').relative_to(output)))
            verify_dir = folder/'continuous'
            # continuous exports this frozen ZIP, audits its hash and performs
            # inference wholly inside Lua. No agent or Python decision pauses.
            code = stage(row, 'continuous', 'experiments.rl.native_continuous',
                         ['--model', model, '--output', verify_dir, '--difficulty', difficulty,
                          '--attempts', 3, '--speed', 'fast'], folder)
            verified = read_result(verify_dir/'result.json', 'astra.rl-continuous.v1')
            if verified.get('native_timing') is not True:
                raise RuntimeError('Continuous run did not certify native-time action scheduling')
            attempts = verified.get('attempts', [])
            if verified.get('model_sha256') != row['model_sha256'] or verified.get('difficulty') != difficulty:
                raise RuntimeError('Continuous model or difficulty mismatch')
            if not 1 <= len(attempts) <= 3 or any(a.get('outcome') not in ('loss', 'rl_gameplay_clear') or not a.get('audit', {}).get('ok') for a in attempts):
                raise RuntimeError('Continuous attempt audit/outcome invalid')
            if any(a['outcome'] == 'rl_gameplay_clear' and (a.get('match_wins') != 11 or len(a.get('matches', [])) != 11) for a in attempts):
                raise RuntimeError('Clear requires eleven audited native match wins')
            clears = sum(a['outcome'] == 'rl_gameplay_clear' for a in attempts)
            if code != (0 if clears else 1) or (not clears and len(attempts) != 3):
                raise RuntimeError('Continuous exit code or attempt budget mismatch')
            row.update(status='complete', stage='complete', clears=clears,
                       attempts=[{'id': a['id'], 'outcome': a['outcome'], 'match_wins': a.get('match_wins'),
                                  'failed_match': (a.get('matches') or [None])[-1] if a['outcome'] == 'loss' else None} for a in attempts],
                       verification_result=str((verify_dir/'result.json').relative_to(output)))
            save()
            print(json.dumps({'cycle': ordinal, 'steps': steps_per_cycle, 'clears': clears,
                              'outcomes': [a['outcome'] for a in attempts]}), flush=True)
            if clears:
                result.update(status='complete', goal_achieved=True, success_cycle=ordinal,
                              selected_model=str(model.relative_to(output)), model_sha256=row['model_sha256'])
                break
        else:
            result.update(status='complete', goal_achieved=False, stop_reason='cycle_budget_exhausted')
    except BaseException as error:
        result.update(status='invalid', goal_achieved=False, error=f'{type(error).__name__}: {error}')
        if result['cycles'] and result['cycles'][-1]['status'] == 'running':
            result['cycles'][-1].update(status='invalid', error=result['error'])
    finally:
        result['completed_training_steps'] = sum(row.get('training_steps', 0) for row in result['cycles'])
        result['wall_seconds'] = time.monotonic()-started
        result['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--init-model', type=Path)
    parser.add_argument('--workers', type=int, choices=range(1, 9), default=8)
    parser.add_argument('--cycles', type=int, default=5)
    parser.add_argument('--steps-per-cycle', type=int, default=102400)
    parser.add_argument('--block', type=int, choices=(1, 16, 32, 64, 128, 256), default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--stage-timeout', type=int, default=3600)
    args = parser.parse_args()
    def interrupted(_signal, _frame):
        raise KeyboardInterrupt('SIGTERM')
    signal.signal(signal.SIGTERM, interrupted)
    result = campaign(**vars(args))
    print(json.dumps(result, indent=2), flush=True)
    return 2 if result['status'] != 'complete' else 0 if result['goal_achieved'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
