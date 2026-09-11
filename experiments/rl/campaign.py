"""Bounded autonomous PPO cycles and immutable natural-coin neural clear attempts."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from astra_play_sf2.config import atomic_json
from astra_play_sf2.runner import sha256
from .dataset import load_dataset


def stop_owned(process):
    """Graceful child cleanup; group kill is confined to our newly owned session."""
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=55)
    except subprocess.TimeoutExpired:
        if os.name != 'nt':
            os.killpg(process.pid, signal.SIGKILL)
        else:
            subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
        process.wait(timeout=5)


def run_stage(module, arguments, log_path, timeout):
    command = [sys.executable, '-m', module, *map(str, arguments)]
    with Path(log_path).open('wb') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=os.name != 'nt')
        try:
            return process.wait(timeout=timeout)
        finally:
            stop_owned(process)


def read_result(path, schema):
    result = json.loads(Path(path).read_text(encoding='utf-8'))
    if result.get('schema') != schema or result.get('status') != 'complete':
        raise RuntimeError(f'Invalid child result: {path}: {result.get("status")}: {result.get("error")}')
    return result


def campaign(dataset, output, init_model=None, workers=8, cycles=5,
             steps_per_cycle=102400, eval_every=102400, seed=42,
             stage_timeout=3600, stage_runner=run_stage):
    if type(workers) is not int or workers not in range(1, 9):
        raise ValueError('workers must be 1..8')
    if any(type(n) is not int or n < 1 for n in (cycles, steps_per_cycle, eval_every, stage_timeout)):
        raise ValueError('Cycle, step and timeout budgets must be positive integers')
    if steps_per_cycle % (workers*256) or eval_every % (workers*256) or steps_per_cycle % eval_every:
        raise ValueError('Steps must divide eval-every and both must be multiples of workers*256')
    dataset = Path(dataset).resolve()
    _, difficulty = load_dataset(dataset)
    if difficulty != 3:
        raise ValueError('This campaign targets Normal difficulty 3')
    initial = Path(init_model).resolve() if init_model else None
    initial_hash = sha256(initial) if initial else None
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = {'schema': 'astra.rl-campaign.v1', 'status': 'running',
              'goal': 'one neural Normal native eleven-match clear; no continue, state load or in-match pause',
              'difficulty': difficulty, 'seed': seed, 'workers': workers,
              'cycles_requested': cycles, 'steps_per_cycle': steps_per_cycle,
              'maximum_training_steps': cycles*steps_per_cycle, 'eval_every': eval_every,
              'development_leads': [2], 'holdout_opened': False,
              'verification_attempts_per_cycle': 3, 'stage_timeout_seconds': stage_timeout,
              'dataset_sha256': sha256(dataset), 'initial_model_sha256': initial_hash,
              'frozen_v4_certification': False, 'cycles': [],
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'sources': {p.name: sha256(p) for p in Path(__file__).parent.iterdir() if p.suffix in ('.py', '.lua')}}
    def save():
        atomic_json(output/'result.json', result)
    def stage(row, name, module, arguments, folder):
        row['stage'] = name
        row.setdefault('commands', {})[name] = [sys.executable, '-m', module, *map(str, arguments)]
        save()
        code = stage_runner(module, arguments, folder/f'{name}.log', stage_timeout)
        row.setdefault('exit_codes', {})[name] = code
        save()
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
                         '--steps', steps_per_cycle, '--eval-every', eval_every,
                         '--eval-leads', '2', '--skip-holdout', '--seed', seed+ordinal-1]
            if model:
                arguments += ['--init-model', model]
            code = stage(row, 'train', 'experiments.rl.train', arguments, folder)
            trained = read_result(train_dir/'result.json', 'astra.rl-scaled.v1')
            if code != 0 or trained.get('actual_steps') != steps_per_cycle or trained.get('difficulty') != difficulty:
                raise RuntimeError('Training exit, step budget or difficulty mismatch')
            if not trained.get('skip_holdout') or 'selected_holdout' in trained:
                raise RuntimeError('Training unexpectedly used final holdout')
            model = train_dir/'best-dev.zip'
            if sha256(model) != trained.get('model_sha256'):
                raise RuntimeError('Selected model checksum mismatch')
            row.update(training_steps=trained['actual_steps'], best_dev_steps=trained['best_dev_steps'],
                       model_sha256=trained['model_sha256'], training_result=str((train_dir/'result.json').relative_to(output)))
            verify_dir = folder/'continuous'
            # continuous exports this frozen ZIP, audits its hash and performs
            # inference wholly inside Lua. No agent or Python decision pauses.
            code = stage(row, 'continuous', 'experiments.rl.continuous',
                         ['--model', model, '--output', verify_dir, '--difficulty', difficulty,
                          '--attempts', 3, '--speed', 'fast'], folder)
            verified = read_result(verify_dir/'result.json', 'astra.rl-continuous.v1')
            attempts = verified.get('attempts', [])
            if verified.get('model_sha256') != row['model_sha256'] or verified.get('difficulty') != difficulty:
                raise RuntimeError('Continuous model or difficulty mismatch')
            if not 1 <= len(attempts) <= 3 or any(a.get('outcome') not in ('loss', 'rl_gameplay_clear') or not a.get('audit', {}).get('ok') for a in attempts):
                raise RuntimeError('Continuous attempt audit/outcome invalid')
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
    parser.add_argument('--eval-every', type=int, default=102400)
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
