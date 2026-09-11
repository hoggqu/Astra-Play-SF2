"""One-generation PPO prefetch alongside immutable full-20 evaluation (POSIX)."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from astra_play_sf2.config import atomic_json, config_path
from astra_play_sf2.runner import sha256
from .campaign import read_result
from .dataset import load_dataset
from .native_campaign import freeze_sources, require_sources
from .reliability_campaign import INTERFACE, audit_full_twenty, execution_sources

HERE = Path(__file__).resolve().parent


def live_group_members(pgid):
    """Exclude dead zombies; only the session created by OwnedStage is eligible."""
    text = subprocess.check_output(['ps', '-axo', 'pid=,pgid=,stat='], text=True)
    members = []
    for line in text.splitlines():
        columns = line.split()
        if len(columns) >= 3 and int(columns[1]) == pgid and not columns[2].startswith('Z'):
            members.append(int(columns[0]))
    return members


class OwnedStage:
    """Explicit child handle; never discover or terminate another controller."""
    def __init__(self, code, module, arguments, log_path, timeout):
        if os.name != 'posix':
            raise RuntimeError('This experimental pipeline requires POSIX owned process groups; use the serial campaign on Windows')
        self.started = time.monotonic()
        self.timeout = timeout
        self.log = Path(log_path).open('xb')
        env = dict(os.environ, PYTHONPATH=str(code/'src')+os.pathsep+str(code))
        for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            env[name] = '1'
        try:
            self.process = subprocess.Popen([sys.executable, '-m', module, *map(str, arguments)],
                cwd=code, env=env, stdin=subprocess.DEVNULL, stdout=self.log,
                stderr=subprocess.STDOUT, start_new_session=True)
        except BaseException:
            self.log.close()
            raise
        self.pid = self.process.pid
        self.pgid = self.pid

    def poll(self):
        code = self.process.poll()
        if code is None and time.monotonic()-self.started > self.timeout:
            raise TimeoutError(f'Owned stage {self.pid} exceeded {self.timeout}s')
        return code

    def _signal_group(self, number):
        if self.pgid == os.getpgrp():
            raise RuntimeError('Refusing to signal the orchestrator process group')
        try:
            os.killpg(self.pgid, number)
        except ProcessLookupError:
            pass

    def stop(self, reason, grace=50):
        """Let the trainer flush its result, then clean surviving owned descendants."""
        before = self.process.poll()
        record = {'reason': reason, 'pid': self.pid, 'pgid': self.pgid,
                  'returncode_before_stop': before, 'cancel_requested': before is None}
        if before is None:
            try:
                self.process.terminate()
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                pass
        # This also handles an already-exited parent with live workers/MAME.
        remaining = live_group_members(self.pgid)
        if remaining:
            self._signal_group(signal.SIGTERM)
            deadline = time.monotonic()+min(grace, 1)
            while live_group_members(self.pgid) and time.monotonic() < deadline:
                self.process.poll()
                time.sleep(.02)
            if live_group_members(self.pgid):
                self._signal_group(signal.SIGKILL)
        self.process.wait(timeout=5)
        deadline = time.monotonic()+5
        while live_group_members(self.pgid) and time.monotonic() < deadline:
            time.sleep(.02)
        remaining = live_group_members(self.pgid)
        self.log.close()
        record.update(returncode=self.process.returncode, remaining_live_pids=remaining)
        if remaining:
            raise RuntimeError(f'Owned process group did not stop: {record}')
        return record

    def finish(self):
        code = self.process.poll()
        if code is None:
            raise RuntimeError('Cannot finish a running stage')
        if live_group_members(self.pgid):
            self.stop('unexpected_descendants_after_stage_exit')
            raise RuntimeError('Stage exited with live owned descendants; cleaned and retained as invalid')
        self.log.close()
        return code


def preserved_training_progress(folder):
    """Report only checksum-verified complete saved updates; do not rewrite child data."""
    source = folder/'result.json'
    if not source.is_file():
        return {'result_present': False}
    data = json.loads(source.read_text(encoding='utf-8'))
    record = {key: data.get(key) for key in ('status', 'error', 'completed_update_steps',
              'optimizer_epochs_completed', 'actual_steps', 'last_checkpoint')}
    checkpoint = data.get('last_checkpoint')
    if checkpoint:
        path = (folder/checkpoint['path']).resolve()
        if not path.is_relative_to(folder.resolve()) or sha256(path) != checkpoint['sha256']:
            raise RuntimeError('Retained completed checkpoint path/hash invalid')
        record['last_checkpoint_verified'] = True
    return record


def pipeline(dataset, output, init_model, training_code, training_package,
             verification_code, verification_package, initial_training_result=None,
             initial_budget_steps=0, workers=8, cycles=5, steps_per_cycle=409600,
             block=64, seed=129, stage_timeout=7200, stage_factory=OwnedStage,
             poll_interval=.1):
    if type(workers) is not int or workers not in range(1, 9):
        raise ValueError('workers must be 1..8')
    if any(type(n) is not int or n < 1 for n in (cycles, steps_per_cycle, stage_timeout)):
        raise ValueError('Positive candidate, decision and timeout budgets required')
    if (steps_per_cycle % (workers*256) or block not in (1, 16, 32, 64, 128, 256)
            or type(initial_budget_steps) is not int or initial_budget_steps < 0):
        raise ValueError('Invalid completed-update budget or block')
    dataset, model = Path(dataset).resolve(), Path(init_model).resolve()
    training_code, verification_code = Path(training_code).resolve(), Path(verification_code).resolve()
    groups, difficulty = load_dataset(dataset)
    if difficulty != 3:
        raise ValueError('This pipeline targets Normal difficulty 3')
    paths = {}
    for role, root, package in (('train', training_code, training_package), ('verify', verification_code, verification_package)):
        paths.update({role+'/'+name: path for name, path in execution_sources(root, package).items()})
    for name in ('reliability_pipeline.py', 'reliability_campaign.py', 'campaign.py', 'native_campaign.py', 'dataset.py', '__init__.py'):
        paths['orchestrator/'+name] = HERE/name
    paths['orchestrator/parent_init.py'] = HERE.parent/'__init__.py'
    import astra_play_sf2
    production = Path(astra_play_sf2.__file__).resolve().parent
    paths.update({'orchestrator/production/'+p.relative_to(production).as_posix(): p
                  for p in production.rglob('*') if p.is_file() and p.suffix in ('.py', '.lua', '.json')})
    paths['dataset'] = dataset
    for i, sample in enumerate(groups.get('train', [])):
        paths[f'train_checkpoint/{i}'] = Path(sample['path'])
    if config_path().is_file():
        paths['mame_config'] = config_path()
    initial_hash = sha256(model)
    initial = None
    if initial_training_result:
        source = Path(initial_training_result).resolve()
        initial = read_result(source, 'astra.rl-batch-prototype.actions16.v1')
        if initial.get('model_sha256') != initial_hash:
            raise ValueError('Initial completed training result identifies another model')
        paths['initial_training_result'] = source
    pins = freeze_sources(paths)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = {'schema': 'astra.rl-reliability-pipeline.v1', 'status': 'running', 'goal_achieved': False,
        'goal': 'one fixed model clears at least 10 of its complete 20 natural coin attempts',
        'difficulty': 3, 'actions': 16, 'action_interface': INTERFACE,
        'initial_model_sha256': initial_hash, 'initial_training_result': str(initial_training_result) if initial_training_result else None,
        'initial_training_steps': initial.get('actual_steps') if initial else None,
        'initial_opponent_sampling': initial.get('opponent_sampling') if initial else None,
        'reported_initial_budget_steps': initial_budget_steps, 'cycles_requested': cycles,
        'maximum_attempts': cycles*20, 'verification_attempts_per_candidate': 20, 'required_clears': 10,
        'maximum_new_training_steps': (cycles-1)*steps_per_cycle, 'steps_per_training_cycle': steps_per_cycle,
        'prefetch_generations': 1, 'maximum_concurrent_training_stages': 1,
        'maximum_concurrent_verification_stages': 1, 'workers': workers, 'block': block, 'seed': seed,
        'stage_timeout_seconds': stage_timeout, 'sources': pins, 'cycles': [],
        'holdout_opened': False, 'frozen_v4_certification': False,
        'continuation': 'PPO parameters/optimizer continue; each training environment/RNG restarts. No partial evaluation guides training.',
        'created_utc': datetime.now(timezone.utc).isoformat()}
    active = {}

    def save():
        atomic_json(output/'result.json', result)

    def guard(input_model, expected):
        require_sources(paths, pins)
        if sha256(input_model) != expected:
            raise RuntimeError('Frozen input model changed')

    def start(row, kind, root, module, args, input_model, expected):
        guard(input_model, expected)
        folder = output/f"cycle-{row['ordinal']:03d}"
        folder.mkdir(exist_ok=True)
        info = {'status': 'starting', 'cwd': str(root), 'module': module,
                'arguments': list(map(str, args)), 'input_model': str(input_model), 'input_model_sha256': expected,
                'started_elapsed_seconds': time.monotonic()-started}
        row[kind] = info
        save()
        handle = stage_factory(root, module, args, folder/(kind+'.log'), stage_timeout)
        # Register immediately: failure to write metadata must still stop this child.
        active[kind] = (handle, row, input_model, expected)
        info.update(status='running', pid=handle.pid)
        save()

    def finish(kind):
        handle, row, input_model, expected = active[kind]
        code = handle.finish()
        row[kind]['exit_code'] = code
        guard(input_model, expected)
        folder = output/f"cycle-{row['ordinal']:03d}"
        if kind == 'train':
            trained = read_result(folder/'train/result.json', 'astra.rl-batch-prototype.actions16.v1')
            if (code != 0 or trained.get('actual_steps') != steps_per_cycle or trained.get('difficulty') != 3
                    or any(trained.get(k) is not False for k in ('benchmark', 'parity', 'native_parity'))
                    or trained.get('parameters_changed') is not True or trained.get('init_model_sha256') != expected
                    or trained.get('dataset_sha256') != pins['dataset'] or trained.get('action_interface') != INTERFACE
                    or trained.get('actions') != 16
                    or trained.get('optimizer_initialization', {}).get('loaded_from_init_model') is not True):
                raise RuntimeError('Prefetched training budget, identity, optimizer or real-update mismatch')
            next_model = folder/'train/ppo-batch.zip'
            if sha256(next_model) != trained.get('model_sha256'):
                raise RuntimeError('Completed prefetched model checksum mismatch')
            row.update(model=str(next_model), model_sha256=trained['model_sha256'], training_steps=trained['actual_steps'],
                status='ready', optimizer_initialization=trained.get('optimizer_initialization'), opponent_sampling=trained.get('opponent_sampling'))
        else:
            verified = read_result(folder/'continuous/result.json', 'astra.rl-continuous.actions16.v1')
            row.update(audit_full_twenty(verified, expected, code))
            row.update(status='complete', attempts=[{k: a.get(k) for k in ('id', 'outcome', 'match_wins', 'matches', 'images')}
                                                   for a in verified['attempts']])
        row[kind].update(status='complete', result=str((folder/kind/'result.json').relative_to(output)))
        row[kind]['ended_elapsed_seconds'] = time.monotonic()-started
        del active[kind]
        save()

    def stop_active(reason):
        errors = []
        for kind, (handle, row, input_model, expected) in list(active.items()):
            info = row[kind]
            info['stop_reason'] = reason
            try:
                info['stop'] = handle.stop(reason)
                if reason == 'owner_cancelled_after_verified_goal' and not info['stop']['cancel_requested']:
                    # A completion raced the cancellation request. Validate the
                    # completed stage normally, including any genuine failure.
                    finish(kind)
                    row['status'] = 'not_evaluated_after_goal'
                    continue
                state = 'cancelled' if info['stop']['cancel_requested'] else 'invalid'
                info['status'] = state
                row['status'] = state
                guard(input_model, expected)
                if kind == 'train':
                    info['retained_progress'] = preserved_training_progress(output/f"cycle-{row['ordinal']:03d}"/'train')
            except BaseException as error:
                errors.append(f'{kind}: {type(error).__name__}: {error}')
                info.update(status='invalid', cleanup_error=errors[-1])
            finally:
                info['ended_elapsed_seconds'] = time.monotonic()-started
                active.pop(kind, None)
        return errors

    try:
        current = {'ordinal': 1, 'status': 'pending', 'model': str(model), 'model_sha256': initial_hash, 'training_steps': 0}
        result['cycles'].append(current)
        save()
        for ordinal in range(1, cycles+1):
            model = Path(current['model'])
            folder = output/f'cycle-{ordinal:03d}'
            args = ['--model', model, '--output', folder/'continuous', '--difficulty', 3,
                    '--attempts', 20, '--speed', 'fast', '--all-attempts']
            start(current, 'continuous', verification_code, verification_package+'.native_continuous', args, model, current['model_sha256'])
            following = None
            if ordinal < cycles:
                following = {'ordinal': ordinal+1, 'status': 'prefetching', 'training_steps': 0}
                result['cycles'].append(following)
                next_folder = output/f'cycle-{ordinal+1:03d}'
                args = ['--dataset', dataset, '--output', next_folder/'train', '--init-model', model,
                        '--workers', workers, '--steps', steps_per_cycle, '--block', block, '--seed', seed+ordinal-1]
                start(following, 'train', training_code, training_package+'.batch_train', args, model, current['model_sha256'])
            while active:
                # Check an already-failed trainer before accepting a concurrent goal.
                for kind in ('train', 'continuous'):
                    if kind in active and active[kind][0].poll() is not None:
                        finish(kind)
                if current.get('goal_achieved'):
                    result['verified_goal_cycle'] = ordinal
                    errors = stop_active('owner_cancelled_after_verified_goal')
                    if errors:
                        raise RuntimeError('Goal verified, but owned cleanup failed: '+'; '.join(errors))
                    result.update(status='complete', goal_achieved=True, success_cycle=ordinal,
                                  selected_model=current['model'], model_sha256=current['model_sha256'])
                    break
                if active:
                    time.sleep(poll_interval)
            if result['goal_achieved']:
                break
            # No N+2 starts until N full20 passed its audit and N+1 completed.
            current = following
        else:
            result.update(status='complete', stop_reason='candidate_budget_exhausted')
    except BaseException as error:
        result.update(status='invalid', goal_achieved=False, error=f'{type(error).__name__}: {error}')
        errors = stop_active('owner_cancelled_after_pipeline_invalid')
        if errors:
            result['cleanup_errors'] = errors
    finally:
        result['completed_new_training_steps'] = sum(row['training_steps'] for row in result['cycles'])
        result['completed_candidate_evaluations'] = sum(row.get('continuous', {}).get('status') == 'complete' for row in result['cycles'])
        result['wall_seconds'] = time.monotonic()-started
        save()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('dataset', 'output', 'init-model', 'training-code', 'verification-code'):
        parser.add_argument('--'+key, type=Path, required=True)
    for key in ('training-package', 'verification-package'):
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--initial-training-result', type=Path)
    parser.add_argument('--initial-budget-steps', type=int, default=0)
    parser.add_argument('--workers', type=int, choices=range(1, 9), default=8)
    parser.add_argument('--cycles', type=int, default=5)
    parser.add_argument('--steps-per-cycle', type=int, default=409600)
    parser.add_argument('--block', type=int, default=64)
    parser.add_argument('--seed', type=int, default=129)
    parser.add_argument('--stage-timeout', type=int, default=7200)
    def stopped(_signal, _frame):
        raise KeyboardInterrupt('Owner cancellation; preserve without retry')
    signal.signal(signal.SIGTERM, stopped)
    result = pipeline(**vars(parser.parse_args()))
    print(json.dumps(result, indent=2), flush=True)
    return 2 if result['status'] != 'complete' else 0 if result['goal_achieved'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
