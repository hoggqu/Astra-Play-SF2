"""Collect bounded natural-coin Blanka R1 states, for training only.

No reset, load or game-RAM write is used. Saving can advance native time by a
frame; both observations are retained. Distinct state bytes are not a claim of
statistical independence. The final attempt stops once the dataset is full.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import time

from astra_play_sf2.config import atomic_json, doctor, load_config
from astra_play_sf2.opening import make_opening_guard, require_difficulty
from astra_play_sf2.runner import NAMES, boot_config, mame_command, sha256, stage_runtime
from astra_play_sf2.transport import Bridge, read_json


class BudgetReached(RuntimeError):
    """The collector stopped within its explicit resource budget."""


class BoundedBridge(Bridge):
    def __init__(self, run, process, deadline):
        super().__init__(run, process, timeout=240)
        self.deadline = deadline

    def wait(self, predicate, timeout=None):
        remaining = self.deadline-time.monotonic()
        if remaining <= 0:
            raise BudgetReached('Wall-time budget exhausted; no request replay')
        try:
            return super().wait(predicate, min(remaining, self.timeout if timeout is None else timeout))
        except TimeoutError:
            if time.monotonic() >= self.deadline:
                raise BudgetReached('Wall-time budget exhausted; owned process will stop') from None
            raise

    def send(self, command, snapshot=True):
        if time.monotonic() >= self.deadline:
            raise BudgetReached('Wall-time budget exhausted before command')
        return super().send(command, snapshot)


def split_for(index, samples):
    """Predetermined split by collection order, independent of match results."""
    return 'train' if index < samples-2 else ('dev' if index == samples-2 else 'holdout')


def capture(run, bridge, manifest, attempt, number, initial):
    require_difficulty(initial, manifest['difficulty'])
    if not make_opening_guard(7-manifest['difficulty'])[1](initial, 2):
        raise ValueError('Capture requires a visible full-health Blanka R1')
    identifier = f"a{attempt['ordinal']:03d}-m{number:02d}"
    path = Path('checkpoints')/(identifier+'.sta')
    (run/path).parent.mkdir(exist_ok=True)
    after = bridge.send('checkpoint('+json.dumps((run/path).as_posix(), ensure_ascii=False)+')')
    if not (run/path).is_file():
        raise RuntimeError('Native checkpoint was not saved')
    require_difficulty(after, manifest['difficulty'])
    if not make_opening_guard(7-manifest['difficulty'])[1](after, 2):
        raise ValueError('Post-save observation is no longer a full-health Blanka R1')
    digest = sha256(run/path)
    row = {'id': identifier, 'path': path.as_posix(), 'sha256': digest,
           'difficulty': manifest['difficulty'], 'opponent': 2,
           'attempt': attempt['ordinal'], 'match': number,
           'initial_state': initial, 'post_save_state': after}
    duplicate = next((o['id'] for o in manifest['openings'] if o['sha256'] == digest), None)
    if duplicate:
        row.update(status='duplicate', duplicate_of=duplicate)
        manifest['rejected_openings'].append(row)
    else:
        row.update(status='accepted', split=split_for(len(manifest['openings']), manifest['samples_requested']))
        manifest['openings'].append(row)
    attempt['captures'].append(identifier)
    return row


def collect_attempt(run, bridge, manifest, attempt, save):
    prefix = f"training/collection/a{attempt['ordinal']:03d}"
    (run/prefix).mkdir(parents=True)
    def action(command):
        return bridge.send("speed('fast');"+command)
    def advance(frames):
        return action(f'next_round({frames})')
    def gate(kind):
        state = action(f'wait_{kind}_ready(9000)')
        result = state.get('session_gate')
        attempt['readiness'][kind] = result
        save()
        if not isinstance(result, dict) or result.get('kind') != kind or result.get('status') != 'ready':
            raise RuntimeError(f'Native {kind} readiness failed: {result}')
    gate('coin')
    action("astra_entry.require_ready('coin');act({{3,'C'},{1,''}})")
    gate('start')
    action("astra_entry.require_ready('start');act({{3,'S'},{120,''},{3,'D'},{12,''}})")
    action("act({{6,'LP'},{120,''}})")
    state = advance(1200)
    seen = []
    guard = make_opening_guard(7-manifest['difficulty'])[3]
    for number in range(1, 12):
        opponent = state['p2']['character']
        state = guard(state, opponent, lambda _s, _o, n: advance(n))
        expected = opponent in {0, 1, 2, 3, 5, 6, 7} if number <= 7 else opponent == [10, 11, 9, 8][number-8]
        if not expected or opponent in seen:
            raise RuntimeError('Unexpected native opponent route')
        seen.append(opponent)
        if opponent == 2:
            capture(run, bridge, manifest, attempt, number, state)
            save()
            if len(manifest['openings']) >= manifest['samples_requested']:
                attempt.update(status='stopped_after_final_capture', outcome=None)
                return
        if manifest['matches_started'] >= manifest['max_matches']:
            raise BudgetReached('Match budget exhausted before next match')
        match = f'{prefix}/m{number:02d}-{NAMES[opponent]}'
        row = {'number': number, 'opponent': opponent, 'path': match+'.json', 'status': 'running'}
        attempt['matches'].append(row)
        manifest['matches_started'] += 1
        save()
        bridge.send(f"play_match('{match}',{opponent},{{training_validation=true,speed='fast'}})", snapshot=False)
        def finished():
            result = read_json(run/(match+'-status.json'))
            return result if result and result.get('status') in ('complete', 'invalid') else False
        result = bridge.wait(finished)
        row.update(status=result.get('status'), result=result.get('result'), reason=result.get('reason'))
        save()
        if result.get('status') != 'complete' or result.get('valid_continuous') is not True:
            raise RuntimeError('Collection match invalid: '+str(result.get('reason')))
        if result.get('result') == 'cpu_win':
            attempt.update(status='complete', outcome='loss')
            return
        if result.get('result') != 'ken_win':
            raise RuntimeError('Unexpected collection match result')
        if opponent == 8:
            # The next coin readiness gate consumes ending/attract neutral time.
            attempt.update(status='complete', outcome='route_completed_training_only')
            return
        if number == 3:
            advance(1200)
            action("local s=repeatseq({{6,'D HK'},{6,''}},100);s[#s+1]={740,''};act(s)")
            state = advance(1800)
        elif number in (6, 9):
            advance(1200)
            action("act({{"+str(2000 if number == 6 else 1940)+",''}})")
            state = advance(1800)
        else:
            state = advance(1800 if number == 7 else 1200)


def collect(config, output, samples=6, max_attempts=12, max_matches=132, max_seconds=900, difficulty=7):
    if type(samples) is not int or samples < 3:
        raise ValueError('At least three samples are needed for train/dev/holdout')
    if any(type(n) is not int or n < 1 for n in (max_attempts, max_matches, max_seconds)):
        raise ValueError('Attempt, match and seconds budgets must be positive integers')
    if type(difficulty) is not int or difficulty not in range(3, 8):
        raise ValueError('Difficulty must be 3..7')
    preflight = doctor(config)
    if not preflight['ok']:
        raise RuntimeError('Preflight failed: '+json.dumps(preflight))
    run = Path(output).resolve()
    run.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = {'schema': 'astra.rl-openings.v1', 'training_only': True, 'formal_clear': False,
                'status': 'running', 'created_utc': datetime.now(timezone.utc).isoformat(),
                'difficulty': difficulty, 'opponent': 2, 'samples_requested': samples,
                'max_attempts': max_attempts, 'max_matches': max_matches, 'max_seconds': max_seconds,
                'matches_started': 0, 'openings': [], 'rejected_openings': [], 'attempts': [],
                'preflight': preflight, 'platform': platform.platform(), 'python': platform.python_version(),
                'sampling': 'One boot; natural coins after native game over. One Blanka R1 per attempt. No loads, resets or RAM writes. Saving may advance a frame. Different state hashes do not prove independence.',
                'split_method': 'Collection order: first N-2 train, penultimate dev, last holdout; no outcome-based selection',
                'sound': 'none', 'video': 'none',
                'experiment_sources': {p.name: sha256(p) for p in Path(__file__).parent.iterdir() if p.suffix in ('.py', '.lua')}}
    def save():
        atomic_json(run/'manifest.json', manifest)
    process = log = None
    try:
        save()
        manifest['runtime_sha256'] = stage_runtime(run, difficulty)
        manifest['boot_config_sha256'] = boot_config(run, difficulty)
        command = mame_command(config)+['-sound', 'none', '-video', 'none']
        manifest['command'] = command
        save()
        log = (run/'mame.log').open('wb')
        process = subprocess.Popen(command, cwd=run, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        bridge = BoundedBridge(run, process, started+max_seconds)
        manifest['ready'] = bridge.wait(lambda: read_json(run/'training/ready.json'), 60)
        observed = bridge.send('session_validate()')
        require_difficulty(observed, difficulty)
        manifest['boot_observation'] = observed
        # Detach only this isolated collector's formal lifecycle listeners.
        # Never begin a formal session or classify these matches as certification.
        bridge.send('astra_load_sub:unsubscribe();astra_save_sub:unsubscribe();astra_reset_sub:unsubscribe();observe()')
        manifest['formal_lifecycle_detached_for_collection'] = True
        for ordinal in range(1, max_attempts+1):
            attempt = {'ordinal': ordinal, 'status': 'running', 'outcome': None, 'readiness': {}, 'matches': [], 'captures': []}
            manifest['attempts'].append(attempt)
            save()
            collect_attempt(run, bridge, manifest, attempt, save)
            save()
            print(f"collection attempt {ordinal}: {attempt['outcome'] or attempt['status']}; openings {len(manifest['openings'])}/{samples}", flush=True)
            if len(manifest['openings']) >= samples:
                break
        manifest['status'] = 'complete' if len(manifest['openings']) >= samples else 'budget_exhausted'
    except BudgetReached as error:
        manifest.update(status='budget_exhausted', error=str(error))
    except BaseException as error:
        manifest.update(status='invalid', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        if log is not None:
            log.close()
        if manifest['attempts'] and manifest['attempts'][-1]['status'] == 'running':
            attempt = manifest['attempts'][-1]
            attempt.update(status='interrupted', reason=manifest.get('error', manifest['status']))
            if attempt['matches'] and attempt['matches'][-1]['status'] == 'running':
                attempt['matches'][-1].update(status='interrupted', reason=attempt['reason'])
        manifest['wall_seconds'] = time.monotonic()-started
        manifest['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save()
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=6)
    parser.add_argument('--max-attempts', type=int, default=12)
    parser.add_argument('--max-matches', type=int, default=132)
    parser.add_argument('--max-seconds', type=int, default=900)
    parser.add_argument('--difficulty', type=int, choices=range(3, 8), default=7)
    args = parser.parse_args()
    manifest = collect(load_config(), **vars(args))
    print(json.dumps({'status': manifest['status'], 'samples': len(manifest['openings']), 'wall_seconds': manifest['wall_seconds']}), flush=True)
    return 0 if manifest['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
