"""Run an immutable learned policy through fresh, uninterrupted whole matches.

This is an isolated RL evaluation, never frozen-V4 certification. No training,
state loading/saving, continue, policy swap, or within-match pause is permitted.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import subprocess
import time

from astra_play_sf2.config import atomic_json, doctor, load_config
from astra_play_sf2.opening import require_difficulty
from astra_play_sf2.runner import (_attempt, boot_config, mame_command, seal_run,
                                  session_lock, sha256, stage_runtime)
from astra_play_sf2.transport import Bridge, read_json
from .export import export_policy, lua_literal

HERE = Path(__file__).resolve().parent


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Frozen adapter changed; review RL staging replacement: ' + old[:80])
    return text.replace(old, new, 1)


def stage_policy(run, level, payload):
    stage_runtime(run, level)
    target = run / 'training/runtime'
    (target / 'rl_policy.lua').write_text('return ' + lua_literal(payload) + '\n', encoding='utf-8')
    for source, name in [('nn.lua', 'rl_nn.lua'), ('actions.lua', 'rl_actions.lua'),
                         ('continuous_core.lua', 'rl_continuous_core.lua')]:
        (target / name).write_bytes((HERE / source).read_bytes())
    path = target / 'play.lua'
    text = path.read_text(encoding='utf-8')
    text = replace_once(text, "local Core=assert(loadfile('training/runtime/play_core.lua'))()",
                        "local Core=assert(loadfile('training/runtime/rl_continuous_core.lua'))()\n"
                        "local NN=assert(loadfile('training/runtime/rl_nn.lua'))()\n"
                        "local Model=assert(loadfile('training/runtime/rl_policy.lua'))()\n"
                        "local Actions=assert(loadfile('training/runtime/rl_actions.lua'))()\n"
                        "local Neural=NN.new(Model,Actions)\n"
                        "choose=function(a,b,mode,s,reset) return Neural:choose(assert(s),reset) end")
    text = replace_once(text, "schema='mame.continuous-play.v1'",
                        "schema='mame.rl-continuous-play.v1',policy_kind='ppo',model_sha256=Model.model_sha256")
    text = replace_once(text, 'local function traced_policy(a,b,selected)',
                        'local function traced_policy(a,b,selected,s,reset)')
    text = replace_once(text, 'policy(a,b,selected)', 'policy(a,b,selected,s,reset)')
    text = replace_once(text, 'mode=modes[opponent],opponent=opponent,choose=traced_policy,lead=opponent==8 and 2 or 0,timeout_guard=timeout_guards[opponent]',
                        "mode='rl_ppo',opponent=opponent,choose=traced_policy,lead=0,timeout_guard=false")
    text = replace_once(text, "'training/runtime/settings.lua'}) do",
                        "'training/runtime/settings.lua','training/runtime/rl_policy.lua','training/runtime/rl_nn.lua','training/runtime/rl_actions.lua','training/runtime/rl_continuous_core.lua'}) do")
    text = replace_once(text, "policy_file:write(r.sources['training/runtime/fighter.lua'])",
                        "policy_file:write(r.sources['training/runtime/rl_policy.lua'])")
    text = replace_once(text, 'r.release();Speed.apply(m.video,r.speed)',
                        "r.release();Speed.apply(m.video,r.speed)\n"
                        "  local initial=r.core:rl_prime(opening)\n"
                        "  r.recorder.last_input=initial\n"
                        "  for key in initial:gmatch('%S+') do assert(key~='C' and key~='S' and r.keys[key]):set_value(1) end")
    path.write_text('-- EXPERIMENTAL RL ADAPTER; NOT FROZEN V4 POLICY CERTIFICATION.\n' + text, encoding='utf-8')
    return {p.name: sha256(p) for p in sorted(target.iterdir()) if p.is_file()}


def audit_attempt(run, attempt, model_hash):
    """Require mature whole-match outcomes and unchanged native lifecycle."""
    life = read_json(run / attempt['lifecycle'])
    if not life or life.get('active') is not False or life.get('violation') is not False:
        raise RuntimeError('Unclosed/violated native lifecycle')
    if any(life.get(k) != 0 for k in ('loads', 'saves', 'resets')):
        raise RuntimeError('Native reset/load/save detected')
    wins = 0
    for index, relative in enumerate(attempt['matches']):
        match = read_json(run / relative)
        if not match:
            raise RuntimeError('Missing completed match evidence')
        summary = match['summary']
        if (summary.get('schema') != 'mame.rl-continuous-play.v1' or
                summary.get('policy_kind') != 'ppo' or summary.get('model_sha256') != model_hash or
                summary.get('status') != 'complete' or summary.get('valid_continuous') is not True):
            raise RuntimeError('Invalid learned-policy match identity/result')
        if any(summary.get(k) != 0 for k in ('pauses_during_match', 'loads', 'saves')):
            raise RuntimeError('Match was interrupted')
        if summary.get('effective_difficulty') != attempt['difficulty'] or summary.get('effective_difficulty_checks') != summary['frame']:
            raise RuntimeError('Incomplete native difficulty checks')
        if summary.get('trace_frames') != summary['frame'] or summary.get('telemetry_error'):
            raise RuntimeError('Incomplete all-frame evidence')
        if summary.get('timeout_guard') is not False or summary.get('mode') != 'rl_ppo':
            raise RuntimeError('Unexpected scripted fighting assistance')
        score = summary['score']
        rounds = match['rounds']
        if [sum(r['outcome'] == 'win' for r in rounds), sum(r['outcome'] == 'loss' for r in rounds)] != score:
            raise RuntimeError('Native round/match score disagreement')
        if summary['result'] == 'ken_win' and score[0] == 2 and score[1] < 2:
            wins += 1
        elif summary['result'] == 'cpu_win' and score[1] == 2 and score[0] < 2 and index == len(attempt['matches'])-1:
            if attempt['outcome'] != 'loss':
                raise RuntimeError('Loss was not retained')
        else:
            raise RuntimeError('Invalid native match score')
    if attempt['outcome'] == 'gameplay_clear':
        if wins != 11 or len(attempt['matches']) != 11:
            raise RuntimeError('A clear requires eleven native whole-match wins')
        attempt['outcome'] = 'rl_gameplay_clear'
    elif attempt['outcome'] != 'loss':
        raise RuntimeError('Unknown attempt outcome')
    attempt['match_wins'] = wins
    attempt['audit'] = {'ok': True, 'basis': 'native lifecycle, all-frame match evidence, immutable neural policy; no visual certification'}


def evaluate(config, model, output, difficulty=3, attempts=1, speed='fast', show_window=False):
    if difficulty not in range(3, 8) or attempts < 1 or speed not in ('normal', '2x', '4x', 'fast'):
        raise ValueError('Invalid difficulty, attempts, or speed')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = {'schema': 'astra.rl-continuous.v1', 'status': 'initializing',
              'frozen_v4_certification': False, 'training_during_run': False,
              'difficulty': difficulty, 'speed': speed, 'attempts_requested': attempts,
              'stop_on_first_clear': True, 'attempts': [], 'sound': 'none',
              'show_window': show_window,
              'created_utc': datetime.now(timezone.utc).isoformat()}
    process = log = None
    def save():
        atomic_json(output / 'result.json', result)
    try:
        save()
        preflight = doctor(config)
        result['preflight'] = preflight
        if not preflight['ok']:
            raise RuntimeError('Preflight failed: ' + json.dumps(preflight))
        payload = export_policy(model)
        result['model_sha256'] = payload['model_sha256']
        result['runtime_sha256'] = stage_policy(output, difficulty, payload)
        result['boot_config_sha256'] = boot_config(output, difficulty)
        command = mame_command(config) + ['-sound', 'none']
        if not show_window:
            command += ['-video', 'none']
        result['command'] = command
        with session_lock(output):
            log = (output / 'mame.log').open('wb')
            process = subprocess.Popen(command, cwd=output, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            bridge = Bridge(output, process, timeout=240)
            bridge.wait(lambda: read_json(output / 'training/ready.json'), 60)
            observed = bridge.send('session_validate()')
            atomic_json(output / 'boot-observation.json', observed)
            require_difficulty(observed, difficulty)
            result['status'] = 'running'
            save()
            for ordinal in range(1, attempts+1):
                attempt = {'id': f'l{difficulty}-{ordinal:03d}', 'difficulty': difficulty,
                           'outcome': 'invalid', 'matches': [], 'images': {}}
                result['attempts'].append(attempt)
                save()
                _attempt(output, bridge, attempt, speed, save)
                audit_attempt(output, attempt, payload['model_sha256'])
                save()
                if attempt['outcome'] == 'rl_gameplay_clear':
                    break
            result['status'] = 'complete'
    except (Exception, KeyboardInterrupt) as error:
        result.update(status='invalid', error=f'{type(error).__name__}: {error}')
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
        result['wall_seconds'] = time.monotonic()-started
        result['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save()
        seal_run(output)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--difficulty', type=int, default=3, choices=range(3, 8))
    parser.add_argument('--attempts', type=int, default=1)
    parser.add_argument('--speed', choices=('normal', '2x', '4x', 'fast'), default='fast')
    parser.add_argument('--show-window', action='store_true')
    args = parser.parse_args()
    def interrupted(_signal, _frame):
        raise KeyboardInterrupt('SIGTERM')
    signal.signal(signal.SIGTERM, interrupted)
    result = evaluate(load_config(), args.model, args.output, args.difficulty, args.attempts, args.speed, args.show_window)
    print(json.dumps(result, indent=2), flush=True)
    raise SystemExit(2 if result['status'] != 'complete' else
                     0 if any(a['outcome'] == 'rl_gameplay_clear' for a in result['attempts']) else 1)


if __name__ == '__main__':
    main()
