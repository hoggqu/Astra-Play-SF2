"""Bounded native input coverage for the full Ken interface; no PPO or win-rate claim."""
import argparse
import importlib
import json
from pathlib import Path
import shutil
import sys
import time

from astra_play_sf2.config import atomic_json, load_config
from astra_play_sf2.runner import sha256
from .projectile_probe import select_train

HERE = Path(__file__).resolve().parent


def probe_runtime(original):
    def replace(old, new):
        nonlocal original
        if original.count(old) != 1:
            raise ValueError('Missing/ambiguous native probe anchor: ' + old)
        original = original.replace(old, new)
    replace(' return s\nend', " s.native_ports={m.ioport.ports[':IN1']:read(),m.ioport.ports[':IN2']:read()}\n return s\nend")
    replace("  if core.phase=='complete' then", '  if frames==pending.frames then')
    replace('opening=opening,initial_loads=1,in_play_pauses=0,chain_metrics={}',
            'opening=opening,initial_loads=1,in_play_pauses=0,chain_metrics={},partial=true')
    return original


def run(source, dataset, output, throws=False):
    """One fresh MAME; one verified train-only restore per bounded input case."""
    from lupa.lua54 import LuaRuntime
    source, dataset, output = (Path(p).resolve() for p in (source, dataset, output))
    sample = select_train(dataset, (3,), 1)[0]
    lua = LuaRuntime()
    actions_bytes = (HERE / 'full_actions.lua').read_bytes()
    actions = lua.execute(actions_bytes.decode())
    count = int(actions['count'])
    from .full_actions import raw_id, keys as expected_keys
    cases = ([(raw_id(direction, button), approach)
              for approach in (0, 12, 24, 36, 48, 60)
              for direction in ('F', 'B') for button in ('MP', 'HP', 'MK', 'HK')]
             if throws else [(action, 0) for action in range(count)])
    output.mkdir(parents=True, exist_ok=False)
    code = output / 'code'
    shutil.copytree(source.parent, code, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    package = code / source.name
    (package / 'actions.lua').write_bytes(actions_bytes)
    (package / 'batch_runtime.lua').write_text(probe_runtime((HERE / 'chain_reference_runtime.lua').read_text()))
    sys.path[:0] = [str(code / 'src'), str(code)]
    batch = importlib.import_module(source.name + '.batch_env')
    record = dict(schema='astra.rl-full-actions-native-probe.v1', status='running',
                  training_only=True, formal_clear=False, included_in_win_rates=False,
                  actions=count, decision_frames=12, max_frame_budget_per_case=240 if throws else 180,
                  checkpoint_sha256=sample['sha256'], holdout_opened=False,
                  action_interface=actions['interface'], actions_sha256=sha256(package / 'actions.lua'),
                  scope='throw_input_contexts' if throws else 'all_inputs', cases=[])
    started = time.monotonic()
    env = None
    save = lambda: atomic_json(output / 'result.json', record)
    try:
        save()
        env = batch.BatchEnv(load_config(), output / 'native', 3, checkpoints=[sample])
        record['runtime_sha256'] = env.manifest['runtime_sha256']
        for case_index, (action, approach) in enumerate(cases):
            # Idle until active play, issue one macro, then observe its continuation.
            target_index = 10 + approach // 12
            frame_budget = 180 + approach
            plan = [dict(frame=i * 12, round=1,
                         action=action if i == target_index else (1 if 10 <= i < target_index else 0))
                    for i in range(frame_budget // 12 + 1)]
            response = env.batch_rpc(dict(op='reference_match', reset=dict(checkpoint=0, lead=0),
                                         actions=plan, frames=frame_budget))
            path = output / (f'context-{case_index:03d}.json' if throws else f'action-{action:03d}.json')
            atomic_json(path, response)
            if response['initial_loads'] != 1 or response['in_play_pauses'] != 0:
                raise RuntimeError('Invalid native probe lifecycle')
            rows = response['trace']
            if len(rows) != frame_budget or len(response['decisions']) != len(plan):
                raise RuntimeError('Incomplete native trace')
            before = response['opening']
            period = before['native_frame_period']
            for frame, row in enumerate(rows, 1):
                if row['frame'] != frame or abs(row['state']['emulated_seconds'] - before['emulated_seconds'] - period) > 1e-7:
                    raise RuntimeError('Native frame gap')
                before = row['state']
            decision = response['decisions'][target_index]
            if decision['action'] != action or decision['state']['timer'] >= 99:
                raise RuntimeError('Action not issued during active play')
            state = lua.table_from(decision['state'], recursive=True)
            forward = 'R' if state['p1']['x'] < state['p2']['x'] else 'L'
            transitions = []
            previous = decision['state']['p1']
            for frame, row in enumerate(rows[120 + approach:132 + approach]):
                expected = expected_keys(action, frame, forward=forward)
                if actions['keys'](action, frame, state, forward) != expected:
                    raise RuntimeError('Lua/Python input waveform disagreement')
                if row['consumed_input'] != expected:
                    raise RuntimeError(f'Native input mismatch action={action} frame={frame}')
                pressed = set(expected.split())
                p1_mask = sum(mask for key, mask in dict(R=1, L=2, D=4, U=8, LP=16, MP=32, HP=64).items() if key in pressed)
                p2_mask = sum(mask for key, mask in dict(LK=1, MK=2, HK=4).items() if key in pressed)
                ports = row['state']['native_ports']
                if ports[0] & 127 != 127 ^ p1_mask or ports[1] & 7 != 7 ^ p2_mask:
                    raise RuntimeError(f'Native input port mismatch action={action} frame={frame}')
            for row in rows[120 + approach:]:
                player = row['state']['p1']
                if (player['a'], player['anim']) != (previous['a'], previous['anim']):
                    transitions.append(dict(frame=row['frame'], action=player['a'], animation=player['anim']))
                previous = player
            record['cases'].append(dict(action=action, status='complete', native_input_frames_checked=12,
                                        approach_frames=approach, frames=frame_budget,
                                        telemetry_file=path.name,
                                        telemetry_sha256=sha256(path), action_animation_transitions=transitions))
            save()
        record.update(status='complete', native_frames=sum(r['frames'] for r in record['cases']),
                      limitation='Input coverage only; does not certify every contextual move, hit, or throw succeeds.')
    except BaseException as error:
        record.update(status='invalid', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        if env is not None:
            env.close()
        record['wall_seconds'] = time.monotonic() - started
        save()
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'dataset', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--throws', action='store_true', help='48 fixed approach/direction/button cases; no adaptive input')
    result = run(**vars(parser.parse_args()))
    print(json.dumps({key: result[key] for key in ('status', 'actions', 'native_frames', 'wall_seconds')}))


if __name__ == '__main__':
    main()
