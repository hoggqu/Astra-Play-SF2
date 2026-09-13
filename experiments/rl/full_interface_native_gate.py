"""Compare full85 batch sampling with independent native-time forced execution."""
import argparse
import importlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from astra_play_sf2.config import atomic_json, load_config
from astra_play_sf2.runner import sha256
from .projectile_probe import select_train
from .pulsed_native_gate import patch_snapshot

HERE = Path(__file__).resolve().parent


def reference_runtime():
    text = patch_snapshot((HERE/'chain_reference_runtime.lua').read_text())
    def patch(old, new):
        nonlocal text
        if text.count(old) != 1:
            raise RuntimeError('Reference anchor changed: '+old)
        text = text.replace(old, new)
    patch('local function choose(a,b,mode,s,reset_history)',
          'local history\nlocal function choose(a,b,mode,s,reset_history)')
    patch(' index=index+1', ''' index=index+1
 local f=NN.features(s)
 if reset_history then history={f,f,f,f} else table.remove(history,1);history[#history+1]=f end
 local observation={};for _,row in ipairs(history) do for _,v in ipairs(row) do observation[#observation+1]=v end end''')
    patch('action=planned.action,state=s,reset_history=reset_history}',
          'action=planned.action,state=s,reset_history=reset_history,observation=observation}')
    # The same native Core records the new request AFTER capturing observations.
    patch(' return seq\nend', ' return seq,nil,planned.action\nend')
    patch("  if core.phase=='complete' then", '  if frames==pending.frames then')
    patch('opening=opening,initial_loads=1,in_play_pauses=0,chain_metrics={}',
          'opening=opening,initial_loads=1,in_play_pauses=0,chain_metrics={},partial=true')
    return text


def run(source, dataset, output):
    source, dataset, output = (Path(p).resolve() for p in (source, dataset, output))
    from .full_interface_identity import validate_build
    manifest = json.loads((source.parent/'build.json').read_text())
    validate_build(manifest, source)
    sample = select_train(dataset, (3,), 1)[0]
    output.mkdir(parents=True, exist_ok=False)
    result = dict(schema='astra.rl-full85-native-gate.v1', status='running', training_only=True,
                  formal_clear=False, source_build_sha256=sha256(source.parent/'build.json'),
                  checkpoint_sha256=sample['sha256'], cases=[])
    save = lambda: atomic_json(output/'result.json', result)
    started = time.monotonic()
    env = None
    path_entry = None
    def unload():
        for name in list(sys.modules):
            if name == source.name or name.startswith(source.name+'.'):
                del sys.modules[name]
    try:
        save()
        for section, ids in enumerate((list(range(43)), list(range(43, 85)))):
            plan = [0]*10 + ids + [0]*3
            traces, observations = {}, {}
            for role in ('batch', 'manual_reset', 'native'):
                name = f'{section:02d}-{role}'
                root = output/'sources'/name
                shutil.copytree(source.parent, root, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
                package = root/source.name
                runtime = reference_runtime() if role == 'native' else patch_snapshot((source/'batch_runtime.lua').read_text())
                (package/'batch_runtime.lua').write_text(runtime)
                unload()
                path_entry = str(root)
                sys.path.insert(0, path_entry)
                batch = importlib.import_module(source.name+'.batch_env')
                env = batch.BatchEnv(load_config(), output/name, 3, checkpoints=[sample])
                options = dict(checkpoint=0, lead=0)
                if role == 'native':
                    actions = [dict(round=1, frame=i*12, action=a) for i, a in enumerate(plan+[0])]
                    response = env.batch_rpc(dict(op='reference_match', reset=options, actions=actions, frames=len(plan)*12))
                    atomic_json(output/(name+'.json'), response)
                    if response['initial_loads'] != 1 or response['in_play_pauses'] != 0:
                        raise RuntimeError('Native reference lifecycle mismatch')
                    traces[role] = response['trace']
                    observations[role] = [d['observation'] for d in response['decisions'][:-1]]
                else:
                    traces[role], observations[role] = [], []
                    if role == 'manual_reset':
                        obs, _ = env.reset(options=options)
                        if obs.shape != (800,): raise RuntimeError('Reset observation dimension mismatch')
                    offset = 0
                    for chunk in (23, len(plan)-23):
                        response = env.batch_rpc(dict(op='reset_rollout' if role == 'batch' and offset == 0 else 'rollout',
                            reset=options, actions=plan[offset:offset+chunk], count=chunk,
                            resets=[options]*chunk, capture_trace=1))
                        atomic_json(output/f'{name}-{offset:03d}.json', response)
                        traces[role].extend(response['trace'])
                        observations[role].extend(t['observation'] for t in response['transitions'])
                        offset += chunk
                env.close();env = None
                sys.path.remove(path_entry);path_entry = None
            reference = traces['native']
            if len(reference) != len(plan)*12:
                raise RuntimeError('Native frame count mismatch')
            for role in ('batch', 'manual_reset'):
                if len(traces[role]) != len(reference): raise RuntimeError('Batch frame count mismatch')
                for actual, expected in zip(traces[role], reference):
                    if any(actual[k] != expected[k] for k in ('frame','state','phase','round','events')):
                        atomic_json(output/'mismatch.json', dict(role=role, actual=actual, expected=expected))
                        raise RuntimeError('Native state/feedback mismatch')
                actual = np.asarray(observations[role], dtype=np.float32)
                expected = np.asarray(observations['native'], dtype=np.float32)
                if actual.shape != (len(plan), 800): raise RuntimeError('Observation dimension mismatch')
                np.testing.assert_array_equal(actual, expected)
            result['cases'].append(dict(section=section, actions=ids, frames=len(reference),
                decisions=len(plan), status='pass', observations=800,
                comparisons=['batch/native','manual_reset/native'], feedback_and_ports=True))
            save()
        result.update(status='complete', actions_covered=85,
                      unique_reference_frames=sum(c['frames'] for c in result['cases']))
        validate_build(json.loads((source.parent/'build.json').read_text()), source)
    except BaseException as error:
        result.update(status='invalid', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        if env is not None: env.close()
        if path_entry in sys.path: sys.path.remove(path_entry)
        unload()
        result['wall_seconds'] = time.monotonic()-started
        save()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source','dataset','output'):
        p.add_argument('--'+name, type=Path, required=True)
    result = run(**vars(p.parse_args()))
    print(json.dumps({k:v for k,v in result.items() if k != 'cases'}, indent=2))


if __name__ == '__main__': main()
