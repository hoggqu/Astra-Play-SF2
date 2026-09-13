"""Build a fresh full85/visible-feedback package without modifying its parent."""
import argparse
import json
from pathlib import Path
import shutil
from .full_interface_identity import (INPUT_NAMES, INTERFACE, OBSERVATION_INTERFACE,
                                      SCHEMA, derive, digest, parent_validate, validate_build)

HERE = Path(__file__).resolve().parent


def build(source, output):
    source = Path(source).resolve(); output = Path(output).resolve()
    parent, captured = parent_validate(source)
    inputs = {n: (HERE/n).read_bytes() for n in INPUT_NAMES}
    derived = derive(captured, inputs)
    for name, body in derived.items():
        if name.endswith('.py'): compile(body, name, 'exec')
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(source, output/'parent-source', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copytree(source/'src', output/'src', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    package = output/'astra_sf2_rl_full85'; package.mkdir()
    for name, body in derived.items(): (package/name).write_bytes(body)
    (output/'integration-inputs').mkdir()
    for name, body in inputs.items(): (output/'integration-inputs'/name).write_bytes(body)
    shutil.copy2(__file__, output/'integration-inputs/full_interface_builder.py')
    # Main guard is essential for multiprocessing spawn on all three platforms.
    launcher = """from pathlib import Path
import runpy
import sys
if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    sys.path[:0] = [str(root/'src'), str(root)]
    module = sys.argv.pop(1) if len(sys.argv) > 1 else ''
    if module not in ('initialize', 'batch_train', 'export', 'native_continuous'):
        raise SystemExit('Use initialize, batch_train, export or native_continuous')
    runpy.run_module('astra_sf2_rl_full85.'+module, run_name='__main__')
"""
    (output/'launch.py').write_text(launcher)
    frozen = {p.relative_to(output).as_posix(): digest(p.read_bytes())
              for folder in ('parent-source', 'src', 'integration-inputs')
              for p in (output/folder).rglob('*') if p.is_file()}
    frozen['launch.py'] = digest(launcher.encode())
    manifest = {'schema':SCHEMA, 'status':'candidate', 'native_validated':False,
                'package':package.name, 'action_interface':INTERFACE,
                'observation_interface':OBSERVATION_INTERFACE, 'actions':85,
                'observations':800, 'frame_features':200, 'history':4, 'decision_frames':12,
                'training_protocol':parent.get('training_protocol', 'native_match_round_episodes_v1'),
                'parent_build_sha256':digest((source/'build.json').read_bytes()),
                'derived_sha256':{n:digest(b) for n,b in derived.items()},
                'frozen_files_sha256':frozen,
                'scope':'Full85 action identity and visible feedback; unchanged native settlement, reward, GAE and 12-frame action timing. Fresh model only.'}
    (output/'build.json').write_text(json.dumps(manifest, indent=2)+'\n')
    validate_build(manifest, package)
    # Recheck the original parent after copying and compilation.
    _, final = parent_validate(source)
    if final != captured: raise RuntimeError('Parent changed during build')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Frozen specialist-pulse root containing build.json')
    parser.add_argument('--output', type=Path, required=True)
    result = build(**vars(parser.parse_args()))
    print(json.dumps({k:result[k] for k in ('schema','package','actions','observations')}, indent=2))


if __name__ == '__main__': main()
