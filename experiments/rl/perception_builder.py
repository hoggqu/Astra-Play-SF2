"""Build a separate screen-perception PPO package from a frozen full85 v1."""
import argparse
import json
from pathlib import Path
import shutil
import tempfile

from .perception_identity import (INPUT_NAMES, PACKAGE, OBSERVATION_INTERFACE,
                                 FRAME_FEATURES, OBSERVATIONS, SCHEMA, digest,
                                 derive, validate_parent, validate_build)

HERE = Path(__file__).resolve().parent


def build(source, output):
    source = Path(source).resolve(); output = Path(output).resolve()
    parent, captured = validate_parent(source)
    inputs = {name:(HERE/name).read_bytes() for name in INPUT_NAMES}
    derived = derive(captured, inputs)
    for name, body in derived.items():
        if name.endswith('.py'): compile(body, name, 'exec')
    output.mkdir(parents=True, exist_ok=False)
    ignore = shutil.ignore_patterns('__pycache__', '*.pyc')
    shutil.copytree(source, output/'perception-parent', ignore=ignore)
    shutil.copytree(source/'src', output/'src', ignore=ignore)
    package = output/PACKAGE; package.mkdir()
    for name, body in derived.items(): (package/name).write_bytes(body)
    folder = output/'perception-inputs'; folder.mkdir()
    for name, body in inputs.items(): (folder/name).write_bytes(body)
    launcher = (source/'launch.py').read_text().replace(parent['package'], PACKAGE)
    (output/'launch.py').write_text(launcher)
    frozen = {p.relative_to(output).as_posix():digest(p.read_bytes())
              for name in ('perception-parent','perception-inputs','src')
              for p in (output/name).rglob('*') if p.is_file()}
    frozen['launch.py'] = digest(launcher.encode())
    manifest = {'schema':SCHEMA, 'package':PACKAGE, 'status':'candidate',
                'native_validated':False, 'action_interface':parent['action_interface'],
                'observation_interface':OBSERVATION_INTERFACE, 'observations':OBSERVATIONS,
                'frame_features':FRAME_FEATURES, 'history':4, 'actions':85,
                'decision_frames':12, 'training_protocol':parent['training_protocol'],
                'parent_build_sha256':digest((source/'build.json').read_bytes()),
                'derived_sha256':{n:digest(b) for n,b in derived.items()},
                'frozen_files_sha256':frozen,
                'scope':'Structured current screen perception; unchanged85 actions/reward/native settlement. New observation identity requires fresh model.'}
    (output/'build.json').write_text(json.dumps(manifest, indent=2)+'\n')
    validate_build(manifest, package)
    _, final = validate_parent(source)
    if final != captured: raise RuntimeError('Full85 parent changed during build')
    return manifest


def bootstrap(output):
    """No private source, ROM, model or training state is needed to build."""
    from .full_interface_bootstrap import build as build_parent
    with tempfile.TemporaryDirectory(prefix='astra-perception-') as folder:
        parent = Path(folder)/'full85'
        build_parent(parent)
        return build(parent, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, help='Optional existing full85 v1 source root')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = build(args.source,args.output) if args.source else bootstrap(args.output)
    print(json.dumps({k:result[k] for k in ('schema','package','actions','observations')}, indent=2))


if __name__ == '__main__': main()
