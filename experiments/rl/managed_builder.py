"""Build portable time-bounded training from public perception128 timing and draw-settlement sources."""
import argparse
import json
from pathlib import Path
import shutil
import tempfile
from .managed_identity import INPUT_NAMES, PACKAGE, SCHEMA, STOP_PROTOCOL, SAMPLING_PROTOCOL, digest, derive, validate_parent, validate_build
HERE = Path(__file__).resolve().parent


def build(source, output):
    source = Path(source).resolve(); output = Path(output).resolve()
    parent, captured = validate_parent(source)
    inputs = {n:(HERE/n).read_bytes() for n in INPUT_NAMES}
    derived = derive(captured, inputs)
    for name, body in derived.items():
        if name.endswith('.py'): compile(body, name, 'exec')
    output.mkdir(parents=True, exist_ok=False)
    ignore = shutil.ignore_patterns('__pycache__','*.pyc')
    shutil.copytree(source, output/'managed-parent', ignore=ignore)
    shutil.copytree(source/'src', output/'src', ignore=ignore)
    package = output/PACKAGE; package.mkdir()
    for name, body in derived.items(): (package/name).write_bytes(body)
    (output/'managed-inputs').mkdir()
    for name, body in inputs.items(): (output/'managed-inputs'/name).write_bytes(body)
    launcher = (source/'launch.py').read_text().replace(parent['package'], PACKAGE)
    (output/'launch.py').write_text(launcher)
    frozen = {p.relative_to(output).as_posix():digest(p.read_bytes())
              for name in ('managed-parent','managed-inputs','src')
              for p in (output/name).rglob('*') if p.is_file()}
    frozen['launch.py'] = digest(launcher.encode())
    manifest = dict(parent)
    manifest.update(schema=SCHEMA, package=PACKAGE, cooperative_stop_protocol=STOP_PROTOCOL,
                    opponent_sampling_protocol=SAMPLING_PROTOCOL, rollout_protocol='exact_global_rollout_per_worker_gae_v1',
                    devices_supported=['cpu','cuda','mps','auto'], status='candidate', native_validated=False,
                    parent_build_sha256=digest((source/'build.json').read_bytes()),
                    derived_sha256={n:digest(b) for n,b in derived.items()}, frozen_files_sha256=frozen,
                    scope='Adaptive training-opponent sampling, custom workers, explicit Torch device and cooperative update-boundary stop; identical Lua, observation, reward and combat execution.')
    (output/'build.json').write_text(json.dumps(manifest,indent=2)+'\n')
    validate_build(manifest, package)
    if validate_parent(source)[1] != captured: raise RuntimeError('Parent changed during managed build')
    return manifest


def bootstrap(output):
    from .perception128_draw_builder import bootstrap as make_parent
    with tempfile.TemporaryDirectory(prefix='astra-managed-') as folder:
        parent = Path(folder)/'parent'; make_parent(parent)
        return build(parent, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path); parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    result = build(args.source,args.output) if args.source else bootstrap(args.output)
    print(json.dumps({k:result[k] for k in ('schema','package','cooperative_stop_protocol','devices_supported')}))


if __name__ == '__main__': main()
