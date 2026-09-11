"""Build an isolated, auditable last-frame-release normal-button candidate."""
import argparse
import hashlib
import json
from pathlib import Path
from astra_play_sf2.config import atomic_json
from astra_play_sf2.runner import sha256
from .pulsed_normals_identity import derive_sources, validate_parent, INTERFACE, IDENTITY

HERE=Path(__file__).resolve().parent


def build(output, source):
    source=Path(source).resolve();source_root=source.parent
    parent_bytes=(source_root/'build.json').read_bytes();parent=json.loads(parent_bytes)
    if parent['package']!=source.name or not source.name.isidentifier():
        raise ValueError('Expected the declared parent package directory')
    captured=validate_parent(parent,source)
    ancestors={p.name:p.read_bytes() for p in source_root.glob('*.json')}
    production={p.relative_to(source_root/'src').as_posix():p.read_bytes()
        for p in (source_root/'src').rglob('*') if p.is_file() and p.suffix in ('.py','.lua','.json')}
    if 'astra_play_sf2/__init__.py' not in production:
        raise ValueError('Parent must include frozen production src')
    derived=derive_sources(captured)
    derived['pulsed_normals_identity.py']=(HERE/'pulsed_normals_identity.py').read_bytes().replace(
        b'from .round_chain_identity import validate_chain_build',b'from .chain_identity import validate_chain_build')
    launcher=("from pathlib import Path\nimport runpy, sys\n"
        "root=Path(__file__).resolve().parent\nsys.path[:0]=[str(root/'src'),str(root)]\n"
        "module=sys.argv.pop(1) if len(sys.argv)>1 else ''\n"
        "if module not in ('batch_train','native_campaign','native_continuous'): raise SystemExit('Use batch_train, native_campaign or native_continuous')\n"
        f"runpy.run_module('{source.name}.'+module,run_name='__main__')\n").encode()
    for name,body in derived.items():
        if name.endswith('.py'):compile(body,name,'exec')
    # Reject races before publishing a fresh snapshot.
    if any((source/name).read_bytes()!=body for name,body in captured.items()) or (source_root/'build.json').read_bytes()!=parent_bytes:
        raise RuntimeError('Parent changed while building')
    if any((source_root/name).read_bytes()!=body for name,body in ancestors.items()):
        raise RuntimeError('Parent ancestor changed while building')
    if any((source_root/'src'/name).read_bytes()!=body for name,body in production.items()):
        raise RuntimeError('Frozen production changed while building')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    package=output/source.name;package.mkdir()
    for name,body in derived.items():(package/name).write_bytes(body)
    for name,body in captured.items():
        p=output/'parent-source'/source.name/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(body)
    for name,body in ancestors.items():(output/'parent-source'/name).write_bytes(body)
    for name,body in production.items():
        p=output/'src'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(body)
    (output/'launch.py').write_bytes(launcher)
    digest=lambda b:hashlib.sha256(b).hexdigest()
    manifest={'schema':'astra.rl-pulsed-normals-build.v1','status':'candidate','native_validated':False,
        'package':source.name,'pulse_identity':IDENTITY,'action_interface':INTERFACE,'actions':16,
        'observations':344,'native_decision_frames':12,'training_protocol':parent['training_protocol'],
        'pulse_parent_build_sha256':digest(parent_bytes),'pulse_source_sha256':parent['derived_sha256'],
        'parent_manifest_sha256':{n:digest(b) for n,b in ancestors.items()},
        'derived_sha256':{n:digest(b) for n,b in derived.items()},
        'frozen_production_sha256':{n:digest(b) for n,b in production.items()},
        'builder_sha256':sha256(Path(__file__)),'identity_sha256':digest(derived['pulsed_normals_identity.py']),
        'launcher_sha256':digest(launcher),'runtime_sha256':digest(derived['batch_runtime.lua']),
        'scope':'Only action6..11 frame11 release attack buttons; actions8/9 retain D. Explicit interface labels and provenance checks updated; NN/reward/sampling unchanged.'}
    atomic_json(output/'build.json',manifest)
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    print(json.dumps(build(**vars(p.parse_args())),indent=2))
if __name__=='__main__':main()
