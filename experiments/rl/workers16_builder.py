"""Isolated pulsed-input derivative permitting up to 16 PPO workers; no runtime launch."""
import argparse
import hashlib
import json
from pathlib import Path
from astra_play_sf2.config import atomic_json
from astra_play_sf2.runner import sha256
from .pulsed_normals_identity import validate_pulsed_build, INTERFACE

HERE=Path(__file__).resolve().parent

def patch(text, old, new, count=1):
    if text.count(old)!=count: raise RuntimeError('Ambiguous workers16 anchor: '+old)
    return text.replace(old,new)

def upper_bound(text, native=False):
    text=patch(text,'range(1, 9)','range(1, 17)',2 if native else 1)
    return patch(text,'workers must be 1..8','workers must be 1..16') if native else text

def identity_source(body):
    text=body.decode()
    text=patch(text,'def derive_sources(captured):','def derive_sources(captured, worker_limit=16):')
    text=patch(text,"    return {name: text.encode('utf-8') for name, text in derived.items()}",
        "    if worker_limit == 16:\n"
        "        patch('batch_train.py', 'choices=range(1, 9)', 'choices=range(1, 17)')\n"
        "        for old, new in [('workers not in range(1, 9)', 'workers not in range(1, 17)'),\n"
        "                         ('choices=range(1, 9)', 'choices=range(1, 17)'),\n"
        "                         ('workers must be 1..8', 'workers must be 1..16')]:\n"
        "            patch('native_campaign.py', old, new)\n"
        "    elif worker_limit != 8: raise RuntimeError('Unsupported worker limit')\n"
        "    return {name: text.encode('utf-8') for name, text in derived.items()}")
    text=patch(text,"    derived = derive_sources(captured)\n    if set(manifest['derived_sha256'])", "    validate_worker_revision(manifest, root, captured)\n    derived = derive_sources(captured)\n    if set(manifest['derived_sha256'])")
    text=patch(text,"    return paths\n", "    paths['workers/parent-build.json'] = root/'workers-parent-build.json'\n    paths['workers/parent-identity.py'] = root/'workers-parent-identity.py'\n    return paths\n")
    text=patch(text,"    return {'identity':IDENTITY,'action_interface':INTERFACE", "    return {'worker_limit_revision':m['worker_limit_revision'], 'identity':IDENTITY,'action_interface':INTERFACE")
    text+='''\n\ndef validate_worker_revision(manifest, root, captured):
    revision=manifest.get('worker_limit_revision',{})
    if revision.get('maximum_workers') != 16 or revision.get('per_worker_rollout_steps') != 256:
        raise RuntimeError('Invalid worker-limit revision')
    old_path=root/'workers-parent-build.json'
    if sha256(old_path) != revision.get('parent_pulse_build_sha256'):
        raise RuntimeError('Worker parent pulse build changed')
    old=json.loads(old_path.read_text())
    expected={n:hashlib.sha256(b).hexdigest() for n,b in derive_sources(captured,worker_limit=8).items()}
    expected['pulsed_normals_identity.py']=sha256(root/'workers-parent-identity.py')
    if old.get('derived_sha256') != expected:
        raise RuntimeError('Worker parent is not the exact original pulsed source')
    for key in ('schema','package','action_interface','actions','observations','pulse_identity',
                'pulse_source_sha256','pulse_parent_build_sha256','parent_manifest_sha256',
                'frozen_production_sha256','launcher_sha256','runtime_sha256'):
        if manifest.get(key) != old.get(key):
            raise RuntimeError('Worker derivation changed unrelated identity: '+key)
'''
    return text.encode()

def build(output, source):
    source=Path(source).resolve();root=source.parent
    before={p.relative_to(root).as_posix():p.read_bytes() for p in root.rglob('*')
            if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.lua','.json')}
    manifest=json.loads(before['build.json']);validate_pulsed_build(manifest,source)
    if manifest.get('worker_limit_revision'): raise ValueError('Expected original pulsed parent')
    derived={name:before[source.name+'/'+name] for name in manifest['derived_sha256']}
    derived['batch_train.py']=upper_bound(derived['batch_train.py'].decode()).encode()
    derived['native_campaign.py']=upper_bound(derived['native_campaign.py'].decode(),native=True).encode()
    derived['pulsed_normals_identity.py']=identity_source(derived['pulsed_normals_identity.py'])
    for name,body in derived.items():
        if name.endswith('.py'): compile(body,name,'exec')
    if any((root/name).read_bytes()!=body for name,body in before.items()): raise RuntimeError('Source changed while building')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    for name,body in before.items():
        p=output/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(body)
    for name,body in derived.items(): (output/source.name/name).write_bytes(body)
    (output/'workers-parent-build.json').write_bytes(before['build.json'])
    (output/'workers-parent-identity.py').write_bytes(before[source.name+'/pulsed_normals_identity.py'])
    digest=lambda b:hashlib.sha256(b).hexdigest()
    manifest.update(derived_sha256={n:digest(b) for n,b in derived.items()},identity_sha256=digest(derived['pulsed_normals_identity.py']),
        worker_limit_revision={'maximum_workers':16,'per_worker_rollout_steps':256,
            'global_rollout_at_16_workers':4096,'parent_pulse_build_sha256':digest(before['build.json']),
            'builder_sha256':sha256(Path(__file__)),'native_validated':False,
            'scope':'Only worker CLI limits and their exact derivation validator; optimizer and PPO hyperparameters unchanged.'})
    atomic_json(output/'build.json',manifest)
    return manifest


def build_driver(output):
    """Copy the reviewed scheduler, with pulsed identity and worker limit only."""
    names=('__init__.py','campaign.py','native_campaign.py','dataset.py','reliability_campaign.py','reliability_pipeline.py')
    captured={n:(HERE/n).read_bytes() for n in names};derived=dict(captured)
    for name in ('reliability_campaign.py','reliability_pipeline.py'):
        text=derived[name].decode()
        text=patch(text,'range(1,9)' if name=='reliability_campaign.py' else 'range(1, 9)',
                        'range(1,17)' if name=='reliability_campaign.py' else 'range(1, 17)',2)
        text=patch(text,'workers must be 1..8','workers must be 1..16')
        if name=='reliability_campaign.py':
            text=patch(text,"INTERFACE = 'ken_actions16_lp_mp_uppercut_v1'",f'INTERFACE = {INTERFACE!r}')
            text=patch(text,'    return paths\n',
                "    paths.update({'provenance/'+p.relative_to(code).as_posix():p for p in code.rglob('*')\n"
                "                  if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.lua','.json')})\n"
                '    return paths\n')
        derived[name]=text.encode();compile(text,name,'exec')
    production=HERE.parents[1]/'src'
    prod={p.relative_to(production).as_posix():p.read_bytes() for p in production.rglob('*') if p.is_file() and p.suffix in ('.py','.lua','.json')}
    parent_init=(HERE.parent/'__init__.py').read_bytes()
    if any((HERE/n).read_bytes()!=b for n,b in captured.items()):raise RuntimeError('Scheduler source changed')
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    files={'experiments/__init__.py':parent_init,**{'experiments/rl/'+n:b for n,b in derived.items()},**{'src/'+n:b for n,b in prod.items()}}
    for name,body in files.items():
        p=out/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(body)
    digest=lambda b:hashlib.sha256(b).hexdigest()
    m={'schema':'astra.rl-workers16-driver.v1','worker_limit':16,'action_interface':INTERFACE,
       'source_sha256':{n:digest(b) for n,b in captured.items()},'derived_sha256':{n:digest(b) for n,b in files.items()},
       'builder_sha256':sha256(Path(__file__)),'native_validated':False,
       'scope':'Same full20 and process-group audits; pulse interface, 16-worker limits, recursive provenance pins.'}
    atomic_json(out/'source-manifest.json',m);return m

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--driver-output',type=Path)
    a=p.parse_args();result={'training':build(a.output,a.source)}
    if a.driver_output:result['driver']=build_driver(a.driver_output)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
