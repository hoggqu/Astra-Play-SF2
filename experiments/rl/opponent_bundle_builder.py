"""Copy a pulsed native verifier into an explicit opponent-bundle candidate."""
import argparse
import hashlib
import json
from pathlib import Path
from astra_play_sf2.config import atomic_json
from astra_play_sf2.runner import sha256
from .pulsed_normals_identity import validate_pulsed_build
from .opponent_bundle_identity import derive
HERE=Path(__file__).resolve().parent

def build(output,source):
    source=Path(source).resolve();root=source.parent
    inputs={p.relative_to(root).as_posix():p.read_bytes() for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.lua','.json')}
    parent=json.loads(inputs['build.json']);validate_pulsed_build(parent,source)
    captured={n:inputs[source.name+'/'+n] for n in parent['derived_sha256']}
    derived=derive(captured,(HERE/'opponent_bundle_nn.lua').read_bytes())
    for name in ('opponent_bundle.py','opponent_bundle_identity.py'):derived[name]=(HERE/name).read_bytes()
    for name,body in derived.items():
        if name.endswith('.py'):compile(body,name,'exec')
    if any((root/n).read_bytes()!=b for n,b in inputs.items()):raise RuntimeError('Parent changed during build')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    frozen={}
    for n,b in inputs.items():
        target='bundle-parent-source/'+n;p=output/target;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b);frozen[target]=hashlib.sha256(b).hexdigest()
        if n.startswith('src/'):
            p=output/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b);frozen[n]=hashlib.sha256(b).hexdigest()
    package=output/source.name;package.mkdir()
    for name,body in derived.items():(package/name).write_bytes(body)
    launch=("from pathlib import Path\nimport runpy,sys\nroot=Path(__file__).resolve().parent\nsys.path[:0]=[str(root/'src'),str(root)]\n"
            f"runpy.run_module('{source.name}.native_continuous',run_name='__main__')\n").encode()
    (output/'launch.py').write_bytes(launch);frozen['launch.py']=hashlib.sha256(launch).hexdigest()
    m={'schema':'astra.rl-opponent-bundle-code.v1','status':'offline_candidate','native_validated':False,'package':source.name,
       'action_interface':'ken_actions16_pulsed_normals_v2','actions':16,'observations':344,'decision_frames':12,
       'policy_kind':'ppo_opponent_bundle','parent_build_sha256':hashlib.sha256(inputs['build.json']).hexdigest(),
       'derived_sha256':{n:hashlib.sha256(b).hexdigest() for n,b in derived.items()},'frozen_inputs_sha256':frozen,
       'builder_sha256':sha256(Path(__file__)),'training_entry_supported':False,'shared_policy_identity':False,
       'scope':'Fixed eleven-branch bundle only; original native/lifecycle/score audits retained, all branches loaded before boot.'}
    atomic_json(output/'build.json',m);return m

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    print(json.dumps(build(**vars(p.parse_args())),indent=2))
if __name__=='__main__':main()
