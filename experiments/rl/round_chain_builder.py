"""Build an isolated round-chain candidate from a frozen 15/16-action package."""
import argparse
import hashlib
import json
from pathlib import Path
from astra_play_sf2.config import atomic_json
from astra_play_sf2.runner import sha256
from .actions16_builder import FILES

HERE=Path(__file__).resolve().parent
PACKAGE='astra_sf2_rl_round_chain'


def build(output, source=HERE):
    source=Path(source).resolve()
    names=list(FILES)+[n for n in ('support.py','migrate.py') if (source/n).is_file()]
    captured={name:(source/name).read_bytes() for name in names}
    runtime=(HERE/'round_chain_runtime.lua').read_bytes()
    parent_bytes=(source.parent/'build.json').read_bytes() if 'support.py' in names else None
    parent=json.loads(parent_bytes) if parent_bytes else None
    if parent and (parent.get('schema')!='astra.rl-actions16-build.v1' or any(parent['derived_sha256'].get(n)!=hashlib.sha256(v).hexdigest() for n,v in captured.items())):
        raise ValueError('Parent actions16 package does not match its build manifest')
    derived={name:body.decode('utf-8') for name,body in captured.items()}
    for name in names:
        for old in ('experiments.rl','astra_sf2_rl16'):
            derived[name]=derived[name].replace(old,PACKAGE)
    derived['native_campaign.py']=derived['native_campaign.py'].replace("'experiments/rl/'+name",repr(PACKAGE+'/')+'+name')
    derived['batch_runtime.lua']=runtime.decode('utf-8')
    def patch(name,old,new):
        if derived[name].count(old)!=1:raise ValueError(f'Missing/ambiguous anchor in {name}: {old}')
        derived[name]=derived[name].replace(old,new)
    patch('batch_env.py',"if result.get('training_only') is not True or", "if result.get('round_chain') is not True or result.get('training_only') is not True or")
    patch('batch_env.py',"self.batch_partial = result.get('partial_episode') or None", "self.batch_partial = result.get('partial_episode') or None\n                    self.chain_metrics = result['chain_metrics']")
    patch('batch_train.py',"'training_only': True, 'formal_clear': False,", "'training_only': True, 'formal_clear': False, 'round_chain': True,")
    patch('batch_train.py', "gathered[w].update(observation=chunk['observation'], episode_start=chunk['episode_start'])",
          "gathered[w].update(observation=chunk['observation'], episode_start=chunk['episode_start'], chain_metrics=chunk['chain_metrics'])")
    patch('batch_train.py', "                result['iterations'].append(row)",
          "                row['chain_metrics'] = [chunk['chain_metrics'] for chunk in gathered]\n                result['iterations'].append(row)")
    # The old parity helper would compare two copies of this new sampler. Do not
    # let it masquerade as independent chain/deployment parity evidence.
    patch('batch_train.py',"    output = args.output.resolve();output.mkdir(parents=True, exist_ok=False)",
          "    if is_parity: parser.error('Round-chain requires its independent cross-round harness; old R1 parity is not valid')\n    output = args.output.resolve();output.mkdir(parents=True, exist_ok=False)")
    if parent:
        old="""    if (manifest.get('schema') != 'astra.rl-actions16-build.v1'
            or manifest.get('action_interface') != INTERFACE or manifest.get('actions') != 16):
        raise RuntimeError('Invalid actions16 build identity')"""
        patch('support.py', old, '    validate_chain_build(manifest,package)')
        patch('support.py', 'from pathlib import Path', 'from pathlib import Path\nfrom .chain_identity import validate_chain_build')
        patch('support.py', "    if sha256(snapshot['manifest']) != snapshot['manifest_sha256']:",
              "    validate_chain_build(json.loads(snapshot['manifest'].read_text()),snapshot['package'])\n    if sha256(snapshot['manifest']) != snapshot['manifest_sha256']:")
        patch('support.py', "    output.mkdir(parents=True,exist_ok=True)",
              "    result['training_protocol'] = 'native_match_round_episodes_v1'\n    output.mkdir(parents=True,exist_ok=True)")
        patch('native_campaign.py', "'__init__.py', 'support.py', 'native_campaign.py'",
              "'__init__.py', 'support.py', 'chain_identity.py', 'native_campaign.py'")
        derived['chain_identity.py']=(HERE/'round_chain_identity.py').read_text()
    for name,body in derived.items():
        if name.endswith('.py'):compile(body,PACKAGE+'/'+name,'exec')
    if any((source/name).read_bytes()!=body for name,body in captured.items()):raise RuntimeError('Source changed during build')
    if parent_bytes and (source.parent/'build.json').read_bytes()!=parent_bytes:raise RuntimeError('Parent manifest changed during build')
    if (HERE/'round_chain_runtime.lua').read_bytes()!=runtime:raise RuntimeError('Candidate runtime changed during build')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    package=output/PACKAGE;package.mkdir()
    for name,body in derived.items():(package/name).write_text(body,encoding='utf-8')
    manifest={'schema':'astra.rl-round-chain-build.v1','status':'candidate','native_validated':False,
              'package':PACKAGE,'training_protocol':'native_match_round_episodes_v1',
              'source_sha256':{n:hashlib.sha256(v).hexdigest() for n,v in captured.items()},
              'runtime_sha256':hashlib.sha256(runtime).hexdigest(),'builder_sha256':sha256(Path(__file__)),
              'derived_sha256':{n:sha256(package/n) for n in derived}}
    if parent:
        (output/'parent-build.json').write_bytes(parent_bytes)
        manifest.update(parent_build_sha256=hashlib.sha256(parent_bytes).hexdigest(),
            action_interface=parent['action_interface'],actions=parent['actions'],observations=344,native_decision_frames=12)
    atomic_json(output/'build.json',manifest);return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=HERE)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    print(json.dumps(build(args.output,args.source),indent=2))
if __name__=='__main__':main()
