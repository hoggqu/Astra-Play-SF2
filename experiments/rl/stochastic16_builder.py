"""Derive a verification-only categorical evaluator from an immutable actions16 snapshot."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from astra_play_sf2.config import atomic_json
from astra_play_sf2.runner import sha256

HERE=Path(__file__).resolve().parent
PACKAGE='astra_sf2_rl_stochastic16'


def build(source,output,policy_seed):
    if type(policy_seed)is not int or not 1<=policy_seed<2147483647:raise ValueError('policy-seed must be 1..2147483646')
    source=Path(source).resolve();parent_path=source.parent/'build.json'
    parent_bytes=parent_path.read_bytes();parent=json.loads(parent_bytes)
    if (parent.get('schema')!='astra.rl-actions16-build.v1' or parent.get('actions')!=16
            or parent.get('observations')!=344 or parent.get('native_decision_frames')!=12):
        raise ValueError('Only original actions16 snapshots are supported')
    captured={name:(source/name).read_bytes() for name in parent['derived_sha256']}
    if any(hashlib.sha256(body).hexdigest()!=parent['derived_sha256'][name] for name,body in captured.items()):
        raise ValueError('Parent source differs from build manifest')
    extras={name:(HERE/name).read_bytes() for name in ('stochastic_continuous.py','stochastic_nn.lua','stochastic16_identity.py')}
    derived={n:b.decode('utf-8').replace('astra_sf2_rl16',PACKAGE) for n,b in captured.items()}
    patches=[]
    def patch(name,old,new,count=1):
        if derived[name].count(old)!=count:raise ValueError('Missing/ambiguous anchor '+name+': '+old[:70])
        derived[name]=derived[name].replace(old,new);patches.append(dict(file=name,old=old,new=new,count=count))
    patch('export.py',"'selection': 'deterministic_argmax'",f"'selection': 'categorical_softmax', 'policy_seed': {policy_seed}, 'policy_prng': 'park_miller_48271_v1'")
    patch('nn.lua',"model.selection=='deterministic_argmax'","model.selection=='categorical_softmax'")
    old="""    if (manifest.get('schema') != 'astra.rl-actions16-build.v1'
            or manifest.get('action_interface') != INTERFACE or manifest.get('actions') != 16):
        raise RuntimeError('Invalid actions16 build identity')"""
    patch('support.py',old,'    validate_build(manifest,package)')
    patch('support.py','from pathlib import Path','from pathlib import Path\nfrom .stochastic_identity import validate_build')
    patch('support.py',"'selection':'deterministic_argmax'","'selection':'categorical_softmax'",2)
    patch('support.py',"    return {'package':package",f"    if payload.get('policy_seed') != manifest['policy_seed'] or payload.get('policy_prng') != 'park_miller_48271_v1':\n        raise RuntimeError('Export seed/PRNG differs from build')\n    return {{'package':package")
    patch('support.py',"('rl_settlement.lua','settlement.lua')):","('rl_settlement.lua','settlement.lua'),('rl_stochastic_nn.lua','stochastic_nn.lua')):")
    patch('support.py',"    if sha256(snapshot['manifest']) != snapshot['manifest_sha256']:","    validate_build(json.loads(snapshot['manifest'].read_text()),snapshot['package'])\n    if sha256(snapshot['manifest']) != snapshot['manifest_sha256']:")
    patch('native_continuous.py','from . import continuous','from . import continuous\nfrom .stochastic_identity import stage_sampling, finish_sampling')
    patch('native_continuous.py','    path.write_text(source)','    source=stage_sampling(run,payload,source,HERE)\n    path.write_text(source)')
    patch('native_continuous.py','    return interface_evaluate(_evaluate_native, globals(), args, kwargs)',
          '    result=interface_evaluate(_evaluate_native, globals(), args, kwargs)\n    return finish_sampling(result,args,kwargs,HERE)')
    # This namespace is evaluation-only; never run training against changed NN semantics.
    for name in ('batch_train.py','native_campaign.py','campaign.py','migrate.py'):
        patch(name,'def main():',"def main():\n    raise RuntimeError('This categorical namespace is verification-only')")
    derived['stochastic_nn.lua']=extras['stochastic_nn.lua'].decode('utf-8')
    patch('stochastic_nn.lua',"  local logits_model={};for k,v in pairs(model) do logits_model[k]=v end\n  logits_model.selection='deterministic_argmax'",
          "  assert(actions.count==16 and actions.interface=='ken_actions16_lp_mp_uppercut_v1','Wrong action interface')")
    patch('stochastic_nn.lua','Base.predict(logits_model,obs)','Base.predict(model,obs)')
    patch('stochastic_nn.lua',"  -- Base.predict computes logits plus an argmax. Its private compatibility copy\n  -- satisfies the original assertion; that argmax is discarded, never deployed.",
          "  -- The derived Base accepts the actual categorical identity. Its extra\n  -- argmax result is discarded; only softmax sampling selects an action.")
    # Reuse the exact reviewed math/audit functions without their old evaluator imports.
    text=extras['stochastic_continuous.py'].decode('utf-8');tree=ast.parse(text)
    wanted={'next_random','stochastic_payload','features','audit_sampling'}
    fragments=[ast.get_source_segment(text,n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in wanted]
    if len(fragments)!=4:raise ValueError('Original sampling helpers changed')
    assignments=[ast.get_source_segment(text,n) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('MODULUS','MULTIPLIER','REASON') for t in n.targets)]
    if len(assignments)!=3:raise ValueError('Original sampling constants changed')
    derived['sampling.py']='from collections import deque\nimport copy,json,re\nimport numpy as np\n'+'\n\n'.join(assignments+fragments)+'\n'
    derived['stochastic_identity.py']=extras['stochastic16_identity.py'].decode('utf-8')
    for name,body in derived.items():
        if name.endswith('.py'):compile(body,PACKAGE+'/'+name,'exec')
    if parent_path.read_bytes()!=parent_bytes or any((source/n).read_bytes()!=b for n,b in captured.items()) or any((HERE/n).read_bytes()!=b for n,b in extras.items()):
        raise RuntimeError('Source changed during build')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False);package=output/PACKAGE;package.mkdir()
    for name,body in derived.items():(package/name).write_text(body,encoding='utf-8')
    (output/'parent-build.json').write_bytes(parent_bytes)
    launcher=(source.parent/'launch.py').read_text().replace('astra_sf2_rl16',PACKAGE)
    launcher=launcher.replace("('migrate','batch_train','native_continuous','native_campaign','export')","('native_continuous','export')")
    launcher=launcher.replace('migrate|batch_train|native_continuous|native_campaign|export','native_continuous|export')
    (output/'launch.py').write_text(launcher,encoding='utf-8')
    manifest={'schema':'astra.rl-stochastic16-build.v1','status':'complete','package':PACKAGE,
        'action_interface':parent['action_interface'],'actions':16,'observations':344,'native_decision_frames':12,
        'selection':'categorical_softmax','policy_seed':policy_seed,'policy_prng':'park_miller_48271_v1',
        'parent_build_sha256':hashlib.sha256(parent_bytes).hexdigest(),'source_sha256':dict(parent['derived_sha256']),
        'sampling_source_sha256':{n:hashlib.sha256(b).hexdigest() for n,b in extras.items()},
        'builder_sha256':sha256(Path(__file__)),'launcher_sha256':sha256(output/'launch.py'),
        'derived_sha256':{n:sha256(package/n) for n in derived},'patches':patches}
    atomic_json(output/'build.json',manifest);return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--policy-seed',type=int,required=True)
    print(json.dumps(build(**vars(p.parse_args())),indent=2))


if __name__=='__main__':main()
