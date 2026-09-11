"""Build candidate-only 368-observation inference/training from a frozen actions16 chain."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
from astra_play_sf2.config import atomic_json
from astra_play_sf2.runner import sha256
from .round_chain_identity import validate_chain_build
from .projectile_identity import INTERFACE, ACTION_INTERFACE, validate_projectile_build
HERE=Path(__file__).resolve().parent
PACKAGE='astra_sf2_rl_projectiles6'


def build(source,output):
    source=Path(source).resolve();output=Path(output).resolve()
    parent_bytes=(source.parent/'build.json').read_bytes();parent=json.loads(parent_bytes)
    validate_chain_build(parent,source)
    captured={n:(source/n).read_bytes() for n in parent['derived_sha256']}
    if any(hashlib.sha256(v).hexdigest()!=parent['derived_sha256'][n] for n,v in captured.items()):
        raise ValueError('Source package differs from frozen chain manifest')
    derived={n:v.decode('utf-8') for n,v in captured.items()}
    for n in derived:
        if n.endswith('.py'):
            derived[n]=derived[n].replace(source.name,PACKAGE).replace('344','368').replace('.actions16.v1','.projectiles6.v1')
        if n=='nn.lua':derived[n]=derived[n].replace('344','368').replace('.actions16.v1','.projectiles6.v1')
    def patch(name,old,new):
        if derived[name].count(old)!=1:raise ValueError(f'Missing/ambiguous anchor {name}: {old}')
        derived[name]=derived[name].replace(old,new)
    # One RAM reader supplies both native deployment and chain training snapshots.
    wave_load="local Projectiles=assert(loadfile('training/runtime/rl_projectiles.lua'))()\n"
    for name in ('batch_runtime.lua','runtime.lua','corrected_reference_runtime.lua'):
        if name not in derived:continue
        patch(name,'local m=manager.machine',wave_load+'local m=manager.machine')
        patch(name,' return s\nend',' s.projectile_observation=Projectiles.read(mem,s)\n return s\nend')
    patch('env.py','shape=(4*86,)','shape=(4*92,)')
    patch('env.py','    return np.clip(np.asarray(values, dtype=np.float32), -1, 1)',
          "    extra=state['projectile_observation']\n    if len(extra)!=6 or not np.isfinite(extra).all(): raise ValueError('Invalid projectile observation')\n    values += list(extra)\n    return np.clip(np.asarray(values, dtype=np.float32), -1, 1)")
    patch('env.py',"[('runtime.lua', 'rl.lua'), ('actions.lua', 'rl_actions.lua')]",
          "[('runtime.lua', 'rl.lua'), ('actions.lua', 'rl_actions.lua'), ('projectile_features.lua', 'rl_projectiles.lua')]")
    patch('env.py',"('rl.lua', 'rl_actions.lua', 'rl_checkpoint.lua')","('rl.lua', 'rl_actions.lua', 'rl_checkpoint.lua', 'rl_projectiles.lua')")
    derived['nn.lua']=derived['nn.lua'].replace('4 x 86 features','4 x 92 features')
    patch('nn.lua','assert(#out==86);return out',
          'assert(#out==86);assert(type(s.projectile_observation)=="table" and #s.projectile_observation==6)\n for _,v in ipairs(s.projectile_observation) do assert(type(v)=="number" and v==v);append(out,v) end\n assert(#out==92);return out')
    patch('nn.lua',"assert(#obs==368",f"assert(model.observation_interface=='{INTERFACE}')\n assert(#obs==368")
    # Training and export refuse an unidentified model even if its tensor shapes fit.
    for name in ('export.py','batch_train.py'):
        derived[name]=derived[name].replace("if getattr(model, 'astra_action_interface', None) !=",f"if getattr(model, 'astra_observation_interface', None) != '{INTERFACE}' or getattr(model, 'astra_action_interface', None) !=")
        derived[name]=derived[name].replace("'schema': 'astra.rl-policy.projectiles6.v1',",f"'schema': 'astra.rl-policy.projectiles6.v1', 'observation_interface': '{INTERFACE}',")
    patch('batch_train.py',"            model.astra_action_interface =",f"            model.astra_observation_interface = '{INTERFACE}'\n            model.astra_action_interface =")
    patch('batch_train.py',"'schema': 'astra.rl-batch-prototype.projectiles6.v1',",f"'schema': 'astra.rl-batch-prototype.projectiles6.v1', 'observation_interface': '{INTERFACE}', 'observations':368,")
    patch('continuous.py',"[('nn.lua', 'rl_nn.lua'),", "[('projectile_features.lua', 'rl_projectiles.lua'), ('nn.lua', 'rl_nn.lua'),")
    patch('continuous.py',"action_interface=Model.action_interface,policy_kind=", "action_interface=Model.action_interface,observation_interface=Model.observation_interface,policy_kind=")
    patch('continuous.py',"'training/runtime/rl_policy.lua','training/runtime/rl_nn.lua'", "'training/runtime/rl_projectiles.lua','training/runtime/rl_policy.lua','training/runtime/rl_nn.lua'")
    patch('continuous.py',"'schema': 'astra.rl-continuous.projectiles6.v1',",f"'schema': 'astra.rl-continuous.projectiles6.v1', 'observation_interface':'{INTERFACE}', 'observations':368,")
    # Both entry adapters receive the same read-only state extension. The native
    # adapter subsequently retains its existing timing and all-frame audit gates.
    patch('continuous.py', "    path.write_text('-- EXPERIMENTAL RL ADAPTER; NOT FROZEN V4 POLICY CERTIFICATION.\\n' + text, encoding='utf-8')",
          "    text = replace_once(text, 'local m=manager.machine', "+repr(wave_load+"local m=manager.machine")+")\n"
          "    text = replace_once(text, 's.emulated_seconds=m.time:as_double()', 's.emulated_seconds=m.time:as_double();s.projectile_observation=Projectiles.read(mem,s)')\n"
          "    path.write_text('-- EXPERIMENTAL RL ADAPTER; NOT FROZEN V4 POLICY CERTIFICATION.\\n' + text, encoding='utf-8')")
    patch('support.py','from .chain_identity import validate_chain_build','from .projectile_identity import validate_projectile_build')
    derived['support.py']=derived['support.py'].replace('validate_chain_build(', 'validate_projectile_build(')
    patch('support.py',"    if (getattr(model, 'astra_action_interface', None)",f"    if (getattr(model, 'astra_observation_interface', None) != '{INTERFACE}'\n            or getattr(model, 'astra_action_interface', None)")
    patch('support.py',"'actions':16,'observations':368,'decision_frames':12,",f"'actions':16,'observations':368,'observation_interface':'{INTERFACE}','decision_frames':12,")
    patch('support.py',"('rl_settlement.lua','settlement.lua')", "('rl_settlement.lua','settlement.lua'),('rl_projectiles.lua','projectile_features.lua')")
    patch('support.py',"if (result.get('schema')",f"if (result.get('observation_interface') != '{INTERFACE}' or result.get('observations') != 368\n            or result.get('schema')")
    patch('support.py',"if (summary.get('schema')",f"if (summary.get('observation_interface') != '{INTERFACE}'\n                    or summary.get('schema')")
    patch('support.py',"return {'ok':True,'action_interface':INTERFACE",f"return {{'ok':True,'observation_interface':'{INTERFACE}','action_interface':INTERFACE")
    patch('support.py',"'schema':'astra.rl-continuous.projectiles6.v1','status':'invalid',",f"'schema':'astra.rl-continuous.projectiles6.v1','status':'invalid','observation_interface':'{INTERFACE}','observations':368,")
    patch('native_campaign.py',"'__init__.py', 'support.py', 'chain_identity.py'", "'__init__.py', 'support.py', 'projectile_identity.py', 'projectile_features.lua', 'chain_identity.py'")
    derived['migrate.py']=(HERE/'projectile_migrate.py').read_text()
    derived['projectile_identity.py']=(HERE/'projectile_identity.py').read_text()
    derived['projectile_features.lua']=(HERE/'projectile_features.lua').read_text()
    # Any Python references to a derived package are changed, but Lua behavior outside observations is retained.
    for n,body in derived.items():
        if n.endswith('.py'):compile(body,PACKAGE+'/'+n,'exec')
    if (source.parent/'build.json').read_bytes()!=parent_bytes or any((source/n).read_bytes()!=b for n,b in captured.items()):
        raise RuntimeError('Source changed during build')
    output.mkdir(parents=True,exist_ok=False);package=output/PACKAGE;package.mkdir()
    for n,body in derived.items():(package/n).write_text(body,encoding='utf-8')
    (output/'parent-build.json').write_bytes(parent_bytes)
    # Preserve the complete original ancestry as immutable, non-executing provenance.
    if (source.parent/'parent-build.json').is_file():shutil.copy2(source.parent/'parent-build.json',output/'grandparent-build.json')
    manifest={'schema':'astra.rl-projectiles6-build.v1','status':'candidate_offline_only','native_validated':False,
        'package':PACKAGE,'action_interface':ACTION_INTERFACE,'observation_interface':INTERFACE,'observations':368,'actions':16,
        'native_decision_frames':12,'training_protocol':'native_match_round_episodes_v1','parent_build_sha256':hashlib.sha256(parent_bytes).hexdigest(),
        'builder_sha256':sha256(Path(__file__)),'source_sha256':dict(parent['derived_sha256']),
        'derived_sha256':{n:sha256(package/n) for n in derived}}
    atomic_json(output/'build.json',manifest);validate_projectile_build(manifest,package);return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    print(json.dumps(build(**vars(p.parse_args())),indent=2))
if __name__=='__main__':main()
