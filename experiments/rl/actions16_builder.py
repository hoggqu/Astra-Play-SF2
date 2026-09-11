"""Build a fresh, isolated 16-action PPO package without editing shared sources."""
import argparse
import hashlib
import json
from pathlib import Path

from astra_play_sf2.config import atomic_json
from astra_play_sf2.runner import sha256

HERE=Path(__file__).resolve().parent
PACKAGE='astra_sf2_rl16'
INTERFACE='ken_actions16_lp_mp_uppercut_v1'
FILES=('__init__.py','env.py','actions.lua','nn.lua','export.py','dataset.py','vector.py',
       'batch_env.py','batch_runtime.lua','batch_train.py','runtime.lua','corrected_reference_runtime.lua',
       'continuous.py','continuous_core.lua','native_continuous.py','native_continuous_core.lua',
       'settlement.lua','campaign.py','native_campaign.py')


def build(output):
    captured={name:(HERE/name).read_bytes() for name in (*FILES,'actions16_migrate.py','actions16_support.py')}
    original={name:captured[name].decode('utf-8') for name in FILES}
    derived=dict(original);patches=[]
    def replace(name,old,new,count=1):
        actual=derived[name].count(old)
        if actual!=count:raise ValueError(f'{name}: expected {count} patch anchors, got {actual}: {old[:80]}')
        derived[name]=derived[name].replace(old,new)
        patches.append({'file':name,'old':old,'new':new,'count':count})
    for name in FILES:
        if 'experiments.rl' in derived[name]:
            replace(name,'experiments.rl',PACKAGE,derived[name].count('experiments.rl'))
    replace('native_campaign.py',"'experiments/rl/'+name","'"+PACKAGE+"/'+name")
    replace('native_campaign.py',"'__init__.py', 'native_campaign.py'","'__init__.py', 'support.py', 'native_campaign.py'")
    replace('native_campaign.py','from .dataset import load_dataset','from .dataset import load_dataset\nfrom .support import validate_model_file')
    replace('native_campaign.py','    initial_hash = sha256(initial) if initial else None','    if initial is not None: validate_model_file(initial)\n    initial_hash = sha256(initial) if initial else None')
    replace('native_continuous.py','from . import continuous','from . import continuous\nfrom .support import interface_evaluate')
    replace('native_continuous.py','def evaluate(*args, **kwargs):','def _evaluate_native(*args, **kwargs):')
    replace('native_continuous.py','def main():','def evaluate(*args, **kwargs):\n    return interface_evaluate(_evaluate_native, globals(), args, kwargs)\n\n\ndef main():')
    replace('env.py',"'uppercut', 'jump_heavy_kick')","'uppercut', 'jump_heavy_kick', 'medium_uppercut')")
    replace('env.py',"'actions': ACTION_NAMES,",f"'action_interface': '{INTERFACE}', 'actions': ACTION_NAMES,")
    replace('actions.lua','local A={frames=12,count=15}',f"local A={{frames=12,count=16,interface='{INTERFACE}'}}")
    replace('actions.lua'," else return frame<3 and 'U '..forward", " elseif action==15 then\n  if frame<2 then return forward elseif frame<4 then return 'D' elseif frame<6 then return 'D '..forward..' MP' else return '' end\n else return frame<3 and 'U '..forward")
    replace('nn.lua',"model.schema=='astra.rl-policy.v1' and model.observations==344 and model.actions==15",f"model.schema=='astra.rl-policy.actions16.v1' and model.observations==344 and model.actions==16 and model.action_interface=='{INTERFACE}'")
    replace('nn.lua','assert(#x==15)','assert(#x==16)')
    replace('nn.lua','features and 15 actions','features and 16 actions')
    replace('nn.lua','function N.new(model,actions)',f"function N.new(model,actions)\n assert(actions.count==16 and actions.interface=='{INTERFACE}','Wrong action interface')")
    replace('export.py',"    policy = model.policy",f"    if getattr(model, 'astra_action_interface', None) != '{INTERFACE}':\n        raise ValueError('Model lacks the explicit actions16 interface identity')\n    policy = model.policy")
    replace('export.py',"'n', None) != 15","'n', None) != 16")
    replace('export.py','344 observations and 15 discrete actions','344 observations and 16 discrete actions')
    replace('export.py',"'schema': 'astra.rl-policy.v1'",f"'schema': 'astra.rl-policy.actions16.v1', 'action_interface': '{INTERFACE}'")
    replace('export.py',"'actions': 15","'actions': 16")
    replace('batch_train.py','    policy = model.policy',f"    if getattr(model, 'astra_action_interface', None) != '{INTERFACE}' or model.action_space.n != 16:\n        raise ValueError('Expected an explicitly identified actions16 PPO model')\n    policy = model.policy")
    replace('batch_train.py',"'schema': 'astra.rl-policy.v1'",f"'schema': 'astra.rl-policy.actions16.v1', 'action_interface': '{INTERFACE}'")
    replace('batch_train.py',"'actions': 15","'actions': 16")
    replace('batch_train.py','from .env import MameEnv','from .env import MameEnv\nfrom .support import parity_actions, validate_model_file, optimizer_provenance')
    replace('batch_train.py','np.random.default_rng(seed).integers(0, 15, count).tolist()','parity_actions(count, seed)',2)
    replace('batch_train.py',"'episodes': len(result['episodes'])}","'episodes': len(result['episodes']), 'action_counts': np.bincount(actions,minlength=16).tolist()}")
    replace('batch_train.py',"'episodes':len(new['episodes'])}","'episodes':len(new['episodes']), 'action_counts':np.bincount(actions,minlength=16).tolist()}")
    replace('batch_train.py','    output = args.output.resolve();output.mkdir(parents=True, exist_ok=False)',
            "    if is_parity and args.steps < 16: parser.error('Actions16 parity requires 16..256 steps')\n    if args.init_model: validate_model_file(args.init_model)\n    output = args.output.resolve();output.mkdir(parents=True, exist_ok=False)")
    replace('batch_train.py',"            result['effective_ppo'] =", "            result['optimizer_initialization'] = optimizer_provenance(model, args.init_model is not None)\n            result['effective_ppo'] =")
    replace('batch_train.py',"'parity': args.parity, 'native_parity': args.native_parity,", "'parity': args.parity, 'native_parity': args.native_parity, 'parity_policy': 'forced actions; init-model only validates identity' if is_parity else None,")
    replace('batch_train.py','            if args.init_model:',f"            model.astra_action_interface = '{INTERFACE}'\n            if args.init_model:")
    replace('batch_train.py','                if model.n_steps != 256:',f"                if getattr(model, 'astra_action_interface', None) != '{INTERFACE}':\n                    raise ValueError('Migrate a 15-action model before actions16 training')\n                if model.n_steps != 256:")
    replace('batch_train.py',"'training_only': True, 'formal_clear': False,",f"'training_only': True, 'formal_clear': False, 'action_interface': '{INTERFACE}', 'actions': 16,")
    replace('continuous.py',"schema='mame.rl-continuous-play.v1',policy_kind='ppo'",f"schema='mame.rl-continuous-play.actions16.v1',action_interface=Model.action_interface,policy_kind='ppo'")
    replace('continuous.py',"'mame.rl-continuous-play.v1'","'mame.rl-continuous-play.actions16.v1'")
    replace('continuous.py','        payload = export_policy(model)\n','')
    replace('continuous.py','        preflight = doctor(config)','        payload = export_policy(model)\n        preflight = doctor(config)')
    replace('continuous.py',"        score = summary['score']",f"        if summary.get('action_interface') != '{INTERFACE}':\n            raise RuntimeError('Wrong actions16 interface identity')\n        score = summary['score']")
    replace('continuous.py',"'frozen_v4_certification': False,",f"'action_interface': '{INTERFACE}', 'actions': 16, 'frozen_v4_certification': False,")
    replace('native_campaign.py',"'frozen_v4_certification': False,",f"'action_interface': '{INTERFACE}', 'actions': 16, 'frozen_v4_certification': False,")
    for name,old,new in (
        ('batch_train.py','astra.rl-batch-prototype.v1','astra.rl-batch-prototype.actions16.v1'),
        ('native_campaign.py','astra.rl-batch-prototype.v1','astra.rl-batch-prototype.actions16.v1'),
        ('continuous.py','astra.rl-continuous.v1','astra.rl-continuous.actions16.v1'),
        ('native_campaign.py','astra.rl-continuous.v1','astra.rl-continuous.actions16.v1'),
        ('native_campaign.py','astra.rl-native-campaign.v1','astra.rl-native-campaign.actions16.v1')):
        replace(name,old,new)
    derived['migrate.py']=captured['actions16_migrate.py'].decode('utf-8')
    derived['support.py']=captured['actions16_support.py'].decode('utf-8')
    for name,source in derived.items():
        if name.endswith('.py'):compile(source,PACKAGE+'/'+name,'exec')
    if any((HERE/name).read_bytes()!=content for name,content in captured.items()):
        raise RuntimeError('Source changed while building; generate a fresh snapshot after it stabilizes')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    package=output/PACKAGE;package.mkdir()
    for name,source in derived.items():(package/name).write_text(source,encoding='utf-8')
    launcher=output/'launch.py'
    launcher.write_text('''"""Portable launcher; subprocesses inherit this isolated package path."""
import os
from pathlib import Path
import subprocess
import sys

if len(sys.argv)<2 or sys.argv[1] not in ('migrate','batch_train','native_continuous','native_campaign','export'):
    raise SystemExit('Usage: launch.py migrate|batch_train|native_continuous|native_campaign|export [arguments]')
env=dict(os.environ)
env['PYTHONPATH']=str(Path(__file__).resolve().parent)+os.pathsep+env.get('PYTHONPATH','')
raise SystemExit(subprocess.call([sys.executable,'-m','astra_sf2_rl16.'+sys.argv[1],*sys.argv[2:]],env=env))
''',encoding='utf-8')
    manifest={'schema':'astra.rl-actions16-build.v1','status':'complete','package':PACKAGE,
              'action_interface':INTERFACE,'observations':344,'actions':16,'native_decision_frames':12,
              'source_sha256':{name:hashlib.sha256(captured[name]).hexdigest() for name in FILES},
              'builder_sha256':sha256(Path(__file__)),'migration_source_sha256':hashlib.sha256(captured['actions16_migrate.py']).hexdigest(),
              'support_source_sha256':hashlib.sha256(captured['actions16_support.py']).hexdigest(),
              'launcher_sha256':sha256(launcher),
              'derived_sha256':{name:sha256(package/name) for name in derived},'patches':patches,
              'training_entry':PACKAGE+'.batch_train','verification_entry':PACKAGE+'.native_continuous',
              'migration_entry':PACKAGE+'.migrate','campaign_entry':PACKAGE+'.native_campaign'}
    atomic_json(output/'build.json',manifest)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=build(args.output)
    print(json.dumps({k:result[k] for k in ('status','package','action_interface','training_entry','verification_entry')},indent=2))


if __name__=='__main__':main()
