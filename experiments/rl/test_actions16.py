"""Offline isolated 16-action migration, macro, export and Lua parity checks."""
import importlib
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch
from lupa.lua54 import LuaRuntime
from stable_baselines3 import PPO

from astra_play_sf2.runner import sha256
from .actions16_builder import build, PACKAGE, INTERFACE, FILES
from .actions16_migrate import migrate_model, migrate

HERE=Path(__file__).resolve().parent


class OldSpaces(gym.Env):
    observation_space=gym.spaces.Box(-1,1,(344,),dtype=np.float32)
    action_space=gym.spaces.Discrete(15)


class Actions16Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name)
        cls.old=PPO('MlpPolicy',OldSpaces(),seed=16,n_steps=256,batch_size=64,n_epochs=4,
                    policy_kwargs={'net_arch':{'pi':[64,64],'vf':[64,64]}},device='cpu')
        cls.old.num_timesteps=102400
        cls.new=migrate_model(cls.old,seed=42)
        cls.before={name:sha256(HERE/name) for name in FILES}
        cls.manifest=build(cls.root/'code');cls.pkg=cls.root/'code'/PACKAGE
        sys.path.insert(0,str(cls.root/'code'))
        cls.export=importlib.import_module(PACKAGE+'.export')
        cls.model_path=cls.root/'new.zip';cls.new.save(cls.model_path)
        cls.payload=cls.export.export_policy(cls.model_path)

    @classmethod
    def tearDownClass(cls):
        sys.path.remove(str(cls.root/'code'))
        for key in list(sys.modules):
            if key==PACKAGE or key.startswith(PACKAGE+'.'):del sys.modules[key]
        cls.temp.cleanup()

    def test_builder_keeps_original_sources_and_derives_distinct_namespace(self):
        self.assertEqual(self.before,{name:sha256(HERE/name) for name in FILES})
        self.assertEqual(self.manifest['package'],PACKAGE)
        self.assertEqual(self.manifest['actions'],16)
        self.assertIn(PACKAGE+'.batch_train',(self.pkg/'native_campaign.py').read_text())
        for name in self.manifest['derived_sha256']:
            if name.endswith('.py'):compile((self.pkg/name).read_text(),name,'exec')
        with self.assertRaises(FileExistsError):build(self.root/'code')

    def test_probability_split_preserves_other_actions_and_dp_family(self):
        obs=torch.tensor(np.random.default_rng(1).uniform(-1,1,(100,344)),dtype=torch.float32)
        with torch.no_grad():
            old_prob=self.old.policy.get_distribution(obs).distribution.probs
            new_prob=self.new.policy.get_distribution(obs).distribution.probs
            old_value=self.old.policy.predict_values(obs);new_value=self.new.policy.predict_values(obs)
        for action in range(15):
            if action!=13:torch.testing.assert_close(new_prob[:,action],old_prob[:,action],rtol=1e-6,atol=1e-8)
        torch.testing.assert_close(new_prob[:,13],old_prob[:,13]/2,rtol=1e-6,atol=1e-8)
        torch.testing.assert_close(new_prob[:,15],old_prob[:,13]/2,rtol=1e-6,atol=1e-8)
        torch.testing.assert_close(new_value,old_value,rtol=0,atol=0)
        self.assertFalse(self.new.policy.optimizer.state_dict()['state'])
        self.assertEqual(self.new.num_timesteps,0);self.assertEqual(self.new._n_updates,0)
        self.assertEqual(self.new.astra_action_interface,INTERFACE)
        old=self.old.policy.state_dict();new=self.new.policy.state_dict()
        for key in old:
            if not key.startswith('action_net.'):torch.testing.assert_close(new[key],old[key],rtol=0,atol=0)

    def test_old15_actions_identical_new15_is_mp_uppercut(self):
        lua=LuaRuntime(unpack_returned_tuples=True)
        old=lua.execute((HERE/'actions.lua').read_text());new=lua.execute((self.pkg/'actions.lua').read_text())
        self.assertEqual(new.count,16);self.assertEqual(new.interface,INTERFACE)
        for facing in ('L','R'):
            for frame in range(12):
                for action in range(15):
                    self.assertEqual(new['keys'](action,frame,lua.table(),facing),old['keys'](action,frame,lua.table(),facing))
                expected=old['keys'](13,frame,lua.table(),facing).replace('LP','MP')
                self.assertEqual(new['keys'](15,frame,lua.table(),facing),expected)

    def test_export_lua_and_torch_logits_actions_agree(self):
        lua=LuaRuntime(unpack_returned_tuples=True);nn=lua.execute((self.pkg/'nn.lua').read_text())
        payload=lua.execute('return '+self.export.lua_literal(self.payload))
        obs=np.random.default_rng(5).uniform(-1,1,(50,344)).astype(np.float32)
        with torch.no_grad():expected=self.new.policy.action_net(self.new.policy.mlp_extractor.policy_net(torch.tensor(obs))).numpy()
        for row,logits in zip(obs,expected):
            action,actual=nn.predict(payload,lua.table_from(row.tolist()))
            np.testing.assert_allclose(list(actual.values()),logits,atol=2e-7,rtol=1e-4)
            self.assertEqual(action,int(logits.argmax()))
        self.assertEqual(self.payload['actions'],16)
        self.assertEqual(self.payload['action_interface'],INTERFACE)
        self.assertEqual(self.payload['schema'],'astra.rl-policy.actions16.v1')

    def test_wrong_model_or_lua_interface_is_rejected(self):
        old_path=self.root/'old.zip';self.old.save(old_path)
        with self.assertRaisesRegex(ValueError,'identity'):self.export.export_policy(old_path)
        unidentified=migrate_model(self.old);del unidentified.astra_action_interface
        path=self.root/'untagged.zip';unidentified.save(path)
        with self.assertRaisesRegex(ValueError,'identity'):self.export.export_policy(path)
        lua=LuaRuntime(unpack_returned_tuples=True);old_nn=lua.execute((HERE/'nn.lua').read_text())
        payload=lua.execute('return '+self.export.lua_literal(self.payload))
        with self.assertRaises(Exception):old_nn.predict(payload,lua.table_from([0.]*344))
        new_nn=lua.execute((self.pkg/'nn.lua').read_text())
        old_actions=lua.execute((HERE/'actions.lua').read_text())
        with self.assertRaises(Exception):new_nn.new(payload,old_actions)

    def test_batch_and_native_staging_use_16_identity(self):
        batch=importlib.import_module(PACKAGE+'.batch_train')
        payload=batch.live_policy(self.new)
        self.assertEqual(payload['actions'],16);self.assertEqual(payload['action_interface'],INTERFACE)
        with self.assertRaisesRegex(ValueError,'actions16'):batch.live_policy(self.old)
        native=importlib.import_module(PACKAGE+'.native_continuous')
        run=self.root/'stage-native';sources=native.stage_native_policy(run,3,self.payload)
        source=(run/'training/runtime/play.lua').read_text()
        self.assertIn('action_interface=Model.action_interface',source)
        self.assertIn('mame.rl-continuous-play.actions16.v1',source)
        self.assertEqual(sources['rl_actions.lua'],sha256(self.pkg/'actions.lua'))
        self.assertEqual(importlib.import_module(PACKAGE+'.env').ACTION_NAMES[-1],'medium_uppercut')

    def test_migration_report_and_portable_launcher(self):
        old_path=self.root/'report-old.zip';self.old.save(old_path)
        output=self.root/'migration-report'
        report=migrate(old_path,output,seed=72)
        restored=PPO.load(output/report['model'],device='cpu')
        self.assertTrue(report['optimizer_reinitialized'])
        self.assertEqual(report['source_steps'],102400)
        self.assertEqual(report['model_sha256'],sha256(output/report['model']))
        self.assertEqual(restored.astra_action_interface,INTERFACE)
        for name in ('n_steps','batch_size','n_epochs','gamma','gae_lambda','ent_coef','vf_coef','max_grad_norm'):
            self.assertEqual(getattr(restored,name),getattr(self.old,name))
        command=[sys.executable,str(self.root/'code/launch.py'),'native_campaign','--help']
        result=subprocess.run(command,capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('--verification-attempts',result.stdout)

    def test_wrong_activation_is_rejected(self):
        wrong=PPO('MlpPolicy',OldSpaces(),n_steps=256,policy_kwargs={
            'net_arch':{'pi':[64,64],'vf':[64,64]},'activation_fn':torch.nn.ReLU})
        with self.assertRaisesRegex(ValueError,'Linear/Tanh'):migrate_model(wrong)

    def test_native_parity_guarantees_all_actions_and_keeps_default_protocol(self):
        support=importlib.import_module(PACKAGE+'.support')
        for seed in (1,42,73):
            sequence=support.parity_actions(256,seed)
            self.assertEqual(sequence,support.parity_actions(256,seed))
            self.assertEqual(sorted(sequence[:16]),list(range(16)))
        with self.assertRaises(ValueError):support.parity_actions(15,42)
        for name in ('settlement.lua','batch_runtime.lua','native_continuous_core.lua'):
            self.assertEqual((self.pkg/name).read_bytes(),(HERE/name).read_bytes())
        campaign=importlib.import_module(PACKAGE+'.native_campaign')
        self.assertIn(PACKAGE+'/support.py',campaign.source_paths())
        info=support.optimizer_provenance(self.new,True)
        self.assertEqual(info['origin'],'reinitialized_at_actions16_migration')
        self.assertTrue(info['migration']['optimizer_reinitialized'])
        self.assertEqual(info['initial_state_entries'],0)

    def test_every_model_entry_rejects_old_before_mame(self):
        path=self.root/'rejected-old.zip';self.old.save(path)
        batch=importlib.import_module(PACKAGE+'.batch_train')
        for extra in ([],['--native-parity']):
            argv=['batch_train','--dataset','unused.json','--output',str(self.root/'not-created'),
                  '--init-model',str(path),'--workers','1','--steps','256',*extra]
            with patch.object(sys,'argv',argv),patch.object(batch,'ManagedVec') as vector:
                with self.assertRaisesRegex(ValueError,'actions16'):batch.main()
                vector.assert_not_called()
        campaign=importlib.import_module(PACKAGE+'.native_campaign')
        with patch.object(campaign,'load_dataset',return_value=({},3)):
            with self.assertRaisesRegex(ValueError,'actions16'):
                campaign.campaign('unused.json',self.root/'not-created',init_model=path)
        native=importlib.import_module(PACKAGE+'.native_continuous')
        with patch.object(native.continuous,'doctor') as doctor:
            result=native.evaluate(None,path,self.root/'rejected-native')
            self.assertEqual(result['status'],'invalid')
            self.assertIn('identity',result['error'])
            doctor.assert_not_called()

    def test_interface_audit_is_earned_and_tampering_invalidates_results(self):
        support=importlib.import_module(PACKAGE+'.support')
        native=importlib.import_module(PACKAGE+'.native_continuous')
        cases=(None,'package','model','actions','nn','payload','match','record','export')
        for case in cases:
            with self.subTest(case=case):
                run=self.root/('audit-'+str(case))
                package_source=self.pkg/'actions.lua';old_source=package_source.read_bytes()
                old_model=self.model_path.read_bytes()
                namespace={'HERE':self.pkg,'stage_native_policy':native.stage_native_policy}
                def fake_evaluate(config,model,output):
                    output.mkdir()
                    payload=dict(self.payload)
                    if case=='export':payload['actions']=15
                    hashes=namespace['stage_native_policy'](output,3,payload)
                    summary={'schema':'mame.rl-continuous-play.actions16.v1',
                             'action_interface':INTERFACE,'model_sha256':self.payload['model_sha256'],
                             'policy_kind':'ppo'}
                    if case=='match':summary['action_interface']='other-interface'
                    (output/'match.json').write_text(json.dumps({'summary':summary}))
                    if case=='package':package_source.write_bytes(old_source+b'\n-- changed')
                    if case=='model':self.model_path.write_bytes(old_model+b'changed')
                    if case in ('actions','nn','payload'):
                        name={'actions':'rl_actions.lua','nn':'rl_nn.lua','payload':'rl_policy.lua'}[case]
                        target=output/'training/runtime'/name
                        target.write_bytes(target.read_bytes()+b'\n-- changed')
                    if case=='record':hashes['rl_nn.lua']='0'*64
                    return {'schema':'astra.rl-continuous.actions16.v1','status':'complete',
                            'action_interface':INTERFACE,'actions':16,'model_sha256':self.payload['model_sha256'],
                            'runtime_sha256':hashes,'attempts':[{'matches':['match.json']}],
                            'native_timing_audit':{'ok':True}}
                try:
                    result=support.interface_evaluate(fake_evaluate,namespace,(None,self.model_path,run),{})
                    self.assertEqual(result['status'],'complete' if case is None else 'invalid')
                    self.assertEqual(result['action_interface_audit']['ok'],case is None)
                    if case is None:
                        self.assertEqual(result['action_interface_audit']['checked_matches'],1)
                        self.assertGreater(result['action_interface_audit']['checked_staged_files'],4)
                    sealed=json.loads((run/'evidence-sha256.json').read_text())
                    self.assertEqual(sealed['files']['result.json'],sha256(run/'result.json'))
                finally:
                    package_source.write_bytes(old_source);self.model_path.write_bytes(old_model)

    def test_interface_audit_rejects_changed_build_before_evaluator(self):
        support=importlib.import_module(PACKAGE+'.support')
        native=importlib.import_module(PACKAGE+'.native_continuous')
        source=self.pkg/'nn.lua';before=source.read_bytes()
        try:
            source.write_bytes(before+b'\n-- tampered before run')
            with patch.object(native,'_evaluate_native') as evaluator:
                result=native.evaluate(None,self.model_path,self.root/'reject-before-start')
                evaluator.assert_not_called()
            self.assertEqual(result['status'],'invalid')
            self.assertFalse(result['action_interface_audit']['ok'])
        finally:source.write_bytes(before)


if __name__=='__main__':unittest.main()
