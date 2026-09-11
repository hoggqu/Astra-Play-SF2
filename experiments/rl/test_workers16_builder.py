"""Offline sixteen-worker limits, immutable derivation and full20 audit tests."""
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from astra_play_sf2.runner import sha256
from .actions16_builder import build as action_build, PACKAGE as ACTION_PACKAGE
from .round_chain_builder import build as chain_build, PACKAGE
from .pulsed_normals_builder import build as pulse_build
from .workers16_builder import build, build_driver
from .test_reliability_campaign import verification

class Workers16Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name)
        action_build(cls.root/'actions');chain_build(cls.root/'chain',cls.root/'actions'/ACTION_PACKAGE)
        shutil.copytree(Path(__file__).resolve().parents[2]/'src',cls.root/'chain/src',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        pulse_build(cls.root/'pulse',cls.root/'chain'/PACKAGE)
        cls.parent=cls.root/'pulse'/PACKAGE
        cls.original={p.relative_to(cls.root/'pulse').as_posix():sha256(p) for p in (cls.root/'pulse').rglob('*') if p.is_file()}
        cls.manifest=build(cls.root/'worker16',cls.parent)
        cls.driver=build_driver(cls.root/'driver');cls.package=cls.root/'worker16'/PACKAGE
        sys.path.insert(0,str(cls.root/'worker16'));cls.identity=importlib.import_module(PACKAGE+'.pulsed_normals_identity')
    @classmethod
    def tearDownClass(cls):
        sys.path.remove(str(cls.root/'worker16'))
        for n in list(sys.modules):
            if n==PACKAGE or n.startswith(PACKAGE+'.'):del sys.modules[n]
        cls.tmp.cleanup()
    def test_only_limit_and_identity_changed(self):
        self.assertEqual(self.original,{p.relative_to(self.root/'pulse').as_posix():sha256(p) for p in (self.root/'pulse').rglob('*') if p.is_file()})
        changed=[n for n in self.manifest['derived_sha256'] if sha256(self.package/n)!=sha256(self.parent/n)]
        self.assertEqual(set(changed),{'batch_train.py','native_campaign.py','pulsed_normals_identity.py'})
        self.identity.validate_pulsed_build(self.manifest,self.package)
        self.assertEqual(self.manifest['worker_limit_revision']['global_rollout_at_16_workers'],4096)
        self.assertEqual((self.package/'batch_train.py').read_text().replace('choices=range(1, 17)','choices=range(1, 9)'),(self.parent/'batch_train.py').read_text())
    def test_parent_identity_and_runtime_tamper_rejected(self):
        for p in [self.root/'worker16/workers-parent-build.json',self.root/'worker16/workers-parent-identity.py',self.package/'actions.lua']:
            old=p.read_bytes()
            try:
                p.write_bytes(old+b'\n')
                with self.assertRaises(RuntimeError):self.identity.validate_pulsed_build(self.manifest,self.package)
            finally:p.write_bytes(old)
    def run_cli(self,code,args):
        env=dict(os.environ,PYTHONPATH=str(code/'src')+os.pathsep+str(code))
        return subprocess.run([sys.executable,*args],cwd=code,env=env,text=True,capture_output=True,timeout=30)
    def test_training_16_accepts_parser_but_rejects_incomplete_update_before_mame(self):
        r=self.run_cli(self.root/'worker16',['-m',PACKAGE+'.batch_train','--workers','16','--dataset','absent','--output',str(self.root/'none'),'--steps','2048'])
        self.assertEqual(r.returncode,2);self.assertIn('workers*256',r.stderr);self.assertFalse((self.root/'none').exists())
        r=self.run_cli(self.root/'worker16',['-m',PACKAGE+'.batch_train','--workers','17'])
        self.assertEqual(r.returncode,2);self.assertIn('invalid choice',r.stderr)
    def test_driver_full20_requires_pulse_and_both_audits(self):
        fixture=verification(10,'abc');fixture['action_interface']='ken_actions16_pulsed_normals_v2'
        path=self.root/'fixture.json';path.write_text(json.dumps(fixture))
        script="""import json,sys,copy
from experiments.rl.reliability_campaign import audit_full_twenty
r=json.load(open(sys.argv[1]));assert audit_full_twenty(r,'abc',0)['goal_achieved']
for mutation in ('old','native','interface','prefix'):
 t=copy.deepcopy(r)
 if mutation=='old':t['action_interface']='ken_actions16_lp_mp_uppercut_v1'
 elif mutation=='prefix':t['attempts']=t['attempts'][:10]
 else:t[('native_timing_audit' if mutation=='native' else 'action_interface_audit')]['ok']=False
 try:audit_full_twenty(t,'abc',0)
 except RuntimeError:pass
 else:raise AssertionError(mutation)
"""
        r=self.run_cli(self.root/'driver',['-c',script,str(path)]);self.assertEqual(r.returncode,0,r.stderr)
        for module in ('reliability_campaign','reliability_pipeline'):
            r=self.run_cli(self.root/'driver',['-m','experiments.rl.'+module,'--help'])
            self.assertEqual(r.returncode,0,r.stderr);self.assertIn('15,16}',r.stdout)
    def test_16_env_load_preserves_weights_adam_and_256_step_buffer(self):
        import numpy as np
        import torch
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        from .test_rollout_builder import DummyGame
        torch.set_num_threads(1)
        old_env=DummyVecEnv([DummyGame]*8);new_env=DummyVecEnv([DummyGame]*16)
        self.addCleanup(old_env.close);self.addCleanup(new_env.close)
        model=PPO('MlpPolicy',old_env,n_steps=256,batch_size=64,n_epochs=4,seed=4)
        x=torch.zeros((2,344));v,lp,_=model.policy.evaluate_actions(x,torch.tensor([6,11]))
        (v.square().mean()+lp.mean()).backward();model.policy.optimizer.step();model._n_updates=12
        path=self.root/'dummy.zip';model.save(path)
        loaded=PPO.load(path,env=new_env,device='cpu',seed=5)
        self.assertEqual((loaded.n_envs,loaded.n_steps,loaded.rollout_buffer.n_envs,loaded.rollout_buffer.buffer_size),(16,256,16,256))
        self.assertEqual(loaded._n_updates,12)
        for k,v in model.policy.state_dict().items():self.assertTrue(torch.equal(v,loaded.policy.state_dict()[k]))
        before=model.policy.optimizer.state_dict();after=loaded.policy.optimizer.state_dict()
        self.assertEqual(before['param_groups'],after['param_groups'])
        for k,state in before['state'].items():
            for n,v in state.items():self.assertTrue(torch.equal(v,after['state'][k][n]))

    def test_provenance_includes_worker_parent_and_entire_frozen_pulse_parent(self):
        paths=self.identity.provenance_paths(self.package)
        self.assertIn('workers/parent-build.json',paths);self.assertIn('workers/parent-identity.py',paths)
        self.assertTrue(any('parent-source/' in k for k in paths))
        text=(self.root/'driver/experiments/rl/reliability_campaign.py').read_text()
        self.assertIn("code.rglob('*')",text)

if __name__=='__main__':unittest.main()
