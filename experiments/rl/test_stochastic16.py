"""Offline categorical16 math, source identity and three-audit integration tests."""
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import gymnasium as gym
import numpy as np
import torch
from lupa.lua54 import LuaRuntime
from stable_baselines3 import PPO
from astra_play_sf2.runner import sha256
from .actions16_builder import build as build16
from .stochastic16_builder import build, PACKAGE


class Spaces(gym.Env):
    observation_space=gym.spaces.Box(-1,1,(344,),dtype=np.float32)
    action_space=gym.spaces.Discrete(16)


class Stochastic16Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name)
        build16(cls.root/'parent');cls.parent=cls.root/'parent/astra_sf2_rl16'
        cls.before={p.name:sha256(p) for p in cls.parent.iterdir() if p.is_file()}
        cls.manifest=build(cls.parent,cls.root/'candidate',42);cls.pkg=cls.root/'candidate'/PACKAGE
        sys.path.insert(0,str(cls.root/'candidate'))
        cls.export=importlib.import_module(PACKAGE+'.export')
        cls.native=importlib.import_module(PACKAGE+'.native_continuous')
        cls.support=importlib.import_module(PACKAGE+'.support')
        cls.sampling=importlib.import_module(PACKAGE+'.sampling')
        model=PPO('MlpPolicy',Spaces(),n_steps=256,seed=8,policy_kwargs={'net_arch':{'pi':[64,64],'vf':[64,64]}})
        model.astra_action_interface='ken_actions16_lp_mp_uppercut_v1'
        cls.model=cls.root/'fixed-model.zip';model.save(cls.model);cls.model_hash=sha256(cls.model)
        cls.payload=cls.export.export_policy(cls.model)
        cls.state={'p1':{'hp':144,'x':100,'y':40,'a':0,'char':4},
                   'p2':{'hp':144,'x':200,'y':40,'a':0,'char':2},'timer':99,'emulated_seconds':0.}

    @classmethod
    def tearDownClass(cls):
        sys.path.remove(str(cls.root/'candidate'))
        for name in list(sys.modules):
            if name==PACKAGE or name.startswith(PACKAGE+'.'):sys.modules.pop(name)
        cls.temp.cleanup()

    def lua_policy(self):
        lua=LuaRuntime(unpack_returned_tuples=True)
        base=lua.execute((self.pkg/'nn.lua').read_text())
        sampling=lua.execute((self.pkg/'stochastic_nn.lua').read_text())(base)
        actions=lua.execute((self.pkg/'actions.lua').read_text())
        payload=lua.execute('return '+self.export.lua_literal(self.payload))
        return lua,base,sampling,actions,payload

    def test_parent_weights_macros_unchanged_and_candidate_identity_explicit(self):
        self.assertEqual(self.before,{p.name:sha256(p) for p in self.parent.iterdir() if p.is_file()})
        self.assertEqual(self.model_hash,sha256(self.model))
        self.assertEqual((self.pkg/'actions.lua').read_bytes(),(self.parent/'actions.lua').read_bytes())
        self.assertEqual(self.payload['selection'],'categorical_softmax')
        self.assertEqual(self.payload['actions'],16);self.assertEqual(self.payload['policy_seed'],42)
        self.support.capture_interface(self.model,self.pkg)
        for module in ('batch_train','native_campaign'):
            with self.assertRaisesRegex(RuntimeError,'verification-only'):
                importlib.import_module(PACKAGE+'.'+module).main()

    def test_categorical_lua_matches_torch_without_false_argmax_identity(self):
        lua,base,sampling,actions,payload=self.lua_policy()
        self.assertNotIn('logits_model',(self.pkg/'stochastic_nn.lua').read_text())
        state=42;rng=np.random.default_rng(7)
        lua.execute('math.random=function() error("global RNG forbidden") end;math.randomseed=math.random')
        for _ in range(100):
            state,u=self.sampling.next_random(state)
            logits=rng.normal(0,4,16);prob=torch.softmax(torch.tensor(logits),dim=0).numpy()
            selected=int(np.searchsorted(np.cumsum(prob),u,side='right'))
            action,logp=sampling.sample(lua.table_from(logits.tolist()),u)
            self.assertEqual(action,selected)
            self.assertAlmostEqual(logp,float(torch.log_softmax(torch.tensor(logits),dim=0)[selected]),places=12)
        policy=sampling.new(payload,actions);state=42
        for reset in (True,False,True,True,True):
            seq,_=policy.choose(policy,lua.table_from(self.state,recursive=True),reset)
            state,_=self.sampling.next_random(state)
            self.assertEqual(policy.rng_state,state);self.assertEqual(len(seq),12)
        self.assertEqual(policy.draws,5)
        payload.selection='deterministic_argmax'
        with self.assertRaises(Exception):sampling.new(payload,actions)
        with self.assertRaises(Exception):base.predict(payload,lua.table_from([0.]*344))

    def test_three_audits_pass_and_sampling_reset_or_dependency_tamper_is_invalid(self):
        for case in (None,'reset','dependency'):
            with self.subTest(case=case):
                output=self.root/('evaluate-'+str(case));dependency=self.pkg/'sampling.py';before=dependency.read_bytes()
                lua,base,sampling,actions,payload=self.lua_policy();policy=sampling.new(payload,actions)
                def fake_native(config,model,run,*args,**kwargs):
                    run.mkdir();hashes=self.native.stage_native_policy(run,3,self.payload)
                    result={'schema':'astra.rl-continuous.actions16.v1','status':'complete',
                        'action_interface':self.payload['action_interface'],'actions':16,'model_sha256':self.model_hash,
                        'runtime_sha256':hashes,'attempts':[]}
                    count=0
                    for coin in range(2):
                        decisions=[]
                        for round_,frame in ((1,0),(2,12)):
                            _,reason=policy.choose(policy,lua.table_from(self.state,recursive=True),True)
                            if case=='reset' and coin==1:reason=reason.replace('policy_draw=3','policy_draw=1')
                            decisions.append({'frame':frame,'round':round_,'reason':reason,
                                              'ken':self.state['p1'],'cpu':self.state['p2']})
                        summary={'schema':'mame.rl-continuous-play.actions16.v1','action_interface':self.payload['action_interface'],
                                 'model_sha256':self.model_hash,'policy_kind':'ppo','native_timing':True,
                                 'frame':24,'native_frame_period':1/60,'native_period_checks':24,
                                 **{k:self.payload[k] for k in ('selection','policy_seed','policy_prng')}}
                        match={'summary':summary,'trace':{'columns':['frame','timer','emulated_seconds'],
                            'rows':[[f,99,f/60] for f in range(1,25)],'decisions':decisions},
                            'events':[{'kind':'match_start','state':self.state},{'kind':'round_stop','round':1,'frame':12},
                                      {'kind':'round_start','round':2,'frame':12},{'kind':'round_stop','round':2,'frame':24}]}
                        count+=self.native.audit_native_match(match)['decisions']
                        relative=f'match-{coin}.json';(run/relative).write_text(json.dumps(match))
                        result['attempts'].append({'matches':[relative],'outcome':'loss'})
                    result['native_timing_audit']={'ok':True,'decisions':count}
                    if case=='dependency':dependency.write_bytes(before+b'\n# tampered')
                    return result
                try:
                    with patch.object(self.native,'_evaluate_native',fake_native):
                        result=self.native.evaluate(None,self.model,output)
                    self.assertEqual(result['status'],'complete' if case is None else 'invalid')
                    if case is None:
                        for name in ('action_interface_audit','native_timing_audit','sampling_audit'):
                            self.assertTrue(result[name]['ok'])
                        self.assertEqual(result['sampling_audit']['decisions'],4)
                        self.assertEqual(result['action_interface_audit']['selection'],'categorical_softmax')
                        self.assertIn('rl_stochastic_nn.lua',result['runtime_sha256'])
                    else:self.assertFalse(result['sampling_audit']['ok'])
                finally:dependency.write_bytes(before)

    def test_parent_manifest_tamper_and_invalid_seeds_are_rejected(self):
        for seed in (0,-1,2147483647,True):
            with self.assertRaises(ValueError):build(self.parent,self.root/'unused',seed)
        parent=self.root/'candidate/parent-build.json';before=parent.read_bytes()
        try:
            parent.write_bytes(before+b' ')
            with self.assertRaisesRegex(RuntimeError,'parent manifest changed'):
                self.support.capture_interface(self.model,self.pkg)
        finally:parent.write_bytes(before)


if __name__=='__main__':unittest.main()
