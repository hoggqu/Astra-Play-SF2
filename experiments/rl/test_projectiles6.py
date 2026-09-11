"""No-emulator checks of the candidate interface, migration and staged source."""
import copy
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from lupa.lua54 import LuaRuntime
from astra_play_sf2.runner import sha256
from . import projectile_migrate as migration
from .actions16_builder import build as build16
from .round_chain_builder import build as buildchain
from .projectile_builder import build, PACKAGE
from .projectile_identity import validate_projectile_build
from .export import lua_literal
HERE=Path(__file__).resolve().parent

class OldSpace(gym.Env):
    observation_space=gym.spaces.Box(-1,1,(344,),dtype=np.float32)
    action_space=gym.spaces.Discrete(16)

class ProjectilesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name)
        build16(cls.root/'parent');buildchain(cls.root/'chain',cls.root/'parent/astra_sf2_rl16')
        cls.manifest=build(cls.root/'chain/astra_sf2_rl_round_chain',cls.root/'candidate')
        cls.package=cls.root/'candidate'/PACKAGE
        sys.path.insert(0,str(cls.root/'candidate'))
        cls.support=importlib.import_module(PACKAGE+'.support')
        cls.export=importlib.import_module(PACKAGE+'.export')
        cls.native=importlib.import_module(PACKAGE+'.native_continuous')
        cls.env=importlib.import_module(PACKAGE+'.env')
        cls.old=PPO('MlpPolicy',OldSpace(),seed=91,n_steps=8,batch_size=4,
                    policy_kwargs={'net_arch':{'pi':[64,64],'vf':[64,64]}})
        cls.old.astra_action_interface=migration.ACTION_INTERFACE
        # Seed real Adam entries without rollouts or an emulator.
        sum((p.square().sum() for p in cls.old.policy.parameters())).backward()
        cls.old.policy.optimizer.step();cls.old.policy.optimizer.zero_grad()
        cls.new=migration.migrate_model(cls.old)
        cls.model_path=cls.root/'expanded.zip';cls.new.save(cls.model_path)

    @classmethod
    def tearDownClass(cls):
        sys.path.remove(str(cls.root/'candidate'))
        for name in list(sys.modules):
            if name==PACKAGE or name.startswith(PACKAGE+'.'):del sys.modules[name]
        cls.tmp.cleanup()

    def lua(self):
        lua=LuaRuntime(unpack_returned_tuples=True)
        return lua,lua.execute((HERE/'projectile_features.lua').read_text())

    def state(self):
        return {'p1':{'x':500,'y':40,'hp':110,'a':12,'char':4},'p2':{'x':650,'y':40,'hp':90,'a':8,'char':3},'timer':81}

    def object(self,owner=0x86c6,kind=3,slot=6,**kw):
        return dict({'slot':slot,'base':0xff938a+0xc0*slot,'status':257,'hp':256,'type':kind,'owner':owner,'x':650,'y':90},**kw)

    def encode(self,objects,links=None):
        lua,mod=self.lua();links=links or [0,objects[0]['base']%65536]
        out,diag=mod.encode(lua.table_from(self.state(),recursive=True),lua.table_from(objects,recursive=True),lua.table_from(links))
        return list(out.values()),dict(diag)

    def test_type_owner_status_hp_and_zero_absence(self):
        for owner in (0x83c6,0x86c6):
            for kind in (0,1,3,4):
                o=self.object(owner,kind);links=[o['base']%65536 if owner==0x83c6 else 0,o['base']%65536 if owner==0x86c6 else 0]
                out,_=self.encode([o],links)
                expected=[1,150/512,50/256]
                self.assertEqual(out,expected+[0,0,0] if owner==0x83c6 else [0,0,0]+expected)
        for kw in ({'type':2},{'hp':255},{'hp':0},{'status':256},{'owner':0},{'slot':8},{'x':1.5}):
            out,_=self.encode([self.object(**kw)])
            self.assertEqual(out,[0]*6)
        self.assertEqual(self.encode([self.object()],links=[0,0])[0],[0]*6)

    def test_two_owners_and_clipping(self):
        a=self.object(owner=0x83c6,slot=6,x=-1000,y=-1000);b=self.object(slot=7,x=2000,y=2000)
        out,_=self.encode([a,b],[a['base']%65536,b['base']%65536]);self.assertEqual(out,[1,-1,-1,1,1,1])
        # Reciprocal owner link admits at most one actual slot per owner.
        out,diag=self.encode([a,self.object(owner=0x83c6,slot=7,x=510)],[a['base']%65536,0])
        self.assertEqual(diag['own_slot'],6);self.assertEqual(out[:3],[1,-1,-1])

    def test_read_uses_exact_u16_reciprocal_addresses(self):
        lua,mod=self.lua();s=lua.table_from(self.state(),recursive=True)
        b=0xff938a+6*0xc0
        lua.globals().ram=lua.table_from({b:257,b+0x2a:256,b+0x20:3,b+0x26:0x86c6,b+6:650,b+10:90,0xff86c6+0x1d4:b%65536})
        mem=lua.execute('return {read_u16=function(_,a)return ram[a] or 0 end,read_i16=function(_,a)return ram[a] or 0 end,read_u8=function(_,a)return ram[a] or 0 end}')
        out,_=mod.read(mem,s);self.assertEqual(list(out.values()),[0,0,0,1,150/512,50/256])

    def test_weights_and_named_adam_are_exactly_mapped(self):
        new_weights=self.new.policy.state_dict()
        for name,value in self.old.policy.state_dict().items():
            expected=migration.expand(value) if name in migration.FIRST else value
            self.assertTrue(torch.equal(new_weights[name],expected),name)
        new_params=dict(self.new.policy.named_parameters())
        for name,param in self.old.policy.named_parameters():
            for key,value in self.old.policy.optimizer.state[param].items():
                expected=migration.expand(value) if name in migration.FIRST and value.ndim>0 else value
                self.assertTrue(torch.equal(self.new.policy.optimizer.state[new_params[name]][key],expected),name+'/'+key)
        self.assertEqual(self.new.num_timesteps,self.old.num_timesteps)
        self.assertEqual(self.new._n_updates,self.old._n_updates)

    def test_float_outputs_argmax_and_save_reload(self):
        rng=np.random.default_rng(4);x=rng.uniform(-1,1,(37,344)).astype(np.float32)
        self.assertTrue(migration.compare(self.old,PPO.load(self.model_path),x,rng.uniform(-1,1,(37,4,6)))['argmax_equal'])
        tied=copy.deepcopy(self.new)
        with torch.no_grad():tied.policy.action_net.bias[0]+=100
        with self.assertRaises(RuntimeError):migration.compare(self.old,tied,x)

    def test_adam_amsgrad_and_serialization_preserve_each_state(self):
        old=copy.deepcopy(self.old)
        for group in old.policy.optimizer.param_groups:group['amsgrad']=True
        for state in old.policy.optimizer.state.values():state['max_exp_avg_sq']=state['exp_avg_sq'].clone()+.001
        new=migration.migrate_model(old);path=self.root/'amsgrad.zip';new.save(path)
        loaded=PPO.load(path);params=dict(loaded.policy.named_parameters())
        for name,p in old.policy.named_parameters():
            for key,value in old.policy.optimizer.state[p].items():
                expected=migration.expand(value) if name in migration.FIRST and value.ndim else value
                self.assertTrue(torch.equal(loaded.policy.optimizer.state[params[name]][key],expected),name+'/'+key)
        self.assertEqual(old.policy.optimizer.param_groups[0]['eps'],loaded.policy.optimizer.param_groups[0]['eps'])
        self.assertTrue(loaded.policy.optimizer.param_groups[0]['amsgrad'])

    def test_lua_actor_value_and_candidate_forward(self):
        from .projectile_acceptance import lua_check
        x=np.random.default_rng(4).uniform(-1,1,(12,344)).astype(np.float32)
        result=lua_check(self.old,self.new,self.package,x)
        self.assertTrue(result['lua_old_new_exact']);self.assertTrue(result['argmax_equal'])

    def test_staging_hashes_shared_reader_and_model_identity(self):
        payload=self.export.export_policy(self.model_path)
        self.assertEqual(payload['observation_interface'],migration.INTERFACE)
        snapshot=self.support.capture_interface(self.model_path,self.package)
        run=self.root/'staged';run.mkdir()
        hashes=self.native.stage_native_policy(run,3,payload)
        self.support.check_staged_interface(snapshot,run,hashes)
        self.assertEqual(hashes['rl_projectiles.lua'],sha256(HERE/'projectile_features.lua'))
        source=(run/'training/runtime/play.lua').read_text()
        self.assertIn('s.projectile_observation=Projectiles.read(mem,s)',source)
        lua=LuaRuntime()
        for path in (run/'training/runtime').glob('*.lua'):lua.execute('assert(load(...))',path.read_text())
        for path in self.package.glob('*.lua'):lua.execute('assert(load(...))',path.read_text())
        self.assertIn('observation_interface=Model.observation_interface',source)
        invalid=copy.deepcopy(payload);invalid['observation_interface']='old'
        nn=lua.execute((self.package/'nn.lua').read_text())
        with self.assertRaises(Exception):nn.predict(lua.table_from(invalid,recursive=True),lua.table_from([0]*368))

    def test_shared_feature_history_layout(self):
        s=self.state();s['projectile_observation']=[1,.2,.3,0,0,0]
        lua=LuaRuntime(unpack_returned_tuples=True);nn=lua.execute((self.package/'nn.lua').read_text())
        actual=list(nn.features(lua.table_from(s,recursive=True)).values())
        np.testing.assert_allclose(actual,self.env.features(s),atol=1e-7)
        self.assertEqual(actual[86:],s['projectile_observation']);self.assertEqual(len(actual),92)

    def test_provenance_and_unchanged_actions_core_reward_sampler(self):
        validate_projectile_build(self.manifest,self.package)
        parent=self.root/'chain/astra_sf2_rl_round_chain'
        for n in ('actions.lua','native_continuous_core.lua','settlement.lua','dataset.py','batch_env.py'):
            self.assertEqual((parent/n).read_bytes(),(self.package/n).read_bytes())
        old=(parent/'batch_runtime.lua').read_text();new=(self.package/'batch_runtime.lua').read_text()
        self.assertEqual(new.replace("local Projectiles=assert(loadfile('training/runtime/rl_projectiles.lua'))()\n",'').replace(' s.projectile_observation=Projectiles.read(mem,s)\n',''),old)
        manifest=copy.deepcopy(self.manifest);manifest['observations']=344
        with self.assertRaises(RuntimeError):validate_projectile_build(manifest,self.package)
        with self.assertRaises(ValueError):self.support.validate_model(self.old)

if __name__=='__main__':unittest.main()
