"""Offline integration checks. No emulator or game training is started."""
import importlib
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import sys

import numpy as np
from lupa import LuaRuntime
import torch

from .full_interface_bootstrap import build
from .full_interface_identity import validate_build

ROOT = Path(__file__).resolve().parents[2]
class FullInterfaceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)/'code'
        cls.manifest = build(cls.root)
        cls.package = cls.root/'astra_sf2_rl_full85'
        sys.path.insert(0, str(cls.root))
        cls.modules = {n:importlib.import_module('astra_sf2_rl_full85.'+n)
                       for n in ('initialize','export','support','env','batch_train','native_continuous','dataset')}
        cls.model = cls.modules['initialize'].create_model(851)
        cls.zip = Path(cls.temp.name)/'model.zip'
        cls.model.save(cls.zip)

    @classmethod
    def tearDownClass(cls):
        sys.path.remove(str(cls.root))
        for n in list(sys.modules):
            if n == 'astra_sf2_rl_full85' or n.startswith('astra_sf2_rl_full85.'):
                del sys.modules[n]
        cls.temp.cleanup()

    def lua(self):
        lua = LuaRuntime(unpack_returned_tuples=True)
        mapping = {'rl_actions.lua':'actions.lua','rl_nn.lua':'nn.lua',
                   'rl_visible_feedback.lua':'visible_feedback.lua',
                   'rl_visible_animation_map.lua':'visible_animation_map.lua'}
        lua.globals().read_source = lambda name:(self.package/mapping[name.split('/')[-1]]).read_text()
        lua.execute('loadfile=function(name) return assert(load(read_source(name))) end')
        return lua

    def test_export_shapes_and_old_identity_rejected(self):
        payload = self.modules['export'].export_policy(self.zip)
        self.assertEqual((payload['observations'],payload['actions']), (800,85))
        self.assertEqual(len(payload['layers'][0]['weight'][0]),800)
        self.assertEqual(len(payload['layers'][-1]['bias']),85)
        old = self.model.astra_observation_interface
        try:
            self.model.astra_observation_interface = 'old'
            with self.assertRaises(ValueError): self.modules['support'].validate_model(self.model)
        finally: self.model.astra_observation_interface = old

    def test_policy_lua_torch_and_real_optimizer_update(self):
        model = self.modules['initialize'].create_model(852)
        model._setup_learn(256)
        rng = np.random.default_rng(11)
        observations = rng.uniform(-1,1,(256,800)).astype(np.float32)
        with torch.no_grad():
            action, values, logp = model.policy(torch.as_tensor(observations))
        payload = self.modules['batch_train'].live_policy(model)
        lua = self.lua(); nn = lua.execute((self.package/'nn.lua').read_text())
        literal = self.modules['export'].lua_literal
        lp = lua.execute('return '+literal(payload))
        with torch.no_grad():
            latent = model.policy.mlp_extractor.forward_actor(torch.as_tensor(observations[:3]))
            logits = model.policy.action_net(latent).numpy()
        for i in range(3):
            chosen, actual = nn.predict(lp, lua.table_from(observations[i].tolist()))
            np.testing.assert_allclose(list(actual.values()), logits[i], atol=2e-6)
            self.assertEqual(chosen, int(logits[i].argmax()))
        before = model.policy.action_net.weight.detach().clone()
        for i in range(256):
            model.rollout_buffer.add(observations[i:i+1], action[i:i+1].numpy(),
                np.asarray([(-1.)**i],np.float32), np.asarray([i==0]), values[i:i+1], logp[i:i+1])
        model.rollout_buffer.compute_returns_and_advantage(torch.zeros(1), np.asarray([True]))
        model.train()
        self.assertFalse(torch.equal(before,model.policy.action_net.weight))
        self.assertGreater(len(model.policy.optimizer.state),0)
        out = Path(self.temp.name)/'optimized.zip'; model.save(out)
        self.assertEqual(self.modules['export'].export_policy(out)['actions'],85)

    def test_feature_encoding_and_no_same_step_label_leak(self):
        lua = self.lua()
        lua.execute("A=assert(loadfile('rl_actions.lua'))();F=assert(loadfile('rl_visible_feedback.lua'))();N=assert(loadfile('rl_nn.lua'))();observer=F.new(A)")
        literal = self.modules['export'].lua_literal
        state = {'p1':{'hp':144,'x':100,'y':40,'a':0,'anim':1,'char':4},
                 'p2':{'hp':144,'x':220,'y':40,'a':0,'anim':1,'char':0},'timer':99}
        lua.execute('s='+literal(state)+';observer:reset(s);before=s.visible_feedback;observer:request(81,s)')
        self.assertEqual(lua.globals().s.visible_feedback.request_action,-1)
        lua.execute('observer:tick(s)')
        self.assertEqual(lua.globals().s.visible_feedback.request_action,81)
        self.assertEqual(lua.globals().before.request_action,-1)
        feedback = dict(lua.globals().s.visible_feedback.items());state['visible_feedback']=feedback
        expected = self.modules['env'].features(state)
        actual = list(lua.globals().N.features(lua.globals().s).values())
        np.testing.assert_allclose(actual, expected, atol=1e-7)
        self.assertEqual(len(actual),200)
        lua.execute('observer:reset(s)')
        self.assertEqual(lua.globals().s.visible_feedback.request_action,-1)

    def test_staged_native_dependencies_and_recording(self):
        run = Path(self.temp.name)/'stage'; run.mkdir()
        snapshot = self.modules['support'].capture_interface(self.zip,self.package)
        hashes = self.modules['native_continuous'].stage_native_policy(run,3,snapshot['payload'])
        self.modules['support'].check_staged_interface(snapshot,run,hashes)
        lua = LuaRuntime()
        for name in hashes:
            if name.endswith('.lua'):
                lua.execute('assert(load(...))', (run/'training/runtime'/name).read_text())
        text = (run/'training/runtime/play.lua').read_text()
        self.assertIn('visible_feedback=copy(assert(s.visible_feedback))',text)
        self.assertIn('return seq,reason,action',text)
        self.assertIn("trace_record(r,'after',effects,r.latest)",text)
        self.assertEqual(hashes['rl_visible_feedback.lua'],self.manifest['derived_sha256']['visible_feedback.lua'])

    def test_native_monitor_shared_hooks_and_round_reset(self):
        lua = self.lua()
        # Original Core behavior is a minimal spy, so this checks the wrapper's
        # ordering independently of the observer implementation and game logic.
        lua.execute('''Base={};Base.__index=Base
function Base.new(options,s) return setmetatable({round=1,choose=options.choose},Base) end
function Base:tick(s) assert(s.visible_feedback.request_age>=0);return {} end
function Base:begin_round(s,e) assert(s.visible_feedback.request_action==-1);return e end
oldloadfile=loadfile
loadfile=function(name)
 if name:find('play_core.lua') then return function() return Base end end
 if name:find('rl_settlement.lua') then return function() return function(c)return c end end end
 return oldloadfile(name)
end''')
        lua.globals().core_source = (self.package/'native_continuous_core.lua').read_text()
        lua.execute('''Core=assert(load(core_source))()
s={p1={hp=144,x=100,y=40,a=0,anim=1,char=4},p2={hp=144,x=220,y=40,a=0,anim=1,char=0},timer=99}
core=Core.new({choose=function(a,b,m,state,reset)
 assert(state.visible_feedback.request_action==-1)
 local seq={};for i=1,12 do seq[i]={1,''} end;return seq,'test',81
end},s)
core:rl_prime(s)
assert(s.visible_feedback.request_action==-1)
core:tick(s);assert(s.visible_feedback.request_action==81)
core:begin_round(s,{})
assert(s.visible_feedback.request_action==-1)
''')

    def test_tampering_fails_and_reward_unchanged(self):
        import ast
        old = ast.parse((self.root/'parent-source/astra_sf2_rl_round_chain/env.py').read_text())
        new = ast.parse((self.package/'env.py').read_text())
        select = lambda t:ast.dump(next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='reward'))
        self.assertEqual(select(old),select(new))
        target = self.package/'nn.lua'; original = target.read_bytes()
        try:
            target.write_bytes(original+b'\n--tamper')
            with self.assertRaises(RuntimeError):validate_build(self.manifest,self.package)
        finally: target.write_bytes(original)

    def test_all_op_dataset_and_split_rejection(self):
        folder = Path(self.temp.name)/'dataset'; folder.mkdir()
        opponents = [0,1,2,3,5,6,7,8,9,10,11]
        data = {'schema':'astra.rl-openings.v2','status':'complete','difficulty':3,
                'opponents':opponents,'openings':[]}
        for op in opponents:
            for i in range(10):
                body = f'{op}:{i}'.encode(); name = f'{op}-{i}.sta'
                (folder/name).write_bytes(body)
                data['openings'].append({'id':name,'path':name,'status':'accepted',
                    'sha256':hashlib.sha256(body).hexdigest(), 'opponent':op,'difficulty':3,
                    'split':'train' if i<8 else 'dev' if i==8 else 'holdout'})
        manifest = folder/'manifest.json'; manifest.write_text(json.dumps(data))
        groups, level = self.modules['dataset'].load_dataset(manifest)
        self.assertEqual(level,3)
        self.assertEqual({k:len(v) for k,v in groups.items()},{'train':88,'dev':11,'holdout':11})
        data['openings'][8]['split']='train'; manifest.write_text(json.dumps(data))
        with self.assertRaises(ValueError):self.modules['dataset'].load_dataset(manifest)
        data['openings'][8]['split']='dev'
        data['openings'][8]['sha256']=data['openings'][0]['sha256']
        manifest.write_text(json.dumps(data))
        with self.assertRaises(ValueError):self.modules['dataset'].load_dataset(manifest)

    def test_complete_source_capture_includes_observer_mapping(self):
        for name in ('visible_feedback.lua','visible_animation_map.lua','full_actions.py','dataset.py'):
            self.assertIn(name,self.manifest['derived_sha256'])
        target = self.root/'integration-inputs/visible_animation_map.lua'
        old = target.read_bytes()
        try:
            target.write_bytes(old+b'\n-- changed mapping')
            with self.assertRaises(RuntimeError):validate_build(self.manifest,self.package)
        finally:target.write_bytes(old)


if __name__ == '__main__': unittest.main()
