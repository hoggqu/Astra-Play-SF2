"""Portable observer/model/recorder integration, without starting an emulator."""
import ast
import copy
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch
from lupa import LuaRuntime, lua_type

from .perception_builder import bootstrap
from .perception_identity import OBSERVATIONS, FRAME_FEATURES, validate_build
from .full_interface_identity import INPUT_NAMES as V1_INPUT_NAMES

HERE=Path(__file__).resolve().parent


def python_value(value):
    if lua_type(value)!='table':return value
    keys=list(value.keys())
    if keys and set(keys)==set(range(1,len(keys)+1)):
        return [python_value(value[i]) for i in range(1,len(keys)+1)]
    return {k:python_value(value[k]) for k in keys}


class PerceptionIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp=tempfile.TemporaryDirectory();cls.work=Path(cls.temp.name)
        cls.v1_inputs={n:(HERE/n).read_bytes() for n in (*V1_INPUT_NAMES,'full_interface_builder.py')}
        cls.code=cls.work/'code';cls.manifest=bootstrap(cls.code)
        cls.package=cls.code/cls.manifest['package'];sys.path.insert(0,str(cls.code))
        cls.modules={n:importlib.import_module(cls.manifest['package']+'.'+n)
                     for n in ('initialize','export','support','env','perception_features','batch_train','native_continuous')}
        cls.model=cls.modules['initialize'].create_model(852)
        cls.model_path=cls.work/'model.zip';cls.model.save(cls.model_path)

    @classmethod
    def tearDownClass(cls):
        sys.path.remove(str(cls.code))
        for name in list(sys.modules):
            if name==cls.manifest['package'] or name.startswith(cls.manifest['package']+'.'):
                del sys.modules[name]
        cls.temp.cleanup()

    def lua(self):
        lua=LuaRuntime(unpack_returned_tuples=True)
        def source(name):
            name=Path(name).name
            if name.startswith('rl_'):name=name[3:]
            if name=='play_core.lua':
                return (self.code/'src/astra_play_sf2/assets/play_core.lua').read_text()
            return (self.package/name).read_text()
        lua.globals().read_source=source
        lua.execute("loadfile=function(name) return assert(load(read_source(name))) end")
        return lua

    def unknown_state(self,lua):
        # No emulator globals means the screen reader must report unknown,
        # rather than constructing a visible scene from hidden actor fields.
        lua.execute('''memory={read_u8=function()return 0 end,read_u16=function()return 0 end,
 read_i16=function()return 0 end,read_u32=function()return 0 end}
A=assert(loadfile('rl_actions.lua'))();F=assert(loadfile('rl_fighter_perception.lua'))()
P=assert(loadfile('rl_visible_projectiles.lua'))();E=assert(loadfile('rl_perception_execution.lua'))()
N=assert(loadfile('rl_perception_features.lua'))()
s={p1={x=100,y=40,hp=144,displayed_hp=144,a=0,anim=1,char=4,wins=0},
 p2={x=220,y=40,hp=144,displayed_hp=144,a=0,anim=1,char=0,wins=0},
 timer=99,emulated_seconds=0,native_frame_period=1/60}
f=F.new();p=P.new();e=E.new(A)
F.read(memory,s);f:reset(s);p:reset(s,P.read(memory,s));e:reset(s)
''')
        return python_value(lua.globals().s)

    def test_visible_feature_python_lua_and_hidden_field_independence(self):
        lua=self.lua();state=self.unknown_state(lua)
        encode=self.modules['perception_features'].features
        original=encode(state)
        self.assertEqual(len(original),FRAME_FEATURES)
        np.testing.assert_allclose(original,list(lua.globals().N.features(lua.globals().s).values()),atol=1e-7)
        changed=copy.deepcopy(state)
        for key in ('p1','p2'):
            changed[key].update(hp=-999,a=255,anim=987654,next_move=42,remaining_hitstun=300,
                                hidden_dizzy_counter=255,intent=81,x=-4000,y=9999,facing_render=99)
        np.testing.assert_array_equal(original,encode(changed))
        changed['p1']['displayed_hp']=72
        self.assertEqual(encode(changed)[0],.5)
        changed['p1']['displayed_hp']=-1
        self.assertEqual(encode(changed)[0],0)
        visible=copy.deepcopy(state)
        visible['fighter_perception']['p1'].update(position_known=True,visible_x=96,visible_y=112)
        visible['fighter_perception']['p2'].update(position_known=True,visible_x=288,visible_y=112)
        values=encode(visible)
        np.testing.assert_array_equal(values[3:8],np.asarray([.5,.25,.75,.5,.5],dtype=np.float32))

    def test_runtime_rejects_unavailable_global_render_reader(self):
        lua=self.lua();self.unknown_state(lua)
        lua.globals().mem=lua.globals().memory
        core=lua.execute((self.package/'native_continuous_core.lua').read_text())
        with self.assertRaisesRegex(Exception,'unsupported-render-reader'):
            core.new(lua.table_from({'opponent':0}),lua.globals().s)

    def test_new_model_rejects_old_identity_and_lua_logits_match(self):
        payload=self.modules['export'].export_policy(self.model_path)
        self.assertEqual((payload['observations'],payload['actions']),(OBSERVATIONS,85))
        saved=self.model.astra_observation_interface
        try:
            self.model.astra_observation_interface='sf2_state86_visible_feedback114_history4_v1'
            with self.assertRaises(ValueError):self.modules['support'].validate_model(self.model)
        finally:self.model.astra_observation_interface=saved
        lua=self.lua();nn=lua.execute((self.package/'nn.lua').read_text())
        literal=self.modules['export'].lua_literal;model=lua.execute('return '+literal(payload))
        rng=np.random.default_rng(19);obs=rng.uniform(-1,1,(4,OBSERVATIONS)).astype(np.float32)
        with torch.no_grad():
            latent=self.model.policy.mlp_extractor.forward_actor(torch.as_tensor(obs))
            expected=self.model.policy.action_net(latent).numpy()
        for i in range(4):
            action,logits=nn.predict(model,lua.table_from(obs[i].tolist()))
            np.testing.assert_allclose(list(logits.values()),expected[i],atol=3e-6)
            self.assertEqual(action,int(expected[i].argmax()))

    def test_real_ppo_update_save_export_dimensions(self):
        model=self.modules['initialize'].create_model(853);model._setup_learn(256)
        rng=np.random.default_rng(20);obs=rng.uniform(-1,1,(256,OBSERVATIONS)).astype(np.float32)
        with torch.no_grad():a,v,lp=model.policy(torch.as_tensor(obs))
        before=model.policy.action_net.weight.detach().clone()
        for i in range(256):
            model.rollout_buffer.add(obs[i:i+1],a[i:i+1].numpy(),np.asarray([(-1.)**i],np.float32),
                                     np.asarray([i==0]),v[i:i+1],lp[i:i+1])
        model.rollout_buffer.compute_returns_and_advantage(torch.zeros(1),np.asarray([True]));model.train()
        self.assertFalse(torch.equal(before,model.policy.action_net.weight))
        self.assertGreater(len(model.policy.optimizer.state),0)
        path=self.work/'optimized.zip';model.save(path)
        self.assertEqual(self.modules['export'].export_policy(path)['observations'],OBSERVATIONS)

    def test_stage_all_observer_dependencies_and_capture_fields(self):
        run=self.work/'stage';run.mkdir()
        snapshot=self.modules['support'].capture_interface(self.model_path,self.package)
        hashes=self.modules['native_continuous'].stage_native_policy(run,3,snapshot['payload'])
        self.modules['support'].check_staged_interface(snapshot,run,hashes)
        lua=LuaRuntime()
        for name in hashes:
            if name.endswith('.lua'):lua.execute('assert(load(...))',(run/'training/runtime'/name).read_text())
        text=(run/'training/runtime/play.lua').read_text()
        for field in ('fighter_perception','visible_projectiles'):
            self.assertIn('copy(assert(s.'+field+'))',text)
        for name in ('rl_fighter_perception.lua','rl_fighter_animation_map.lua','rl_fighter_render.lua','rl_visible_projectiles.lua','rl_visible_sprite_buffer.lua','rl_perception_execution.lua','rl_perception_features.lua'):
            self.assertIn(name,hashes)

    def test_reward_actions_and_parent_unchanged_and_tamper_fails(self):
        for name,body in self.v1_inputs.items():self.assertEqual((HERE/name).read_bytes(),body)
        parent=self.code/'perception-parent/astra_sf2_rl_full85'
        self.assertEqual((parent/'actions.lua').read_bytes(),(self.package/'actions.lua').read_bytes())
        def function(path,name):
            return ast.dump(next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name))
        self.assertEqual(function(parent/'env.py','reward'),function(self.package/'env.py','reward'))
        target=self.package/'fighter_animation_map.lua';body=target.read_bytes()
        try:
            target.write_bytes(body+b'\n-- tamper')
            with self.assertRaises(RuntimeError):validate_build(self.manifest,self.package)
        finally:target.write_bytes(body)

    def test_relocated_self_contained_package(self):
        clone=self.work/'clean checkout';clone.mkdir()
        for folder in ('src','experiments'):
            shutil.copytree(HERE.parents[1]/folder,clone/folder,
                            ignore=shutil.ignore_patterns('__pycache__','*.pyc','.local','.git','MAME','skills'))
        env=dict(os.environ,PYTHONPATH=os.pathsep.join((str(clone/'src'),str(clone))),PYTHONNOUSERSITE='1')
        output=self.work/'public build'
        done=subprocess.run([sys.executable,'-m','experiments.rl.perception_builder','--output',str(output)],
                            cwd=clone,env=env,capture_output=True,text=True,timeout=60)
        self.assertEqual(done.returncode,0,done.stdout+done.stderr)
        self.assertFalse((clone/'.local').exists())
        shutil.rmtree(clone)
        relocated=self.work/'relocated public build';output.rename(relocated)
        env['PYTHONPATH']=os.pathsep.join((str(relocated/'src'),str(relocated)))
        code="""from pathlib import Path
import json
import astra_play_sf2
from astra_sf2_rl_perception.perception_identity import validate_build
root=Path.cwd();assert Path(astra_play_sf2.__file__).resolve().is_relative_to(root/'src')
validate_build(json.loads((root/'build.json').read_text()),root/'astra_sf2_rl_perception')
"""
        done=subprocess.run([sys.executable,'-c',code],cwd=relocated,env=env,capture_output=True,text=True,timeout=60)
        self.assertEqual(done.returncode,0,done.stdout+done.stderr)


if __name__=='__main__':unittest.main()
