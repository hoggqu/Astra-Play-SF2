"""Architecture-only portability and actual Lua/Torch inference checks."""
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
from lupa import LuaRuntime
from .perception128_builder import bootstrap
from .perception128_identity import validate_build, ARCHITECTURE


class Perception128Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name)/'code'
        cls.manifest=bootstrap(cls.root);cls.package=cls.root/cls.manifest['package']
        sys.path.insert(0,str(cls.root))
        cls.mod={n:importlib.import_module(cls.manifest['package']+'.'+n)
                 for n in ('initialize','export','support','batch_train')}
        cls.model=cls.mod['initialize'].create_model(855)
        cls.path=Path(cls.temp.name)/'model.zip';cls.model.save(cls.path)

    @classmethod
    def tearDownClass(cls):
        sys.path.remove(str(cls.root))
        for name in list(sys.modules):
            if name.startswith(cls.manifest['package']): del sys.modules[name]
        cls.temp.cleanup()

    def test_model_identity_width_and_export_lua_parity(self):
        self.assertEqual(self.model.policy.net_arch,{'pi':[128,128],'vf':[128,128]})
        self.assertEqual(len(self.model.policy.optimizer.state),0)
        payload=self.mod['export'].export_policy(self.path)
        self.assertEqual(payload['architecture'],ARCHITECTURE)
        lua=LuaRuntime(unpack_returned_tuples=True)
        lua.globals().source=lambda n:(self.package/Path(n).name.removeprefix('rl_')).read_text()
        lua.execute('loadfile=function(n) return assert(load(source(n))) end')
        nn=lua.execute((self.package/'nn.lua').read_text())
        dense_source=(self.package/'nn.lua').read_text().replace(
            '   if index==1 then\n    for _,j in ipairs(nonzero) do value=value+row[j]*x[j] end\n   else\n    for j,w in ipairs(row) do value=value+w*x[j] end\n   end',
            '   for j,w in ipairs(row) do value=value+w*x[j] end')
        dense=lua.execute(dense_source)
        lm=lua.execute('return '+self.mod['export'].lua_literal(payload))
        live=self.mod['batch_train'].live_policy(self.model)
        self.assertEqual(live['architecture'],ARCHITECTURE)
        self.assertEqual(live['layers'],payload['layers'])
        rng=np.random.default_rng(8)
        obs=rng.uniform(-1,1,(4,4516)).astype(np.float32)
        obs[1,rng.random(4516)<.94]=0
        obs[2]=0;obs[3]=1
        with torch.no_grad(): expected=self.model.policy.action_net(self.model.policy.mlp_extractor.forward_actor(torch.as_tensor(obs))).numpy()
        for row,want in zip(obs,expected):
            action,got=nn.predict(lm,lua.table_from(row.tolist()))
            np.testing.assert_allclose(list(got.values()),want,atol=3e-6)
            self.assertEqual(action,int(want.argmax()))
            dense_action,dense_got=dense.predict(lm,lua.table_from(row.tolist()))
            self.assertEqual(action,dense_action)
            self.assertEqual(list(got.values()),list(dense_got.values()))
        bad=copy.deepcopy(payload);bad['layers'][0]['bias']=bad['layers'][0]['bias'][:64]
        bad['layers'][0]['weight']=bad['layers'][0]['weight'][:64]
        with self.assertRaisesRegex(Exception,'width'):
            nn.predict(lua.execute('return '+self.mod['export'].lua_literal(bad)),lua.table_from(obs[0].tolist()))

    def test_wrong_policy_or_critic_width_and_missing_identity_rejected(self):
        from stable_baselines3 import PPO
        for architecture in ({'pi':[64,64],'vf':[128,128]}, {'pi':[128,128],'vf':[64,64]}):
            model=PPO('MlpPolicy',self.mod['initialize'].ShapeEnv(),device='cpu',policy_kwargs={'net_arch':architecture})
            model.astra_action_interface=self.model.astra_action_interface
            model.astra_observation_interface=self.model.astra_observation_interface
            model.astra_architecture=ARCHITECTURE
            with self.assertRaisesRegex(ValueError,'architecture'):self.mod['support'].validate_model(model)
        model=copy.deepcopy(self.model);del model.astra_architecture
        with self.assertRaisesRegex(ValueError,'architecture'):self.mod['batch_train'].live_policy(model)

    def test_exact_parent_runtime_and_tamper(self):
        parent=self.root/'width-parent/astra_sf2_rl_perception'
        changed={p.name for p in parent.iterdir() if p.is_file() and (self.package/p.name).read_bytes()!=p.read_bytes()}
        self.assertEqual(changed,{'initialize.py','support.py','batch_train.py','export.py','nn.lua'})
        target=self.package/'nn.lua';old=target.read_bytes()
        try:
            target.write_bytes(old+b'\n-- changed')
            with self.assertRaises(RuntimeError):validate_build(self.manifest,self.package)
        finally:target.write_bytes(old)

    def test_relocation_without_build_checkout(self):
        clone=Path(self.temp.name)/'clean checkout';clone.mkdir()
        repo=Path(__file__).resolve().parents[2]
        for directory in ('src','experiments'):
            shutil.copytree(repo/directory,clone/directory,
                            ignore=shutil.ignore_patterns('__pycache__','*.pyc','.local','.git','MAME','skills'))
        output=Path(self.temp.name)/'public build'
        env=dict(os.environ,PYTHONPATH=os.pathsep.join((str(clone/'src'),str(clone))),PYTHONNOUSERSITE='1')
        done=subprocess.run([sys.executable,'-m','experiments.rl.perception128_builder','--output',str(output)],
                            cwd=clone,env=env,capture_output=True,text=True,timeout=60)
        self.assertEqual(done.returncode,0,done.stdout+done.stderr)
        shutil.rmtree(clone)
        moved=Path(self.temp.name)/'relocated';output.rename(moved)
        env=dict(os.environ,PYTHONPATH=os.pathsep.join((str(moved/'src'),str(moved))),PYTHONNOUSERSITE='1')
        code="""import json
from pathlib import Path
from astra_sf2_rl_perception128.perception128_identity import validate_build
r=Path.cwd();validate_build(json.loads((r/'build.json').read_text()),r/'astra_sf2_rl_perception128')
"""
        done=subprocess.run([sys.executable,'-c',code],cwd=moved,env=env,capture_output=True,text=True,timeout=60)
        self.assertEqual(done.returncode,0,done.stdout+done.stderr)
        help_run=subprocess.run([sys.executable,str(moved/'launch.py'),'batch_train','--help'],
                                cwd=moved,env=env,capture_output=True,text=True,timeout=60)
        self.assertEqual(help_run.returncode,0,help_run.stderr)
        self.assertIn('13,14,15,16',help_run.stdout)


if __name__=='__main__':unittest.main()
