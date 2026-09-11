"""Offline waveform, immutable derivation and metadata-only model migration."""
import importlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
import zipfile

import numpy as np
import torch
from lupa.lua54 import LuaRuntime
from stable_baselines3 import PPO
from astra_play_sf2.runner import sha256
from .actions16_builder import build as actions_build, PACKAGE as ACTIONS_PACKAGE
from .round_chain_builder import build as chain_build, PACKAGE
from .pulsed_normals_builder import build
from .pulsed_normals_identity import validate_pulsed_build, INTERFACE, OLD
from .pulsed_normals_migrate import migrate, identical
from .width_migrate import SpacesOnly


class PulsedNormalsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name)
        actions_build(cls.root/'actions')
        chain_build(cls.root/'parent',cls.root/'actions'/ACTIONS_PACKAGE)
        shutil.copytree(Path(__file__).resolve().parents[2]/'src',cls.root/'parent/src',
                        ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        cls.parent=cls.root/'parent'/PACKAGE
        cls.parent_hashes={p.name:sha256(p) for p in cls.parent.iterdir() if p.is_file()}
        cls.manifest=build(cls.root/'candidate',cls.parent);cls.package=cls.root/'candidate'/PACKAGE
        sys.path.insert(0,str(cls.root/'candidate'))
        cls.support=importlib.import_module(PACKAGE+'.support')
        cls.export=importlib.import_module(PACKAGE+'.export')
        cls.identity=importlib.import_module(PACKAGE+'.pulsed_normals_identity')
        cls.old=PPO('MlpPolicy',SpacesOnly(),n_steps=256,batch_size=64,seed=17,
                    policy_kwargs={'net_arch':{'pi':[64,64],'vf':[64,64]}})
        cls.old.astra_action_interface=OLD
        # Populate real Adam moments with one disposable tensor step, no rollout.
        x=torch.from_numpy(np.random.default_rng(17).uniform(-1,1,(4,344)).astype(np.float32))
        cls.old.policy.optimizer.zero_grad()
        value,logprob,_=cls.old.policy.evaluate_actions(x,torch.tensor([0,6,10,15]))
        (value.square().sum()+logprob.sum()).backward()
        cls.old.policy.optimizer.step()
        cls.original=cls.root/'old.zip';cls.old.save(cls.original)
        cls.migration=migrate(cls.original,cls.root/'migration')
        cls.model=cls.root/'migration/ppo-pulsed.zip'

    @classmethod
    def tearDownClass(cls):
        sys.path.remove(str(cls.root/'candidate'))
        for name in list(sys.modules):
            if name==PACKAGE or name.startswith(PACKAGE+'.'):del sys.modules[name]
        cls.temp.cleanup()

    def test_all_16_actions_two_directions_twelve_frames_exact_waveform(self):
        lua=LuaRuntime();old=lua.execute((self.parent/'actions.lua').read_text())
        new=lua.execute((self.package/'actions.lua').read_text());state=lua.table()
        self.assertEqual(new.interface,INTERFACE);self.assertEqual(new.count,16);self.assertEqual(new.frames,12)
        differences=0
        for action in range(16):
            for direction in ('L','R'):
                for frame in range(12):
                    before=old['keys'](action,frame,state,direction)
                    expected=('D' if action in (8,9) else '') if 6<=action<=11 and frame==11 else before
                    actual=new['keys'](action,frame,state,direction)
                    self.assertEqual(actual,expected,(action,direction,frame))
                    differences+=before!=actual
        self.assertEqual(differences,12)
        for action in range(6,12):
            wave=[new['keys'](action,f%12,state,'R') for f in range(24)]
            self.assertEqual(wave[11],wave[23]);self.assertNotEqual(wave[11],wave[12])

    def test_parent_unchanged_and_exact_derivation_checks_all_dependencies(self):
        self.assertEqual({p.name:sha256(p) for p in self.parent.iterdir() if p.is_file()},self.parent_hashes)
        validate_pulsed_build(self.manifest,self.package)
        self.identity.validate_pulsed_build(self.manifest,self.package)
        for name in ('batch_env.py','batch_runtime.lua','settlement.lua','continuous_core.lua','native_continuous_core.lua'):
            # Identity strings may change; behavior text otherwise is identical.
            self.assertEqual((self.package/name).read_text(),(self.parent/name).read_text().replace(OLD,INTERFACE))
        self.assertEqual((self.package/'nn.lua').read_text(),(self.parent/'nn.lua').read_text().replace(OLD,INTERFACE))
        paths=self.identity.provenance_paths(self.package)
        self.assertIn('pulse/parent-source/build.json',paths)
        self.assertIn('pulse/parent-source/'+PACKAGE+'/actions.lua',paths)

    def test_reject_tampering_even_when_changed_action_hash_is_relabelled(self):
        p=self.package/'actions.lua';original=p.read_bytes()
        try:
            p.write_bytes(original.replace(b'frame==11',b'frame==10'))
            manifest=json.loads(json.dumps(self.manifest));manifest['derived_sha256']['actions.lua']=sha256(p)
            with self.assertRaisesRegex(RuntimeError,'exact allowed derivation'):
                validate_pulsed_build(manifest,self.package)
        finally:p.write_bytes(original)
        for p in (self.root/'candidate/parent-source/build.json',
                  self.root/'candidate/parent-source/parent-build.json',
                  self.root/'candidate/parent-source'/PACKAGE/'actions.lua',
                  self.root/'candidate/src/astra_play_sf2/assets/play_core.lua'):
            original=p.read_bytes()
            try:
                p.write_bytes(original+b'\nchanged')
                with self.assertRaises(RuntimeError):validate_pulsed_build(self.manifest,self.package)
            finally:p.write_bytes(original)

    def test_only_explicit_metadata_changes_weights_and_adam_exact(self):
        with zipfile.ZipFile(self.original) as old,zipfile.ZipFile(self.model) as new:
            self.assertEqual(old.namelist(),new.namelist())
            for name in old.namelist():
                if name!='data':self.assertEqual(old.read(name),new.read(name),name)
            a=json.loads(old.read('data'));b=json.loads(new.read('data'))
            self.assertEqual({k for k in a.keys()|b.keys() if a.get(k)!=b.get(k)},
                             {'astra_action_interface','astra_pulsed_normals'})
        reloaded=PPO.load(self.model,device='cpu')
        identical(self.old.policy.state_dict(),reloaded.policy.state_dict())
        identical(self.old.policy.optimizer.state_dict(),reloaded.policy.optimizer.state_dict())
        self.assertEqual(len(reloaded.policy.optimizer.state),12)
        with self.assertRaises(ValueError):migrate(self.model,self.root/'must-not-exist')
        self.assertFalse((self.root/'must-not-exist').exists())

    def test_old_model_rejected_export_capture_and_new_model_accepted(self):
        with self.assertRaises(ValueError):self.support.validate_model_file(self.original)
        with self.assertRaises(ValueError):self.export.export_policy(self.original)
        with self.assertRaises(ValueError):self.support.capture_interface(self.original,self.package)
        snapshot=self.support.capture_interface(self.model,self.package)
        self.assertEqual(snapshot['payload']['action_interface'],INTERFACE)
        self.assertEqual(snapshot['model_sha256'],self.migration['model_sha256'])
        self.assertEqual(snapshot['payload']['observations'],344)
        self.assertEqual(snapshot['payload']['actions'],16)
        lua=LuaRuntime();nn=lua.execute((self.package/'nn.lua').read_text())
        payload=lua.execute(snapshot['payload_text']);observations=lua.table_from([0.]*344)
        nn.predict(payload,observations)
        payload.action_interface=OLD
        with self.assertRaises(Exception):nn.predict(payload,observations)
        relabelled=dict(self.manifest,action_interface=OLD)
        with self.assertRaisesRegex(RuntimeError,'interface identity'):
            validate_pulsed_build(relabelled,self.package)


if __name__=='__main__':unittest.main()
