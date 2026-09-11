import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from lupa.lua54 import LuaRuntime
from .projectile_probe import HERE,probe_runtime,select_train,summarize
from .env import features
import numpy as np

class ProjectileProbeTests(unittest.TestCase):
    def test_reference_transform_compiles_and_keeps_nn_observation_contract(self):
        code=probe_runtime((HERE/'chain_reference_runtime.lua').read_text())
        LuaRuntime().execute('assert(load(...))',code)
        self.assertIn('policy:choose(s,reset_history)',code)
        self.assertIn('pending.frames<=3600',code)
        self.assertNotIn('mem:write',code)
        state={'timer':99,'p1':{'hp':144,'x':100,'y':40,'a':0,'char':4},'p2':{'hp':144,'x':200,'y':40,'a':0,'char':3}}
        before=features(state);state.update(projectiles=[{'hp':256,'x':123}],projectile_pointers=[1,2])
        np.testing.assert_array_equal(before,features(state))
    def test_selection_never_opens_dev_or_holdout_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows=[]
            for opponent in (3,9):
                for i,split in enumerate(('train','train','dev','holdout')):
                    p=root/f'{opponent}-{i}.sta';body=f'{opponent}-{i}'.encode()
                    if split=='train':p.write_bytes(body)
                    rows.append({'id':p.stem,'path':p.name,'sha256':hashlib.sha256(body).hexdigest(),'status':'accepted','difficulty':3,'opponent':opponent,'split':split})
            manifest=root/'manifest.json';manifest.write_text(json.dumps({'status':'complete','difficulty':3,'openings':rows}))
            selected=select_train(manifest)
            self.assertEqual(len(selected),4);self.assertTrue(all(s['split']=='train' for s in selected))
    def test_summary_rejects_missing_native_frames(self):
        opening={'native_frame_period':.1,'emulated_seconds':0}
        with self.assertRaisesRegex(RuntimeError,'cadence'):
            summarize({'opening':opening,'trace':[{'frame':2,'state':{'emulated_seconds':.1}}]})
