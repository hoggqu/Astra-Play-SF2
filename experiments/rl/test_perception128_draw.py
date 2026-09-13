"""Portable result-only revision preserves policy, optimizer and observer code."""
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from .perception128_draw_builder import bootstrap
from .perception128_draw_identity import validate_build

class SettlementBuildTests(unittest.TestCase):
    def test_exact_derivation_parent_model_and_relocation(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'code';manifest=bootstrap(root);package=root/manifest['package']
            parent=root/'settlement-parent/astra_sf2_rl_perception128_timing'
            changed={p.name for p in parent.iterdir() if p.is_file() and p.read_bytes()!=(package/p.name).read_bytes()}
            self.assertEqual(changed,{'initialize.py','support.py','batch_train.py','settlement.lua'})
            for name in ('initialize.py','support.py','batch_train.py'):
                expected=(parent/name).read_bytes().replace(b'from .perception128_timing_identity import',b'from .perception128_draw_identity import')
                self.assertEqual((package/name).read_bytes(),expected)
            sys.path.insert(0,str(root))
            try:
                initialize=importlib.import_module(manifest['package']+'.initialize')
                model=initialize.create_model(12)
                self.assertEqual(model.policy.net_arch,{'pi':[128,128],'vf':[128,128]})
                support=importlib.import_module(manifest['package']+'.support');support.validate_model(model)
                metadata=importlib.import_module(manifest['package']+'.perception128_draw_identity').training_metadata(package)
                self.assertEqual(metadata['settlement_protocol'],manifest['settlement_protocol'])
            finally:
                sys.path.remove(str(root))
                for name in list(sys.modules):
                    if name.startswith(manifest['package']):del sys.modules[name]
            file=package/'settlement.lua';old=file.read_bytes()
            try:
                file.write_bytes(old+b'\n-- mutation')
                with self.assertRaises(RuntimeError):validate_build(manifest,package)
            finally:file.write_bytes(old)
            moved=Path(temp)/'relocated';root.rename(moved)
            env=dict(os.environ,PYTHONPATH=os.pathsep.join((str(moved/'src'),str(moved))),PYTHONNOUSERSITE='1')
            code="""import json
from pathlib import Path
from astra_sf2_rl_perception128_draw.perception128_draw_identity import validate_build
r=Path.cwd();validate_build(json.loads((r/'build.json').read_text()),r/'astra_sf2_rl_perception128_draw')
"""
            done=subprocess.run([sys.executable,'-c',code],cwd=moved,env=env,capture_output=True,text=True,timeout=60)
            self.assertEqual(done.returncode,0,done.stdout+done.stderr)

if __name__=='__main__':unittest.main()
