import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from .managed_runtime import stop_request
from .managed_builder import bootstrap
from .managed_identity import validate_build, PACKAGE


class ManagedTests(unittest.TestCase):
    def test_public_build_preserves_every_lua_file_and_rejects_mutation(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'code';m=bootstrap(root);package=root/PACKAGE
            parent=root/'managed-parent';p=json.loads((parent/'build.json').read_text())
            for name in m['derived_sha256']:
                if name.endswith('.lua'):
                    self.assertEqual((package/name).read_bytes(),(parent/p['package']/name).read_bytes())
            self.assertEqual(m['cooperative_stop_protocol'],'ppo-update-stop-file-v1')
            self.assertIn('device=model.device',(package/'rollout_control.py').read_text())
            self.assertIn('OpponentSampler(', (package/'batch_train.py').read_text())
            self.assertIn('set_opponent_probabilities', (package/'batch_env.py').read_text())
            self.assertEqual((package/'continuous.py').read_bytes(),(parent/p['package']/'continuous.py').read_bytes())
            self.assertEqual((package/'native_continuous.py').read_bytes(),(parent/p['package']/'native_continuous.py').read_bytes())
            (package/'managed_runtime.py').write_text('changed')
            with self.assertRaises(RuntimeError):validate_build(m,package)

    def test_stop_request_requires_valid_scope_and_reason(self):
        with tempfile.TemporaryDirectory() as folder,patch.dict(os.environ,{'ASTRA_RL_STOP_FILE':str(Path(folder)/'stop.json')}):
            p=Path(folder)/'stop.json';self.assertIsNone(stop_request())
            p.write_text(json.dumps({'schema':'astra.rl-stop-request.v1','reason':'max_duration'}))
            self.assertEqual(stop_request(),'max_duration')
            p.write_text(json.dumps({'schema':'wrong','reason':'max_duration'}))
            with self.assertRaises(ValueError):stop_request()


if __name__=='__main__':unittest.main()
