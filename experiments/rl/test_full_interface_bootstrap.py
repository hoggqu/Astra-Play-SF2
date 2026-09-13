"""Prove public-source construction and relocation without the private workspace."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from .full_interface_bootstrap import build, HERE, REPO


class FullInterfaceBootstrapTests(unittest.TestCase):
    def test_clean_copy_builds_and_relocated_package_validates(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            clone = work/'clean checkout'; clone.mkdir()
            ignore = shutil.ignore_patterns('__pycache__', '*.pyc', '.local', '.git', 'MAME', 'skills')
            for folder in ('experiments', 'src'):
                shutil.copytree(REPO/folder, clone/folder, ignore=ignore)
            self.assertFalse((clone/'.local').exists())
            env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(clone/'src'), str(clone))))
            env['PYTHONNOUSERSITE'] = '1'
            code = work/'generated code'
            result = subprocess.run(
                [sys.executable, '-m', 'experiments.rl.full_interface_bootstrap', '--output', str(code)],
                cwd=clone, env=env, text=True, capture_output=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            manifest = json.loads((code/'build.json').read_text(encoding='utf-8'))
            self.assertEqual((manifest['actions'], manifest['observations']), (85, 800))
            self.assertNotIn('specialist_identity.py', manifest['derived_sha256'])
            self.assertEqual(manifest['derived_sha256']['settlement.lua'],
                             hashlib.sha256((HERE/'settlement_v5.lua').read_bytes()).hexdigest())
            self.assertFalse((clone/'.local').exists())

            # Delete the entire source copy, and move the result. Its preserved
            # public ancestry must validate without the build-time directory.
            shutil.rmtree(clone)
            relocated = work/'relocated code'; code.rename(relocated)
            env['PYTHONPATH'] = os.pathsep.join((str(relocated/'src'), str(relocated)))
            validate = """import json
from pathlib import Path
import astra_play_sf2
import astra_sf2_rl_full85.full_interface_identity as identity
root = Path.cwd()
assert Path(astra_play_sf2.__file__).resolve().is_relative_to(root/'src')
manifest = json.loads((root/'build.json').read_text())
identity.validate_build(manifest, root/'astra_sf2_rl_full85')
print('relocated validation passed')
"""
            result = subprocess.run([sys.executable, '-c', validate], cwd=relocated,
                                    env=env, text=True, capture_output=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            # The portable launch entry imports the new package without any
            # dependency setup, ROM audit, emulator, or actual model training.
            result = subprocess.run([sys.executable, str(relocated/'launch.py'), 'initialize', '--help'],
                                    cwd=relocated, env=env, text=True, capture_output=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            self.assertIn('--seed', result.stdout)

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/'existing'; output.mkdir()
            marker = output/'keep.txt'; marker.write_text('keep', encoding='utf-8')
            with self.assertRaises(FileExistsError):
                build(output)
            self.assertEqual(marker.read_text(encoding='utf-8'), 'keep')
            self.assertEqual(list(output.iterdir()), [marker])


if __name__ == '__main__':
    unittest.main()
