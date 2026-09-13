"""Portable viewer orchestration tests; never launch MAME."""
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile
from . import watch

class WatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve();self.model=self.root/'checkpoint.zip'
        with zipfile.ZipFile(self.model,'w') as z:
            for name in ('data','policy.pth','policy.optimizer.pth'):z.writestr(name,'fixture')

    def test_campaign_published_latest_relative_and_hash(self):
        pointer=self.root/'result.json'
        pointer.write_text(json.dumps({'latest_model':self.model.name,'latest_model_sha256':watch.digest(self.model)}))
        path,sha=watch.resolve_model(campaign=self.root)
        self.assertEqual(path,self.model);self.assertEqual(sha,watch.digest(path))

    def test_autotrain_top_level_resolves_run_pointer(self):
        run=self.root/'run';run.mkdir()
        (run/'result.json').write_text(json.dumps({'latest_model':'../checkpoint.zip','latest_model_sha256':watch.digest(self.model)}))
        self.assertEqual(watch.resolve_model(campaign=self.root)[0],self.model)

    def test_default_build_uses_current_managed_bootstrap(self):
        calls=[]
        def build(path):
            calls.append(path);path.mkdir()
            (path/'build.json').write_text(json.dumps({'package':'astra_sf2_rl_managed'}))
        with patch('experiments.rl.managed_builder.bootstrap',side_effect=build), patch('experiments.rl.managed_identity.validate_build') as validate:
            code,_=watch.prepare_code(None,self.root)
        self.assertEqual(calls,[self.root/'code']);validate.assert_called_once()
        self.assertEqual(code,self.root/'code')

    def test_missing_latest_never_guesses_newest_file(self):
        (self.root/'result.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'published'):watch.resolve_model(campaign=self.root)

    def test_snapshot_pins_hash_and_rejects_bad_hash(self):
        target=self.root/'snapshot.zip';sha=watch.snapshot_model(self.model,target)
        self.assertEqual(watch.digest(target),sha)
        with self.assertRaisesRegex(ValueError,'SHA256'):watch.snapshot_model(self.model,target,'0'*64)

    def test_viewer_launch_is_visible_silent_native_and_independent(self):
        code=self.root/'code';code.mkdir();calls=[]
        config=self.root/'config.json';config.write_text('{}')
        def runner(command,**kwargs):
            calls.append((command,kwargs));output=Path(command[command.index('--output')+1]);output.mkdir()
            (output/'result.json').write_text(json.dumps({'status':'complete','attempts':[{'id':'l3-001','outcome':'loss','match_wins':4}]}))
            return types.SimpleNamespace(returncode=1)
        with patch.object(watch,'prepare_code',return_value=(code,'buildhash')):
            result,output=watch.watch(model=self.model,config=config,output=self.root/'watch',runner=runner)
        command,kwargs=calls[0]
        self.assertIn('native_continuous',command);self.assertIn('--show-window',command)
        self.assertIn('--all-attempts',command)
        self.assertEqual(command[command.index('--speed')+1],'normal')
        self.assertEqual(command[command.index('--attempts')+1],'1')
        self.assertEqual(kwargs['env']['ASTRA_SF2_CONFIG'],str(config))
        self.assertEqual(result['status'],'complete') # valid loss is a completed viewing session
        self.assertFalse(result['training_statistics']);self.assertFalse(result['automatic_verification'])
        self.assertEqual(result['sound'],'none');self.assertFalse(result['always_on_top'])
        self.assertEqual(Path(command[command.index('--model')+1]),output/'model.zip')
        self.assertFalse((self.root/'result.json').exists())

    def test_child_failure_is_recorded_and_never_retried(self):
        calls=[]
        def fail(command,**kwargs):calls.append(command);return types.SimpleNamespace(returncode=2)
        with patch.object(watch,'prepare_code',return_value=(self.root,'h')):
            result,output=watch.watch(model=self.model,output=self.root/'watch',runner=fail,speed='2x')
        self.assertEqual(len(calls),1);self.assertEqual(result['status'],'invalid')
        self.assertEqual(json.loads((output/'watch.json').read_text())['status'],'invalid')

    def test_ssh_wslg_environment_uses_existing_display_socket(self):
        with patch.dict('os.environ',{},clear=True), patch.object(watch.sys,'platform','linux'), patch.object(Path,'is_dir',return_value=True), patch.object(Path,'is_socket',return_value=True):
            env=watch.viewer_environment()
        self.assertEqual(env['DISPLAY'],':0');self.assertEqual(env['SDL_VIDEODRIVER'],'x11')

    def test_explicit_display_is_never_replaced(self):
        with patch.dict('os.environ',{'DISPLAY':':7','SDL_VIDEODRIVER':'wayland'},clear=True), patch.object(watch.sys,'platform','linux'):
            env=watch.viewer_environment()
        self.assertEqual(env['DISPLAY'],':7');self.assertEqual(env['SDL_VIDEODRIVER'],'wayland')

    def test_unknown_code_package_is_rejected(self):
        code=self.root/'code';code.mkdir()
        (code/'build.json').write_text(json.dumps({'package':'arbitrary'}))
        with self.assertRaisesRegex(ValueError,'Unsupported viewer source'):
            watch.prepare_code(code,self.root)

    def test_managed_code_uses_its_own_validator(self):
        code=self.root/'code';code.mkdir()
        (code/'build.json').write_text(json.dumps({'package':'astra_sf2_rl_managed'}))
        calls=[];module=types.ModuleType('experiments.rl.managed_identity')
        module.validate_build=lambda manifest,package:calls.append((manifest,package))
        with patch.dict('sys.modules',{'experiments.rl.managed_identity':module}):
            actual,sha=watch.prepare_code(code,self.root)
        self.assertEqual(actual,code);self.assertEqual(len(calls),1)
        self.assertEqual(calls[0][1],code/'astra_sf2_rl_managed')

    def test_existing_output_never_overwritten(self):
        with self.assertRaises(FileExistsError):watch.watch(model=self.model,output=self.root)

if __name__=='__main__':unittest.main()
