"""Pure action A/B coverage and immutable staged-source identities."""
from pathlib import Path
import json
import tempfile
import unittest
from lupa.lua54 import LuaRuntime

from astra_play_sf2.runner import sha256
from .fast_fire_continuous import stage_policy, INTERFACES

HERE=Path(__file__).resolve().parent


class FastFireTests(unittest.TestCase):
    def setUp(self):
        self.lua=LuaRuntime(unpack_returned_tuples=True)
        self.base=self.lua.execute((HERE/'actions.lua').read_text())
        self.lua.globals().base=self.base
        self.lua.execute("function loadfile(path) assert(path=='training/runtime/rl_actions_base.lua');return function() return base end end")
        self.fast=self.lua.execute((HERE/'fast_fire_actions.lua').read_text())
        self.state=self.lua.table()

    def test_every_other_action_is_exactly_identical_on_both_sides(self):
        self.assertEqual(self.fast.frames,12);self.assertEqual(self.fast.count,15)
        for side in ('L','R'):
            for action in range(15):
                if action==12:continue
                for frame in range(12):
                    self.assertEqual(self.fast['keys'](action,frame,self.state,side),self.base['keys'](action,frame,self.state,side))

    def test_only_action12_changes_to_two_two_two_six_with_fixed_facing(self):
        for side in ('L','R'):
            expected=['D']*2+['D '+side]*2+[side+' LP']*2+['']*6
            self.assertEqual([self.fast['keys'](12,f,self.state,side) for f in range(12)],expected)
            original=['D']*3+['D '+side]*3+[side+' LP']*2+['']*4
            self.assertEqual([self.base['keys'](12,f,self.state,side) for f in range(12)],original)

    def test_original_argument_validation_remains(self):
        for action,frame in ((-1,0),(15,0),(12,-1),(12,12)):
            with self.assertRaises(Exception):self.fast['keys'](action,frame,self.state,'R')

    def test_staging_marks_both_variants_and_pins_extra_base_source(self):
        payload={'model_sha256':'fixed','selection':'deterministic_argmax'}
        with tempfile.TemporaryDirectory() as folder:
            for variant in ('original','fast'):
                run=Path(folder)/variant;sources=stage_policy(run,3,payload,variant)
                metadata=json.loads((run/'action-interface.json').read_text())
                self.assertEqual(metadata['action_interface'],INTERFACES[variant])
                source=(run/'training/runtime/play.lua').read_text()
                self.assertIn("action_interface='"+INTERFACES[variant]+"'",source)
                if variant=='original':
                    self.assertEqual(sources['rl_actions.lua'],sha256(HERE/'actions.lua'))
                    self.assertNotIn('rl_actions_base.lua',sources)
                else:
                    self.assertEqual(sources['rl_actions_base.lua'],sha256(HERE/'actions.lua'))
                    self.assertEqual(sources['rl_actions.lua'],sha256(HERE/'fast_fire_actions.lua'))
                    self.assertIn("'training/runtime/rl_actions_base.lua','training/runtime/rl_actions.lua'",source)
                self.assertEqual(metadata['model_sha256'],'fixed')


if __name__=='__main__':unittest.main()
