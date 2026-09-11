import unittest
from pathlib import Path
from lupa.lua54 import LuaRuntime
from .basic_button_probe import actions_source,runtime_source
HERE=Path(__file__).resolve().parent
class BasicButtonProbeTests(unittest.TestCase):
    def test_only_last_frame_of_basic_lp_mk_changes(self):
        lua=LuaRuntime();original=lua.execute((HERE/'actions.lua').read_text())
        changed=lua.execute(actions_source((HERE/'actions.lua').read_text()))
        for action in range(15):
            for frame in range(12):
                for forward in ('L','R'):
                    before=original['keys'](action,frame,lua.table(),forward)
                    self.assertEqual(changed['keys'](action,frame,lua.table(),forward,False),before)
                    self.assertEqual(changed['keys'](action,frame,lua.table(),forward,True),'' if action in (6,10) and frame==11 else before)
    def test_reference_preserves_native_runtime_shape(self):
        text=runtime_source((HERE/'chain_reference_runtime.lua').read_text())
        LuaRuntime().execute('assert(load(...))',text)
        self.assertIn('pending.pulse',text);self.assertIn('s.native_ports=',text)
        self.assertIn('frames==pending.frames',text)
if __name__=='__main__':unittest.main()
