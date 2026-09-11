import unittest
from lupa.lua54 import LuaRuntime
from .pulsed_native_gate import HERE,action_plan,patch_snapshot,reference_runtime
class PulsedGateTests(unittest.TestCase):
    def test_plan_covers_all_actions_and_crosses_basic_batch_edge(self):
        plan=action_plan();self.assertEqual(len(plan),64);self.assertEqual(set(plan),set(range(16)))
        self.assertEqual(plan[:10],[0]*10);self.assertEqual(plan[32:34],[6,6])
    def test_instrumentation_is_read_only_and_reference_has_stop_boundary_plan(self):
        lua=LuaRuntime()
        for text in (patch_snapshot((HERE/'round_chain_runtime.lua').read_text()),reference_runtime((HERE/'chain_reference_runtime.lua').read_text())):
            lua.execute('assert(load(...))',text)
            self.assertIn('s.native_ports=',text);self.assertNotIn('mem:write',text)
        text=reference_runtime((HERE/'chain_reference_runtime.lua').read_text())
        self.assertIn('if frames==pending.frames then',text)
if __name__=='__main__':unittest.main()
