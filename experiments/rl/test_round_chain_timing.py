"""A resumed batch must not expose action0 at the paused native timestamp."""
import unittest
from .test_round_chain import RoundChainTests
from .test_batch import HERE

class ResumeTimingTest(unittest.TestCase):
    def test_resume_defers_input_until_native_time_advances(self):
        fixture=RoundChainTests();fixture.setUp()
        fixture.lua.execute((HERE/'round_chain_runtime.lua').read_text())
        fixture.reset()
        fixture.request({'id':2,'op':'rollout','count':1,'actions':[6], 'resets':[{'checkpoint':0,'lead':0}]})
        for _ in range(12):fixture.tick()
        self.assertIsNotNone(fixture.reply(2))
        fixture.request({'id':3,'op':'rollout','count':1,'actions':[1], 'resets':[{'checkpoint':0,'lead':0}]})
        g=fixture.lua.globals()
        self.assertEqual(list(g.active.keys()),[])
        for _ in range(3):g.on_rpc()
        self.assertEqual(list(g.active.keys()),[])
        g.native_time=g.native_time+1;g.on_rpc()
        self.assertEqual(list(g.active.keys()),['R'])

    def test_initial_paused_reset_has_same_deferred_start(self):
        fixture=RoundChainTests();fixture.setUp()
        fixture.lua.execute((HERE/'round_chain_runtime.lua').read_text())
        fixture.reset()
        fixture.request({'id':2,'op':'rollout','count':1,'actions':[1], 'resets':[{'checkpoint':0,'lead':0}]})
        g=fixture.lua.globals()
        self.assertEqual(list(g.active.keys()),[])
        for _ in range(3):g.on_rpc()
        self.assertEqual(list(g.active.keys()),[])
        g.native_time=g.native_time+1;g.on_rpc()
        self.assertEqual(list(g.active.keys()),['R'])
