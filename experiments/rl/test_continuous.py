"""Offline neural export, exact input cadence, and continuous evidence checks."""
import json
from pathlib import Path
import tempfile
import unittest

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from lupa.lua54 import LuaRuntime

from astra_play_sf2.config import atomic_json
from .continuous import audit_attempt, stage_policy
from .env import features
from .export import export_policy, lua_literal

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parents[1] / 'src/astra_play_sf2/assets'


class Dummy(gym.Env):
    observation_space = gym.spaces.Box(-1, 1, (344,), dtype=np.float32)
    action_space = gym.spaces.Discrete(15)


class ContinuousTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.model = PPO('MlpPolicy', Dummy(), seed=72, n_steps=8, batch_size=4, device='cpu')
        cls.path = Path(cls.tmp.name) / 'policy.zip'
        cls.model.save(cls.path)
        cls.payload = export_policy(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        self.lua = LuaRuntime(unpack_returned_tuples=True)
        self.nn = self.lua.execute((HERE / 'nn.lua').read_text())
        self.export = self.lua.execute('return ' + lua_literal(self.payload))

    def test_logits_and_actions_agree_with_torch_for_100_observations(self):
        inputs = np.random.default_rng(6).uniform(-1, 1, (100, 344)).astype(np.float32)
        with torch.no_grad():
            expected = self.model.policy.action_net(self.model.policy.mlp_extractor.policy_net(torch.tensor(inputs))).numpy()
        for obs, logits in zip(inputs, expected):
            action, actual = self.nn.predict(self.export, self.lua.table_from(obs.tolist()))
            np.testing.assert_allclose(list(actual.values()), logits, atol=1e-7, rtol=1e-4)
            self.assertEqual(action, int(logits.argmax()))

    def state(self):
        return {'p1': {'hp': 120, 'x': 301, 'y': 40, 'a': 12, 'char': 4, 'anim': 1, 'wins': 0},
                'p2': {'hp': 72, 'x': 240, 'y': 83, 'a': 22, 'char': 2, 'anim': 1, 'wins': 0}, 'timer': 71}

    def test_feature_order_clipping_and_history_match_gym(self):
        state = self.state()
        state['p1']['hp'] = -400
        state['p2']['x'] = 2048
        actual = self.nn.features(self.lua.table_from(state, recursive=True))
        np.testing.assert_allclose(list(actual.values()), features(state), atol=1e-7)
        actions = self.lua.execute((HERE / 'actions.lua').read_text())
        policy = self.nn.new(self.export, actions)
        policy.choose(policy, self.lua.table_from(state, recursive=True), True)
        self.assertEqual(len(policy.history), 4)
        original = features(state)
        state['timer'] = 13
        policy.choose(policy, self.lua.table_from(state, recursive=True), False)
        for i in range(1, 4):
            np.testing.assert_allclose(list(policy.history[i].values()), original, atol=1e-7)
        np.testing.assert_allclose(list(policy.history[4].values()), features(state), atol=1e-7)

    def test_native_frame_macro_does_not_cancel_on_hit_or_crossup(self):
        self.lua.globals().modules = self.lua.table()
        self.lua.globals().modules['training/runtime/play_core.lua'] = self.lua.execute((ASSETS / 'play_core.lua').read_text())
        self.lua.globals().modules['training/runtime/rl_actions.lua'] = self.lua.execute((HERE / 'actions.lua').read_text())
        self.lua.execute('function loadfile(path) return function() return assert(modules[path]) end end')
        core = self.lua.execute((HERE / 'continuous_core.lua').read_text())
        self.lua.globals().Core = core
        self.lua.execute('''
            s={timer=99,p1={char=4,hp=144,x=100,y=40,a=0,anim=1,wins=0},
                        p2={char=2,hp=144,x=200,y=40,a=0,anim=1,wins=0}}
            calls=0; resets={}; local A=modules['training/runtime/rl_actions.lua']
            c=Core.new({mode='rl_ppo',opponent=2,timeout_guard=false,choose=function(a,b,mode,state,reset)
                calls=calls+1;resets[calls]=reset
                local seq={};local f=a.x<b.x and 'R' or 'L'
                for frame=0,11 do seq[#seq+1]={1,A.keys(12,frame,state,f)} end
                return seq
            end},s)
            consumed={};held=c:rl_prime(s)
            for i=1,24 do
                consumed[#consumed+1]=held
                if i==4 then s.p1.a=14;s.p1.x=250 end
                local e=c:tick(s);held=e.input or held
            end
        ''')
        samples = list(self.lua.globals().consumed.values())
        self.assertEqual(samples[:12], ['D']*3 + ['D R']*3 + ['R LP']*2 + ['']*4)
        self.assertEqual(samples[12:], ['D']*3 + ['D L']*3 + ['L LP']*2 + ['']*4)
        self.assertEqual(self.lua.globals().calls, 3)  # initial, frame12, frame24
        self.assertTrue(self.lua.globals().resets[1])
        self.assertFalse(self.lua.globals().resets[2])

    def test_each_new_native_round_reinitializes_history_before_first_action(self):
        self.lua.globals().modules = self.lua.table()
        self.lua.globals().modules['training/runtime/play_core.lua'] = self.lua.execute((ASSETS / 'play_core.lua').read_text())
        self.lua.globals().modules['training/runtime/rl_actions.lua'] = self.lua.execute((HERE / 'actions.lua').read_text())
        self.lua.execute('function loadfile(path) return function() return assert(modules[path]) end end')
        self.lua.globals().Core = self.lua.execute((HERE / 'continuous_core.lua').read_text())
        self.lua.execute("""
            s={timer=99,p1={char=4,hp=144,x=100,y=40,a=0,anim=1,wins=0},
                        p2={char=2,hp=144,x=200,y=40,a=0,anim=1,wins=0}}
            reset_flags={}
            c=Core.new({mode='rl_ppo',opponent=2,choose=function(a,b,mode,state,reset)
                reset_flags[#reset_flags+1]=reset
                local seq={};for i=1,12 do seq[i]={1,'D'} end;return seq
            end},s)
            c:rl_prime(s)
            for i=1,12 do c:drive(s,{events={}}) end
            c:begin_round(s,{events={}})
            after=c:drive(s,{events={}})
        """)
        self.assertEqual(list(self.lua.globals().reset_flags.values()), [True, False, True])
        self.assertEqual(self.lua.globals().c.rl_elapsed, 0)
        self.assertEqual(self.lua.globals().after.input, 'D')

    def test_staged_adapter_compiles_and_keeps_frozen_sources_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            hashes = stage_policy(run, 3, self.payload)
            runtime = run / 'training/runtime'
            for name in ('fighter.lua', 'play_core.lua', 'selection.json'):
                self.assertEqual((runtime / name).read_bytes(), (ASSETS / name).read_bytes())
            for path in runtime.glob('*.lua'):
                self.lua.execute('assert(load(...))', path.read_text())
            text = (runtime / 'play.lua').read_text()
            self.assertIn('timeout_guard=false', text)
            self.assertIn('r.core:rl_prime(opening)', text)
            self.assertIn('rl_policy.lua', hashes)

    def test_invalid_lifecycle_is_not_a_clear(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            atomic_json(run / 'life.json', {'active': False, 'violation': False, 'loads': 1, 'saves': 0, 'resets': 0})
            attempt = {'lifecycle': 'life.json', 'matches': [], 'difficulty': 3, 'outcome': 'gameplay_clear'}
            with self.assertRaisesRegex(RuntimeError, 'reset/load/save'):
                audit_attempt(run, attempt, 'hash')

    def test_ten_matches_cannot_be_reported_as_clear(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            atomic_json(run / 'life.json', {'active': False, 'violation': False, 'loads': 0, 'saves': 0, 'resets': 0})
            attempt = {'lifecycle': 'life.json', 'matches': [], 'difficulty': 3, 'outcome': 'gameplay_clear'}
            for index in range(10):
                path = f'm{index}.json'
                attempt['matches'].append(path)
                atomic_json(run / path, {'summary': {
                    'schema': 'mame.rl-continuous-play.v1', 'policy_kind': 'ppo', 'model_sha256': 'hash',
                    'status': 'complete', 'valid_continuous': True, 'pauses_during_match': 0, 'loads': 0, 'saves': 0,
                    'effective_difficulty': 3, 'effective_difficulty_checks': 100, 'frame': 100, 'trace_frames': 100,
                    'telemetry_error': False, 'timeout_guard': False, 'mode': 'rl_ppo', 'score': [2, 0], 'result': 'ken_win'},
                    'rounds': [{'outcome': 'win'}, {'outcome': 'win'}]})
            with self.assertRaisesRegex(RuntimeError, 'eleven'):
                audit_attempt(run, attempt, 'hash')


if __name__ == '__main__':
    unittest.main()
