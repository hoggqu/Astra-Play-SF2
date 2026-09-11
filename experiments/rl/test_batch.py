"""Offline batch cadence, terminal resets, and old-policy PPO buffer checks."""
from pathlib import Path
import json
import unittest

import gymnasium as gym
import numpy as np
import torch
from lupa.lua54 import LuaRuntime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from .batch_train import fill_buffer, live_policy
from .export import lua_literal

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parents[1] / 'src/astra_play_sf2/assets'


class BatchRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.lua = LuaRuntime(unpack_returned_tuples=True)
        self.published = {}
        self.lua.globals().publish_python = self.published.__setitem__
        self.lua.execute('''
            active={};keys={};loads=0;pauses=0;native_time=0
            for _,key in ipairs({'R','L','D','U','LP','HP','MK','HK'}) do
                local name=key;keys[name]={set_value=function(_,v) active[name]=v end}
            end
            function release() active={} end
            function play_busy() return false end
            play_bridge_generation=1;astra_difficulty_bits=0
            astra_difficulty={check=function() return {effective_difficulty=7} end}
            function fighter(i) return {char=i==0 and 4 or 2,hp=144,x=100+i*100,y=40,a=0,anim=1} end
            mem={read_u8=function(_,addr) return addr==0xff8ace and 0x99 or 0 end,read_i16=function() return 144 end}
            manager={machine={paused=true,time={as_double=function() return native_time end},screens={[':screen']={frame_period=1}},video={speed_factor=1000,throttle_rate=1,throttled=true},
                load=function() loads=loads+1;must_load=true end}}
            emu={pause=function() pauses=pauses+1;manager.machine.paused=true end,
                 unpause=function() manager.machine.paused=false end,
                 add_machine_post_load_notifier=function(fn) on_load=fn end,
                 add_machine_frame_notifier=function(fn) on_frame=fn end,
                 register_frame_done=function(fn) on_rpc=fn end}
            modules={}
            modules['training/runtime/rl_settlement.lua']=function(c) return c end
            modules['training/runtime/rl_checkpoint.lua']={'/isolated/cp.sta'}
            modules['training/runtime/rl_batch_checkpoints.lua']={2}
            modules['training/runtime/status_io.lua']={publish=function(path,body) publish_python(path,body) end,
                append=function() end}
            modules['training/runtime/play_core.lua']={opening=function() return true end,new=function()
                local c={frame=0,phase='fighting',rounds={}}
                function c:tick(s)
                    self.frame=self.frame+1
                    if self.frame==terminal_at then self.phase='between';self.rounds[1]={outcome='win',settled=s} end
                    return {input='HP'}
                end
                return c
            end}
            function loadfile(path) return function() return assert(modules[path],path) end end
            io.open=function(path,mode)
                assert(path=='training/rl-batch-request.lua')
                if not request then return nil end
                return {read=function() return request end,close=function() end}
            end
            os.remove=function() request=nil;return true end
        ''')
        for name, source in [('speed.lua', ASSETS/'speed.lua'), ('rl_nn.lua', HERE/'nn.lua'), ('rl_actions.lua', HERE/'actions.lua')]:
            self.lua.globals().modules['training/runtime/'+name] = self.lua.execute(source.read_text())
        self.lua.execute((HERE/'batch_runtime.lua').read_text())

    def request(self, body):
        self.lua.globals().request = 'return ' + lua_literal(body)
        self.lua.globals().on_rpc()

    def tick(self):
        g = self.lua.globals()
        if g.must_load:
            g.must_load = False
            g.on_load()
        g.native_time = g.native_time + 1
        g.on_frame()
        g.on_rpc()

    def reply(self, index):
        body = self.published.get(f'training/rl-batch-reply-{index:08d}.json')
        return json.loads(body) if body else None

    def reset(self):
        self.request({'id': 1, 'op': 'reset', 'reset': {'checkpoint': 0, 'lead': 0}})
        self.tick();self.tick()
        self.assertEqual(len(self.reply(1)['observation']), 344)

    def test_multiple_actions_run_without_intermediate_file_rpc(self):
        self.reset()
        self.request({'id': 2, 'op': 'rollout', 'count': 2, 'actions': [12,13],
                      'resets': [{'checkpoint': 0, 'lead': 0}]*2})
        samples = []
        for _ in range(24):
            self.assertFalse(self.lua.globals().manager.machine.paused)
            samples.append(' '.join(sorted(self.lua.globals().active.keys())))
            self.tick()
        result = self.reply(2)
        self.assertEqual([row['frames'] for row in result['transitions']], [12,24])
        self.assertEqual(samples[:12], ['D']*3 + ['D R']*3 + ['LP R']*2 + ['']*4)
        self.assertEqual(samples[12:], ['R']*2 + ['D']*2 + ['D LP R']*2 + ['']*6)
        self.assertEqual(self.lua.globals().pauses, 2)  # reset boundary + whole batch
        self.assertEqual(result['partial_episode']['steps'], 2)
        self.assertEqual(result['partial_episode']['frames'], 24)

    def test_duplicate_native_time_callback_does_not_shorten_action(self):
        self.reset()
        self.request({'id': 2, 'op': 'rollout', 'count': 1, 'actions': [0],
                      'resets': [{'checkpoint': 0, 'lead': 0}]})
        for _ in range(5):
            self.lua.globals().on_frame()
        self.assertIsNone(self.reply(2))
        for _ in range(11):
            self.tick()
            self.lua.globals().on_frame()  # Same native time; must not consume another frame.
        self.assertIsNone(self.reply(2))
        self.tick()
        self.assertEqual(self.reply(2)['transitions'][0]['frames'], 12)

    def test_terminal_resets_and_starts_are_preserved_inside_batch(self):
        self.reset()
        self.lua.globals().terminal_at = 13
        self.request({'id': 2, 'op': 'rollout', 'count': 3, 'actions': [0,1,2],
                      'resets': [{'checkpoint': 0, 'lead': 0}]*3})
        for _ in range(27):
            self.tick()
        result = self.reply(2)
        self.assertEqual([r['done'] for r in result['transitions']], [False, True, False])
        self.assertEqual([r['episode_start'] for r in result['transitions']], [True, False, True])
        self.assertEqual([r['reward'] for r in result['transitions']], [0,1,0])
        self.assertEqual(result['episodes'][0]['outcome'], 'win')
        self.assertEqual(self.lua.globals().loads, 2)
        self.assertEqual(self.lua.globals().pauses, 2)

    def test_checkpoint_actor_mismatch_invalidates_reset(self):
        self.lua.globals().modules['training/runtime/rl_batch_checkpoints.lua'][1] = 3
        self.request({'id': 1, 'op': 'reset', 'reset': {'checkpoint': 0, 'lead': 0}})
        self.tick();self.tick()
        self.assertIn('expected full-health Ken R1 opponent', self.reply(1)['error'])

    def test_unknown_settlement_keeps_evidence_and_is_never_a_done_transition(self):
        self.lua.execute('''
            modules['training/runtime/play_core.lua'].new=function()
                local c={frame=0,phase='fighting',rounds={},score={0,0}}
                function c:tick(s)
                    self.frame=self.frame+1
                    self.round_stop=s;self.terminal_frame=1;self.phase='invalid'
                    return {terminal={valid=false,reason='unresolved native result'}}
                end
                return c
            end
        ''')
        self.reset()
        self.request({'id':2,'op':'rollout','count':1,'actions':[0],
                      'resets':[{'checkpoint':0,'lead':0}]})
        self.tick()
        self.assertIn('unresolved native result', self.reply(2)['error'])
        self.assertNotIn('transitions', self.reply(2))
        evidence=json.loads(self.published['training/rl-batch-unresolved-settlement.json'])
        self.assertEqual(evidence['score'], [0,0])
        self.assertEqual(evidence['current_state']['p2']['char'], 2)
        self.assertEqual(len(evidence['trace']), 1)
        self.assertEqual(self.lua.globals().loads, 1)

    def test_stochastic_uniform_sampling_returns_actual_logprob(self):
        self.reset()
        model = {'schema':'astra.rl-policy.v1','model_sha256':'test','observations':344,'actions':15,
                 'activation':'tanh','selection':'deterministic_argmax',
                 'layers':[{'weight': [[0]*344 for _ in range(15)], 'bias':[0]*15}]}
        self.request({'id':2,'op':'rollout','count':2,'model':model,'uniforms':[0,.999],
                      'resets':[{'checkpoint':0,'lead':0}]*2})
        for _ in range(24): self.tick()
        rows = self.reply(2)['transitions']
        self.assertEqual([r['action'] for r in rows], [0,14])
        np.testing.assert_allclose([r['logprob'] for r in rows], [-np.log(15)]*2, atol=1e-12)


class Dummy(gym.Env):
    observation_space = gym.spaces.Box(-1,1,(344,),dtype=np.float32)
    action_space = gym.spaces.Discrete(15)


class BufferTests(unittest.TestCase):
    def test_batched_values_returns_and_logprob_match_standard_sb3_buffer(self):
        torch.set_num_threads(1)
        env = DummyVecEnv([Dummy, Dummy])
        model = PPO('MlpPolicy', env, n_steps=4, batch_size=4, seed=6, device='cpu')
        obs = np.random.default_rng(2).uniform(-1,1,(4,2,344)).astype(np.float32)
        actions = np.asarray([[0,1],[2,3],[4,5],[6,7]])
        starts = np.asarray([[True,True],[False,False],[True,False],[False,False]])
        with torch.no_grad():
            values, logp, _ = model.policy.evaluate_actions(torch.tensor(obs.reshape(-1,344)), torch.tensor(actions.reshape(-1)))
        values, logp = values.reshape(4,2), logp.reshape(4,2)
        chunks = []
        for w in range(2):
            rows=[{'observation':obs[t,w].tolist(),'action':int(actions[t,w]),'reward':float(t),
                   'episode_start':bool(starts[t,w]),'done':bool(starts[t+1,w]) if t<3 else w==0,'logprob':float(logp[t,w])} for t in range(4)]
            chunks.append({'transitions':rows,'observation':obs[-1,w].tolist(),'episode_start':w==0})
        error = fill_buffer(model, chunks)
        self.assertLess(error, 1e-6)
        actual = model.rollout_buffer.returns.copy()
        model.rollout_buffer.reset()
        for t in range(4):
            model.rollout_buffer.add(obs[t],actions[t],np.asarray([t,t]),starts[t],values[t],logp[t])
        with torch.no_grad():
            last = model.policy.predict_values(torch.tensor(obs[-1]))
        model.rollout_buffer.compute_returns_and_advantage(last,np.asarray([True,False]))
        np.testing.assert_array_equal(model.rollout_buffer.returns, actual)
        chunks[0]['transitions'][0]['logprob'] += .01
        with self.assertRaisesRegex(RuntimeError, 'logprob disagreement'):
            fill_buffer(model, chunks)
        self.assertEqual(live_policy(model)['observations'], 344)
        env.close()


if __name__ == '__main__':
    unittest.main()
