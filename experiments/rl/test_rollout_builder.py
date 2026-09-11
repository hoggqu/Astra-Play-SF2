"""Offline long-rollout identity, optimizer and terminal/GAE checks; no MAME."""
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.vec_env import DummyVecEnv
from astra_play_sf2.runner import sha256
from .actions16_builder import build as actions_build, PACKAGE as ACTIONS_PACKAGE
from .round_chain_builder import build as chain_build, PACKAGE as CHAIN_PACKAGE
from .rollout_builder import build, PACKAGE


class DummyGame(gym.Env):
    observation_space = gym.spaces.Box(-1, 1, shape=(344,), dtype=np.float32)
    action_space = gym.spaces.Discrete(16)
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(344, dtype=np.float32), {}
    def step(self, action):
        return np.zeros(344, dtype=np.float32), 0., False, False, {}


class RolloutBuilderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name).resolve()
        actions_build(cls.root/'actions')
        chain_build(cls.root/'parent', cls.root/'actions'/ACTIONS_PACKAGE)
        production = Path(__file__).resolve().parents[2]/'src'
        shutil.copytree(production, cls.root/'parent/src', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        cls.parent = cls.root/'parent'/CHAIN_PACKAGE
        cls.original = {p.name: sha256(p) for p in cls.parent.iterdir() if p.is_file()}
        cls.manifest = build(cls.root/'candidate', cls.parent)
        cls.package = cls.root/'candidate'/PACKAGE
        sys.path.insert(0, str(cls.root/'candidate'))
        cls.module = importlib.import_module(PACKAGE+'.batch_train')
        cls.identity = importlib.import_module(PACKAGE+'.rollout_identity')

    @classmethod
    def tearDownClass(cls):
        sys.path.remove(str(cls.root/'candidate'))
        for name in list(sys.modules):
            if name == PACKAGE or name.startswith(PACKAGE+'.'):
                del sys.modules[name]
        cls.temporary.cleanup()

    def test_default_256_explicit_1024_and_reject_incomplete_8192_update(self):
        body = (self.package/'batch_train.py').read_text()
        self.assertIn("choices=(256,1024), default=256", body)
        self.assertIn('range(0, args.rollout_steps, args.block)', body)
        self.assertIn('n_steps=args.rollout_steps)', body)
        self.assertEqual(self.manifest['default_rollout_steps'], 256)
        env = dict(os.environ, PYTHONPATH=str(self.root/'candidate/src')+os.pathsep+str(self.root/'candidate'))
        result = subprocess.run([sys.executable, '-m', PACKAGE+'.batch_train', '--dataset', 'unused',
            '--output', str(self.root/'must-not-exist'), '--workers', '8', '--rollout-steps', '1024', '--steps', '2048'],
            env=env, cwd=self.root/'candidate', capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 2)
        self.assertIn('workers*rollout-steps', result.stderr)
        self.assertFalse((self.root/'must-not-exist').exists())

    def test_parent_lua_production_and_uniform_sampler_unchanged(self):
        self.assertEqual({p.name: sha256(p) for p in self.parent.iterdir() if p.is_file()}, self.original)
        for name in self.original:
            if name.endswith('.lua'):
                self.assertEqual(sha256(self.package/name), self.original[name])
        self.assertEqual((self.package/'batch_env.py').read_bytes(), (self.parent/'batch_env.py').read_bytes())
        for name, digest in self.manifest['frozen_production_sha256'].items():
            self.assertEqual(sha256(self.root/'candidate/src'/name), digest)
        self.identity.validate_rollout_build(self.manifest, self.package)

    def test_identity_rejects_changed_lua_and_parent_manifest(self):
        for path in (self.package/'batch_runtime.lua', self.root/'candidate/rollout-parent-build.json'):
            original = path.read_bytes()
            try:
                path.write_bytes(original+b'\nchanged')
                with self.assertRaises((RuntimeError, ValueError)):
                    self.identity.validate_rollout_build(self.manifest, self.package)
            finally:
                path.write_bytes(original)

    def test_loaded_1024_buffer_preserves_parameters_adam_moments_and_update_age(self):
        torch.set_num_threads(1)
        vector = DummyVecEnv([DummyGame, DummyGame])
        self.addCleanup(vector.close)
        model = PPO('MlpPolicy', vector, n_steps=256, batch_size=64, n_epochs=4, learning_rate=3e-4, seed=42)
        model.astra_action_interface = 'ken_actions16_lp_mp_uppercut_v1'
        optimizer = model.policy.optimizer
        optimizer.zero_grad()
        sum(p.square().sum() for p in model.policy.parameters()).backward()
        optimizer.step(); model._n_updates = 123
        path = self.root/'tensor-model.zip'; model.save(path)
        loaded = PPO.load(path, env=vector, device='cpu', seed=42, n_steps=1024)
        self.assertEqual((loaded.n_steps, loaded.rollout_buffer.buffer_size, loaded.n_envs), (1024, 1024, 2))
        self.assertEqual(loaded._n_updates, 123)
        for key, value in model.policy.state_dict().items():
            self.assertTrue(torch.equal(value, loaded.policy.state_dict()[key]), key)
        before, after = optimizer.state_dict(), loaded.policy.optimizer.state_dict()
        self.assertEqual(before['param_groups'], after['param_groups'])
        self.assertEqual(len(before['state']), 12)
        for parameter, values in before['state'].items():
            for key, value in values.items():
                self.assertTrue(torch.equal(value, after['state'][parameter][key]), (parameter, key))
        metadata = self.identity.training_metadata(self.package, 1024, path)
        self.assertEqual(metadata['initial_model_rollout_steps'], 256)
        self.assertFalse(metadata['optimizer_reinitialized'])

    def test_1024_buffer_preserves_time_worker_axes_and_round_terminal_gae_masks(self):
        steps, workers = 1024, 2
        observations = np.zeros((steps, workers, 344), dtype=np.float32)
        observations[:, :, 0] = np.arange(steps)[:, None]/2048 + np.arange(workers)[None, :]/10
        rewards = np.full((steps, workers), .01, dtype=np.float32)
        dones = np.zeros((steps, workers), dtype=bool)
        dones[[31, 255, 735], 0] = True
        dones[[63, 256, 1023], 1] = True
        rewards[dones] = 1
        starts = np.zeros_like(dones); starts[0] = True; starts[1:] = dones[:-1]
        final_obs = np.zeros((workers, 344), dtype=np.float32)
        final_obs[:, 0] = [.8, 100]  # Artificial refill value must be masked for worker 1.
        class Policy:
            def evaluate_actions(self, obs, actions):
                return obs[:, :1], torch.zeros(len(obs)), torch.zeros(len(obs))
            def predict_values(self, obs):
                return obs[:, :1]
        buffer = RolloutBuffer(steps, DummyGame.observation_space, DummyGame.action_space,
                              n_envs=workers, gamma=.99, gae_lambda=.95)
        model = SimpleNamespace(n_envs=workers, n_steps=steps, policy=Policy(), rollout_buffer=buffer)
        chunks = []
        for worker in range(workers):
            chunks.append({'transitions': [{'observation': observations[t, worker].tolist(), 'action': (t+worker)%16,
                'reward': float(rewards[t, worker]), 'episode_start': bool(starts[t, worker]),
                'done': bool(dones[t, worker]), 'logprob': 0.} for t in range(steps)],
                'observation': final_obs[worker].tolist(), 'episode_start': bool(dones[-1, worker])})
        self.assertEqual(self.module.fill_buffer(model, chunks), 0)
        np.testing.assert_array_equal(buffer.observations, observations)
        expected = np.zeros((steps, workers), dtype=np.float64)
        for worker in range(workers):
            gae = 0
            for t in range(steps-1, -1, -1):
                next_value = final_obs[worker, 0] if t == steps-1 else observations[t+1, worker, 0]
                mask = 1-int(dones[t, worker])
                delta = rewards[t, worker]+.99*next_value*mask-observations[t, worker, 0]
                gae = delta+.99*.95*mask*gae
                expected[t, worker] = gae+observations[t, worker, 0]
        np.testing.assert_allclose(buffer.returns, expected, atol=2e-6, rtol=1e-6)
        np.testing.assert_allclose(buffer.returns[dones], 1, atol=1e-6)
        self.assertAlmostEqual(buffer.returns[-1, 0], .01+.99*.8, places=6)
        chunks[0]['transitions'][256]['episode_start'] = False
        with self.assertRaisesRegex(RuntimeError, 'Terminal flags'):
            self.module.fill_buffer(model, chunks)

    def test_telemetry_is_read_only(self):
        helper = importlib.import_module(PACKAGE+'.ppo_metrics')
        values = {'train/entropy_loss': -.5, 'train/n_updates': 7, 'train/approx_kl': float('nan')}
        before = dict(values)
        row = helper.ppo_update_metrics(SimpleNamespace(logger=SimpleNamespace(name_to_value=values)))
        self.assertEqual(row['policy_entropy'], .5)
        self.assertEqual(row['n_updates'], 7)
        self.assertIsNone(row['approx_kl'])
        self.assertEqual(values, before)


if __name__ == '__main__':
    unittest.main()
