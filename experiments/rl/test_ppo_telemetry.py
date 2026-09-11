"""Real CPU SB3 update and finite JSON telemetry; no emulator or game fixtures."""
import json
from pathlib import Path
import tempfile
import unittest

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure

from astra_play_sf2.config import atomic_json
from .batch_train import ppo_update_metrics


class TinyEnv(gym.Env):
    observation_space = gym.spaces.Box(-1, 1, shape=(3,), dtype=np.float32)
    action_space = gym.spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_index = 0
        return np.zeros(3, dtype=np.float32), {}

    def step(self, action):
        self.step_index += 1
        observation = np.array([self.step_index/4, float(action), -self.step_index/4], dtype=np.float32)
        return observation, float(self.step_index + action), self.step_index == 4, False, {}


class PPOTelemetryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def trained_model(self):
        model = PPO('MlpPolicy', TinyEnv(), n_steps=8, batch_size=4, n_epochs=2,
                    policy_kwargs={'net_arch': {'pi': [8], 'vf': [8]}}, seed=11, device='cpu')
        model.set_logger(configure(format_strings=[]))
        model.learn(total_timesteps=8)
        return model

    def test_real_update_metrics_survive_result_serialization(self):
        model = self.trained_model()
        metrics = ppo_update_metrics(model)
        self.assertEqual(metrics['unavailable'], {})
        self.assertEqual(metrics['n_updates'], 2)
        self.assertGreater(metrics['policy_entropy'], 0)
        for name in metrics.keys()-{'policy_entropy', 'unavailable'}:
            self.assertAlmostEqual(metrics[name], float(model.logger.name_to_value['train/'+name]))
        self.assertEqual(metrics['policy_entropy'], -metrics['entropy_loss'])
        result = {'iterations': [{'steps': model.num_timesteps, 'ppo': metrics}]}
        json.dumps(result, allow_nan=False)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'result.json'
            atomic_json(path, result)
            self.assertEqual(json.loads(path.read_text()), result)

    def test_nonfinite_missing_and_nonscalar_are_explicit_null_without_mutating_logger(self):
        model = self.trained_model()
        # Exercise only sanitization here; the preceding test uses actual values.
        recorded = model.logger.name_to_value
        recorded['train/explained_variance'] = np.float32('nan')
        recorded['train/approx_kl'] = float('inf')
        recorded['train/clip_fraction'] = np.array([0.5])
        del recorded['train/entropy_loss']
        metrics = ppo_update_metrics(model)
        for key in ('explained_variance', 'approx_kl', 'clip_fraction', 'entropy_loss', 'policy_entropy'):
            self.assertIsNone(metrics[key])
        self.assertEqual(metrics['unavailable']['explained_variance'], 'nonfinite')
        self.assertEqual(metrics['unavailable']['clip_fraction'], 'not_a_real_scalar')
        self.assertEqual(metrics['unavailable']['entropy_loss'], 'missing')
        self.assertNotIn('train/entropy_loss', recorded)
        self.assertTrue(np.isnan(recorded['train/explained_variance']))
        json.dumps(metrics, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
