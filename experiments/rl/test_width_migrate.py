"""Width expansion preserves functions and real Adam state, without an emulator."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from stable_baselines3 import PPO

from .width_migrate import (SpacesOnly, INTERFACE, expand_model, check_embedding,
                            check_function, check_roundtrip, check_learning, outputs)


class WidthMigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def old_model(self):
        old = PPO('MlpPolicy', SpacesOnly(), n_steps=256, batch_size=64, seed=13,
                  policy_kwargs={'net_arch': {'pi': [64, 64], 'vf': [64, 64]}}, device='cpu')
        old.astra_action_interface = INTERFACE
        # Genuine nonzero Adam moments in every parameter, with no game involved.
        loss = sum(parameter.square().sum() for parameter in old.policy.parameters())
        old.policy.optimizer.zero_grad();loss.backward();old.policy.optimizer.step()
        old.num_timesteps = 409600;old._n_updates = 800
        return old

    def test_function_moments_random_new_features_and_roundtrip(self):
        old = self.old_model()
        before = {name: value.clone() for name, value in old.policy.state_dict().items()}
        new = expand_model(old, seed=21)
        self.assertTrue(check_embedding(old, new)['ok'])
        observations = np.random.default_rng(8).uniform(-1, 1, (128, 344)).astype(np.float32)
        self.assertTrue(check_function(old, new, observations)['ok'])
        for name, value in old.policy.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        self.assertEqual(new.num_timesteps, 409600)
        self.assertEqual(new._n_updates, 800)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'wide.zip';new.save(path)
            restored = PPO.load(path, device='cpu')
            self.assertTrue(check_roundtrip(new, restored)['ok'])
            self.assertTrue(check_learning(restored, observations)['ok'])
            # The disposable gradient probe leaves the saved artifact unchanged.
            self.assertTrue(check_embedding(old, PPO.load(path, device='cpu'))['ok'])

    def test_preserves_amsgrad_state_and_nondefault_optimizer_group(self):
        old = self.old_model()
        old.policy.optimizer = torch.optim.Adam(old.policy.parameters(), lr=0.00017,
            betas=(0.8, 0.97), eps=2e-5, amsgrad=True)
        loss = sum(parameter.square().sum() for parameter in old.policy.parameters())
        old.policy.optimizer.zero_grad();loss.backward();old.policy.optimizer.step()
        new = expand_model(old)
        self.assertTrue(check_embedding(old, new)['ok'])
        self.assertEqual(new.policy.optimizer.param_groups[0]['betas'], (0.8, 0.97))
        self.assertTrue(all('max_exp_avg_sq' in value for value in new.policy.optimizer.state.values()))

    def test_function_checks_default_chunk_sizes_and_partial_tail(self):
        old = self.old_model();new = expand_model(old)
        observations = np.random.default_rng(9).uniform(-1, 1, (35, 344)).astype(np.float32)
        with patch('experiments.rl.width_migrate.outputs', wraps=outputs) as calls:
            result = check_function(old, new, observations)
        self.assertEqual(result['batch_sizes'], [1, 32, 256])
        self.assertEqual([result['chunks'][str(n)]['chunks_per_model'] for n in (1, 32, 256)], [35, 2, 1])
        self.assertEqual({len(call.args[1]) for call in calls.call_args_list}, {1, 3, 32, 35})
        for chunk in result['chunks'].values():
            for key in ('old_vs_new', 'old_vs_full_batch', 'new_vs_full_batch'):
                self.assertEqual(chunk[key]['argmax_mismatches'], 0)
        with self.assertRaises(ValueError):check_function(old, new, observations, batch_sizes=(0,))

    def test_rejects_wrong_interface_and_already_expanded_model(self):
        old = self.old_model();new = expand_model(old)
        with self.assertRaises(ValueError):expand_model(new)
        old.astra_action_interface = 'unknown'
        with self.assertRaises(ValueError):expand_model(old)


if __name__ == '__main__':unittest.main()
