"""Checkpoint transaction and interrupted-update preservation, without MAME."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO

from astra_play_sf2.runner import sha256
from .batch_train import checkpoint_interval, main, save_model_checkpoint


class CheckpointTests(unittest.TestCase):
    def test_interval_rounds_up_to_complete_rollout(self):
        self.assertEqual(checkpoint_interval(20480, 8*256), 20480)
        self.assertEqual(checkpoint_interval(20480, 3*256), 20736)
        with self.assertRaises(ValueError):
            checkpoint_interval(0, 2048)

    def test_actual_ppo_optimizer_and_parameters_round_trip(self):
        class Dummy(gym.Env):
            observation_space = gym.spaces.Box(-1, 1, (344,), dtype=np.float32)
            action_space = gym.spaces.Discrete(15)
        torch.set_num_threads(1)
        model = PPO('MlpPolicy', Dummy(), n_steps=256, batch_size=64, seed=4, device='cpu')
        # Produce real Adam state without an emulator; checkpoint must retain it.
        loss = sum(parameter.square().sum() for parameter in model.policy.parameters())
        model.policy.optimizer.zero_grad();loss.backward();model.policy.optimizer.step()
        model.num_timesteps = 20480;model._n_updates = 4
        with tempfile.TemporaryDirectory() as folder:
            row = save_model_checkpoint(model, folder, 20480)
            self.assertEqual(row['path'], 'checkpoint-000020480.zip')
            self.assertEqual(sha256(Path(folder)/row['path']), row['sha256'])
            loaded = PPO.load(Path(folder)/row['path'], device='cpu')
            self.assertEqual(loaded.num_timesteps, 20480)
            self.assertEqual(loaded._n_updates, 4)
            for key, tensor in model.policy.state_dict().items():
                torch.testing.assert_close(loaded.policy.state_dict()[key], tensor)
            old = model.policy.optimizer.state_dict()['state']
            new = loaded.policy.optimizer.state_dict()['state']
            self.assertTrue(old)
            for key, values in old.items():
                for field, value in values.items():
                    torch.testing.assert_close(new[key][field], value)
            with self.assertRaises(FileExistsError):
                save_model_checkpoint(model, folder, 20480)
            self.assertFalse(list(Path(folder).glob('.*.tmp.zip')))

    def test_partial_zip_never_published_and_previous_checkpoint_survives(self):
        model = Mock(num_timesteps=256, _n_updates=4)
        with tempfile.TemporaryDirectory() as folder:
            model.save.side_effect = lambda path: Path(path).write_bytes(b'complete model')
            first = save_model_checkpoint(model, folder, 256)
            model.num_timesteps = 512
            def broken(path):
                Path(path).write_bytes(b'partial zip')
                raise OSError('disk failure')
            model.save.side_effect = broken
            with self.assertRaisesRegex(OSError, 'disk failure'):
                save_model_checkpoint(model, folder, 512)
            self.assertEqual(sha256(Path(folder)/first['path']), first['sha256'])
            self.assertEqual([p.name for p in Path(folder).iterdir()], [first['path']])

    def test_failed_later_update_retains_previous_complete_model_and_invalid_run(self):
        policy = Mock(net_arch={'pi':[64,64], 'vf':[64,64]})
        policy.state_dict.return_value = {'weight': torch.tensor([1.])}
        model = Mock(policy=policy, _n_updates=0, n_steps=256, gamma=.99, gae_lambda=.95,
                     ent_coef=.01, learning_rate=.0003, n_epochs=4, batch_size=64)
        updates = 0
        def train():
            nonlocal updates
            updates += 1
            model._n_updates += 4
            if updates == 2:
                raise RuntimeError('interrupted halfway through optimizer update')
        model.train.side_effect = train
        model.save.side_effect = lambda path: Path(path).write_bytes(f'optimizer-{model._n_updates}'.encode())
        vector = Mock(close_errors=[])
        vector.env_method.return_value = [{'model_sha256':'policy', 'transitions':[],
                                          'observation':[0]*344, 'episode_start':False}]
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/'run';dataset = Path(folder)/'dataset.json';dataset.write_text('{}')
            argv = ['batch_train', '--dataset', str(dataset), '--output', str(output), '--workers', '1',
                    '--steps', '512', '--block', '256', '--checkpoint-every', '256']
            with patch('sys.argv', argv), patch('experiments.rl.batch_train.load_config', return_value={}), \
                 patch('experiments.rl.batch_train.load_dataset', return_value=({'train':[]},3)), \
                 patch('experiments.rl.batch_train.ManagedVec', return_value=vector), \
                 patch('experiments.rl.batch_train.PPO', return_value=model), \
                 patch('experiments.rl.batch_train.live_policy', return_value={'model_sha256':'policy'}), \
                 patch('experiments.rl.batch_train.fill_buffer', return_value=0), \
                 patch('experiments.rl.batch_train.signal.signal'), redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, 'halfway'):
                    main()
            result = json.loads((output/'result.json').read_text())
            self.assertEqual(result['status'], 'invalid')
            self.assertEqual(result['actual_steps'], 256)
            self.assertEqual(result['completed_update_steps'], 256)
            self.assertEqual(result['last_checkpoint']['steps'], 256)
            self.assertEqual(len(result['checkpoints']), 1)
            self.assertEqual((output/result['last_checkpoint']['path']).read_bytes(), b'optimizer-4')
            self.assertFalse((output/'ppo-batch.zip').exists())
            vector.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
