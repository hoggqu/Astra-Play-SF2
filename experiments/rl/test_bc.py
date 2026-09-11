"""Offline BC validation, leakage guards, and policy-only update checks."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from stable_baselines3 import PPO
from astra_play_sf2.runner import sha256
from .bc import episode_partition, load_demonstrations, train
from .projection_teacher import ProjectionTeacher


def fixture(root):
    dataset = root/'dataset'
    dataset.mkdir()
    openings = []
    for index, split in enumerate(('train', 'dev', 'holdout')):
        state = dataset/f'{index}.sta'
        state.write_bytes(bytes([index]))
        openings.append({'id': str(index), 'status': 'accepted', 'split': split,
                         'path': state.name, 'sha256': sha256(state), 'difficulty': 3, 'opponent': 2})
    manifest = dataset/'manifest.json'
    manifest.write_text(json.dumps({'schema': 'astra.rl-openings.v2', 'status': 'complete',
                                   'opponents': [2], 'difficulty': 3, 'openings': openings}))
    demos = root/'demos'
    demos.mkdir()
    np.savez_compressed(demos/'demonstrations.npz', observations=np.zeros((8, 344), dtype=np.float32),
                        actions=np.full(8, 12, dtype=np.int64), episode_ids=np.repeat(np.arange(4), 2),
                        checkpoint_indices=np.zeros(8, dtype=np.int64))
    rows = [{'checkpoint': 0, 'steps': 2, 'opponent': 2, 'outcome': 'win',
             'phase': 'projected-teacher-train', 'baseline': False} for _ in range(4)]
    result = {'schema': 'astra.rl-projected-teacher.v1', 'status': 'complete', 'split': 'train',
              'training_only': True, 'formal_clear': False, 'difficulty': 3,
              'teacher': ProjectionTeacher().identity, 'dataset_sha256': sha256(manifest),
              'demonstrations_sha256': sha256(demos/'demonstrations.npz'),
              'demonstration_count': 8, 'steps': 8, 'round_budget': 4, 'episodes': rows}
    (demos/'result.json').write_text(json.dumps(result))
    return demos, manifest


class BCTests(unittest.TestCase):
    def test_whole_episode_partition_keeps_opponents_in_both_sets(self):
        episodes = np.repeat(np.arange(4), 3)
        opponents = np.repeat([1, 2, 1, 2], 3)
        fit, diagnostic = episode_partition(episodes, opponents)
        self.assertFalse(set(episodes[fit]) & set(episodes[diagnostic]))
        self.assertEqual(set(opponents[fit]), {1, 2})
        self.assertEqual(set(opponents[diagnostic]), {1, 2})

    def test_validated_demo_and_rejects_evaluation_split(self):
        with tempfile.TemporaryDirectory() as folder:
            demos, dataset = fixture(Path(folder))
            arrays, _ = load_demonstrations(demos, dataset)
            self.assertEqual(arrays['observations'].shape, (8, 344))
            record = json.loads((demos/'result.json').read_text())
            record['split'] = 'dev'
            (demos/'result.json').write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, 'train-split'):
                load_demonstrations(demos, dataset)

    def test_modified_array_and_teacher_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            demos, dataset = fixture(Path(folder))
            record = json.loads((demos/'result.json').read_text())
            record['teacher']['projection_sha256'] = '0'*64
            (demos/'result.json').write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, 'identity'):
                load_demonstrations(demos, dataset)
            record['teacher'] = ProjectionTeacher().identity
            (demos/'result.json').write_text(json.dumps(record))
            with (demos/'demonstrations.npz').open('ab') as file:
                file.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                load_demonstrations(demos, dataset)

    def test_policy_updates_value_unchanged_and_sb3_reload(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            demos, dataset = fixture(root)
            args = argparse.Namespace(demonstrations=demos, dataset=dataset, output=root/'bc',
                                      init_model=None, epochs=2, batch_size=4, learning_rate=.001,
                                      seed=42, balance_opponents=True)
            result = train(args)
            self.assertEqual(result['status'], 'complete')
            self.assertTrue(result['policy_parameters_changed'])
            self.assertTrue(result['value_and_other_parameters_unchanged'])
            self.assertEqual(result['environment_steps'], 0)
            model = PPO.load(root/'bc/ppo-bc.zip', device='cpu')
            self.assertEqual(model.policy.observation_space.shape, (344,))
            self.assertEqual(model.policy.action_space.n, 15)
            # Existing learned value parameters are preserved on initialization too.
            args.init_model, args.output, args.epochs = root/'bc/ppo-bc.zip', root/'bc2', 1
            self.assertTrue(train(args)['value_and_other_parameters_unchanged'])


if __name__ == '__main__':
    unittest.main()
