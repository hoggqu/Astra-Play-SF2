import json
from pathlib import Path
import tempfile
import unittest
from astra_play_sf2.runner import sha256
from .dataset import load_dataset


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = dict(schema='astra.rl-openings.v1', status='complete', difficulty=7, opponent=2, openings=[])
        for split in ('train', 'dev', 'holdout'):
            p = self.root/f'{split}.sta'
            p.write_bytes(split.encode())
            self.data['openings'].append(dict(id=split, path=p.name, sha256=sha256(p), split=split,
                                             difficulty=7, opponent=2, status='accepted'))

    def load(self):
        p = self.root/'manifest.json'
        p.write_text(json.dumps(self.data))
        return load_dataset(p)

    def test_disjoint_validated_splits_use_absolute_local_paths(self):
        groups, level = self.load()
        self.assertEqual(level, 7)
        self.assertTrue(all(Path(rows[0]['path']).is_absolute() for rows in groups.values()))
        self.assertEqual([groups[key][0]['id'] for key in groups], ['train','dev','holdout'])

    def test_duplicate_checkpoint_cannot_cross_split(self):
        self.data['openings'][1].update(path='train.sta', sha256=sha256(self.root/'train.sta'))
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.load()

    def test_tampering_and_incomplete_collection_rejected(self):
        (self.root/'train.sta').write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            self.load()
        self.data['status'] = 'invalid'
        with self.assertRaisesRegex(ValueError, 'completed'):
            self.load()

    def test_multi_opponent_preserves_identity_and_requires_per_split_coverage(self):
        self.data.update(schema='astra.rl-openings.v2', opponents=[0, 2])
        for split in ('train', 'dev', 'holdout'):
            p = self.root/f'ryu-{split}.sta'
            p.write_bytes(p.name.encode())
            self.data['openings'].append(dict(id=p.stem, path=p.name, sha256=sha256(p),
                split=split, difficulty=7, opponent=0, status='accepted'))
        groups, _ = self.load()
        self.assertTrue(all({row['opponent'] for row in rows} == {0, 2} for rows in groups.values()))
        self.data['openings'].pop()
        with self.assertRaisesRegex(ValueError, 'coverage'):
            self.load()

    def test_escape_path_and_wrong_difficulty_rejected(self):
        self.data['openings'][0]['path'] = '../escape.sta'
        with self.assertRaisesRegex(ValueError, 'outside'):
            self.load()
        self.data['openings'][0]['path'] = 'train.sta'
        self.data['openings'][0]['difficulty'] = 3
        with self.assertRaisesRegex(ValueError, 'metadata'):
            self.load()


if __name__ == '__main__':
    unittest.main()
