import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from .scenario_dataset import build
from .versioned_campaign import digest
from .managed_runtime import select_device


class ScenarioDatasetTests(unittest.TestCase):
    def fixture(self,root,label,opponents,count):
        folder=root/label;folder.mkdir();rows=[]
        for op in opponents:
            for i in range(count):
                f=folder/f'{op}-{i}.sta';f.write_bytes(f'{label}:{op}:{i}'.encode())
                rows.append(dict(id=f'{op}-{i}',path=f.name,sha256=digest(f),opponent=op,
                    status='accepted',difficulty=3,attempt=i+1,
                    split='train' if i<count-2 else 'dev' if i==count-2 else 'holdout'))
        path=folder/'manifest.json';path.write_text(json.dumps(dict(schema='astra.rl-openings.v2',
            status='complete',difficulty=3,opponents=list(opponents),openings=rows)))
        return path

    def test_same_budget_preserved_and_no_evaluation_attempt_enters_training(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);base=self.fixture(root,'base',(0,1,2,3,5,6,7,8,9,10,11),10)
            fresh=self.fixture(root,'fresh',(0,2,5),8)
            results=build(base,fresh,root/'out')
            a,b,e=[json.loads(Path(results[k]['path']).read_text()) for k in ('A','B','evaluation')]
            train=lambda d:[r for r in d['openings'] if r['split']=='train']
            self.assertEqual(len(train(a)),88);self.assertEqual(len(train(b)),88)
            bt={r['sha256'] for r in train(b)}
            self.assertFalse(bt & {r['sha256'] for r in e['openings'] if r['split']!='train'})
            for op in (0,2,5):
                added=[r for r in train(b) if r['opponent']==op and r['id'].startswith('fresh-')]
                self.assertEqual(len(added),4)
                self.assertTrue(all(r['attempt']<7 for r in added))
            self.assertEqual([r for r in a['openings'] if r['split']!='train'],[r for r in b['openings'] if r['split']!='train'])

    def test_incomplete_collection_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);base=self.fixture(root,'base',(0,2,5),10);fresh=self.fixture(root,'fresh',(0,2,5),8)
            d=json.loads(fresh.read_text());d['status']='budget_exhausted';fresh.write_text(json.dumps(d))
            with self.assertRaises(ValueError):build(base,fresh,root/'out')


class GPUSelectionTests(unittest.TestCase):
    def test_explicit_mps_and_cuda_never_silently_fall_back(self):
        with patch('torch.cuda.is_available',return_value=False),patch('torch.backends.mps.is_available',return_value=False):
            for device in ('mps','cuda'):
                with self.assertRaises(RuntimeError):select_device(device)
            self.assertEqual(select_device('auto'),'cpu')
        with patch('torch.cuda.is_available',return_value=False),patch('torch.backends.mps.is_available',return_value=True):
            self.assertEqual(select_device('mps'),'mps');self.assertEqual(select_device('auto'),'mps')
        with patch('torch.cuda.is_available',return_value=True),patch('torch.backends.mps.is_available',return_value=True):
            self.assertEqual(select_device('auto'),'cuda');self.assertEqual(select_device('mps'),'mps')


if __name__=='__main__':unittest.main()
