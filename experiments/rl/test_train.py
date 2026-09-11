"""Offline split, evaluation and checkpoint-pool regression checks."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from astra_play_sf2.runner import sha256
from .dataset import load_dataset
from .env import EVAL_LEADS
from .train import evaluate, summarize, selection_rank
from .test_runtime import LuaRuntime
from . import test_runtime


class DatasetTests(unittest.TestCase):
    def dataset(self, folder):
        root = Path(folder)
        rows = []
        for index, split in enumerate(('train', 'dev', 'holdout')):
            sample = root/f'{split}.sta'
            sample.write_bytes(f'native state {index}'.encode())
            rows.append({'id': split, 'status': 'accepted', 'split': split,
                         'path': sample.name, 'sha256': sha256(sample),
                         'difficulty': 7, 'opponent': 2})
        data = {'schema': 'astra.rl-openings.v1', 'status': 'complete',
                'difficulty': 7, 'opponent': 2, 'openings': rows}
        path = root/'manifest.json'
        path.write_text(json.dumps(data))
        return path, data

    def test_loads_separate_portable_groups(self):
        with tempfile.TemporaryDirectory() as folder:
            path, _ = self.dataset(folder)
            groups, difficulty = load_dataset(path)
            self.assertEqual(difficulty, 7)
            self.assertEqual(set(groups), {'train', 'dev', 'holdout'})
            self.assertEqual(groups['dev'][0]['id'], 'dev')
            self.assertTrue(Path(groups['train'][0]['path']).is_absolute())

    def test_rejects_duplicate_state_across_splits(self):
        with tempfile.TemporaryDirectory() as folder:
            path, data = self.dataset(folder)
            data['openings'][1].update(path=data['openings'][0]['path'], sha256=data['openings'][0]['sha256'])
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                load_dataset(path)

    def test_rejects_incomplete_or_modified_dataset(self):
        with tempfile.TemporaryDirectory() as folder:
            path, data = self.dataset(folder)
            data['status'] = 'budget_exhausted'
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'completed'):
                load_dataset(path)
            data['status'] = 'complete'
            path.write_text(json.dumps(data))
            (Path(folder)/'holdout.sta').write_bytes(b'changed state')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                load_dataset(path)

    def test_rejects_path_escape_before_reading_external_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path, data = self.dataset(folder)
            data['openings'][0]['path'] = '../outside.sta'
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'outside'):
                load_dataset(path)


class EvaluationTests(unittest.TestCase):
    def test_explicitly_covers_each_checkpoint_and_lead_deterministically(self):
        env = Mock()
        env.checkpoints = [{}, {}]
        env.episodes = [{'outcome': 'win', 'return': 1.25}]
        env.reset.return_value = ('observation', {})
        env.step.return_value = ('next', 1.25, True, False, {})
        model = Mock()
        model.predict.return_value = (3, None)
        result = evaluate(env, model, 'dev')
        self.assertEqual([c.kwargs['options'] for c in env.reset.call_args_list],
                         [{'checkpoint': index, 'lead': lead} for index in range(2) for lead in EVAL_LEADS])
        self.assertEqual(result['episodes'], 2*len(EVAL_LEADS))
        self.assertEqual(result['win_rate'], 1)
        self.assertTrue(all(c.kwargs == {'deterministic': True} for c in model.predict.call_args_list))
        model.learn.assert_not_called()

    def test_macro_selection_prioritizes_weakest_opponent(self):
        rows = [{'opponent': 0, 'outcome': 'win', 'return': 1}]*9
        rows += [{'opponent': 2, 'outcome': 'loss', 'return': -1}]
        stats = summarize(rows)
        self.assertEqual(stats['win_rate'], .9)
        self.assertEqual(stats['macro_win_rate'], .5)
        self.assertEqual(selection_rank(stats)[:2], (0, .5))
        balanced = summarize([{'opponent': opponent, 'outcome': outcome, 'return': 0}
                             for opponent in (0, 2) for outcome in ('win', 'loss')])
        self.assertGreater(selection_rank(balanced), selection_rank(stats))

    def test_predeclared_single_lead_reduces_dev_cost(self):
        env = Mock()
        env.checkpoints = [{}, {}]
        env.episodes = [{'opponent': 0, 'outcome': 'win', 'return': 1}]
        env.reset.return_value = ('observation', {})
        env.step.return_value = ('next', 1, True, False, {})
        model = Mock()
        model.predict.return_value = (3, None)
        self.assertEqual(evaluate(env, model, 'dev', leads=(2,))['episodes'], 2)
        self.assertEqual([c.kwargs['options']['lead'] for c in env.reset.call_args_list], [2, 2])

    def test_baseline_never_uses_model_predictions(self):
        env = Mock()
        env.checkpoints = [{}]
        env.episodes = [{'outcome': 'loss', 'return': -1}]
        env.reset.return_value = ('observation', {})
        env.step.return_value = ('next', -1, True, False, {})
        model = Mock()
        stats = evaluate(env, model, 'baseline-dev', baseline=True)
        self.assertTrue(env.baseline)
        self.assertEqual(stats['rounds'], {'loss': len(EVAL_LEADS)})
        model.predict.assert_not_called()


@unittest.skipUnless(LuaRuntime, 'Install lupa to execute the real Lua runtime')
class LuaPoolTests(unittest.TestCase):
    def test_pool_index_and_baseline_flags_are_independent(self):
        harness = test_runtime.RuntimeBoundaryTests(methodName='runTest')
        harness.setUp()
        lua = harness.lua
        lua.globals().modules['training/runtime/rl_checkpoint.lua'] = lua.table_from(['/first.sta', '/second.sta'])
        lua.globals().enqueue(1, 'reset', 0, 2)
        self.assertEqual(lua.globals().loaded_path, '/second.sta')
        lua.globals().on_load()
        harness.advance(2)
        self.assertTrue(harness.reply(1)['reset_confirmed'])
        self.assertNotEqual(lua.globals().core_options.choose, lua.globals().choose)
        lua.globals().enqueue(2, 'reset', 0, 3)
        self.assertEqual(lua.globals().loaded_path, '/second.sta')
        lua.globals().on_load()
        harness.advance(2)
        self.assertTrue(harness.reply(2)['reset_confirmed'])
        self.assertTrue(lua.eval('core_options.choose == choose'))

    def test_out_of_range_index_fails_without_loading(self):
        harness = test_runtime.RuntimeBoundaryTests(methodName='runTest')
        harness.setUp()
        harness.lua.globals().modules['training/runtime/rl_checkpoint.lua'] = harness.lua.table_from(['/only.sta'])
        harness.lua.globals().enqueue(1, 'reset', 0, 2)
        self.assertIn('Invalid checkpoint index', harness.reply(1)['error'])
        self.assertIsNone(harness.lua.globals().loaded_path)


class ManagedVectorTests(unittest.TestCase):
    def test_close_handles_broken_pipe_without_waiting_for_a_result(self):
        from .vector import ManagedVec
        vector = ManagedVec.__new__(ManagedVec)
        vector.closed = False
        vector.waiting = True
        vector.close_timeout = 0
        vector.close_errors = []
        bad, good = Mock(), Mock()
        bad.send.side_effect = BrokenPipeError('worker exited')
        vector.remotes = (bad, good)
        vector.work_remotes = (Mock(), Mock())
        workers = [Mock(), Mock()]
        for worker in workers:
            worker.is_alive.return_value = False
        vector.processes = workers
        vector.close()
        good.send.assert_called_once_with(('close', None))
        for remote in (bad, good):
            remote.recv.assert_not_called()
            remote.close.assert_called_once()
        self.assertTrue(vector.closed)
        self.assertFalse(vector.waiting)
        self.assertTrue(vector.close_errors)
        vector.close()
        good.send.assert_called_once()

    def test_stuck_owned_worker_is_terminated_then_killed_with_bounded_joins(self):
        from .vector import ManagedVec
        vector = ManagedVec.__new__(ManagedVec)
        vector.closed = False
        vector.waiting = False
        vector.close_timeout = 0
        vector.close_errors = []
        vector.remotes = ()
        vector.work_remotes = ()
        process = Mock()
        process.is_alive.side_effect = [True, True, False]
        vector.processes = [process]
        vector.close()
        process.terminate.assert_called_once()
        process.kill.assert_called_once()
        self.assertEqual(process.join.call_count, 3)
        self.assertTrue(all('timeout' in call.kwargs and 0 <= call.kwargs['timeout'] <= 5
                            for call in process.join.call_args_list))

    def test_partial_constructor_failure_reaps_already_started_worker(self):
        from unittest.mock import patch
        from .vector import ManagedVec
        worker = Mock()
        worker.is_alive.return_value = False
        pipe = Mock()
        def fail(instance, *_args, **_kwargs):
            instance.processes = [worker]
            instance.remotes = (pipe,)
            raise RuntimeError('worker initialization failed')
        with patch('experiments.rl.vector.SubprocVecEnv.__init__', fail):
            with self.assertRaisesRegex(RuntimeError, 'initialization'):
                ManagedVec([], close_timeout=0)
        pipe.send.assert_called_once_with(('close', None))
        pipe.close.assert_called_once()
        worker.join.assert_called_once()


if __name__ == '__main__':
    unittest.main()
