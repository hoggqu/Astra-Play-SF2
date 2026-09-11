"""Bounded prefetch fixtures and actual harmless owned-process cleanup tests."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from astra_play_sf2.config import atomic_json
from astra_play_sf2.runner import sha256
from .reliability_pipeline import OwnedStage, INTERFACE, live_group_members, pipeline
from .test_reliability_campaign import verification


class PrefetchTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.dataset = self.root/'data.json'; atomic_json(self.dataset, {})
        self.model = self.root/'model.zip'; self.model.write_bytes(b'initial')
        self.source = self.root/'source.py'; self.source.write_text('frozen')
        self.handles = []; self.events = []; self.scores = [9, 10]
        self.train_delay = 1; self.verify_delay = 3
        self.bad_verify = False; self.bad_train = False; self.mutation = None
        self.fail_start = False
        self.race_complete_on_stop = False

    def factory(self, code, module, args, log, timeout):
        fixture = self
        kind = 'train' if module.endswith('batch_train') else 'continuous'
        if self.fail_start and kind == 'train':
            raise RuntimeError('Cannot start training')
        strings = list(map(str, args))
        folder = Path(strings[strings.index('--output')+1]); folder.mkdir()
        model = Path(strings[strings.index('--init-model' if kind == 'train' else '--model')+1])
        ordinal = int(folder.parent.name.split('-')[-1])
        self.events.append(('start', kind, ordinal))
        log.write_text('retained log')
        fixture = self
        class Handle:
            def __init__(self):
                self.pid = 100+len(fixture.handles)
                self.kind = kind; self.ordinal = ordinal; self.args = strings
                self.finished = False; self.stopped = False; self.polls = 0
                self.delay = fixture.train_delay if kind == 'train' else fixture.verify_delay

            def poll(self):
                self.polls += 1
                return 0 if self.polls >= self.delay else None

            def finish(self):
                self.finished = True
                fixture.events.append(('finish', kind, ordinal))
                if fixture.mutation:
                    target = fixture.source if fixture.mutation == 'source' else model
                    target.write_text('changed')
                if kind == 'train':
                    trained = folder/'ppo-batch.zip'; trained.write_bytes(f'model-{ordinal}'.encode())
                    atomic_json(folder/'result.json', {
                        'schema': 'astra.rl-batch-prototype.actions16.v1',
                        'status': 'invalid' if fixture.bad_train else 'complete',
                        'actual_steps': 2048, 'difficulty': 3, 'benchmark': False, 'parity': False,
                        'native_parity': False, 'parameters_changed': True, 'init_model_sha256': sha256(model),
                        'dataset_sha256': sha256(fixture.dataset), 'model_sha256': sha256(trained),
                        'actions': 16, 'action_interface': INTERFACE,
                        'optimizer_initialization': {'loaded_from_init_model': True},
                        'opponent_sampling': {'identity': 'fixed_weights'}})
                    return 2 if fixture.bad_train else 0
                score = fixture.scores[min(ordinal-1, len(fixture.scores)-1)]
                value = verification(score, sha256(model))
                if fixture.bad_verify:
                    value['attempts'] = value['attempts'][:10]
                atomic_json(folder/'result.json', value)
                return 0 if score else 1

            def stop(self, reason):
                self.stopped = True
                fixture.events.append(('stop', kind, ordinal))
                if fixture.race_complete_on_stop and kind == 'train' and not self.finished:
                    self.finish()
                if not self.finished:
                    data = {'status': 'invalid', 'error': 'KeyboardInterrupt: SIGTERM'}
                    if kind == 'train':
                        checkpoint = folder/'checkpoint-000002048.zip'; checkpoint.write_bytes(b'saved optimizer')
                        data.update(completed_update_steps=3072, last_checkpoint={
                            'path': checkpoint.name, 'steps': 2048, 'sha256': sha256(checkpoint), 'complete_update': True})
                    atomic_json(folder/'result.json', data)
                return {'reason': reason, 'cancel_requested': not self.finished, 'remaining_live_pids': []}
        handle = Handle()
        self.handles.append(handle)
        self.assertLessEqual(sum(not h.finished and not h.stopped for h in self.handles), 2)
        return handle

    def run_pipeline(self, cycles=3):
        with patch('experiments.rl.reliability_pipeline.load_dataset', return_value=({}, 3)), \
             patch('experiments.rl.reliability_pipeline.execution_sources', return_value={'source.py': self.source}):
            return pipeline(self.dataset, self.root/'out', self.model, self.root/'train-code', 'trainer',
                self.root/'verify-code', 'verifier', cycles=cycles, steps_per_cycle=2048,
                stage_factory=self.factory, poll_interval=0)

    def test_one_generation_prefetch_waits_for_full20_and_preserves_optimizer(self):
        result = self.run_pipeline(cycles=2)
        self.assertTrue(result['goal_achieved'])
        self.assertEqual([r['clears'] for r in result['cycles']], [9, 10])
        self.assertLess(self.events.index(('finish', 'train', 2)), self.events.index(('finish', 'continuous', 1)))
        self.assertLess(self.events.index(('finish', 'continuous', 1)), self.events.index(('start', 'continuous', 2)))
        self.assertEqual(result['completed_new_training_steps'], 2048)
        self.assertEqual(result['cycles'][1]['opponent_sampling'], {'identity': 'fixed_weights'})
        train = next(h for h in self.handles if h.kind == 'train')
        self.assertEqual(train.args[train.args.index('--init-model')+1], str(self.model))
        self.assertEqual(train.args[train.args.index('--seed')+1], '129')
        for handle in self.handles:
            if handle.kind == 'continuous':
                self.assertIn('--all-attempts', handle.args)
                self.assertEqual(handle.args[handle.args.index('--attempts')+1], '20')

    def test_goal_cancels_only_prefetch_and_retains_verified_checkpoint(self):
        self.scores = [10]; self.train_delay = 100
        result = self.run_pipeline()
        self.assertTrue(result['goal_achieved'])
        self.assertEqual(result['completed_candidate_evaluations'], 1)
        self.assertEqual(result['completed_new_training_steps'], 0)
        future = result['cycles'][1]
        self.assertEqual(future['status'], 'cancelled')
        self.assertEqual(future['train']['stop_reason'], 'owner_cancelled_after_verified_goal')
        self.assertTrue(future['train']['retained_progress']['last_checkpoint_verified'])
        self.assertEqual(future['train']['retained_progress']['last_checkpoint']['steps'], 2048)
        self.assertEqual(future['train']['retained_progress']['completed_update_steps'], 3072)
        self.assertEqual([e for e in self.events if e[0] == 'stop'], [('stop', 'train', 2)])
        self.assertEqual(len(result['cycles']), 2)

    def test_two_nines_never_pool_no_generation_beyond_budget(self):
        self.scores = [9, 9]
        result = self.run_pipeline(cycles=2)
        self.assertEqual(result['status'], 'complete'); self.assertFalse(result['goal_achieved'])
        self.assertEqual(result['maximum_new_training_steps'], 2048)
        self.assertEqual(len([h for h in self.handles if h.kind == 'train']), 1)

    def test_single_candidate_never_trains(self):
        self.scores = [10]
        result = self.run_pipeline(cycles=1)
        self.assertTrue(result['goal_achieved']); self.assertEqual(len(self.handles), 1)

    def test_incomplete_group_stops_prefetch_without_replacement_coins(self):
        self.bad_verify = True; self.train_delay = 100
        result = self.run_pipeline()
        self.assertEqual(result['status'], 'invalid'); self.assertFalse(result['goal_achieved'])
        self.assertEqual(len(self.handles), 2)
        self.assertTrue(next(h for h in self.handles if h.kind == 'train').stopped)

    def test_failed_training_stops_evaluation_not_mislabelled_as_goal_cancellation(self):
        self.bad_train = True; self.scores = [10]
        result = self.run_pipeline()
        self.assertEqual(result['status'], 'invalid'); self.assertFalse(result['goal_achieved'])
        self.assertEqual(len(self.handles), 2)
        self.assertTrue(next(h for h in self.handles if h.kind == 'continuous').stopped)
        self.assertEqual(result['cycles'][1]['train']['status'], 'invalid')

    def test_source_change_invalidates_and_cleans_both_stages(self):
        self.mutation = 'source'
        result = self.run_pipeline()
        self.assertEqual(result['status'], 'invalid')
        self.assertTrue(all(h.stopped for h in self.handles))

    def test_model_change_invalidates_without_accepting_changed_child_identity(self):
        self.mutation = 'model'
        result = self.run_pipeline()
        self.assertEqual(result['status'], 'invalid'); self.assertFalse(result['goal_achieved'])
        self.assertTrue(all(h.stopped for h in self.handles))

    def test_training_completion_racing_goal_is_validated_not_called_cancelled(self):
        self.scores = [10]; self.train_delay = 100; self.race_complete_on_stop = True
        result = self.run_pipeline()
        self.assertTrue(result['goal_achieved'])
        self.assertEqual(result['completed_new_training_steps'], 2048)
        self.assertEqual(result['cycles'][1]['status'], 'not_evaluated_after_goal')
        self.assertEqual(result['cycles'][1]['train']['status'], 'complete')

    def test_training_failure_racing_goal_stays_invalid(self):
        self.scores = [10]; self.train_delay = 100; self.race_complete_on_stop = True; self.bad_train = True
        result = self.run_pipeline()
        self.assertEqual(result['status'], 'invalid'); self.assertFalse(result['goal_achieved'])
        self.assertEqual(result['verified_goal_cycle'], 1)
        self.assertEqual(result['cycles'][0]['clears'], 10)

    def test_second_launch_error_cleans_first_child(self):
        self.fail_start = True
        result = self.run_pipeline()
        self.assertEqual(result['status'], 'invalid')
        self.assertEqual(len(self.handles), 1); self.assertTrue(self.handles[0].stopped)


@unittest.skipUnless(os.name == 'posix', 'Owned process-group pipeline is a POSIX experiment')
class RealProcessTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

    def stage(self, source, timeout=10):
        (self.root/'probe.py').write_text(source)
        stage = OwnedStage(self.root, 'probe', [], self.root/'probe.log', timeout)
        self.addCleanup(lambda: stage.stop('test_cleanup', grace=.2))
        return stage

    def wait_file(self, path):
        deadline = time.monotonic()+5
        while not path.exists() and time.monotonic() < deadline:
            time.sleep(.02)
        self.assertTrue(path.exists())

    def test_stop_reaps_owned_descendant_but_leaves_unrelated_process(self):
        unrelated = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(30)'], start_new_session=True)
        self.addCleanup(lambda: (unrelated.terminate(), unrelated.wait()))
        stage = self.stage("import subprocess,sys,pathlib,time\np=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])\npathlib.Path('child.pid').write_text(str(p.pid))\ntime.sleep(30)\n")
        self.wait_file(self.root/'child.pid')
        self.assertIn(int((self.root/'child.pid').read_text()), live_group_members(stage.pgid))
        stopped = stage.stop('verified_goal', grace=.2)
        self.assertTrue(stopped['cancel_requested']); self.assertEqual(live_group_members(stage.pgid), [])
        self.assertIsNone(unrelated.poll())

    def test_finished_parent_with_orphan_is_cleaned_and_invalid(self):
        stage = self.stage("import subprocess,sys,pathlib\np=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])\npathlib.Path('child.pid').write_text(str(p.pid))\n")
        self.wait_file(self.root/'child.pid'); stage.process.wait(timeout=5)
        with self.assertRaisesRegex(RuntimeError, 'live owned descendants'):
            stage.finish()
        self.assertEqual(live_group_members(stage.pgid), [])

    def test_timeout_stops_owned_process_and_keeps_log(self):
        stage = self.stage("import time\nprint('retained',flush=True)\ntime.sleep(30)\n", timeout=.05)
        time.sleep(.1)
        with self.assertRaises(TimeoutError):
            stage.poll()
        stage.stop('pipeline_timeout', grace=.2)
        self.assertEqual(live_group_members(stage.pgid), [])
        self.assertIn('retained', (self.root/'probe.log').read_text())

    def test_sigterm_ignoring_descendant_is_killed_only_in_our_group(self):
        source = ("import subprocess,sys,pathlib,time\n"
                  "child='import signal,pathlib,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);pathlib.Path(\"ready\").write_text(\"yes\");time.sleep(30)'\n"
                  "p=subprocess.Popen([sys.executable,'-c',child])\ntime.sleep(30)\n")
        stage = self.stage(source)
        self.wait_file(self.root/'ready')
        stage.stop('forced_cleanup', grace=.1)
        self.assertEqual(live_group_members(stage.pgid), [])


if __name__ == '__main__':
    unittest.main()
