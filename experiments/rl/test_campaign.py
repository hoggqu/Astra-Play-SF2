"""Exercise autonomous orchestration with structured child artifacts, no MAME."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from astra_play_sf2.config import atomic_json
from astra_play_sf2.runner import sha256
from .campaign import campaign


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.dataset = self.root/'dataset.json'
        atomic_json(self.dataset, {})
        self.calls = []
        self.outcomes = ['loss', 'rl_gameplay_clear']
        self.bad_code = False
        self.interrupt = False
        self.train_calls = 0

    def stage(self, module, arguments, log, timeout):
        args = [str(a) for a in arguments]
        self.calls.append((module, args))
        Path(log).write_text('child retained log')
        output = Path(args[args.index('--output')+1])
        output.mkdir()
        if self.interrupt:
            raise KeyboardInterrupt('test SIGTERM')
        if module == 'experiments.rl.train':
            self.train_calls += 1
            model = output/'best-dev.zip'
            model.write_bytes(f'policy-{self.train_calls}'.encode())
            atomic_json(output/'result.json', dict(schema='astra.rl-scaled.v1', status='complete',
                difficulty=3, actual_steps=2048, skip_holdout=True, best_dev_steps=2048,
                model_sha256=sha256(model)))
            return 0
        model = Path(args[args.index('--model')+1])
        outcome = self.outcomes[min(self.train_calls-1, len(self.outcomes)-1)]
        attempts = [dict(id=f'l3-{i:03d}', outcome=outcome, match_wins=11 if outcome == 'rl_gameplay_clear' else 0,
                         matches=['training/attempts/l3-001/m01-blanka.json'], audit={'ok': True})
                    for i in range(1, 2 if outcome == 'rl_gameplay_clear' else 4)]
        atomic_json(output/'result.json', dict(schema='astra.rl-continuous.v1', status='complete',
            difficulty=3, model_sha256=sha256(model), attempts=attempts))
        return 0 if self.bad_code or outcome == 'rl_gameplay_clear' else 1

    def run_campaign(self, **kwargs):
        with patch('experiments.rl.campaign.load_dataset', return_value=({}, 3)):
            return campaign(self.dataset, self.root/'campaign', workers=8, cycles=5,
                            steps_per_cycle=2048, eval_every=2048, stage_runner=self.stage, **kwargs)

    def test_loss_advances_with_previous_best_then_clear_stops(self):
        result = self.run_campaign()
        self.assertEqual(result['status'], 'complete')
        self.assertTrue(result['goal_achieved'])
        self.assertEqual(result['success_cycle'], 2)
        self.assertEqual(len(result['cycles']), 2)
        self.assertEqual(len(result['cycles'][0]['attempts']), 3)
        self.assertEqual(result['cycles'][0]['attempts'][0]['failed_match'], 'training/attempts/l3-001/m01-blanka.json')
        self.assertNotIn('--init-model', self.calls[0][1])
        second = self.calls[2][1]
        inherited = Path(second[second.index('--init-model')+1])
        self.assertEqual(inherited, self.root/'campaign/cycle-001/train/best-dev.zip')
        self.assertTrue(all('--skip-holdout' in args and args[args.index('--eval-leads')+1] == '2'
                            for module, args in self.calls if module.endswith('.train')))
        self.assertEqual(result['completed_training_steps'], 4096)

    def test_valid_losses_exhaust_budget_without_claiming_success(self):
        self.outcomes = ['loss']
        result = self.run_campaign()
        self.assertEqual(result['status'], 'complete')
        self.assertFalse(result['goal_achieved'])
        self.assertEqual(len(result['cycles']), 5)
        self.assertEqual(sum(len(c['attempts']) for c in result['cycles']), 15)

    def test_exit_code_disagreement_invalidates_without_retry(self):
        self.bad_code = True
        result = self.run_campaign()
        self.assertEqual(result['status'], 'invalid')
        self.assertIn('exit code', result['error'])
        self.assertEqual(len(self.calls), 2)
        self.assertTrue((self.root/'campaign/cycle-001/continuous/result.json').is_file())

    def test_interrupt_retained_and_no_next_stage(self):
        self.interrupt = True
        result = self.run_campaign()
        self.assertEqual(result['status'], 'invalid')
        self.assertIn('KeyboardInterrupt', result['error'])
        self.assertEqual(len(self.calls), 1)
        saved = json.loads((self.root/'campaign/result.json').read_text())
        self.assertEqual(saved['cycles'][0]['status'], 'invalid')

    def test_initial_model_identity_is_retained(self):
        model = self.root/'initial.zip'
        model.write_bytes(b'initial')
        self.outcomes = ['rl_gameplay_clear']
        result = self.run_campaign(init_model=model)
        self.assertEqual(result['initial_model_sha256'], sha256(model))
        self.assertIn(str(model), self.calls[0][1])

    def test_invalid_budgets_fail_before_launch(self):
        with self.assertRaises(ValueError):
            campaign(self.dataset, self.root/'invalid', steps_per_cycle=123, stage_runner=self.stage)
        self.assertFalse(self.calls)


if __name__ == '__main__':
    unittest.main()
