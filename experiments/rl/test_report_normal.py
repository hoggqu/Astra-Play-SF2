"""Small offline reporting fixtures; source artifacts are never modified."""
import json
from pathlib import Path
import tempfile
import unittest

from .report_normal import render_markdown, summarize


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def fixture(root, native=False):
    campaign = root/'campaign'
    write(campaign/'result.json', {'schema': 'astra.rl-native-campaign.v1' if native else 'astra.rl-campaign.v1',
                                  'status': 'complete', 'difficulty': 3, 'seed': 42,
                                  'cycles': [{'ordinal': 1, 'status': 'complete', 'model_sha256': 'model'}]})
    train = campaign/'cycle-001/train'
    write(train/'result.json', {'status': 'complete', 'actual_steps': 36, 'model_sha256': 'model'})
    row = {'phase': 'train', 'baseline': False, 'episode': 1, 'opponent': 2, 'checkpoint': 0,
           'outcome': 'win', 'steps': 12, 'frames': 144}
    path = train/'worker-00/episodes.jsonl'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(row)+'\n'+json.dumps(row)+'\n')
    pending = dict(row, episode=2, outcome='loss')
    native_path = train/'worker-00/training/rl-batch-episodes.jsonl'
    native_path.parent.mkdir()
    native_path.write_text(json.dumps(row)+'\n'+json.dumps(pending)+'\n')
    play = campaign/'cycle-001/continuous'
    write(play/'result.json', {'status': 'complete', 'model_sha256': 'model', 'native_timing': native,
                               'native_timing_audit': {'ok': native},
                               'attempts': [{'id': 'l3-001', 'outcome': 'loss', 'audit': {'ok': True},
                                             'matches': ['training/m01.json']}]})
    write(play/'training/m01.json', {'summary': {'status': 'complete', 'valid_continuous': True,
                                                'model_sha256': 'model', 'opponent': 0, 'result': 'cpu_win', 'score': [0, 2]},
                                      'rounds': [{'outcome': 'loss'}, {'outcome': 'loss'}]})
    return campaign


class ReportTests(unittest.TestCase):
    def test_repeated_campaign_and_native_copies_never_double_count(self):
        with tempfile.TemporaryDirectory() as folder:
            campaign = fixture(Path(folder), native=True)
            before = (campaign/'result.json').read_bytes()
            report = summarize([campaign, campaign/'.'])
            self.assertEqual(len(report['campaigns']), 1)
            self.assertEqual(len(report['duplicate_campaign_paths_ignored']), 1)
            cycle = report['campaigns'][0]['cycles'][0]
            self.assertEqual(cycle['training']['rounds']['2']['win'], 1)
            self.assertEqual(cycle['training']['native_unconfirmed_rounds'], 1)
            self.assertEqual(cycle['training']['native_unconfirmed_by_opponent']['2']['loss'], 1)
            self.assertEqual(cycle['continuous']['rounds']['0']['loss'], 2)
            self.assertEqual(cycle['continuous']['attempts'][0]['failed_opponent_name'], 'Ryu')
            self.assertEqual((campaign/'result.json').read_bytes(), before)
            self.assertIn('待核对', render_markdown(report))

    def test_native_invalid_run_retains_observations_but_not_verified_scores(self):
        with tempfile.TemporaryDirectory() as folder:
            campaign = fixture(Path(folder), native=True)
            path = campaign/'cycle-001/continuous/result.json'
            result = json.loads(path.read_text())
            result['native_timing_audit']['ok'] = False
            write(path, result)
            play = summarize([campaign])['campaigns'][0]['cycles'][0]['continuous']
            self.assertEqual(play['attempt_counts'], {'invalid': 1})
            self.assertEqual(play['rounds'], {})
            self.assertEqual(play['unverified_observed_rounds']['0']['loss'], 2)

    def test_running_attempts_are_pending_not_invalid(self):
        with tempfile.TemporaryDirectory() as folder:
            campaign = fixture(Path(folder), native=True)
            path = campaign/'cycle-001/continuous/result.json'
            result = json.loads(path.read_text())
            result['status'] = 'running'
            result['attempts'][0]['outcome'] = 'invalid'  # Native runner's in-flight placeholder.
            write(path, result)
            play = summarize([campaign])['campaigns'][0]['cycles'][0]['continuous']
            self.assertEqual(play['attempt_counts'], {'pending': 1})
            self.assertEqual(play['rounds'], {})
            self.assertEqual(play['unverified_observed_rounds']['0']['loss'], 2)

    def test_underlying_complete_before_native_final_audit_remains_pending(self):
        with tempfile.TemporaryDirectory() as folder:
            campaign = fixture(Path(folder), native=True)
            path = campaign/'cycle-001/continuous/result.json'
            result = json.loads(path.read_text())
            del result['native_timing_audit']
            del result['native_timing']
            write(path, result)
            play = summarize([campaign])['campaigns'][0]['cycles'][0]['continuous']
            self.assertEqual(play['attempt_counts'], {'pending': 1})
            self.assertEqual(play['audit_state'], 'pending')

    def test_clear_claim_requires_eleven_distinct_native_match_wins(self):
        with tempfile.TemporaryDirectory() as folder:
            campaign = fixture(Path(folder))
            path = campaign/'cycle-001/continuous/result.json'
            result = json.loads(path.read_text())
            result['attempts'][0]['outcome'] = 'rl_gameplay_clear'
            write(path, result)
            play = summarize([campaign])['campaigns'][0]['cycles'][0]['continuous']
            self.assertEqual(play['attempt_counts'], {'invalid': 1})
            self.assertEqual(play['attempts'][0]['reported_outcome'], 'rl_gameplay_clear')

    def test_incomplete_jsonl_tail_is_reported_and_prior_episode_retained(self):
        with tempfile.TemporaryDirectory() as folder:
            campaign = fixture(Path(folder))
            with (campaign/'cycle-001/train/worker-00/episodes.jsonl').open('a') as stream:
                stream.write('{"episode":')
            run = summarize([campaign])['campaigns'][0]
            self.assertTrue(any('unparsed' in issue.get('classification', '') for issue in run['issues']))
            self.assertEqual(run['cycles'][0]['training']['python_completed_rounds'], 1)


if __name__ == '__main__':
    unittest.main()
