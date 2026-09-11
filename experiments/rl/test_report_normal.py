"""Small offline reporting fixtures; source artifacts are never modified."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .report_normal import main, render_markdown, summarize


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def fixture(root, native=False):
    campaign = root/'campaign'
    write(campaign/'result.json', {'schema': 'astra.rl-native-campaign.v1' if native else 'astra.rl-campaign.v1',
                                  'status': 'complete', 'difficulty': 3, 'seed': 42,
                                  'cycles': [{'ordinal': 1, 'status': 'complete', 'model_sha256': 'model'}]})
    train = campaign/'cycle-001/train'
    write(train/'result.json', {'schema': 'astra.rl-batch-prototype.v1', 'difficulty': 3, 'status': 'complete', 'actual_steps': 36, 'model_sha256': 'model'})
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
    write(play/'result.json', {'schema': 'astra.rl-continuous.v1', 'difficulty': 3, 'status': 'complete', 'model_sha256': 'model', 'native_timing': native,
                               'native_timing_audit': {'ok': native},
                               'attempts': [{'id': 'l3-001', 'outcome': 'loss', 'audit': {'ok': True},
                                             'matches': ['training/m01.json']}]})
    write(play/'training/m01.json', {'summary': {'status': 'complete', 'valid_continuous': True,
                                                'model_sha256': 'model', 'opponent': 0, 'result': 'cpu_win', 'score': [0, 2]},
                                      'rounds': [{'outcome': 'loss'}, {'outcome': 'loss'}]})
    return campaign


class ReportTests(unittest.TestCase):
    def test_weighted_sampling_lineage_survives_campaign_and_standalone_reports(self):
        with tempfile.TemporaryDirectory() as folder:
            campaign = fixture(Path(folder), native=True)
            train = campaign/'cycle-001/train'
            metadata = {'identity': 'fixed_opponent_probabilities_v1',
                        'configuration': {'schema': 'astra.rl-fixed-opponent-sampling.v1',
                            'probabilities': {str(op): 1/11 for op in (0,1,2,3,5,6,7,8,9,10,11)},
                            'formula': {'kind': 'uniform_floor_failure_squared', 'alpha': .4},
                            'statistics_source': {'kind': 'training_only', 'sha256': 'stats-hash',
                                                  'round_scope': ['R1', 'R2', 'R3']}},
                        'configuration_file_sha256': 'config-hash', 'input_weights_sha256': 'weights-hash',
                        'build_manifest_sha256': 'build-hash', 'parent_chain_build_sha256': 'parent-hash'}
            originals = {}
            for path in (campaign/'result.json', train/'result.json'):
                result = json.loads(path.read_text())
                result['opponent_sampling'] = metadata
                write(path, result)
                originals[path] = path.read_bytes()
            run = summarize([campaign])['campaigns'][0]
            self.assertEqual(run['opponent_sampling'], metadata)
            self.assertEqual(run['cycles'][0]['training']['opponent_sampling'], metadata)
            standalone = summarize(training_paths=[train])
            self.assertEqual(standalone['standalone'][0]['summary']['opponent_sampling'], metadata)
            for report in (summarize([campaign]), standalone):
                rendered = render_markdown(report)
                for text in ('fixed_opponent_probabilities_v1', 'uniform_floor_failure_squared',
                             'stats-hash', 'config-hash', 'weights-hash', 'build-hash', 'parent-hash'):
                    self.assertIn(text, rendered)
            self.assertEqual(originals, {path: path.read_bytes() for path in originals})
            self.assertEqual(run['aggregate']['training_rounds']['2']['win'], 1)

    def test_sampling_metadata_is_never_inherited_or_invented_for_old_stages(self):
        with tempfile.TemporaryDirectory() as folder:
            campaign = fixture(Path(folder), native=True)
            result = json.loads((campaign/'result.json').read_text())
            result['opponent_sampling'] = {'identity': 'parent-declaration-only'}
            write(campaign/'result.json', result)
            report = summarize([campaign])
            self.assertIsNone(report['campaigns'][0]['cycles'][0]['training']['opponent_sampling'])
            self.assertIn('不推断为均匀采样', render_markdown(report))
            del result['opponent_sampling']
            write(campaign/'result.json', result)
            self.assertIsNone(summarize([campaign])['campaigns'][0]['opponent_sampling'])

    def test_round_chain_training_layers_are_subsets_not_additional_rounds(self):
        with tempfile.TemporaryDirectory() as folder:
            campaign = fixture(Path(folder), native=True)
            train = campaign/'cycle-001/train'
            path = train/'result.json'
            result = json.loads(path.read_text())
            result.update(round_chain=True, training_protocol='native_match_round_episodes_v1')
            write(path, result)
            path = train/'worker-00/episodes.jsonl'
            original = json.loads(path.read_text().splitlines()[0])
            rows = [dict(original, round=1), dict(original, episode=2, round=2, outcome='loss'),
                    dict(original, episode=3, round=4, outcome='draw'),
                    dict(original, episode=4, native_round={'round': 3}),
                    dict(original, episode=5, round=True)]
            path.write_text('\n'.join(json.dumps(row) for row in [rows[0]]+rows)+'\n')
            before = path.read_bytes()
            report = summarize(training_paths=[train, train])
            summary = report['standalone'][0]['summary']
            self.assertEqual(summary['python_completed_rounds'], 5)
            self.assertEqual(summary['rounds']['2']['rounds'], 5)
            self.assertTrue(summary['round_chain'])
            self.assertEqual(summary['training_protocol'], 'native_match_round_episodes_v1')
            self.assertEqual(list(summary['rounds_by_index']), ['1', '2', '4'])
            self.assertEqual(summary['rounds_by_index']['4']['2']['draw'], 1)
            self.assertEqual(summary['rounds_without_index_count'], 2)
            indexed = sum(row['rounds'] for table in summary['rounds_by_index'].values() for row in table.values())
            self.assertEqual(indexed+summary['rounds_without_index_count'], 5)
            self.assertEqual(summary['duplicate_lines_excluded'], 1)
            self.assertEqual(summary['native_unconfirmed_rounds'], 0)
            self.assertEqual(path.read_bytes(), before)
            markdown = render_markdown(report)
            self.assertIn('| R4 | Blanka | 0/0/1 |', markdown)
            self.assertIn('未记录有效轮次', markdown)
            self.assertNotIn('| R3 |', markdown)

    def test_legacy_round_index_absence_stays_unknown_and_campaign_protocol_survives(self):
        with tempfile.TemporaryDirectory() as folder:
            campaign = fixture(Path(folder), native=True)
            path = campaign/'result.json'
            result = json.loads(path.read_text())
            result['training_protocol'] = 'native_match_round_episodes_v1'
            write(path, result)
            report = summarize([campaign])
            run = report['campaigns'][0]
            train = run['cycles'][0]['training']
            self.assertEqual(train['rounds_by_index'], {})
            self.assertEqual(train['rounds_without_index_count'], 1)
            self.assertIsNone(train['round_chain'])
            self.assertIsNone(train['training_protocol'])
            self.assertEqual(run['training_protocol'], 'native_match_round_episodes_v1')
            self.assertIn('运行声明训练协议：native_match_round_episodes_v1', render_markdown(report))

    def test_actions16_mixed_campaigns_and_standalone_keep_separate_identities(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            old = fixture(root/'old', native=True)
            new = fixture(root/'new', native=True)
            train, play = new/'cycle-001/train', new/'cycle-001/continuous'
            for path, schema in ((new, 'astra.rl-native-campaign.actions16.v1'),
                                 (train, 'astra.rl-batch-prototype.actions16.v1'),
                                 (play, 'astra.rl-continuous.actions16.v1')):
                result = json.loads((path/'result.json').read_text())
                result.update(schema=schema, action_interface='ken_actions16_lp_mp_uppercut_v1', actions=16)
                if path == play:
                    result['action_interface_audit'] = {'ok': True}
                write(path/'result.json', result)
            report = summarize([old, new, new], [train], [play])
            self.assertEqual(len(report['campaigns']), 2)
            self.assertEqual(report['standalone'], [])
            self.assertEqual(len(report['duplicate_input_paths_ignored']), 3)
            self.assertEqual([r['action_schema_family'] for r in report['campaigns']],
                             ['legacy_actions15', 'actions16'])
            for run in report['campaigns']:
                self.assertEqual(run['aggregate']['attempt_counts'], {'loss': 1})
                self.assertEqual(run['aggregate']['training_rounds']['2']['win'], 1)
            report = summarize(training_paths=[train], continuous_paths=[play])
            self.assertEqual([r['classification'] for r in report['standalone']], ['complete', 'complete'])
            self.assertTrue(all(r['summary']['actions'] == 16 for r in report['standalone']))
            self.assertIn('ken_actions16_lp_mp_uppercut_v1', render_markdown(report))

    def test_actions16_schema_requires_interface_and_native_audits_without_sidecar(self):
        with tempfile.TemporaryDirectory() as folder:
            campaign = fixture(Path(folder), native=True)
            play = campaign/'cycle-001/continuous'
            path = play/'result.json'
            result = json.loads(path.read_text())
            result['schema'] = 'astra.rl-continuous.actions16.v1'
            write(path, result)
            summary = summarize(continuous_paths=[play])['standalone'][0]['summary']
            self.assertTrue(summary['action_interface_audit_required'])
            self.assertEqual(summary['attempt_counts'], {'pending': 1})
            result['action_interface_audit'] = {'ok': True}
            del result['native_timing_audit']
            write(path, result)
            self.assertEqual(summarize(continuous_paths=[play])['standalone'][0]['classification'], 'pending')
            result['native_timing_audit'] = {'ok': True}
            result['action_interface_audit'] = {'ok': False}
            write(path, result)
            self.assertEqual(summarize(continuous_paths=[play])['standalone'][0]['classification'], 'invalid')
            # Parent schema imposes the interface audit during the child finalization window too.
            parent = json.loads((campaign/'result.json').read_text())
            parent['schema'] = 'astra.rl-native-campaign.actions16.v1'
            write(campaign/'result.json', parent)
            result['schema'] = 'astra.rl-continuous.v1'
            del result['action_interface_audit']
            write(path, result)
            summary = summarize([campaign])['campaigns'][0]['cycles'][0]['continuous']
            self.assertEqual(summary['attempt_counts'], {'pending': 1})

    def test_action_interface_identity_requires_its_final_audit(self):
        with tempfile.TemporaryDirectory() as folder:
            campaign = fixture(Path(folder), native=True)
            play = campaign/'cycle-001/continuous'
            path = play/'result.json'
            result = json.loads(path.read_text())
            interface = {'variant': 'fast', 'action_interface': 'fire2-experiment',
                         'model_sha256': 'model', 'decision_native_frames': 12}
            write(play/'action-interface.json', interface)
            item = summarize(continuous_paths=[play])['standalone'][0]
            self.assertEqual(item['classification'], 'pending')
            self.assertEqual(item['summary']['variant'], 'fast')
            self.assertEqual(item['summary']['action_interface_protocol'], interface)
            self.assertEqual(len(item['summary']['action_interface_protocol_sha256']), 64)
            result.update(variant='fast', action_interface='fire2-experiment', action_interface_audit={'ok': True})
            write(path, result)
            report = summarize(continuous_paths=[play])
            self.assertEqual(report['standalone'][0]['summary']['attempt_counts'], {'loss': 1})
            self.assertIn('fire2-experiment', render_markdown(report))
            result['action_interface_audit'] = {'ok': False, 'reason': 'interface mismatch'}
            write(path, result)
            self.assertEqual(summarize(continuous_paths=[play])['standalone'][0]['classification'], 'invalid')
            (play/'action-interface.json').unlink()
            del result['action_interface_audit']
            write(path, result)
            self.assertEqual(summarize(continuous_paths=[play])['standalone'][0]['classification'], 'pending')

    def test_categorical_identity_and_final_sampling_audit_are_required(self):
        with tempfile.TemporaryDirectory() as folder:
            campaign = fixture(Path(folder), native=True)
            play = campaign/'cycle-001/continuous'
            path = play/'result.json'
            result = json.loads(path.read_text())
            protocol = {'model_sha256': 'model', 'selection': 'categorical_softmax',
                        'policy_seed': 71, 'policy_prng': 'park_miller_48271_v1'}
            write(play/'sampling-protocol.json', protocol)
            item = summarize(continuous_paths=[play])['standalone'][0]
            self.assertEqual(item['classification'], 'pending')
            self.assertEqual(item['summary']['attempt_counts'], {'pending': 1})
            self.assertEqual(item['summary']['selection'], 'categorical_softmax')
            self.assertEqual(item['summary']['policy_seed'], 71)
            result.update(protocol, evaluation_kind='fixed-weight categorical policy', sampling_audit={'ok': True})
            write(path, result)
            report = summarize(continuous_paths=[play])
            summary = report['standalone'][0]['summary']
            self.assertEqual(summary['attempt_counts'], {'loss': 1})
            self.assertEqual(summary['evaluation_kind'], 'fixed-weight categorical policy')
            self.assertEqual(summary['sampling_audit'], {'ok': True})
            self.assertIn('categorical_softmax', render_markdown(report))
            result['sampling_audit'] = {'ok': False, 'reason': 'wrong draw'}
            write(path, result)
            self.assertEqual(summarize(continuous_paths=[play])['standalone'][0]['classification'], 'invalid')
            # A failed execution cannot become pending merely because no audit exists.
            result['status'] = 'invalid'
            del result['sampling_audit']
            write(path, result)
            self.assertEqual(summarize(continuous_paths=[play])['standalone'][0]['classification'], 'invalid')
            # Declared categorical selection requires the audit even without a sidecar.
            result['status'] = 'complete'
            write(path, result)
            (play/'sampling-protocol.json').unlink()
            self.assertEqual(summarize(continuous_paths=[play])['standalone'][0]['classification'], 'pending')

    def test_standalone_stages_keep_identity_and_deduplicate_repeated_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            campaign = fixture(Path(folder), native=True)
            train, play = campaign/'cycle-001/train', campaign/'cycle-001/continuous'
            report = summarize(training_paths=[train, train/'.'], continuous_paths=[play, play/'.'])
            self.assertEqual(report['campaigns'], [])
            self.assertEqual(len(report['standalone']), 2)
            self.assertEqual(len(report['duplicate_input_paths_ignored']), 2)
            self.assertEqual(report['standalone'][0]['summary']['python_completed_rounds'], 1)
            self.assertEqual(report['standalone'][1]['summary']['attempt_counts'], {'loss': 1})
            self.assertEqual(report['standalone'][0]['summary']['model_sha256'], 'model')
            self.assertIn('独立训练', render_markdown(report))
            self.assertIn('独立连续验证', render_markdown(report))

    def test_explicit_campaign_child_stages_are_not_counted_twice(self):
        with tempfile.TemporaryDirectory() as folder:
            campaign = fixture(Path(folder), native=True)
            report = summarize([campaign], [campaign/'cycle-001/train'], [campaign/'cycle-001/continuous'])
            self.assertEqual(len(report['campaigns']), 1)
            self.assertEqual(report['standalone'], [])
            self.assertEqual([row['reason'] for row in report['duplicate_input_paths_ignored']],
                             ['already_reported_campaign_stage']*2)

    def test_standalone_native_audit_failure_is_invalid(self):
        with tempfile.TemporaryDirectory() as folder:
            campaign = fixture(Path(folder), native=True)
            play = campaign/'cycle-001/continuous'
            result = json.loads((play/'result.json').read_text())
            result['native_timing_audit']['ok'] = False
            write(play/'result.json', result)
            item = summarize(continuous_paths=[play])['standalone'][0]
            self.assertEqual(item['classification'], 'invalid')
            self.assertEqual(item['summary']['rounds'], {})

    def test_cli_accepts_standalone_only_and_rejects_zero_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            campaign = fixture(root, native=True)
            output = root/'report'
            with patch('sys.argv', ['report_normal', '--training', str(campaign/'cycle-001/train'), '--output', str(output)]):
                main()
            self.assertTrue((output/'report.json').is_file())
            with self.assertRaises(ValueError):
                summarize()

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
