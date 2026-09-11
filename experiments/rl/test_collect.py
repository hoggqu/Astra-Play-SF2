"""Offline collector contract checks: no simulator or optional ML imports."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from .collect import BudgetReached, BoundedBridge, capture, collect, collect_attempt, split_for, collection_complete


def opening(opponent=2):
    player = {'hp': 144, 'displayed_hp': 144, 'round_wins': 0,
              'x': 100, 'y': 40, 'animation': 1}
    return {'paused': True, 'controller_busy': False, 'difficulty_bits': 0,
            'difficulty_mirror': 7, 'effective_difficulty': 7, 'timer_seconds': 99,
            'p1': dict(player, character=4), 'p2': dict(player, character=opponent)}


def record(samples=6):
    return {'difficulty': 7, 'samples_requested': samples, 'openings': [],
            'rejected_openings': [], 'matches_started': 0, 'max_matches': 132}


def attempt(ordinal=1):
    return {'ordinal': ordinal, 'status': 'running', 'outcome': None,
            'captures': [], 'readiness': {}, 'matches': []}


class FakeBridge:
    def __init__(self, native=None, checkpoint_bytes=b'distinct-native-state'):
        self.commands = []
        self.native = native or opening()
        self.checkpoint_bytes = checkpoint_bytes

    def send(self, command, snapshot=True):
        self.commands.append(command)
        if command.startswith('checkpoint('):
            Path(json.loads(command[len('checkpoint('):-1])).write_bytes(self.checkpoint_bytes)
        state = copy.deepcopy(self.native)
        for kind in ('coin', 'start'):
            if f'wait_{kind}_ready' in command:
                state['session_gate'] = {'kind': kind, 'status': 'ready', 'frames': 25}
        return state

    def wait(self, predicate):
        return {'status': 'complete', 'valid_continuous': True, 'result': 'cpu_win'}


class CollectionTests(unittest.TestCase):
    def test_split_is_predeclared_four_one_one(self):
        self.assertEqual([split_for(i, 6) for i in range(6)], ['train']*4+['dev', 'holdout'])

    def test_capture_rejects_duplicate_without_deleting_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = record()
            first = capture(root, FakeBridge(), manifest, attempt(), 1, opening())
            second = capture(root, FakeBridge(), manifest, attempt(2), 1, opening())
            self.assertEqual(first['split'], 'train')
            self.assertEqual(second['duplicate_of'], first['id'])
            self.assertEqual(len(manifest['openings']), 1)
            self.assertEqual(len(manifest['rejected_openings']), 1)
            self.assertTrue((root/second['path']).is_file())
            self.assertEqual(first['initial_state'], first['post_save_state'])

    def test_capture_requires_native_difficulty_and_full_health(self):
        with tempfile.TemporaryDirectory() as temporary:
            for field, value in [('effective_difficulty', 3), ('timer_seconds', 98)]:
                state = opening()
                state[field] = value
                bridge = FakeBridge()
                with self.assertRaises(ValueError):
                    capture(Path(temporary), bridge, record(), attempt(), 1, state)
                self.assertEqual(bridge.commands, [])

    def test_post_save_health_change_rejected(self):
        state = opening()
        state['p1']['hp'] = 143
        with tempfile.TemporaryDirectory() as temporary:
            manifest = record()
            with self.assertRaisesRegex(ValueError, 'Post-save'):
                capture(Path(temporary), FakeBridge(state), manifest, attempt(), 1, opening())
            self.assertFalse(manifest['openings'])
            self.assertTrue((Path(temporary)/'checkpoints/a001-m01.sta').is_file())

    def test_final_capture_stops_before_combat_and_never_starts_formal_session(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = record(3)
            manifest['openings'] = [{'id': 'previous1', 'sha256': '1'}, {'id': 'previous2', 'sha256': '2'}]
            current = attempt(3)
            bridge = FakeBridge()
            collect_attempt(Path(temporary), bridge, manifest, current, lambda: None)
            self.assertEqual(current['status'], 'stopped_after_final_capture')
            self.assertIsNone(current['outcome'])
            self.assertEqual(manifest['matches_started'], 0)
            text = '\n'.join(bridge.commands)
            for forbidden in ('session_begin', 'session_end', 'restore(', 'reset(', 'play_match('):
                self.assertNotIn(forbidden, text)
            self.assertIn("astra_entry.require_ready('coin')", text)

    def test_loss_retained_and_next_attempt_uses_fresh_coin_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = record()
            bridge = FakeBridge(opening(0))
            for ordinal in (1, 2):
                current = attempt(ordinal)
                collect_attempt(Path(temporary), bridge, manifest, current, lambda: None)
                self.assertEqual(current['outcome'], 'loss')
                self.assertEqual(current['matches'][0]['result'], 'cpu_win')
            self.assertEqual(manifest['matches_started'], 2)
            self.assertEqual(sum('wait_coin_ready' in c for c in bridge.commands), 2)
            self.assertEqual(sum("{3,'C'}" in c for c in bridge.commands), 2)
            self.assertTrue(all('training_validation=true' in c for c in bridge.commands if c.startswith('play_match')))

    def test_match_budget_stops_before_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = record()
            manifest['max_matches'] = 0
            bridge = FakeBridge(opening(0))
            with self.assertRaises(BudgetReached):
                collect_attempt(Path(temporary), bridge, manifest, attempt(), lambda: None)
            self.assertFalse(any(c.startswith('play_match') for c in bridge.commands))

    def test_expired_wall_budget_does_not_send(self):
        bridge = BoundedBridge(Path('/unused'), Mock(), deadline=0)
        with patch('astra_play_sf2.transport.Bridge.send') as parent:
            with self.assertRaises(BudgetReached):
                bridge.send("act({{3,'C'}})")
        parent.assert_not_called()

    def test_multi_opponent_split_counts_separately_and_requires_all(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = record(3)
            manifest['opponents'] = [0, 2]
            root = Path(temporary)
            for ordinal in range(1, 4):
                for opponent in (0, 2):
                    state = opening(opponent)
                    row = capture(root, FakeBridge(state, f'{ordinal}-{opponent}'.encode()),
                                  manifest, attempt(ordinal), opponent+1, state)
                    self.assertEqual(row['split'], ('train', 'dev', 'holdout')[ordinal-1])
                    self.assertEqual(row['opponent'], opponent)
                    self.assertEqual(collection_complete(manifest), ordinal == 3 and opponent == 2)

    def test_invalid_configuration_never_runs_doctor(self):
        with patch('experiments.rl.collect.doctor') as preflight:
            for kwargs in ({'samples': 2}, {'max_attempts': 0}, {'max_matches': 0}, {'max_seconds': -1}, {'difficulty': 2}):
                with self.assertRaises(ValueError):
                    collect({}, Path('/unused'), **kwargs)
        preflight.assert_not_called()


if __name__ == '__main__':
    unittest.main()
