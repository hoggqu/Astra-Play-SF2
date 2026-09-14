"""Ordinary KO may have exactly one adjacent positive timer decrement."""
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch
from . import test_settlement_v6

HERE = Path(__file__).resolve().parent


class KoTimerTailTests(unittest.TestCase):
    def make(self, loss=False, score=0):
        self.fixture = test_settlement_v6.NativeKoTests()
        read = Path.read_text
        def current(path, *args, **kwargs):
            return read(HERE / 'settlement_v11.lua', *args, **kwargs) if path.name == 'settlement_v6.lua' else read(path, *args, **kwargs)
        with patch.object(Path, 'read_text', current):
            terminal = self.fixture.make(loss, score)
        self.core = self.fixture.core
        self.tick = self.fixture.tick
        self.winner, self.loser = self.fixture.winner, self.fixture.loser
        return terminal

    def tail(self, terminal):
        state = self.fixture.awarded(terminal)
        state['timer'] -= 1
        return state

    def test_both_sides_and_final_pip_wait_for_full_maturity(self):
        for loss in (False, True):
            for score in (0, 1):
                with self.subTest(loss=loss, score=score):
                    terminal = self.make(loss, score)
                    state = self.tail(terminal)
                    original = deepcopy(state)
                    self.tick(state, 360)
                    self.assertEqual(len(self.core.rounds), 0)
                    self.tick(state)
                    result = self.core.rounds[1]
                    self.assertEqual(result.outcome, 'loss' if loss else 'win')
                    self.assertEqual(result.recognition, 'rl-native-nontime-ko-timer-tail-v11')
                    self.assertEqual(result.stop.timer, 53)
                    self.assertEqual(result.settled.timer, 52)
                    self.assertEqual(result.stable_award_frames, 360)
                    self.assertEqual(result.ko_award.timer_tail.previous.frame, 1)
                    self.assertEqual(self.core.phase, 'complete' if score else 'between')
                    self.assertEqual(state, original)

    def test_wrong_timer_or_missing_adjacent_terminal_fails_closed(self):
        for kind in ('missing', 'gap', 'wrong_previous_hp', 'wrong_previous_timer',
                     'delayed', 'twice', 'revert', 'drop2', 'increase', 'zero'):
            with self.subTest(kind=kind):
                terminal = self.make()
                state = self.tail(terminal)
                previous = self.core.rl_time_previous
                if kind == 'missing': self.core.rl_time_previous = None
                elif kind == 'gap': previous.frame -= 1
                elif kind == 'wrong_previous_hp': previous.state[self.loser].hp = 0
                elif kind == 'wrong_previous_timer': previous.state.timer = 54
                elif kind == 'delayed': self.tick(terminal)
                elif kind in ('twice', 'revert'):
                    self.tick(state)
                    state['timer'] = 51 if kind == 'twice' else 53
                elif kind == 'drop2': state['timer'] = 51
                elif kind == 'increase': state['timer'] = 54
                elif kind == 'zero': state['timer'] = 0
                self.tick(state, 400)
                self.assertEqual(len(self.core.rounds), 0)
                self.assertTrue(self.core.rl_ko_disqualified)

    def test_wrong_health_display_latch_score_and_early_opening_rejected(self):
        for kind in ('winner_hp', 'loser_hp', 'display', 'latch', 'later_latch',
                     'no_pip', 'wrong_pip', 'both_pips', 'early'):
            with self.subTest(kind=kind):
                terminal = self.make()
                state = self.tail(terminal)
                if kind == 'winner_hp': state[self.winner]['hp'] -= 1
                elif kind == 'loser_hp': state[self.loser]['hp'] = 0
                elif kind == 'display': state[self.loser]['displayed_hp'] = 28
                elif kind == 'latch': state[self.winner]['timeout_hp'] = 71
                elif kind == 'later_latch':
                    self.tick(state)
                    state[self.winner]['timeout_hp'] = 71
                elif kind == 'no_pip': state[self.winner]['wins'] = 0
                elif kind == 'wrong_pip':
                    state[self.winner]['wins'] = 0
                    state[self.loser]['wins'] = 1
                elif kind == 'both_pips': state[self.loser]['wins'] = 1
                self.tick(state, 10 if kind == 'early' else 400)
                if kind == 'early':
                    opening = deepcopy(self.fixture.opening)
                    opening[self.winner]['wins'] = 1
                    self.tick(opening)
                self.assertEqual(len(self.core.rounds), 0)

    def test_begin_round_clears_timer_tail(self):
        state = self.tail(self.make())
        self.tick(state, 361)
        opening = deepcopy(self.fixture.opening)
        opening[self.winner]['wins'] = 1
        self.tick(opening, 90)
        self.assertIsNone(self.core.rl_ko_award)
        self.assertIsNone(self.core.rl_ko_disqualified)

    def test_all_v10_and_older_contracts_remain(self):
        from . import test_settlement_v10
        read = Path.read_text
        def current(path, *args, **kwargs):
            return read(HERE / 'settlement_v11.lua', *args, **kwargs) if path.name == 'settlement_v10.lua' else read(path, *args, **kwargs)
        result = unittest.TestResult()
        with patch.object(Path, 'read_text', current):
            unittest.defaultTestLoader.loadTestsFromModule(test_settlement_v10).run(result)
        self.assertEqual(result.errors + result.failures, [])


if __name__ == '__main__':
    unittest.main()
