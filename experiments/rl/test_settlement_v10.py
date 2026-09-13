"""Zero raw/display HP can precede an unlatched late KO at timer zero."""
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

from lupa.lua54 import LuaRuntime

HERE = Path(__file__).resolve().parent
CORE = HERE.parents[1] / 'src/astra_play_sf2/assets/play_core.lua'


class ZeroHpLateKoTests(unittest.TestCase):
    def tick(self, state, count=1):
        for _ in range(count):
            self.core.tick(self.core, self.lua.table_from(state, recursive=True))

    def make(self, loss=False):
        self.lua = LuaRuntime(unpack_returned_tuples=True)
        C = self.lua.execute(CORE.read_text())
        C = self.lua.execute((HERE / 'settlement_v10.lua').read_text())(C)
        def player(char, x):
            return dict(char=char, x=x, y=40, a=10, anim=12345, hp=144,
                        displayed_hp=144, timeout_hp=0, wins=0)
        self.opening = dict(timer=99, p1=player(4, 100), p2=player(3, 200))
        self.core = C.new(self.lua.table_from(dict(opponent=3, mode='test',
            choose=self.lua.eval('function()return {{12,""}}end'))),
            self.lua.table_from(self.opening, recursive=True))
        self.winner, self.loser = ('p2', 'p1') if loss else ('p1', 'p2')
        state = deepcopy(self.opening)
        state['timer'] = 0
        state[self.winner].update(hp=4, displayed_hp=4)
        state[self.loser].update(hp=0, displayed_hp=0, y=84)
        self.tick(state, 27)
        return state

    def ko(self, state):
        state = deepcopy(state)
        state[self.loser].update(hp=-1, a=20)
        return state

    def award(self, state):
        state = deepcopy(state)
        state[self.winner].update(wins=1, a=16)
        state[self.loser].update(displayed_hp=-1, y=40, a=2)
        return state

    def test_both_sides_require_unique_pip_and_full_maturity(self):
        for loss in (False, True):
            with self.subTest(loss=loss):
                ko = self.ko(self.make(loss))
                self.tick(ko)
                self.assertTrue(self.core.rl_late_ko.zero_hp_edge)
                awarded = self.award(ko)
                original = deepcopy(awarded)
                self.tick(awarded, 360)
                self.assertEqual(len(self.core.rounds), 0)
                self.tick(awarded)
                row = self.core.rounds[1]
                self.assertEqual(row.outcome, 'loss' if loss else 'win')
                self.assertEqual(row.recognition, 'rl-native-zero-hp-time-ko-v10')
                self.assertEqual(row.stable_award_frames, 360)
                self.assertEqual(row.late_ko.previous.state[self.loser].hp, 0)
                self.assertEqual(row.late_ko.previous.state[self.loser].displayed_hp, 0)
                self.assertEqual(awarded, original)

    def test_same_frame_zero_to_ko_and_pip(self):
        state = self.award(self.ko(self.make()))
        self.tick(state, 360)
        self.assertEqual(len(self.core.rounds), 0)
        self.tick(state)
        self.assertEqual(self.core.rounds[1].recognition, 'rl-native-zero-hp-time-ko-v10')

    def test_zero_hp_alone_and_missing_or_inconsistent_edge_rejected(self):
        kinds = ('no_ko', 'missing', 'gap', 'prior_ko', 'prior_display',
                 'regained_display', 'prior_award', 'prior_timer', 'prior_latch',
                 'current_latch', 'changed_winner', 'zero_winner', 'both_ko')
        for kind in kinds:
            with self.subTest(kind=kind):
                state = self.make()
                ko = self.ko(state)
                prev = self.core.rl_time_previous
                if kind == 'no_ko': ko = deepcopy(state)
                elif kind == 'missing': self.core.rl_time_previous = None
                elif kind == 'gap': prev.frame -= 1
                elif kind == 'prior_ko': prev.state[self.loser].hp = -1
                elif kind == 'prior_display': prev.state[self.loser].displayed_hp = 1
                elif kind == 'regained_display': ko[self.loser]['displayed_hp'] = 1
                elif kind == 'prior_award': prev.state[self.winner].wins = 1
                elif kind == 'prior_timer': prev.state.timer = 1
                elif kind == 'prior_latch': prev.state[self.winner].timeout_hp = 4
                elif kind == 'current_latch': ko[self.winner]['timeout_hp'] = 4
                elif kind == 'changed_winner': ko[self.winner].update(hp=3, displayed_hp=3)
                elif kind == 'zero_winner': ko[self.winner].update(hp=0, displayed_hp=0)
                elif kind == 'both_ko': ko[self.winner]['hp'] = -1
                self.tick(ko)
                self.assertIsNone(self.core.rl_late_ko)

    def test_later_latch_or_bad_score_and_early_opening_rejected(self):
        for kind in ('latch', 'missing_pip', 'wrong_pip', 'both_pips', 'early'):
            with self.subTest(kind=kind):
                ko = self.ko(self.make())
                self.tick(ko)
                state = self.award(ko)
                if kind == 'latch': state[self.winner]['timeout_hp'] = 4
                elif kind == 'missing_pip': state[self.winner]['wins'] = 0
                elif kind == 'wrong_pip':
                    state[self.winner]['wins'] = 0
                    state[self.loser]['wins'] = 1
                elif kind == 'both_pips': state[self.loser]['wins'] = 1
                self.tick(state, 10 if kind == 'early' else 500)
                if kind == 'early':
                    opening = deepcopy(self.opening)
                    opening[self.winner]['wins'] = 1
                    self.tick(opening)
                self.assertEqual(len(self.core.rounds), 0)

    def test_all_v9_and_older_contracts_remain(self):
        from . import test_settlement_v9
        read = Path.read_text
        def current(path, *args, **kwargs):
            return read(HERE / 'settlement_v10.lua', *args, **kwargs) if path.name == 'settlement_v9.lua' else read(path, *args, **kwargs)
        result = unittest.TestResult()
        with patch.object(Path, 'read_text', current):
            unittest.defaultTestLoader.loadTestsFromModule(test_settlement_v9).run(result)
        self.assertEqual(result.errors + result.failures, [])


if __name__ == '__main__':
    unittest.main()
