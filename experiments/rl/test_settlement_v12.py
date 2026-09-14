"""An unchanged zero-HP survivor can win a positively observed late damage KO."""
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch
from lupa.lua54 import LuaRuntime
from . import test_settlement_v11

HERE = Path(__file__).resolve().parent
CORE = HERE.parents[1] / 'src/astra_play_sf2/assets/play_core.lua'


class ZeroWinnerTests(unittest.TestCase):
    def tick(self, state, count=1):
        for _ in range(count):
            self.core.tick(self.core, self.lua.table_from(state, recursive=True))

    def make(self, loss=False, score=0):
        self.lua = LuaRuntime(unpack_returned_tuples=True)
        C = self.lua.execute(CORE.read_text())
        C = self.lua.execute((HERE/'settlement_v12.lua').read_text())(C)
        def actor(char, x):
            return dict(char=char, x=x, y=40, a=10, anim=12345, hp=144,
                        displayed_hp=144, timeout_hp=0, wins=0)
        self.opening = dict(timer=99, p1=actor(4, 100), p2=actor(0, 200))
        self.core = C.new(self.lua.table_from(dict(opponent=0, mode='test',
            choose=self.lua.eval('function()return {{12,""}}end'))),
            self.lua.table_from(self.opening, recursive=True))
        self.winner, self.loser = ('p2', 'p1') if loss else ('p1', 'p2')
        self.score = score
        self.core.score = self.lua.table_from([0, score] if loss else [score, 0])
        state = deepcopy(self.opening); state['timer'] = 0
        state[self.winner].update(hp=0, displayed_hp=0, wins=score)
        state[self.loser].update(hp=9, displayed_hp=9, y=84)
        self.tick(state, 21)
        return state

    def ko(self, state):
        state = deepcopy(state); state[self.loser].update(hp=-1, a=20)
        return state

    def award(self, state):
        state = deepcopy(state)
        state[self.winner].update(wins=self.score+1, a=16)
        state[self.loser].update(displayed_hp=-1, y=40, a=2)
        return state

    def test_both_sides_final_pip_and_same_frame_award_wait_for_maturity(self):
        for loss in (False, True):
            for score in (0, 1):
                for same_frame in (False, True):
                    with self.subTest(loss=loss, score=score, same_frame=same_frame):
                        state = self.ko(self.make(loss, score))
                        if not same_frame: self.tick(state)
                        state = self.award(state); original = deepcopy(state)
                        self.tick(state, 360)
                        self.assertEqual(len(self.core.rounds), 0)
                        self.tick(state)
                        row = self.core.rounds[1]
                        self.assertEqual(row.recognition, 'rl-native-zero-winner-time-ko-v12')
                        self.assertEqual(row.outcome, 'loss' if loss else 'win')
                        self.assertEqual(row.stable_award_frames, 360)
                        self.assertEqual(row.settled[self.winner].hp, 0)
                        self.assertEqual(row.late_ko.previous.state[self.loser].hp, 9)
                        self.assertEqual(self.core.phase, 'complete' if score else 'between')
                        self.assertEqual(state, original)

    def test_missing_or_conflicting_chronology_never_qualifies(self):
        for kind in ('missing', 'gap', 'no_ko', 'prior_ko', 'both_zero', 'both_ko',
                     'prior_award', 'prior_timer', 'prior_latch', 'latch',
                     'winner_changed', 'winner_display', 'prior_display', 'prior_character'):
            with self.subTest(kind=kind):
                state = self.ko(self.make()); prev = self.core.rl_time_previous
                if kind == 'missing': self.core.rl_time_previous = None
                elif kind == 'gap': prev.frame -= 1
                elif kind == 'no_ko': state[self.loser]['hp'] = 9
                elif kind == 'prior_ko': prev.state[self.loser].hp = -1
                elif kind == 'both_zero':
                    prev.state[self.loser].hp = 0; prev.state[self.loser].displayed_hp = 0
                    state[self.loser]['displayed_hp'] = 0
                elif kind == 'both_ko': state[self.winner]['hp'] = -1
                elif kind == 'prior_award': prev.state[self.winner].wins = 1
                elif kind == 'prior_timer': prev.state.timer = 1
                elif kind == 'prior_latch': prev.state[self.loser].timeout_hp = 9
                elif kind == 'latch': state[self.loser]['timeout_hp'] = 9
                elif kind == 'winner_changed': prev.state[self.winner].hp = 1
                elif kind == 'winner_display': state[self.winner]['displayed_hp'] = 1
                elif kind == 'prior_display': prev.state[self.loser].displayed_hp = 8
                elif kind == 'prior_character': prev.state[self.loser].char = 3
                self.tick(state)
                self.assertIsNone(self.core.rl_late_ko)

    def test_bad_score_later_latch_and_early_next_round_never_qualify(self):
        for kind in ('missing_pip', 'wrong_pip', 'both_pips', 'latch', 'display_regains',
                     'winner_regains', 'early'):
            with self.subTest(kind=kind):
                ko = self.ko(self.make()); self.tick(ko)
                state = self.award(ko)
                if kind == 'missing_pip': state[self.winner]['wins'] = 0
                elif kind == 'wrong_pip':
                    state[self.winner]['wins'] = 0; state[self.loser]['wins'] = 1
                elif kind == 'both_pips': state[self.loser]['wins'] = 1
                elif kind == 'latch': state[self.loser]['timeout_hp'] = 9
                elif kind == 'display_regains': state[self.loser]['displayed_hp'] = 10
                elif kind == 'winner_regains': state[self.winner].update(hp=1, displayed_hp=1)
                self.tick(state, 10 if kind == 'early' else 400)
                if kind == 'early':
                    opening = deepcopy(self.opening); opening[self.winner]['wins'] = 1
                    self.tick(opening)
                self.assertEqual(len(self.core.rounds), 0)

    def test_all_previous_contracts_remain(self):
        read = Path.read_text
        def current(path, *args, **kwargs):
            return read(HERE/'settlement_v12.lua', *args, **kwargs) if path.name == 'settlement_v11.lua' else read(path, *args, **kwargs)
        result = unittest.TestResult()
        with patch.object(Path, 'read_text', current):
            unittest.defaultTestLoader.loadTestsFromModule(test_settlement_v11).run(result)
        self.assertEqual(result.errors + result.failures, [])


if __name__ == '__main__': unittest.main()
