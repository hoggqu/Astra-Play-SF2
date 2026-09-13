"""An equal TIME latch may force one raw HP to KO on the same native frame."""
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

from lupa.lua54 import LuaRuntime

HERE = Path(__file__).resolve().parent
CORE = HERE.parents[1] / 'src/astra_play_sf2/assets/play_core.lua'


class EqualTimeAdjacentTests(unittest.TestCase):
    def tick(self, state, count=1):
        for _ in range(count):
            self.core.tick(self.core, self.lua.table_from(state, recursive=True))

    def make(self, side=1):
        self.lua = LuaRuntime(unpack_returned_tuples=True)
        C = self.lua.execute(CORE.read_text())
        C = self.lua.execute((HERE / 'settlement_v9.lua').read_text())(C)
        def p(char, x):
            return dict(char=char, x=x, y=40, a=0, anim=12345,
                        hp=144, displayed_hp=144, timeout_hp=0, wins=0)
        self.opening = dict(timer=99, p1=p(4, 100), p2=p(7, 200))
        self.core = C.new(self.lua.table_from(dict(
            opponent=7, mode='test', choose=self.lua.eval('function()return {{12,""}}end'))),
            self.lua.table_from(self.opening, recursive=True))
        live = deepcopy(self.opening)
        live['timer'] = 0
        for fighter in ('p1', 'p2'):
            live[fighter].update(hp=12, displayed_hp=12, a=10)
        self.tick(live, 30)
        settled = deepcopy(live)
        settled['p1']['timeout_hp'] = settled['p2']['timeout_hp'] = 12
        settled[f'p{side}']['hp'] = -1
        settled[f'p{side}']['a'] = 18
        return settled

    def test_equal_latch_requires_maturity_and_native_next_round(self):
        for side in (1, 2):
            with self.subTest(side=side):
                state = self.make(side)
                original = deepcopy(state)
                self.tick(state, 370)
                self.assertEqual(len(self.core.rounds), 0)
                self.assertIsNotNone(self.core.rl_draw_evidence.mature)
                self.tick(self.opening)
                row = self.core.rounds[1]
                self.assertEqual(row.outcome, 'draw')
                self.assertEqual(row.recognition, 'rl-native-equal-time-adjacent-ko-v9')
                self.assertEqual(list(row.score.values()), [0, 0])
                self.assertEqual(row.equal_time_previous.state.p1.hp, 12)
                self.assertEqual(row.equal_time_previous.frame + 1, row.equal_time.frame)
                self.assertGreater(row.confirmation_frame, row.settled_frame)
                self.assertEqual(state, original)

    def test_missing_or_contradictory_chronology_rejected(self):
        for kind in ('missing', 'gap', 'prior_ko', 'unequal', 'display', 'score',
                     'timer', 'latch', 'both_ko'):
            with self.subTest(kind=kind):
                state = self.make()
                previous = self.core.rl_time_previous
                if kind == 'missing': self.core.rl_time_previous = None
                elif kind == 'gap': previous.frame -= 1
                elif kind == 'prior_ko': previous.state.p1.hp = -1
                elif kind == 'unequal': previous.state.p1.hp = 13
                elif kind == 'display': previous.state.p1.displayed_hp = 13
                elif kind == 'score': previous.state.p1.wins = 1
                elif kind == 'timer': previous.state.timer = 1
                elif kind == 'latch': previous.state.p1.timeout_hp = 12
                elif kind == 'both_ko': state['p2']['hp'] = -1
                self.tick(state, 370)
                self.tick(self.opening)
                self.assertEqual(len(self.core.rounds), 0)

    def test_early_confirmation_or_later_mutation_rejected(self):
        for kind in ('early', 'hp', 'display', 'latch', 'pip'):
            with self.subTest(kind=kind):
                state = self.make()
                self.tick(state)
                if kind != 'early':
                    key = dict(hp='hp', display='displayed_hp', latch='timeout_hp', pip='wins')[kind]
                    state['p2'][key] += 1
                    self.tick(state, 370)
                self.tick(self.opening)
                self.assertEqual(len(self.core.rounds), 0)

    def test_native_zeroed_initialization_preserves_mature_evidence(self):
        state = self.make()
        self.tick(state, 370)
        initializing = deepcopy(state)
        for fighter in ('p1', 'p2'):
            initializing[fighter].update(anim=0, hp=0, displayed_hp=0, timeout_hp=0, x=0, y=0)
        self.tick(initializing, 5)
        self.tick(self.opening)
        self.assertEqual(self.core.rounds[1].outcome, 'draw')

    def test_v8_and_older_contracts_remain(self):
        from . import test_settlement_v8
        read = Path.read_text
        def current(path, *args, **kwargs):
            return read(HERE / 'settlement_v9.lua', *args, **kwargs) if path.name == 'settlement_v8.lua' else read(path, *args, **kwargs)
        suite = unittest.defaultTestLoader.loadTestsFromModule(test_settlement_v8)
        result = unittest.TestResult()
        with patch.object(Path, 'read_text', current):
            suite.run(result)
        self.assertEqual(result.errors + result.failures, [])


if __name__ == '__main__':
    unittest.main()
