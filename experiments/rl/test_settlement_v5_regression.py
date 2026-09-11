"""Run old settlement contracts against v5 without changing historical tests."""
from copy import deepcopy
import unittest

from lupa.lua54 import LuaRuntime

from . import test_settlement_v2 as v2
from . import test_settlement_v3 as v3
from . import test_settlement_v4 as v4


class V5Fixture:
    def make(self, opponent=7):
        self.lua = LuaRuntime(unpack_returned_tuples=True)
        core = self.lua.execute((v2.ASSETS / 'play_core.lua').read_text())
        core = self.lua.execute((v2.HERE / 'settlement_v5.lua').read_text())(core)

        def actor(char, x):
            return dict(char=char, x=x, y=40, a=0, anim=1, hp=144,
                        displayed_hp=144, timeout_hp=0, wins=0)

        self.opening = dict(timer=99, p1=actor(4, 100), p2=actor(opponent, 200))
        opts = self.lua.table_from(dict(opponent=opponent, mode='v5-regression',
            choose=self.lua.eval('function() return {{12,""}} end')))
        self.core = core.new(opts, self.table(self.opening))

    def test_missing_live_chronology_equal_hp_or_invalid_health_never_classifies(self):
        # v4's unconditional loser.hp == -1 rejection intentionally changes:
        # v5 accepts it ONLY with an adjacent live/old-pip snapshot. Replace
        # that historical negative with missing/already-KO preceding evidence.
        for prior in ('missing', 'already_ko'):
            with self.subTest(prior=prior):
                s = self.prepare()
                s['p1']['hp'] = -1
                if prior == 'missing':
                    self.core.rl_time_previous = None
                else:
                    previous = deepcopy(s)
                    previous['p2']['wins'] = 0
                    self.tick(previous)
                self.tick(s, 420)
                self.assertIsNone(self.core.rl_time_award)
                self.assertEqual(len(self.core.rounds), 0)
        for field, value in [('timeout_hp', -1), ('timeout_hp', 145),
                             ('timeout_hp', 2.5), ('displayed_hp', 1)]:
            with self.subTest(field=field, value=value):
                s = self.prepare()
                s['p1'][field] = value
                self.tick(s, 420)
                self.assertEqual(len(self.core.rounds), 0)
        s = self.prepare()
        for field in ('hp', 'timeout_hp', 'displayed_hp'):
            s['p1'][field] = 20
        self.tick(s, 420)
        self.assertEqual(len(self.core.rounds), 0)


class TimeV5RegressionTests(V5Fixture, v2.SettlementV2Tests):
    def test_previous_timer_must_already_be_zero(self):
        s = self.prepare()
        previous = deepcopy(s)
        previous['timer'] = 1
        previous['p2']['wins'] = 0
        self.tick(previous)
        s['p1']['hp'] = -1
        self.tick(s, 420)
        self.assertIsNone(self.core.rl_time_award)
        self.assertEqual(len(self.core.rounds), 0)

    def test_previous_core_frame_must_be_adjacent(self):
        s = self.prepare()
        self.core.rl_time_previous.frame -= 1
        s['p1']['hp'] = -1
        self.tick(s, 420)
        self.assertIsNone(self.core.rl_time_award)
        self.assertEqual(len(self.core.rounds), 0)

    def test_begin_round_clears_all_cross_round_evidence(self):
        s = self.prepare()
        self.tick(s)
        self.assertIsNotNone(self.core.rl_time_previous)
        self.assertIsNotNone(self.core.rl_time_award)
        for name in ('rl_draw_evidence', 'rl_draw_disqualified',
                     'rl_double_ko', 'rl_double_ko_disqualified'):
            setattr(self.core, name, self.lua.table_from({'stale': True}))
        self.core.begin_round(self.core, self.table(self.opening),
                              self.lua.table_from({'events': self.lua.table()}))
        for name in ('rl_time_previous', 'rl_time_award', 'rl_draw_evidence',
                     'rl_draw_disqualified', 'rl_double_ko', 'rl_double_ko_disqualified'):
            self.assertIsNone(getattr(self.core, name), name)
        s['p1']['hp'] = -1
        self.tick(s, 420)
        self.assertIsNone(self.core.rl_time_award)
        self.assertEqual(len(self.core.rounds), 0)


class DrawV5RegressionTests(V5Fixture, v3.SettlementV3Tests):
    pass


class DoubleKOV5RegressionTests(V5Fixture, v4.DoubleKOTests):
    pass


if __name__ == '__main__':
    unittest.main()
