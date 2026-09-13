"""Visible-only observer state, attribution, restart and Python/Lua parity tests."""
import unittest
from pathlib import Path
from . import visible_feedback as py
try:
    from lupa import LuaRuntime
except ImportError:
    LuaRuntime = None
HERE = Path(__file__).resolve().parent


@unittest.skipIf(LuaRuntime is None, 'optional lupa is unavailable')
class VisibleFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.lua = LuaRuntime(unpack_returned_tuples=True)
        self.module = self.lua.execute((HERE/'visible_feedback.lua').read_text())
        self.mapping = self.lua.execute((HERE/'visible_animation_map.lua').read_text())
        self.actions = self.lua.execute((HERE/'full_actions.lua').read_text())
        self.observer = self.module.new(self.actions, self.mapping)
        self.initial = self.state()
        self.observer.reset(self.observer, self.initial)

    def state(self, anim=382306, action=0, y=40, enemy_anim=407552, enemy_char=3):
        return self.lua.table_from({'p1': self.lua.table_from(dict(char=4, anim=anim, a=action, x=100, y=y)),
                                    'p2': self.lua.table_from(dict(char=enemy_char, anim=enemy_anim, a=0, x=250, y=40))})

    def tick(self, **kwargs):
        s = self.state(**kwargs);self.observer.tick(self.observer,s);return s

    def request(self, action):
        self.observer.request(self.observer, action, self.initial)

    def test_request_cannot_leak_into_previous_observation(self):
        before=dict(self.initial.visible_feedback)
        self.request(13)
        self.assertEqual(dict(self.initial.visible_feedback),before)
        self.assertEqual(self.tick().visible_feedback.request_action,13)

    def test_special_started_not_hit_and_strength_not_inferred(self):
        self.request(13)
        s=self.tick(anim=390174,action=12)
        self.assertEqual(s.visible_feedback.result,'started')
        self.assertEqual(s.visible_feedback.actual_family,'shoryuken')
        self.assertEqual(s.visible_feedback.actual_strength,'unknown')
        self.assertEqual(s.visible_feedback.start_age,0)
        old=dict(s.visible_feedback)
        s2=self.tick(anim=390174,action=12)
        self.assertEqual(s2.visible_feedback.start_age,1)
        self.assertEqual(dict(s.visible_feedback),old)

    def test_macro_other_move_and_unconfirmed_can_resolve_later(self):
        self.request(13)
        for _ in range(12):s=self.tick()
        self.assertEqual(s.visible_feedback.result,'unconfirmed')
        s=self.tick(anim=383790,action=10)
        self.assertEqual(s.visible_feedback.result,'other')
        self.assertEqual(s.visible_feedback.actual_family,'normal_punch')

    def test_continuous_normal_can_restart_without_state_change(self):
        self.request(6)
        self.tick(anim=383790,action=10)
        self.tick(anim=383814,action=10)
        s=self.tick(anim=383790,action=10)
        self.assertEqual(s.visible_feedback.start_age,0)

    def test_tatsumaki_rotation_loops_are_one_actual_start(self):
        self.request(82)
        self.tick(anim=390690,action=12)
        for address in (390714,390738,390762,390786,390814,390838,390862,390886,390910,390814):
            s=self.tick(anim=address,action=12)
        self.assertEqual(s.visible_feedback.start_age,10)
        self.assertEqual(s.visible_feedback.actual_family,'tatsumaki')

    def test_contextual_throw_and_special_are_not_wrong_button_failure(self):
        for address in (389482,391490,390174):
            self.observer.reset(self.observer,self.initial);self.request(25)
            s=self.tick(anim=address,action=10)
            self.assertEqual(s.visible_feedback.result,'unknown' if address==390174 else 'started')

    def test_visible_jump_uses_native_jump_state_four(self):
        self.request(4)
        s=self.tick(anim=382882,action=4,y=46)
        self.assertEqual(s.visible_feedback.actual_family,'jump')
        self.assertEqual(s.visible_feedback.result,'started')

    def test_raw_wrong_normal_or_light_throw_is_unknown_attribution(self):
        for address in (385966,389482):
            self.observer.reset(self.observer,self.initial);self.request(6)
            s=self.tick(anim=address,action=10)
            self.assertEqual(s.visible_feedback.result,'unknown')

    def test_dizzy_all_characters_and_invalid_unknown(self):
        for char in range(12):
            for _,address in self.mapping.dizzy[char].items():
                s=self.tick(enemy_char=char,enemy_anim=address)
                self.assertEqual(s.visible_feedback.p2_dizzy,'present')
        self.assertEqual(self.tick(enemy_char=12).visible_feedback.p2_dizzy,'unknown')
        self.assertEqual(self.tick(enemy_anim=0).visible_feedback.p2_dizzy,'unknown')
        self.assertEqual(self.tick().visible_feedback.p2_dizzy,'absent')
        self.assertEqual(self.tick(anim=388926).visible_feedback.p1_dizzy,'present')

    def test_python_lua_features_and_cap(self):
        self.request(84)
        for _ in range(130):s=self.tick()
        expected=py.features({'visible_feedback':dict(s.visible_feedback)})
        actual=list(self.module.features(s).values())
        self.assertEqual(len(actual),114);self.assertEqual(actual,expected)
        self.assertEqual(actual[105],1.0)

    def test_reset_forgets_requests_and_does_not_report_midmove_as_start(self):
        self.request(13)
        s=self.state(anim=390222,action=12)
        self.observer.reset(self.observer,s)
        self.assertEqual(s.visible_feedback.result,'none')
        s=self.tick(anim=390246,action=12)
        self.assertEqual(s.visible_feedback.start_age,-1)

    def test_coarse_attack_state_before_animation_is_not_a_second_start(self):
        self.request(6)
        s=self.tick(action=10)
        self.assertEqual(s.visible_feedback.start_age,-1)
        s=self.tick(anim=383790,action=10)
        self.assertEqual(s.visible_feedback.start_age,0)
        self.assertEqual(s.visible_feedback.actual_family,'normal_punch')

    def test_invalid_feedback_is_rejected_in_both_encoders(self):
        for field,bad in [('interface','old'),('request_action',85),('result','maybe'),('actual_family','secret'),('actual_strength','super'),('p2_dizzy','hidden'),('start_age',float('inf')),('request_age',-2),('started_since_request',1)]:
            v=dict(self.initial.visible_feedback);v[field]=bad
            with self.assertRaises((ValueError,TypeError)):
                py.features({'visible_feedback':v})
            with self.assertRaises(Exception):
                self.module.features(self.lua.table_from({'visible_feedback':self.lua.table_from(v)}))

    def test_no_memory_pause_mask_or_hidden_timer_access(self):
        source=(HERE/'visible_feedback.lua').read_text()
        for forbidden in ('mem:', 'emu.', 'manager.', 'set_value', 'remaining_recovery', '0x123', '0x124', '0x5e'):
            self.assertNotIn(forbidden,source)

if __name__=='__main__':unittest.main()
