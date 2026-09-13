"""Execute Lua and verify the complete input interface against independent facts."""
from pathlib import Path
import tempfile
import unittest

from lupa.lua54 import LuaRuntime

from .actions16_builder import build as build_actions16, PACKAGE
from . import full_actions as actions


HERE = Path(__file__).resolve().parent


class FullActionsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lua = LuaRuntime()
        cls.actual = cls.lua.execute((HERE/'full_actions.lua').read_text(encoding='utf-8'))
        # The frozen16 definition derives from the unchanged public builder,
        # plus the established last-frame-release change for ordinary attacks.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'legacy'
            build_actions16(root)
            source = (root/PACKAGE/'actions.lua').read_text(encoding='utf-8')
        anchor = ' if action<12 then return basic[action+1] end'
        assert source.count(anchor) == 1
        source = source.replace(anchor,
            " if action>=6 and action<=11 and frame==11 then return (action==8 or action==9) and 'D' or '' end\n" + anchor)
        cls.legacy = cls.lua.execute(source)

    def wave(self, action, direction='R', state=None):
        return [self.actual['keys'](action, f, state, direction) for f in range(12)]

    def test_actual_lua_all_waveforms_and_metadata_match_python(self):
        self.assertEqual(self.actual.interface, actions.INTERFACE)
        self.assertEqual(self.actual.count, 85)
        self.assertEqual(self.actual.frames, 12)
        self.assertEqual(len(set(actions.ACTION_NAMES)), 85)
        for action in range(85):
            self.assertEqual(dict(self.actual['descriptor'](action)), actions.descriptor(action))
            for forward in ('L', 'R'):
                self.assertEqual(self.wave(action, forward),
                    [actions.keys(action, f, None, forward) for f in range(12)], (action, forward))

    def test_old_sixteen_have_exact_same_executed_lua_waveforms(self):
        for action in range(16):
            for forward in ('L', 'R'):
                self.assertEqual(self.wave(action, forward),
                    [self.legacy['keys'](action, f, None, forward) for f in range(12)])

    def test_nine_directions_times_six_buttons_and_release(self):
        expected_directions = ('', 'R', 'L', 'D', 'D R', 'D L', 'U', 'U R', 'U L')
        seen = set()
        for direction, expected in zip(actions.DIRECTIONS, expected_directions):
            for button in actions.BUTTONS:
                action = actions.raw_id(direction, button)
                seen.add(action)
                held = ' '.join(x for x in (expected, button) if x)
                self.assertEqual(self.wave(action), [held]*11 + [expected])
                self.assertEqual(actions.descriptor(action)['target'], 'contextual')
        self.assertEqual(seen, set(range(16, 79)))

    def test_all_nine_special_requests_and_exact_strengths(self):
        by_family = {'hadouken': {12: 'LP', 79: 'MP', 80: 'HP'},
                     'shoryuken': {13: 'LP', 15: 'MP', 81: 'HP'},
                     'tatsumaki': {82: 'LK', 83: 'MK', 84: 'HK'}}
        for family, variants in by_family.items():
            for action, button in variants.items():
                self.assertEqual(actions.descriptor(action)['target'], family)
                expected = (['D']*3 + ['D R']*3 + ['R '+button]*2 + ['']*4
                    if family == 'hadouken' else
                    ['R']*2 + ['D']*2 + ['D R '+button]*2 + ['']*6
                    if family == 'shoryuken' else
                    ['D']*3 + ['D L']*3 + ['L '+button]*2 + ['']*4)
                self.assertEqual(self.wave(action), expected)

    def test_throw_requests_do_not_check_distance_grounded_or_opponent_state(self):
        unreadable = self.lua.execute("return setmetatable({}, {__index=function() error('Do not inspect game state') end})")
        for direction, key in (('F', 'R'), ('B', 'L')):
            for button in ('MP', 'HP', 'MK', 'HK'):
                self.assertEqual(self.wave(actions.raw_id(direction, button), state=unreadable),
                                 [key+' '+button]*11 + [key])
        for action in range(85):
            self.wave(action, state=unreadable)

    def test_six_air_and_crouch_button_requests_without_state_rewrite(self):
        for button in actions.BUTTONS[1:]:
            self.assertEqual(self.wave(actions.raw_id('N', button)), [button]*11 + [''])
            self.assertEqual(self.wave(actions.raw_id('D', button)), ['D '+button]*11 + ['D'])
        # A learned jump then attack sequence can use any of the six buttons;
        # execution makes no assumption whether the fighter is already airborne.
        self.assertEqual(self.wave(actions.raw_id('U')), ['U']*12)

    def test_invalid_ids_frames_and_facing_fail_closed(self):
        for bad in (-1, 85, .5, True, '12', None):
            with self.assertRaises(Exception): self.actual['keys'](bad, 0, None, 'R')
            with self.assertRaises(ValueError): actions.keys(bad, 0)
        for bad in (-1, 12, .5, True, '0', None):
            with self.assertRaises(Exception): self.actual['keys'](0, bad, None, 'R')
            with self.assertRaises(ValueError): actions.keys(0, bad)
        with self.assertRaises(Exception): self.actual['keys'](0, 0, None, 'F')
        with self.assertRaises(ValueError): actions.keys(0, 0, None, 'F')


if __name__ == '__main__':
    unittest.main()
