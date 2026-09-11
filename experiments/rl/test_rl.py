"""Offline experimental checks; no emulator or ROM required."""
import unittest
import json
import tempfile
import re
from pathlib import Path
import numpy as np
from lupa.lua54 import LuaRuntime, LuaError
from .env import ACTION_NAMES, MameEnv, features, reward


def state(hp1=144, hp2=144):
    return dict(timer=99, p1=dict(hp=hp1, x=200, y=40, a=0, char=4),
                p2=dict(hp=hp2, x=400, y=40, a=0, char=7))


class RLTests(unittest.TestCase):
    def test_baseline_mode_mapping_matches_frozen_selection(self):
        root = Path(__file__).resolve().parents[2]
        selected = json.loads((root/'src/astra_play_sf2/assets/selection.json').read_text())
        text = Path(__file__).with_name('runtime.lua').read_text()
        modes = dict(re.findall(r"\[(\d+)\]='([^']+)'", text))
        self.assertEqual(modes, selected)

    def test_partial_episode_keeps_original_phase_after_evaluation_switch(self):
        with tempfile.TemporaryDirectory() as folder:
            env = MameEnv.__new__(MameEnv)
            env.run = Path(folder)
            env.must_reset = False
            env.phase = 'trained'
            env.episode_phase = 'train'
            env.lead = 0
            env.episode_steps = 5
            env.last_frames = 60
            env.episode_return = -.1
            env.record_partial('reset_before_round_end')
            record = json.loads((env.run/'partial-episodes.jsonl').read_text())
            self.assertEqual(record['phase'], 'train')
            self.assertNotIn('outcome', record)
            self.assertTrue(env.must_reset)

    def test_reward_damage_sign_terminal_and_no_double_count(self):
        initial, hit = state(), state(144, 72)
        self.assertEqual(reward(initial, hit), .125)
        self.assertEqual(reward(hit, hit), 0)
        self.assertEqual(reward(state(), state(72, 144)), -.125)
        self.assertEqual(reward(hit, state(144, -1), 'win'), 1.125)
        self.assertEqual(reward(hit, hit, 'loss'), -1)
        self.assertEqual(reward(hit, hit, 'draw'), 0)

    def test_features_current_public_state_only_bounded_and_finite(self):
        s = state()
        vector = features(s)
        self.assertEqual(vector.shape, (86,))
        self.assertEqual(vector.dtype, np.float32)
        s.update(ai_rank=10000, ai_index=-999, difficulty_mirror=5)
        np.testing.assert_array_equal(vector, features(s))
        s['p1'].update(hp=-1, x=-100, y=800)
        vector = features(s)
        self.assertTrue(np.isfinite(vector).all())
        self.assertTrue((vector >= -1).all() and (vector <= 1).all())

    def test_actions_are_ordinary_inputs_and_exact_directional_macros(self):
        lua = LuaRuntime(unpack_returned_tuples=True)
        actions = lua.execute(Path(__file__).with_name('actions.lua').read_text())
        self.assertEqual(actions.count, len(ACTION_NAMES))
        allowed = {'L','R','U','D','LP','MP','HP','LK','MK','HK'}
        for side in ('L', 'R'):
            for action in range(actions.count):
                sequence = [actions['keys'](action, frame, None, side) for frame in range(actions.frames)]
                self.assertTrue(all(set(keys.split()) <= allowed for keys in sequence))
            self.assertEqual([actions['keys'](12, f, None, side) for f in range(12)],
                             ['D']*3+['D '+side]*3+[side+' LP']*2+['']*4)
            self.assertEqual([actions['keys'](13, f, None, side) for f in range(12)],
                             [side]*2+['D']*2+['D '+side+' LP']*2+['']*6)
        for invalid in (-1, 15, 1.5):
            with self.assertRaises(LuaError):
                actions['keys'](invalid, 0, None, 'R')


if __name__ == '__main__':
    unittest.main()
