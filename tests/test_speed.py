"""CLI speed choices plus actual Lua 5.4 speed application/guard checks."""
from contextlib import redirect_stdout
from importlib.resources import files
import io
from pathlib import Path
import unittest
from unittest.mock import patch
from astra_play_sf2 import cli, runner

try:
    from lupa.lua54 import LuaError, LuaRuntime
except ImportError:
    LuaError = LuaRuntime = None


class SpeedCLITests(unittest.TestCase):
    def test_fixed_speeds_reach_runner_without_changing_budget(self):
        for mode in ('2x', '4x'):
            with self.subTest(mode=mode), patch.object(cli, 'load_config', return_value={}), \
                 patch.object(runner, 'verify', return_value=(Path('fixture-run'), 0)) as verify, \
                 redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(['verify', '--difficulty', '7', '--speed', mode]), 0)
                verify.assert_called_once_with({}, [7], 1, mode, None)


@unittest.skipUnless(LuaRuntime is not None, 'Real Lua speed tests require optional test dependency lupa')
class LuaSpeedTests(unittest.TestCase):
    def setUp(self):
        self.lua = LuaRuntime(unpack_returned_tuples=True)
        self.speed = self.lua.execute(files('astra_play_sf2').joinpath('assets/speed.lua').read_text(encoding='utf-8'))

    def test_each_mode_applies_rate_and_can_restore_normal(self):
        video = self.lua.table_from({'throttled': False, 'throttle_rate': 8, 'speed_factor': 1000})
        for mode, rate in (('normal', 1), ('2x', 2), ('4x', 4), ('fast', 1)):
            self.speed.apply(video, mode)
            self.assertEqual(video.throttle_rate, rate)
            self.assertEqual(video.throttled, mode != 'fast')
            self.speed.check(video, mode)
        self.speed.apply(video, 'normal')
        self.assertTrue(video.throttled)
        self.assertEqual(video.throttle_rate, 1)

    def test_guard_rejects_mid_match_throttle_rate_and_factor_changes(self):
        for mode in ('normal', '2x', '4x', 'fast'):
            for key, bad in (('throttled', mode == 'fast'), ('throttle_rate', 3), ('speed_factor', 2000)):
                with self.subTest(mode=mode, key=key):
                    video = self.lua.table_from({'speed_factor': 1000})
                    self.speed.apply(video, mode)
                    video[key] = bad
                    with self.assertRaisesRegex(LuaError, 'selected game speed changed'):
                        self.speed.check(video, mode)

    def test_unknown_mode_and_nondefault_base_factor_are_rejected(self):
        video = self.lua.table_from({'speed_factor': 1000})
        with self.assertRaisesRegex(LuaError, 'Speed must be'):
            self.speed.apply(video, '3x')
        video.speed_factor = 2000
        with self.assertRaisesRegex(LuaError, 'base speed factor changed'):
            self.speed.apply(video, '2x')
