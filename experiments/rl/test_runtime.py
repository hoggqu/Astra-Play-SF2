"""Offline regression tests for the RL controller's native-frame boundary.

The real runtime, action macros and speed guard execute in Lua 5.4. Only MAME,
the result core and file transport are replaced. These tests do not launch MAME
and do not certify a native game outcome.
"""
import json
from pathlib import Path
import unittest

try:
    from lupa.lua54 import LuaRuntime
except ImportError:
    LuaRuntime = None


HERE = Path(__file__).resolve().parent
ASSETS = HERE.parents[1] / "src" / "astra_play_sf2" / "assets"


@unittest.skipUnless(LuaRuntime, "Install lupa to execute the real Lua runtime")
class RuntimeBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.lua = LuaRuntime(unpack_returned_tuples=True)
        self.published = {}
        self.lua.globals().publish_python = self.published.__setitem__
        self.lua.execute(r'''
            active = {}
            keys = {}
            for _, key in ipairs({'R','L','D','U','LP','MP','HP','LK','MK','HK','C','S'}) do
                local name = key
                keys[name] = {set_value=function(_, value) active[name]=value end}
            end
            function release() active={} end
            function play_busy() return false end
            function choose() return {{1,''}} end
            play_bridge_generation=1
            astra_difficulty_bits=0
            astra_difficulty={check=function(level)
                assert(level==7)
                return {effective_difficulty=7}
            end}
            function fighter(i)
                return {char=i==0 and 4 or 0, hp=144, x=100+i*100,
                        y=40, a=0, anim=1}
            end
            mem={read_u8=function(_, addr) return addr==0xff8ace and 0x99 or 0 end,
                 read_i16=function() return 144 end}
            manager={machine={paused=true,
                video={speed_factor=1000, throttle_rate=1, throttled=true},
                load=function(_, path) loaded_path=path end}}
            local subscription={unsubscribe=function() end}
            emu={pause=function() manager.machine.paused=true end,
                 unpause=function() manager.machine.paused=false end,
                 add_machine_post_load_notifier=function(fn) on_load=fn;return subscription end,
                 add_machine_frame_notifier=function(fn) on_frame=fn;return subscription end,
                 register_frame_done=function(fn) on_rpc=fn end}
            modules={}
            modules['training/runtime/rl_checkpoint.lua']='/isolated/run/training/rl-start.sta'
            modules['training/runtime/status_io.lua']={publish=function(path, text)
                publish_python(path,text)
            end}
            plan={[1]='R',[12]='D LP',[24]='HP'}
            opening_ok=true
            core_count=0
            modules['training/runtime/play_core.lua']={
                opening=function() return opening_ok end,
                new=function(options)
                    core_count=core_count+1
                    core_options=options
                    local core={frame=0,phase='fighting',rounds={}}
                    function core:tick(state)
                        self.frame=self.frame+1
                        if self.frame==invalid_at then
                            return {terminal={valid=false,reason='unresolved settlement'}}
                        end
                        if self.frame==terminal_at then
                            self.phase='between'
                            self.rounds[1]={outcome='win',frame=self.frame,
                                stop={timer=0,marker='native stop'},
                                settled={marker='native settled'},score={1,0}}
                        end
                        return {input=plan[self.frame]}
                    end
                    return core
                end
            }
            function loadfile(path)
                assert(modules[path]~=nil,'Unexpected module '..path)
                return function() return modules[path] end
            end
            io.open=function(path, mode)
                assert(path=='training/rl-request.txt' and mode=='rb')
                if not request then return nil end
                return {read=function() return request end,close=function() end}
            end
            os.remove=function(path)
                assert(path=='training/rl-request.txt')
                request=nil
                return true
            end
            function enqueue(id,op,arg,flag)
                assert(not request)
                request=string.format('%d %s %d %d\n',id,op,arg or 0,flag or 0)
                on_rpc()
            end
        ''')
        self.lua.globals().modules["training/runtime/speed.lua"] = self.lua.execute(
            (ASSETS / "speed.lua").read_text(encoding="utf-8"))
        self.lua.globals().modules["training/runtime/rl_actions.lua"] = self.lua.execute(
            (HERE / "actions.lua").read_text(encoding="utf-8"))
        self.lua.execute((HERE / "runtime.lua").read_text(encoding="utf-8"))

    def reply(self, identifier):
        text = self.published.get(f"training/rl-reply-{identifier:08d}.json")
        return json.loads(text) if text is not None else None

    def reset_baseline(self, lead=0):
        self.lua.globals().enqueue(1, "reset", lead, 1)
        self.lua.globals().on_load()
        for _ in range(2 + lead):
            self.lua.globals().on_frame()
        self.assertTrue(self.reply(1)["reset_confirmed"])

    def advance(self, frames):
        """Sample inputs consumed by a frame before its end notifier runs."""
        samples = []
        for _ in range(frames):
            self.assertFalse(self.lua.globals().manager.machine.paused)
            samples.append(frozenset(self.lua.globals().active.keys()))
            self.lua.globals().on_frame()
        return samples

    def test_baseline_preserves_inputs_and_new_commands_across_12_frame_boundaries(self):
        self.reset_baseline()
        samples = []
        for identifier in (2, 3, 4):
            self.lua.globals().enqueue(identifier, "step", 0, 0)
            samples.extend(self.advance(12))
            result = self.reply(identifier)
            self.assertEqual(result["frames"], (identifier - 1) * 12)
            self.assertFalse(result["done"])
            self.assertTrue(self.lua.globals().manager.machine.paused)
            self.assertEqual(list(self.lua.globals().active.keys()), [])
        # Core's first end-of-frame decision applies on frame 2. Decisions at
        # frame 12 and 24 must survive pause/release and apply on frame 13/25.
        self.assertEqual(samples, [frozenset()] + [frozenset({"R"})] * 11
                         + [frozenset({"D", "LP"})] * 12
                         + [frozenset({"HP"})] * 12)

    def test_terminal_reply_retains_native_round_evidence(self):
        self.reset_baseline()
        self.lua.globals().terminal_at = 25
        for identifier in (2, 3):
            self.lua.globals().enqueue(identifier, "step", 0, 0)
            self.advance(12)
        self.lua.globals().enqueue(4, "step", 0, 0)
        self.advance(1)
        result = self.reply(4)
        self.assertTrue(result["done"])
        self.assertTrue(result["training_only"])
        self.assertEqual(result["outcome"], "win")
        self.assertEqual(result["frames"], 25)
        self.assertEqual(result["native_round"], {
            "outcome": "win", "frame": 25,
            "stop": {"timer": 0, "marker": "native stop"},
            "settled": {"marker": "native settled"},
            "score": {"1": 1, "2": 0},
        })
        self.assertTrue(self.lua.globals().manager.machine.paused)
        self.assertEqual(list(self.lua.globals().active.keys()), [])

    def test_reset_requires_post_load_confirmation_and_all_refresh_frames(self):
        self.lua.globals().enqueue(1, "reset", 4, 1)
        self.assertEqual(self.lua.globals().loaded_path,
                         "/isolated/run/training/rl-start.sta")
        self.advance(20)
        self.assertIsNone(self.reply(1))
        self.assertEqual(self.lua.globals().core_count, 0)
        self.lua.globals().on_load()
        self.advance(5)
        self.assertIsNone(self.reply(1))
        self.advance(1)
        self.assertTrue(self.reply(1)["reset_confirmed"])
        self.assertEqual(self.lua.globals().core_count, 1)

    def test_invalid_native_result_is_an_error_not_a_loss(self):
        self.reset_baseline()
        self.lua.globals().invalid_at = 1
        self.lua.globals().enqueue(2, "step", 0, 0)
        self.advance(1)
        result = self.reply(2)
        self.assertIn("unresolved settlement", result["error"])
        self.assertNotIn("outcome", result)
        self.assertIn("training/rl-error.json", self.published)
        self.assertTrue(self.lua.globals().manager.machine.paused)


if __name__ == "__main__":
    unittest.main()
