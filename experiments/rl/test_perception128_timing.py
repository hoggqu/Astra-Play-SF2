"""Portable source and callback-order regression tests; no emulator or ROM."""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from .perception128_timing_builder import bootstrap
from .perception128_timing_identity import patch_runtime, validate_build


class TimingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory()
        cls.root=Path(cls.temp.name)/'code';cls.manifest=bootstrap(cls.root)
        cls.package=cls.root/cls.manifest['package']
        cls.parent=cls.root/'timing-parent/astra_sf2_rl_perception128_late_ko'
        cls.before=(cls.parent/'batch_runtime.lua').read_bytes()
        cls.after=(cls.package/'batch_runtime.lua').read_bytes()

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_minimal_source_change_and_guard_preserved(self):
        changed={p.name for p in self.parent.iterdir() if p.is_file() and p.read_bytes()!=(self.package/p.name).read_bytes()}
        self.assertEqual(changed,{'initialize.py','support.py','batch_train.py','batch_runtime.lua'})
        self.assertEqual(self.after,patch_runtime(self.before))
        for line in self.before.splitlines():
            if b'deferred_after' in line:self.assertIn(line,self.after.splitlines())
        with self.assertRaises(RuntimeError):patch_runtime(self.after)
        with self.assertRaises(RuntimeError):patch_runtime(self.before+self.before)

    def harness(self,patched=True):
        from lupa.lua54 import LuaRuntime
        lua=LuaRuntime(unpack_returned_tuples=True)
        text=(self.after if patched else self.before).decode()
        # Execute actual captured input(), answer(), reset() and rollout-resume
        # bodies. The harness supplies only their non-emulator dependencies.
        input_source=text[text.index('local held_input') if patched else text.index('local function input'):text.index('local function hp')]
        answer=text[text.index('local function answer()'):text.index('local function fail(')]
        reset=text[text.index('local function reset(choice)'):text.index('local function begin_episode')]
        start=text.index("    assert(pending.op=='rollout'")
        resume=text[start:text.index('\n   end',start)]
        lua.execute('''
held={};latched='';paused=false;clock=12;model=false;obs={};state={};metrics={loads=0}
checkpoints={'checkpoint'};active_action=nil;pending={id=1,transitions={},episodes={}}
core={phase='fighting'};episode_start=false
function release() held={} end
keys=setmetatable({}, {__index=function(t,k) local f={}; function f:set_value(v) held[k]=v~=0 end;t[k]=f;return f end})
function pressed() local out={};for k,v in pairs(held) do if v then out[#out+1]=k end end;table.sort(out);return table.concat(out,' ') end
function poll() if not paused then latched=pressed() end;return latched end
emu={pause=function() paused=true end,unpause=function() paused=false end}
m={time={as_double=function() return clock end},load=function() end}
IO={publish=function() end};function partial() return false end;function json() return '{}' end
function start_action() pending.deferred='R LP' end
''')
        lua.execute(input_source+answer+reset+'''
set_input=input;pause_answer=answer;reset_choice=reset
function rpc_resume()
 pending={op='rollout',id=2}
'''+resume+'''
 emu.unpause()
end
function deferred_hook()
 if pending and pending.deferred~=nil and (not pending.deferred_after or m.time:as_double()>pending.deferred_after) then
  local text=pending.deferred;pending.deferred=nil;pending.deferred_after=nil;input(text)
 end
end
''')
        return lua

    def test_same_time_resume_poll_preserves_old_input(self):
        for old in ('U R','U L','','D LP','R HK'):
            for patched in (False,True):
                with self.subTest(old=old,patched=patched):
                    lua=self.harness(patched);g=lua.globals()
                    g.set_input(old);original=g.poll()
                    g.pause_answer();self.assertEqual(g.pressed(),'')
                    g.rpc_resume();g.deferred_hook()
                    resumed=g.poll()
                    self.assertEqual(resumed,original if patched else '')
                    # First CPU step after resume sees original latch. New
                    # macro begins at next advancing frame_done, never earlier.
                    g.clock=13;g.deferred_hook();self.assertEqual(g.poll(),'LP R')

    def test_reset_clears_tracked_input_before_manual_reset_answer(self):
        lua=self.harness();g=lua.globals()
        g.set_input('U L HK');g.pause_answer();g.rpc_resume()
        g.reset_choice(lua.table_from({'checkpoint':0,'lead':4}))
        g.pause_answer();g.rpc_resume();self.assertEqual(g.poll(),'')

    def test_all_action_boundary_latches_match_continuous(self):
        lua=self.harness();g=lua.globals()
        # Each macro boundary can retain a direction; pause/resume must be a
        # no-op on the input latch for any previous/following waveform pair.
        actions=lua.execute((self.package/'actions.lua').read_text())
        snapshot=lua.table_from({'p1':lua.table_from({'x':100}), 'p2':lua.table_from({'x':200})})
        for action in range(85):
            old=actions['keys'](action,11,snapshot,'R')
            for pause_count in (1,2,7):
                g.set_input(old);expected=g.poll()
                for _ in range(pause_count):
                    g.pause_answer();g.rpc_resume();g.deferred_hook()
                    self.assertEqual(g.poll(),expected,(action,pause_count))

    def test_identity_tampering_rejected(self):
        validate_build(self.manifest,self.package)
        runtime=self.package/'batch_runtime.lua';before=runtime.read_bytes()
        try:
            runtime.write_bytes(before+b'\n-- mutation')
            with self.assertRaises(RuntimeError):validate_build(self.manifest,self.package)
        finally:runtime.write_bytes(before)

    def test_isolated_metadata_uses_new_identity(self):
        env=dict(os.environ,PYTHONPATH=os.pathsep.join((str(self.root/'src'),str(self.root))),PYTHONNOUSERSITE='1')
        code="""from pathlib import Path
from astra_sf2_rl_perception128_timing.perception128_timing_identity import training_metadata
r=training_metadata(Path('astra_sf2_rl_perception128_timing'))
assert r['input_timing_protocol']=='restore-held-input-before-rpc-resume-v1'
from astra_sf2_rl_perception128_timing.support import validate_model
"""
        done=subprocess.run([sys.executable,'-c',code],cwd=self.root,env=env,capture_output=True,text=True,timeout=60)
        self.assertEqual(done.returncode,0,done.stdout+done.stderr)


if __name__=='__main__':unittest.main()
