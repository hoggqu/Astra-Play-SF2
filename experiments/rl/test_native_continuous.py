"""Native deployment staging and phase contract, separate from legacy adapter."""
from pathlib import Path
import tempfile
import unittest

from lupa.lua54 import LuaRuntime

from astra_play_sf2.runner import sha256
from .native_continuous import stage_native_policy, audit_native_match

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parents[1]/'src/astra_play_sf2/assets'


class NativeContinuousTests(unittest.TestCase):
    def test_staging_compiles_and_hashes_the_actual_native_core(self):
        lua = LuaRuntime(unpack_returned_tuples=True)
        payload = {'schema':'astra.rl-policy.v1','model_sha256':'offline','observations':344,'actions':15,
                   'activation':'tanh','selection':'deterministic_argmax',
                   'layers':[{'weight':[[0]*344 for _ in range(15)],'bias':[0]*15}]}
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            hashes = stage_native_policy(run, 3, payload)
            staged = run/'training/runtime'
            self.assertEqual(hashes['rl_continuous_core.lua'],sha256(HERE/'native_continuous_core.lua'))
            self.assertNotEqual(hashes['rl_continuous_core.lua'],sha256(HERE/'continuous_core.lua'))
            for name in ('fighter.lua','play_core.lua','selection.json'):
                self.assertEqual((staged/name).read_bytes(),(ASSETS/name).read_bytes())
            for source in staged.glob('*.lua'):
                lua.execute('assert(load(...))',source.read_text())
            source=(staged/'play.lua').read_text()
            self.assertIn('native_timing=true',source)
            self.assertLess(source.index('if now==r.native_last_time'),source.index('r.speed_checks=r.speed_checks+1'))
            self.assertIn('r.recorder.row[25]=text',source)

    def test_audit_rejects_false_native_flag_duplicate_time_and_short_decisions(self):
        import copy
        match={'summary':{'native_timing':True,'frame':36,'native_period_checks':36,'native_frame_period':1/60},
               'events':[{'kind':'match_start','state':{'emulated_seconds':100.}},{'kind':'round_stop','round':1,'frame':25}],
               'trace':{'columns':['frame','emulated_seconds'],
                        'rows':[[i,100+i/60] for i in range(1,37)],
                        'decisions':[{'frame':i,'round':1} for i in (0,12,24)]}}
        self.assertTrue(audit_native_match(match)['ok'])
        broken=copy.deepcopy(match);broken['summary']['native_timing']=False
        with self.assertRaisesRegex(RuntimeError,'attest'): audit_native_match(broken)
        broken=copy.deepcopy(match);broken['trace']['rows'][1][1]=broken['trace']['rows'][0][1]
        with self.assertRaisesRegex(RuntimeError,'strictly increasing'): audit_native_match(broken)
        broken=copy.deepcopy(match);broken['trace']['decisions']=[]
        with self.assertRaisesRegex(RuntimeError,'coverage'): audit_native_match(broken)
        broken=copy.deepcopy(match);broken['trace']['rows']=[[i,100+i/30] for i in range(1,37)]
        with self.assertRaisesRegex(RuntimeError,'gaps'): audit_native_match(broken)
        broken=copy.deepcopy(match);broken['trace']['decisions'][1]['frame']=11
        with self.assertRaisesRegex(RuntimeError,'twelve'): audit_native_match(broken)

    def test_boundary_frame_zero_is_deferred_and_macro_keeps_twelve_frames(self):
        lua = LuaRuntime(unpack_returned_tuples=True)
        lua.globals().modules=lua.table()
        lua.globals().modules['training/runtime/play_core.lua']=lua.execute((ASSETS/'play_core.lua').read_text())
        lua.globals().modules['training/runtime/rl_actions.lua']=lua.execute((HERE/'actions.lua').read_text())
        lua.execute('function loadfile(path) return function() return assert(modules[path]) end end')
        lua.globals().Core=lua.execute((HERE/'native_continuous_core.lua').read_text())
        lua.execute('''
            s={timer=99,p1={char=4,hp=144,x=100,y=40,a=0,anim=1,wins=0},
                        p2={char=2,hp=144,x=200,y=40,a=0,anim=1,wins=0}}
            local A=modules['training/runtime/rl_actions.lua'];calls=0
            c=Core.new({mode='rl_ppo',opponent=2,choose=function(a,b,mode,state,reset)
                calls=calls+1;local seq={}
                for frame=0,11 do seq[#seq+1]={1,A.keys(12,frame,state,'R')} end
                return seq
            end},s)
            consumed={};deferred={};held=c:rl_prime(s)
            for frame=1,24 do
                consumed[#consumed+1]=held
                local e=c:tick(s)
                if e.input~=nil then held=e.input end
                if e.rl_deferred_input~=nil then
                    assert(held=='')
                    deferred[#deferred+1]=frame
                    held=e.rl_deferred_input -- frame_done phase, no native time advance
                end
            end
        ''')
        expected=['D']*3+['D R']*3+['R LP']*2+['']*4
        self.assertEqual(list(lua.globals().consumed.values()),expected*2)
        self.assertEqual(list(lua.globals().deferred.values()),[12,24])


if __name__=='__main__':
    unittest.main()
