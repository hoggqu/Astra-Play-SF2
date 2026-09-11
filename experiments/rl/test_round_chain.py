"""Candidate native Core lifecycle integration; no emulator claim from mocks."""
from pathlib import Path
import tempfile
import unittest
from .test_batch import BatchRuntimeTests, HERE, ASSETS
from .round_chain_builder import build, PACKAGE


class RoundChainTests(BatchRuntimeTests):
    # Reuse only the environment fixture, not tests tied to reset-every-round.
    def setUp(self):
        super().setUp()
        g=self.lua.globals()
        g.modules['training/runtime/play_core.lua']=self.lua.execute((ASSETS/'play_core.lua').read_text())
        g.modules['training/runtime/rl_settlement.lua']=self.lua.execute((HERE/'settlement.lua').read_text())
        g.modules['training/runtime/rl_native_core.lua']=self.lua.execute((HERE/'native_continuous_core.lua').read_text())
        self.lua.execute('''
            native={timer=99,p1={char=4,hp=144,displayed_hp=144,timeout_hp=0,x=100,y=40,a=0,anim=1,wins=0},
             p2={char=2,hp=144,displayed_hp=144,timeout_hp=0,x=200,y=40,a=0,anim=1,wins=0}}
            function fighter(i)
             local p=native[i==0 and 'p1' or 'p2'];local o={};for k,v in pairs(p) do o[k]=v end;return o
            end
            mem.read_u8=function(_,addr)
             if addr==0xff8ace then return math.floor(native.timer/10)*16+native.timer%10 end
             return native[addr<0xff8956 and 'p1' or 'p2'].wins
            end
            mem.read_i16=function(_,addr)
             local i=addr<0xff86c6 and 0 or 1;local base=0xff83c6+i*0x300
             return native[i==0 and 'p1' or 'p2'][addr-base==0x164 and 'timeout_hp' or 'displayed_hp']
            end
        ''')
        self.lua.execute((HERE/'round_chain_runtime.lua').read_text())

    def win(self,side=1):
        self.lua.execute(f'''
            native.timer=50
            native.p{side}.wins=native.p{side}.wins+1
            native.p{side}.a=16
            native.p{3-side}.hp=-1;native.p{3-side}.displayed_hp=-1;native.p{3-side}.a=0
        ''')
        for _ in range(361):self.tick()

    def opening(self):
        self.lua.execute('''native.timer=99;for _,p in ipairs({native.p1,native.p2}) do
          p.hp=144;p.displayed_hp=144;p.timeout_hp=0;p.a=0;p.anim=1 end''')
        for _ in range(90):self.tick()

    def rollout(self,count=8):
        self.request({'id':2,'op':'rollout','count':count,'actions':[6]*count,
                      'resets':[{'checkpoint':0,'lead':0}]*count})

    def test_round_chain_keeps_core_and_resets_history_without_refill_reward(self):
        self.reset();self.rollout(3)
        self.win();self.assertIsNone(self.reply(2));self.assertEqual(self.lua.globals().loads,1)
        self.opening()
        for _ in range(24):self.tick()
        result=self.reply(2);self.assertIsNotNone(result)
        self.assertEqual([t['done'] for t in result['transitions']],[True,False,False])
        self.assertEqual([t['episode_start'] for t in result['transitions']],[True,True,False])
        self.assertAlmostEqual(result['transitions'][0]['reward'],1.25)
        self.assertEqual([t['reward'] for t in result['transitions'][1:]],[0,0])
        self.assertEqual(len(result['episodes']),1);self.assertEqual(result['episodes'][0]['round'],1)
        self.assertEqual(result['chain_metrics']['natural_rounds'],1)
        self.assertEqual(result['chain_metrics']['loads'],1)
        first=result['transitions'][1]['observation']
        self.assertEqual(first[:86],first[86:172]);self.assertEqual(first[:86],first[-86:])

    def test_budget_at_terminal_waits_for_actionable_next_round(self):
        self.reset();self.rollout(1);self.win();self.assertIsNone(self.reply(2))
        self.opening();result=self.reply(2)
        self.assertTrue(result['episode_start']);self.assertEqual(result['state']['timer'],99)
        self.assertEqual(result['state']['p1']['wins'],1);self.assertEqual(self.lua.globals().loads,1)
        self.assertEqual(len(result['episodes']),1)
        self.assertEqual(self.lua.globals().pauses,2)
        # Next request starts at the already-observed R2 action boundary.
        self.request({'id':3,'op':'rollout','count':1,'actions':[6], 'resets':[{'checkpoint':0,'lead':0}]})
        for _ in range(12):self.tick()
        self.assertTrue(self.reply(3)['transitions'][0]['episode_start'])
        self.assertEqual(self.reply(3)['transitions'][0]['round'],2)

    def test_round3_and_match_end_load_once_only_after_second_pip(self):
        self.reset();self.rollout(4);self.win();self.opening();self.win(2);self.opening()
        self.assertEqual(self.lua.globals().loads,1)
        self.win(1);self.assertEqual(self.lua.globals().loads,2)
        self.lua.execute("native.p1.wins=0;native.p2.wins=0")
        self.opening()  # Simulated checkpoint restoration for the mock load.
        for _ in range(12):
            if self.reply(2):break
            self.tick()
        result=self.reply(2)
        self.assertEqual([e['round'] for e in result['episodes']],[1,2,3])
        self.assertEqual([e['outcome'] for e in result['episodes']],['win','loss','win'])
        self.assertEqual(result['chain_metrics']['matches'],1)
        self.assertEqual(result['chain_metrics']['loads'],2)

    def test_confirmed_draw_naturally_continues_without_rewarding_refill(self):
        self.reset();self.rollout(2)
        self.lua.execute("native.timer=0;for _,p in ipairs({native.p1,native.p2}) do p.hp=0;p.displayed_hp=0;p.timeout_hp=0;p.a=12 end")
        self.tick()
        self.lua.execute("native.p1.a=18;native.p2.hp=-1;native.p2.a=0;native.p2.anim=307368")
        for _ in range(360):self.tick()
        self.assertIsNone(self.reply(2))
        self.opening()
        for _ in range(12):self.tick()
        result=self.reply(2)
        self.assertEqual(result['episodes'][0]['outcome'],'draw')
        self.assertEqual(result['episodes'][0]['final_state']['p2']['hp'],-1)
        self.assertEqual(result['episodes'][0]['confirmation_state']['p2']['hp'],144)
        self.assertEqual([t['reward'] for t in result['transitions']],[0,0])
        self.assertEqual([t['episode_start'] for t in result['transitions']],[True,True])
        self.assertEqual(result['chain_metrics']['loads'],1)

    def test_multiple_actions_run_without_intermediate_file_rpc(self):
        return super().test_multiple_actions_run_without_intermediate_file_rpc()

# Only inherit fixture/cadence methods useful to the new protocol.
for _name in list(BatchRuntimeTests.__dict__):
    if _name.startswith('test_') and _name not in RoundChainTests.__dict__:
        setattr(RoundChainTests,_name,None)


class ChainBuilderTests(unittest.TestCase):
    def test_build_preserves_parent_and_marks_candidate(self):
        original=(HERE/'batch_runtime.lua').read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            result=build(Path(directory)/'chain')
            self.assertEqual(result['status'],'candidate');self.assertFalse(result['native_validated'])
            package=Path(directory)/'chain'/PACKAGE
            self.assertIn('round_chain=true',(package/'batch_runtime.lua').read_text())
            self.assertIn('independent cross-round harness',(package/'batch_train.py').read_text())
        self.assertEqual((HERE/'batch_runtime.lua').read_bytes(),original)

    def test_build_accepts_frozen_actions16_package_without_interface_changes(self):
        from .actions16_builder import build as build16
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);build16(root/'sixteen')
            source=root/'sixteen'/'astra_sf2_rl16'
            result=build(root/'chain',source)
            package=root/'chain'/PACKAGE
            self.assertEqual((package/'actions.lua').read_bytes(),(source/'actions.lua').read_bytes())
            self.assertIn("'actions': 16",(package/'batch_train.py').read_text())
            self.assertIn("'round_chain': True",(package/'batch_train.py').read_text())

    def test_actions16_capture_accepts_chain_identity_and_rejects_parent_tamper(self):
        import importlib
        import json
        import sys
        import gymnasium as gym
        import numpy as np
        from stable_baselines3 import PPO
        from .actions16_builder import build as build16
        class Spaces(gym.Env):
            observation_space=gym.spaces.Box(-1,1,(344,),np.float32)
            action_space=gym.spaces.Discrete(16)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);build16(root/'sixteen')
            build(root/'chain',root/'sixteen'/'astra_sf2_rl16')
            package=root/'chain'/PACKAGE
            model=PPO('MlpPolicy',Spaces(),n_steps=256,policy_kwargs={'net_arch':{'pi':[64,64],'vf':[64,64]}})
            model.astra_action_interface='ken_actions16_lp_mp_uppercut_v1';model.save(root/'model.zip')
            sys.path.insert(0,str(root/'chain'))
            try:
                support=importlib.import_module(PACKAGE+'.support')
                snapshot=support.capture_interface(root/'model.zip',package)
                self.assertEqual(snapshot['payload']['actions'],16)
                parent=root/'chain'/'parent-build.json'
                value=json.loads(parent.read_text());value['actions']=15;parent.write_text(json.dumps(value))
                with self.assertRaisesRegex(RuntimeError,'parent build identity changed'):
                    support.capture_interface(root/'model.zip',package)
            finally:
                sys.path.remove(str(root/'chain'))
                for name in list(sys.modules):
                    if name==PACKAGE or name.startswith(PACKAGE+'.'):sys.modules.pop(name)


    def test_standard_actions16_build_freezes_installed_production_without_src(self):
        import astra_play_sf2
        from .actions16_builder import build as build16
        from .round_chain_trial import freeze_production
        from astra_play_sf2.runner import sha256
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);build16(root/'sixteen')
            self.assertFalse((root/'sixteen'/'src').exists())
            source=root/'sixteen'/'astra_sf2_rl16'
            build(root/'chain',source)
            record=freeze_production(source,root/'chain')
            installed=Path(astra_play_sf2.__file__).resolve().parent
            self.assertEqual(record['origin'],'installed_package')
            self.assertEqual(record['source_package'],str(installed))
            for name,digest in record['files_sha256'].items():
                self.assertEqual(sha256(root/'chain'/'src'/'astra_play_sf2'/name),digest)
            self.assertIn('assets/play_core.lua',record['files_sha256'])
            self.assertEqual(record['files_sha256']['assets/play_core.lua'],sha256(installed/'assets'/'play_core.lua'))
            self.assertTrue((root/'chain'/'production-source.json').is_file())
