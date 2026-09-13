"""An observed simultaneous KO may finish one native timer decrement, once."""
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch
from lupa.lua54 import LuaRuntime
HERE=Path(__file__).resolve().parent
CORE=HERE.parents[1]/'src/astra_play_sf2/assets/play_core.lua'

class DoubleKoTimerTests(unittest.TestCase):
    def table(self,s):return self.lua.table_from(s,recursive=True)
    def tick(self,s,n=1):
        for _ in range(n):self.core.tick(self.core,self.table(s))
    def make(self):
        self.lua=LuaRuntime(unpack_returned_tuples=True)
        C=self.lua.execute(CORE.read_text());C=self.lua.execute((HERE/'settlement_v8.lua').read_text())(C)
        def p(c,x):return dict(char=c,x=x,y=40,a=0,anim=12345,hp=144,displayed_hp=144,timeout_hp=0,wins=0)
        self.opening=dict(timer=99,p1=p(4,100),p2=p(2,200))
        self.core=C.new(self.lua.table_from(dict(opponent=2,mode='test',choose=self.lua.eval('function()return {{12,""}}end'))),self.table(self.opening))
        s=deepcopy(self.opening);s['timer']=36
        s['p1'].update(hp=22,displayed_hp=22,a=10);s['p2'].update(hp=21,displayed_hp=21,a=10)
        self.tick(s);return s
    def ko(self,s):
        k=deepcopy(s);k['p1'].update(hp=-1,a=6);k['p2'].update(hp=-1,a=0);return k
    def finish(self,k,timer=35):
        k=deepcopy(k);k['timer']=timer
        k['p1']['displayed_hp']=-1;k['p2']['displayed_hp']=-1
        self.tick(k,365);return k
    def test_one_next_frame_tail_waits_for_native_same_score_next_round(self):
        k=self.ko(self.make());before=deepcopy(k);self.tick(k)
        t=self.finish(k);self.assertEqual(k,before);self.assertEqual(len(self.core.rounds),0)
        self.assertIsNotNone(self.core.rl_double_ko.mature)
        self.tick(self.opening);row=self.core.rounds[1]
        self.assertEqual(row.outcome,'draw');self.assertEqual(row.recognition,'rl-native-double-ko-timer-tail-v8')
        self.assertEqual(row.settled.timer,35);self.assertEqual(row.stop.timer,36)
        self.assertEqual(row.double_ko_timer_tail.previous_live.state.p1.hp,22)
        self.assertEqual(row.double_ko_timer_tail['from'],36);self.assertEqual(row.double_ko_timer_tail.to,35)
        self.assertEqual([row.score[1],row.score[2]],[0,0])
        self.assertGreater(row.confirmation_frame,row.settled_frame)
    def test_missing_adjacent_live_proof_fails_closed_for_tail(self):
        for kind in ('missing','gap','prior_ko','zero_hp','prior_display','prior_score','prior_timer','prior_latch'):
            s=self.make();p=self.core.rl_time_previous
            if kind=='missing':self.core.rl_time_previous=None
            elif kind=='gap':p.frame-=1
            elif kind=='prior_ko':p.state.p1.hp=-1
            elif kind=='zero_hp':p.state.p1.hp=0;p.state.p1.displayed_hp=0
            elif kind=='prior_display':p.state.p1.displayed_hp=23
            elif kind=='prior_score':p.state.p1.wins=1
            elif kind=='prior_timer':p.state.timer=37
            elif kind=='prior_latch':p.state.p1.timeout_hp=22
            k=self.ko(s);self.tick(k);self.finish(k);self.tick(self.opening)
            self.assertEqual(len(self.core.rounds),0,kind)
    def test_wrong_or_later_timer_change_rejected(self):
        for kind in ('drop2','increase','zero','delayed','twice'):
            k=self.ko(self.make());self.tick(k)
            if kind=='delayed':self.tick(k)
            if kind=='twice':
                t=deepcopy(k);t['timer']=35;self.tick(t)
            self.finish(k,{'drop2':34,'increase':37,'zero':0,'delayed':35,'twice':34}[kind]);self.tick(self.opening)
            self.assertEqual(len(self.core.rounds),0,kind)
    def test_pip_display_mutation_and_early_opening_rejected(self):
        for kind in ('pip','hp','display','early'):
            k=self.ko(self.make());self.tick(k);t=deepcopy(k);t['timer']=35
            if kind=='pip':t['p1']['wins']=2
            if kind=='hp':t['p1']['hp']=1
            if kind=='display':t['p1']['displayed_hp']=23
            self.tick(t)
            if kind!='early':self.finish(t)
            self.tick(self.opening);self.assertEqual(len(self.core.rounds),0,kind)
    def test_unchanged_timer_retains_old_recognition(self):
        k=self.ko(self.make());self.tick(k);self.finish(k,36);self.tick(self.opening)
        self.assertEqual(self.core.rounds[1].recognition,'rl-native-double-ko-next-round-v4')
    def test_all_v7_and_older_contracts_remain(self):
        from . import test_settlement_v7
        read=Path.read_text
        def current(path,*args,**kwargs):
            return read(HERE/'settlement_v8.lua',*args,**kwargs) if path.name=='settlement_v7.lua' else read(path,*args,**kwargs)
        suite=unittest.defaultTestLoader.loadTestsFromModule(test_settlement_v7);result=unittest.TestResult()
        with patch.object(Path,'read_text',current):suite.run(result)
        self.assertEqual(result.errors+result.failures,[])

if __name__=='__main__':unittest.main()
