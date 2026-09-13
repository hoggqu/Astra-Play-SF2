"""Timer zero does not preclude a later, positively observed damage KO."""
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch
from lupa.lua54 import LuaRuntime
HERE=Path(__file__).resolve().parent
CORE=HERE.parents[1]/'src/astra_play_sf2/assets/play_core.lua'

class LateKoTests(unittest.TestCase):
    def table(self,s):return self.lua.table_from(s,recursive=True)
    def tick(self,s,n=1):
        for _ in range(n):self.core.tick(self.core,self.table(s))
    def make(self,loss=False):
        self.lua=LuaRuntime(unpack_returned_tuples=True)
        C=self.lua.execute(CORE.read_text());C=self.lua.execute((HERE/'settlement_v7.lua').read_text())(C)
        def p(c,x):return dict(char=c,x=x,y=40,a=10,anim=12345,hp=144,displayed_hp=144,timeout_hp=0,wins=0)
        self.opening=dict(timer=99,p1=p(4,100),p2=p(7,200))
        self.core=C.new(self.lua.table_from(dict(opponent=7,mode='test',choose=self.lua.eval('function()return {{12,""}}end'))),self.table(self.opening))
        self.winner='p2' if loss else 'p1';self.loser='p1' if loss else 'p2'
        s=deepcopy(self.opening);s['timer']=0
        s[self.winner].update(hp=10,displayed_hp=10)
        s[self.loser].update(hp=15,displayed_hp=15,y=80)
        self.tick(s,6);return s
    def ko(self,s):
        t=deepcopy(s);t[self.loser].update(hp=-1,a=20);return t
    def award(self,t):
        t=deepcopy(t);t[self.winner].update(wins=1,a=16)
        t[self.loser].update(displayed_hp=-1,y=40,a=2);return t
    def test_both_sides_adjacent_ko_then_unique_pip(self):
        for loss in (False,True):
            k=self.ko(self.make(loss));self.tick(k);self.assertIsNotNone(self.core.rl_late_ko)
            t=self.award(k);self.tick(t,360);self.assertEqual(len(self.core.rounds),0)
            self.tick(t);r=self.core.rounds[1]
            self.assertEqual(r.outcome,'loss' if loss else 'win')
            self.assertEqual(r.recognition,'rl-native-unlatched-time-ko-v7')
            self.assertEqual(r.stable_award_frames,360)
            self.assertEqual(r.late_ko.previous.state[self.loser].hp,15)
    def test_same_frame_ko_pip_has_same_maturity(self):
        t=self.award(self.ko(self.make()));self.tick(t,360);self.assertEqual(len(self.core.rounds),0)
        self.tick(t);self.assertEqual(self.core.rounds[1].stable_award_frames,360)
    def test_no_adjacent_live_or_already_awarded_cannot_be_reconstructed(self):
        for kind in ('missing','gap','prior_ko','prior_award','prior_timer','prior_display','prior_zero_hp','prior_latch'):
            s=self.make();k=self.ko(s)
            p=self.core.rl_time_previous
            if kind=='missing':self.core.rl_time_previous=None
            elif kind=='gap':p.frame-=1
            elif kind=='prior_ko':p.state[self.loser].hp=-1
            elif kind=='prior_award':p.state[self.winner].wins=1
            elif kind=='prior_timer':p.state.timer=1
            elif kind=='prior_display':p.state[self.loser].displayed_hp=14
            elif kind=='prior_zero_hp':p.state[self.loser].hp=0;p.state[self.loser].displayed_hp=0
            elif kind=='prior_latch':p.state[self.loser].timeout_hp=15
            self.tick(k);self.tick(self.award(k),500)
            self.assertIsNone(self.core.rl_late_ko,kind);self.assertEqual(len(self.core.rounds),0,kind)
    def test_latched_time_or_post_edge_latch_invalidates_new_evidence(self):
        for when in ('edge','later'):
            k=self.ko(self.make())
            if when=='edge':k[self.loser]['timeout_hp']=15
            self.tick(k);t=self.award(k)
            if when=='later':t[self.winner]['timeout_hp']=10
            self.tick(t,500);self.assertEqual(len(self.core.rounds),0)
    def test_no_pip_wrong_pip_and_early_next_round_do_not_qualify(self):
        for kind in ('none','other','both','early'):
            k=self.ko(self.make());self.tick(k);t=self.award(k)
            if kind=='none':t[self.winner]['wins']=0
            if kind=='other':t[self.winner]['wins']=0;t[self.loser]['wins']=1
            if kind=='both':t[self.loser]['wins']=1
            self.tick(t,10 if kind=='early' else 500)
            if kind=='early':
                n=deepcopy(self.opening);n[self.winner]['wins']=1;self.tick(n)
            self.assertEqual(len(self.core.rounds),0,kind)
    def test_all_v6_and_older_contracts_remain(self):
        from . import test_settlement_v6
        read=Path.read_text
        def current(path,*args,**kwargs):
            return read(HERE/'settlement_v7.lua',*args,**kwargs) if path.name=='settlement_v6.lua' else read(path,*args,**kwargs)
        suite=unittest.defaultTestLoader.loadTestsFromModule(test_settlement_v6);result=unittest.TestResult()
        with patch.object(Path,'read_text',current):suite.run(result)
        self.assertEqual(result.errors+result.failures,[])

if __name__=='__main__':unittest.main()
