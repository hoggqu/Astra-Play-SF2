"""Non-TIME native KO chronology replaces loser-pose exceptions only."""
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch
from lupa.lua54 import LuaRuntime
HERE=Path(__file__).resolve().parent
CORE=HERE.parents[1]/'src/astra_play_sf2/assets/play_core.lua'

class NativeKoTests(unittest.TestCase):
    def test_all_previous_v5_contracts_against_v6_without_editing_old_tests(self):
        from . import test_settlement_v5, test_settlement_v5_regression
        read=Path.read_text
        def current(path,*args,**kwargs):
            return read(HERE/'settlement_v6.lua',*args,**kwargs) if path.name=='settlement_v5.lua' else read(path,*args,**kwargs)
        suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromModule(module)
                                 for module in (test_settlement_v5,test_settlement_v5_regression))
        result=unittest.TestResult()
        with patch.object(Path,'read_text',current): suite.run(result)
        self.assertGreater(result.testsRun,20)
        self.assertEqual(result.errors+result.failures,[])
    def make(self,loss=False,score=0):
        self.lua=LuaRuntime(unpack_returned_tuples=True)
        C=self.lua.execute(CORE.read_text());C=self.lua.execute((HERE/'settlement_v6.lua').read_text())(C)
        def p(c,x):return dict(char=c,x=x,y=40,a=0,anim=1,hp=144,displayed_hp=144,timeout_hp=0,wins=0)
        self.opening=dict(timer=99,p1=p(4,100),p2=p(3,200))
        self.core=C.new(self.lua.table_from(dict(opponent=3,mode='test',choose=self.lua.eval('function()return {{12,""}}end'))),self.table(self.opening))
        if score:self.core.score=self.table([score,score])
        s=deepcopy(self.opening);s['timer']=53
        self.winner='p2' if loss else 'p1';self.loser='p1' if loss else 'p2'
        for p in ('p1','p2'):s[p]['wins']=score
        s[self.winner].update(hp=71,displayed_hp=71,a=16)
        s[self.loser].update(hp=-1,displayed_hp=27,a=20,y=84,anim=12345)
        self.tick(s)
        return s
    def table(self,s):return self.lua.table_from(s,recursive=True)
    def tick(self,s,n=1):
        for _ in range(n):self.core.tick(self.core,self.table(s))
    def awarded(self,s):
        s=deepcopy(s);s[self.winner]['wins']+=1
        s[self.loser].update(displayed_hp=-1,a=2,y=40)
        return s
    def test_both_sides_and_match_winning_native_score(self):
        for loss in (False,True):
            for score in (0,1):
                t=self.awarded(self.make(loss,score));raw=deepcopy(t)
                self.tick(t,360);self.assertEqual(len(self.core.rounds),0)
                self.tick(t);r=self.core.rounds[1]
                self.assertEqual(r.outcome,'loss' if loss else 'win')
                self.assertEqual(r.recognition,'rl-native-nontime-pip-ko-v6')
                self.assertEqual(r.stable_award_frames,360)
                self.assertEqual(self.core.phase,'complete' if score else 'between')
                self.assertEqual(raw,t)
    def test_no_pip_and_early_opening_never_prove_win(self):
        s=self.make();s[self.loser].update(displayed_hp=-1,y=40,a=2);self.tick(s,400)
        self.assertEqual(len(self.core.rounds),0)
        opening=deepcopy(self.opening);opening[self.winner]['wins']=1
        self.tick(opening);self.assertEqual(self.core.phase,'invalid')
    def test_chronology_mutations_disable_fallback_without_inventing_result(self):
        for target,field,value in [('p1','hp',70),('p1','displayed_hp',70),('p2','hp',0),('p2','displayed_hp',28),('p2','wins',1),('s','timer',52)]:
            s=self.make();t=self.awarded(s);self.tick(t)
            bad=deepcopy(t)
            (bad if target=='s' else bad[target])[field]=value
            self.tick(bad);self.tick(t,400)
            self.assertEqual(len(self.core.rounds),0,(target,field))
        s=self.make();t=self.awarded(s);self.tick(t);self.core.frame+=1;self.tick(t,400)
        self.assertEqual(len(self.core.rounds),0)
    def test_airborne_and_wrong_winner_pose_require_later_maturity(self):
        for target,field,value in [('p2','y',84),('p1','a',10)]:
            t=self.awarded(self.make());t[target][field]=value;self.tick(t,500)
            self.assertEqual(len(self.core.rounds),0)
            t[target][field]=40 if field=='y' else 16;self.tick(t)
            self.assertEqual(len(self.core.rounds),1)
    def test_original_known_ko_keeps_original_timing_and_recognition(self):
        t=self.awarded(self.make());t[self.loser]['a']=0
        self.tick(t,359);self.assertEqual(len(self.core.rounds),0)
        self.tick(t);self.assertEqual(len(self.core.rounds),1)
        self.assertIsNone(self.core.rounds[1].recognition)
    def test_begin_round_clears_fallback_and_terminal_core_is_not_revived(self):
        t=self.awarded(self.make());self.tick(t,361)
        s=deepcopy(self.opening);s[self.winner]['wins']=1;self.tick(s,90)
        self.assertIsNone(self.core.rl_ko_award);self.assertIsNone(self.core.rl_ko_disqualified)
        self.core.phase='invalid';self.tick(t);self.assertEqual(self.core.phase,'invalid')

if __name__=='__main__':unittest.main()
