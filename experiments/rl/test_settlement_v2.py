"""Pose-independent time result: native award chronology and adversarial cases."""
from copy import deepcopy
from pathlib import Path
import unittest
from lupa.lua54 import LuaRuntime

HERE=Path(__file__).resolve().parent
ASSETS=HERE.parents[1]/'src/astra_play_sf2/assets'

class SettlementV2Tests(unittest.TestCase):
    def make(self,opponent=7):
        self.lua=LuaRuntime(unpack_returned_tuples=True)
        core=self.lua.execute((ASSETS/'play_core.lua').read_text())
        core=self.lua.execute((HERE/'settlement.lua').read_text())(core)
        def actor(char,x): return dict(char=char,x=x,y=40,a=0,anim=1,hp=144,displayed_hp=144,timeout_hp=0,wins=0)
        self.opening={'timer':99,'p1':actor(4,100),'p2':actor(opponent,200)}
        opts=self.lua.table_from(dict(opponent=opponent,mode='v2-test',choose=self.lua.eval('function() return {{12,""}} end')))
        self.core=core.new(opts,self.table(self.opening))
    def table(self,s):return self.lua.table_from(s,recursive=True)
    def tick(self,s,n=1):
        for _ in range(n): self.core.tick(self.core,self.table(s))
    def prepare(self,win=False):
        self.make();s=deepcopy(self.opening);s['timer']=0
        for p,h in [('p1',20 if win else 2),('p2',2 if win else 20)]:s[p].update(hp=h,displayed_hp=h)
        self.tick(s,30)
        for p in ('p1','p2'):s[p].update(timeout_hp=s[p]['hp'],a=8,anim=12345)
        s['p1' if win else 'p2']['wins']=1
        return s
    def test_unique_native_award_can_classify_either_side_without_pose_whitelist(self):
        for win in (False,True):
            with self.subTest(win=win):
                s=self.prepare(win);self.tick(s)
                original=deepcopy(s)
                s['p2' if win else 'p1']['hp']=-1
                self.tick(s,359);self.assertEqual(len(self.core.rounds),0)
                self.tick(s)
                row=self.core.rounds[1]
                self.assertEqual(row.outcome,'win' if win else 'loss')
                self.assertEqual(row.stable_award_frames,360)
                self.assertEqual(row.time_award.state.p1.hp,original['p1']['hp'])
                self.assertEqual(row.settled.p1.a,8)
                self.assertEqual(row.recognition,'rl-native-time-pip-v2')
    def test_missing_live_chronology_equal_hp_or_invalid_health_never_classifies(self):
        for field,value in [('hp',-1),('timeout_hp',-1),('timeout_hp',145),('timeout_hp',2.5),('displayed_hp',1)]:
            with self.subTest(field=field,value=value):
                s=self.prepare();s['p1'][field]=value;self.tick(s,420)
                self.assertEqual(len(self.core.rounds),0)
        s=self.prepare()
        for field in ('hp','timeout_hp','displayed_hp'):s['p1'][field]=20
        self.tick(s,420);self.assertEqual(len(self.core.rounds),0)
    def test_award_changes_are_invalid_not_restarted_as_another_candidate(self):
        for player,field,value in [('p2','wins',0),('p1','wins',1),('p1','timeout_hp',1),('p2','displayed_hp',19)]:
            with self.subTest(player=player,field=field):
                s=self.prepare();self.tick(s);s[player][field]=value;self.tick(s)
                self.assertEqual(self.core.phase,'invalid');self.assertEqual(len(self.core.rounds),0)
    def test_airborne_or_early_new_round_does_not_bypass_maturity(self):
        s=self.prepare();self.tick(s);s['p1']['hp']=-1;s['p1']['y']=80
        self.tick(s,400);self.assertEqual(len(self.core.rounds),0)
        s['p1']['y']=40;self.tick(s);self.assertEqual(self.core.rounds[1].outcome,'loss')
        s=self.prepare();self.tick(s)
        opening=deepcopy(self.opening);opening['p2']['wins']=1
        self.tick(opening);self.assertEqual(self.core.phase,'invalid');self.assertEqual(len(self.core.rounds),0)
    def test_no_pip_never_becomes_a_guessed_draw(self):
        s=self.prepare();s['p2']['wins']=0;self.tick(s,420)
        self.assertEqual(len(self.core.rounds),0)
    def test_does_not_patch_raw_snapshots(self):
        s=self.prepare();before=deepcopy(s);self.tick(s,400)
        self.assertEqual(s,before)

if __name__=='__main__':unittest.main()
