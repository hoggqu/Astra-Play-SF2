"""Result-only RL adapter: exact native evidence, no inferred wins or RAM edits."""
from copy import deepcopy
from pathlib import Path
import unittest
from lupa.lua54 import LuaRuntime

HERE=Path(__file__).resolve().parent
ASSETS=HERE.parents[1]/'src/astra_play_sf2/assets'

class SettlementTests(unittest.TestCase):
    def make(self, opponent):
        lua=LuaRuntime(unpack_returned_tuples=True)
        core=lua.execute((ASSETS/'play_core.lua').read_text())
        core=lua.execute((HERE/'settlement.lua').read_text())(core)
        self.lua=lua
        self.opening={'timer':99,'p1':self.actor(4),'p2':self.actor(opponent,200)}
        options=lua.table_from({'opponent':opponent,'mode':'rl-test','choose':lua.eval('function() return {{12,""}} end')})
        self.core=core.new(options,self.table(self.opening))
    def table(self,s): return self.lua.table_from(s,recursive=True)
    @staticmethod
    def actor(char,x=100):
        return dict(char=char,x=x,y=40,a=0,anim=1,hp=144,displayed_hp=144,timeout_hp=0,wins=0)
    def tick(self,s,frame=None):
        if frame is not None: self.core.frame=frame-1
        return self.core.tick(self.core,self.table(s))
    def test_zero_time_draw_requires_live_zero_chronology_and_native_maturity(self):
        self.make(5)
        s=deepcopy(self.opening);s['timer']=0
        for p in ('p1','p2'): s[p].update(hp=0,displayed_hp=0)
        self.tick(s,1);s['p1']['a']=18;self.tick(s,31)
        s['p1'].update(hp=-1,a=0,anim=388594);s['p2']['a']=18
        self.tick(s,360);self.assertEqual(len(self.core.rounds),0)
        self.tick(s,361);self.assertEqual(self.core.rounds[1].outcome,'draw')
        self.assertEqual(s['p1']['hp'],-1)
    def test_zero_draw_without_alive_time_loss_pose_stays_unresolved(self):
        self.make(5);s=deepcopy(self.opening);s['timer']=0
        for p in ('p1','p2'): s[p].update(hp=0,displayed_hp=0)
        self.tick(s,1)
        s['p1'].update(hp=-1,a=0,anim=388594);s['p2']['a']=18
        self.tick(s,361);self.assertEqual(len(self.core.rounds),0)
    def guile(self, latch=True):
        self.make(3);s=deepcopy(self.opening);s['timer']=0
        s['p1'].update(hp=6,displayed_hp=6);s['p2'].update(hp=17,displayed_hp=17)
        self.tick(s,1)
        s['p1'].update(a=18,timeout_hp=6);s['p2'].update(a=10,timeout_hp=17,wins=1)
        if latch: self.tick(s,31)
        s['p1'].update(hp=-1,a=8,anim=388594);s['p2']['a']=16
        return s
    def test_guile_late_ko_is_a_loss_after_original_award_and_maturity(self):
        s=self.guile();self.tick(s,390);self.assertEqual(len(self.core.rounds),0)
        self.tick(s,391)
        row=self.core.rounds[1]
        self.assertEqual(row.outcome,'loss');self.assertEqual(row.settled.p1.a,8)
        self.assertEqual(row.settled.p1.hp,-1);self.assertEqual(row.settled.p1.displayed_hp,6)
        self.assertEqual(row.score[2],1)
    def test_guile_missing_latch_or_mismatched_fields_never_awards_result(self):
        s=self.guile(latch=False);self.tick(s,361);self.assertEqual(len(self.core.rounds),0)
        for player,key,value in [('p1','anim',1),('p1','y',41),('p1','timeout_hp',5),('p2','displayed_hp',16),('p2','a',10),('p2','wins',0)]:
            with self.subTest(player=player,key=key):
                s=self.guile();s[player][key]=value;self.tick(s,361)
                self.assertEqual(len(self.core.rounds),0)

if __name__=='__main__': unittest.main()
