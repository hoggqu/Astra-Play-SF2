"""TIME assignment may atomically replace the loser's live HP with KO sentinel."""
from copy import deepcopy
from pathlib import Path
import unittest
from lupa.lua54 import LuaRuntime
HERE=Path(__file__).resolve().parent
ASSETS=HERE.parents[1]/'src/astra_play_sf2/assets'

class AdjacentTimeTests(unittest.TestCase):
    def make(self,win=False):
        self.lua=LuaRuntime(unpack_returned_tuples=True)
        cls=self.lua.execute((ASSETS/'play_core.lua').read_text())
        cls=self.lua.execute((HERE/'settlement_v5.lua').read_text())(cls)
        def actor(c,x):return dict(char=c,x=x,y=40,a=0,anim=1,hp=144,displayed_hp=144,timeout_hp=0,wins=0)
        self.opening=dict(timer=99,p1=actor(4,100),p2=actor(7,200))
        self.core=cls.new(self.lua.table_from(dict(opponent=7,mode='test',choose=self.lua.eval('function() return {{12,""}} end'))),self.table(self.opening))
        s=deepcopy(self.opening);s['timer']=0
        for name,hp in [('p1',42 if win else 1),('p2',1 if win else 42)]:s[name].update(hp=hp,displayed_hp=hp,a=12,anim=12345)
        return s
    def table(self,s):return self.lua.table_from(s,recursive=True)
    def tick(self,s,n=1):
        for _ in range(n):self.core.tick(self.core,self.table(s))
    def award(self,s,win=False):
        t=deepcopy(s)
        for p in ('p1','p2'):t[p]['timeout_hp']=s[p]['hp']
        t['p1' if win else 'p2']['wins']=1;t['p2' if win else 'p1']['hp']=-1
        return t
    def test_both_sides_require_previous_native_state_and_mature_score(self):
        for win in (False,True):
            s=self.make(win);self.tick(s,30);t=self.award(s,win);self.tick(t)
            self.assertEqual(self.core.rl_time_award.previous.state.p1.hp,s['p1']['hp'])
            self.assertEqual(self.core.rl_time_award.state['p2' if win else 'p1'].hp,-1)
            self.tick(t,359);self.assertEqual(len(self.core.rounds),0)
            self.tick(t);r=self.core.rounds[1]
            self.assertEqual(r.outcome,'win' if win else 'loss');self.assertEqual(r.stable_award_frames,360)
            self.assertEqual(r.recognition,'rl-native-time-pip-adjacent-ko-v5')
            self.assertEqual(r.settled['p2' if win else 'p1'].hp,-1)
    def test_missing_previous_chronology_never_classifies(self):
        for field,value in [('hp',0),('displayed_hp',0),('wins',1),('char',3),('hp',1.5)]:
            s=self.make();t=self.award(s);s['p1'][field]=value;self.tick(s,30);self.tick(t,420)
            self.assertEqual(len(self.core.rounds),0,(field,value))
        s=self.make();self.tick(s,30);self.core.rl_time_previous=None;self.tick(self.award(s),420)
        self.assertEqual(len(self.core.rounds),0)
    def test_nonadjacent_saved_frame_is_rejected(self):
        s=self.make();self.tick(s,30);self.core.rl_time_previous.frame-=1
        self.tick(self.award(s),420);self.assertEqual(len(self.core.rounds),0)

    def test_later_hp_reversion_does_not_recreate_old_score_edge(self):
        s=self.make();self.tick(s,30);t=self.award(s);t['p1']['hp']=-2;self.tick(t)
        t['p1']['hp']=-1;self.tick(t,420);self.assertEqual(len(self.core.rounds),0)
    def test_bad_pip_lock_sentinel_equal_hp_never_infers_result(self):
        for p,field,value in [('p1','wins',1),('p2','wins',2),('p1','timeout_hp',2),('p2','displayed_hp',41),('p1','hp',-2),('p2','hp',-1)]:
            s=self.make();self.tick(s,30);t=self.award(s);t[p][field]=value;self.tick(t,420)
            self.assertEqual(len(self.core.rounds),0,(p,field,value))
    def test_post_award_changes_invalid_and_no_raw_mutation(self):
        s=self.make();self.tick(s,30);t=self.award(s);before=deepcopy(t);self.tick(t)
        self.assertEqual(t,before);t['p2']['timeout_hp']=41;self.tick(t)
        self.assertEqual(self.core.phase,'invalid');self.assertEqual(len(self.core.rounds),0)
        self.tick(self.opening);self.assertEqual(self.core.phase,'invalid')
    def test_early_new_round_and_airborne_do_not_bypass_maturity(self):
        s=self.make();self.tick(s,30);t=self.award(s);self.tick(t)
        opening=deepcopy(self.opening);opening['p2']['wins']=1;self.tick(opening)
        self.assertEqual(self.core.phase,'invalid');self.assertEqual(len(self.core.rounds),0)
        s=self.make();self.tick(s,30);t=self.award(s);t['p1']['y']=60;self.tick(t,500)
        self.assertEqual(len(self.core.rounds),0)

if __name__=='__main__':unittest.main()
