"""Non-time double-KO confirmation with strict old-state/next-round separation."""
from copy import deepcopy
from lupa.lua54 import LuaRuntime
from .test_settlement_v2 import SettlementV2Tests,HERE,ASSETS

class DoubleKOTests(SettlementV2Tests):
    def make(self,opponent=2):
        self.lua=LuaRuntime(unpack_returned_tuples=True)
        core=self.lua.execute((ASSETS/'play_core.lua').read_text())
        core=self.lua.execute((HERE/'settlement.lua').read_text())(core)
        def p(c,x):return dict(char=c,x=x,y=40,a=0,anim=1,hp=144,displayed_hp=144,timeout_hp=0,wins=0)
        self.opening=dict(timer=99,p1=p(4,100),p2=p(opponent,200))
        opts=self.lua.table_from(dict(opponent=opponent,mode='v4-test',choose=self.lua.eval('function() return {{12,""}} end')))
        self.core=core.new(opts,self.table(self.opening))
    def double(self,mature=True):
        self.make();s=deepcopy(self.opening);s['timer']=65
        s['p1'].update(hp=-1,displayed_hp=24,a=10);s['p2'].update(hp=-1,displayed_hp=10,a=10)
        self.tick(s)
        s['p1'].update(displayed_hp=-1,a=6,anim=388594);s['p2'].update(displayed_hp=-1,a=0,anim=440986)
        self.tick(s,360 if mature else 359)
        return s
    def test_confirmed_double_ko_preserves_mature_raw_state(self):
        self.double();self.assertEqual(len(self.core.rounds),0)
        init=deepcopy(self.opening)
        for p in ('p1','p2'):init[p].update(hp=0,displayed_hp=0,anim=0,y=0)
        self.tick(init,5);self.tick(self.opening)
        row=self.core.rounds[1]
        self.assertEqual(row.outcome,'draw');self.assertEqual(row.recognition,'rl-native-double-ko-next-round-v4')
        self.assertEqual(row.settled.timer,65);self.assertEqual(row.settled.p1.hp,-1)
        self.assertEqual(row.settled.p1.a,6);self.assertEqual(row.confirmation.p1.hp,144)
        self.assertEqual(self.core.phase,'between');self.assertEqual(self.core.ready_frames,1)
    def test_premature_double_ko_is_not_confirmed(self):
        self.double(False);self.tick(self.opening)
        self.assertEqual(self.core.phase,'invalid');self.assertEqual(len(self.core.rounds),0)
    def test_missing_double_stop_does_not_infer_draw(self):
        self.make();s=deepcopy(self.opening);s['timer']=65
        s['p1'].update(hp=-1,displayed_hp=-1,a=6);s['p2'].update(hp=1,displayed_hp=1,a=6)
        self.tick(s);s['p2'].update(hp=-1,displayed_hp=-1);self.tick(s,400);self.tick(self.opening)
        self.assertEqual(self.core.phase,'invalid')
    def test_change_and_reversion_disqualify_evidence(self):
        for p,field,value in [('p1','wins',1),('p1','hp',0),('p2','displayed_hp',0),('p2','displayed_hp',-.5)]:
            with self.subTest(field=field):
                s=self.double();old=s[p][field];s[p][field]=value;self.tick(s)
                s[p][field]=old;self.tick(s);self.tick(self.opening)
                self.assertEqual(self.core.phase,'invalid');self.assertEqual(len(self.core.rounds),0)
    def test_timer_change_or_airborne_without_maturity_does_not_confirm(self):
        s=self.double();s['timer']=64;self.tick(s);s['timer']=65;self.tick(s);self.tick(self.opening)
        self.assertEqual(self.core.phase,'invalid')
        s=self.double(False);s['p1']['y']=41;self.tick(s,20);self.tick(self.opening)
        self.assertEqual(self.core.phase,'invalid')
    def test_missing_native_ko_sentinels_and_changed_actor_are_rejected(self):
        for value in (-2,-.5,0):
            self.make();s=deepcopy(self.opening);s['timer']=65
            s['p1'].update(hp=value,displayed_hp=-1,a=6)
            s['p2'].update(hp=-1,displayed_hp=-1,a=6)
            self.tick(s,401);self.tick(self.opening)
            self.assertEqual(self.core.phase,'invalid');self.assertEqual(len(self.core.rounds),0)
        self.double();changed=deepcopy(self.opening);changed['p2']['char']=3
        self.tick(changed);self.assertEqual(self.core.phase,'invalid')

    def test_invalid_core_is_never_revived(self):
        self.double();self.core.phase='invalid';self.tick(self.opening)
        self.assertEqual(self.core.phase,'invalid');self.assertEqual(len(self.core.rounds),0)
