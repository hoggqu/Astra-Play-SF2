"""Confirmed native next-round draw evidence and separate previous-round reward."""
from copy import deepcopy
from lupa.lua54 import LuaRuntime
from .test_settlement_v2 import SettlementV2Tests,HERE,ASSETS
from .test_batch import BatchRuntimeTests

class SettlementV3Tests(SettlementV2Tests):
    def make(self,opponent=7):
        self.lua=LuaRuntime(unpack_returned_tuples=True)
        core=self.lua.execute((ASSETS/'play_core.lua').read_text())
        core=self.lua.execute((HERE/'settlement.lua').read_text())(core)
        def actor(char,x):return dict(char=char,x=x,y=40,a=0,anim=1,hp=144,displayed_hp=144,timeout_hp=0,wins=0)
        self.opening={'timer':99,'p1':actor(4,100),'p2':actor(opponent,200)}
        opts=self.lua.table_from(dict(opponent=opponent,mode='v3-test',choose=self.lua.eval('function() return {{12,""}} end')))
        self.core=core.new(opts,self.table(self.opening))
    def draw(self,hp=0):
        self.make();s=deepcopy(self.opening);s['timer']=0
        for p in ('p1','p2'):s[p].update(hp=hp,displayed_hp=hp,timeout_hp=hp,a=12)
        self.tick(s)
        s['p2'].update(hp=-1,a=0,anim=307368);s['p1']['a']=18
        self.tick(s,360)
        return s
    def test_new_round_confirms_draw_but_previous_settled_hp_is_preserved(self):
        for hp in (0,5):
            with self.subTest(hp=hp):
                previous=self.draw(hp);self.assertEqual(len(self.core.rounds),0)
                # Actual game initializes actors through zero pointers before opening.
                init=deepcopy(previous)
                for p in ('p1','p2'):init[p].update(hp=144,displayed_hp=144,timeout_hp=0,anim=0,y=0)
                self.tick(init,3);self.tick(self.opening)
                row=self.core.rounds[1]
                self.assertEqual(row.outcome,'draw');self.assertEqual(row.settled.p2.hp,-1)
                self.assertEqual(row.settled.p1.hp,hp);self.assertEqual(row.confirmation.p1.hp,144)
                self.assertLess(row.settled_frame,row.confirmation_frame)
                self.assertEqual(self.core.phase,'between');self.assertEqual(self.core.ready_frames,1)
    def test_draw_rejects_missing_chronology_maturity_or_complete_native_opening(self):
        self.make();s=deepcopy(self.opening);s['timer']=0
        s['p1'].update(hp=0,displayed_hp=0);s['p2'].update(hp=-1,displayed_hp=0,a=8)
        self.tick(s,400);self.tick(self.opening)
        self.assertEqual(self.core.phase,'invalid');self.assertEqual(len(self.core.rounds),0)
        self.draw();self.core.frame=10;self.core.rl_draw_evidence.mature=None
        self.tick(self.opening);self.assertEqual(self.core.phase,'invalid')
        self.draw();incomplete=deepcopy(self.opening);incomplete['p2']['hp']=143
        self.tick(incomplete);self.assertEqual(len(self.core.rounds),0)
    def test_observed_pip_or_locked_hp_changes_disqualify_draw_even_if_reverted(self):
        for field in ('wins','timeout_hp','displayed_hp'):
            with self.subTest(field=field):
                s=self.draw();s['p2'][field]=1;self.tick(s)
                s['p2'][field]=0;self.tick(s);self.tick(self.opening)
                self.assertEqual(self.core.phase,'invalid');self.assertEqual(len(self.core.rounds),0)
    def test_prior_invalid_is_never_revived_by_a_matching_opening(self):
        self.draw();self.core.phase='invalid';self.tick(self.opening)
        self.assertEqual(self.core.phase,'invalid');self.assertEqual(len(self.core.rounds),0)

class DrawRuntimeTests(BatchRuntimeTests):
    def setUp(self):
        super().setUp()
        self.lua.execute((HERE/'batch_runtime.lua').read_text())
        # Stub terminal rows elsewhere in the inherited cadence tests also need
        # real settled fields, as production Core always supplies them.
        self.lua.execute('''
            modules['training/runtime/play_core.lua'].new=function()
                local c={frame=0,phase='fighting',rounds={}}
                function c:tick(s)
                    self.frame=self.frame+1
                    if self.frame==terminal_at then
                        self.phase='between';self.rounds[1]={outcome='win',settled=s}
                    end
                    return {input='HP'}
                end
                return c
            end
        ''')
    def test_next_round_refill_never_enters_the_previous_draw_reward(self):
        self.lua.execute('''
            modules['training/runtime/play_core.lua'].new=function()
                local c={frame=0,phase='fighting',rounds={}}
                function c:tick(s)
                    self.frame=self.frame+1
                    if self.frame==13 then
                        self.phase='between'
                        local old={timer=0,p1={hp=0},p2={hp=5}}
                        self.rounds[1]={outcome='draw',settled=old,confirmation=s}
                    end
                    return {}
                end
                return c
            end
        ''')
        self.reset()
        self.request({'id':2,'op':'rollout','count':2,'actions':[0,0],
                      'resets':[{'checkpoint':0,'lead':0}]*2})
        for _ in range(15):self.tick()
        result=self.reply(2);terminal=result['transitions'][1]
        self.assertTrue(terminal['done'])
        self.assertAlmostEqual(terminal['reward'],-.25*5/144)
        self.assertEqual(terminal['state']['p1']['hp'],144)
        row=result['episodes'][0]
        self.assertEqual(row['final_state']['p2']['hp'],5)
        self.assertEqual(row['confirmation_state']['p2']['hp'],144)
        self.assertIn('training/rl-draw-action-plan-00001.json',self.published)
