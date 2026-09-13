"""New execution feedback must depend only on already observed visible evidence."""
from pathlib import Path
import unittest
from lupa import LuaRuntime
from .export import lua_literal
from .perception_execution import features

HERE=Path(__file__).resolve().parent


class ScreenExecutionTests(unittest.TestCase):
    def setUp(self):
        self.lua=LuaRuntime(unpack_returned_tuples=True)
        self.lua.globals().A=self.lua.execute((HERE/'full_actions.lua').read_text())
        self.lua.globals().E=self.lua.execute((HERE/'perception_execution.lua').read_text())
        self.lua.execute('''o=E.new(A)
s={p1={a=0,anim=1},p2={a=0,anim=1},fighter_perception={
 p1={status='idle',recognized=true,started=false,family='unknown',strength='unknown',posture='unknown'},
 p2={status='idle',recognized=true,started=false,family='unknown',strength='unknown',posture='unknown'}}}
o:reset(s)
''')

    def test_raw_action_and_animation_changes_do_not_signal_start(self):
        self.lua.execute('''o:request(81,s)
s.p1.a=12;s.p1.anim=123456;o:tick(s)
assert(s.visible_feedback.result=='pending' and not s.visible_feedback.started_since_request)
s.p1.a=10;s.p1.anim=345678;o:tick(s)
assert(s.visible_feedback.result=='pending' and s.visible_feedback.actual_family=='unknown')
''')

    def test_current_request_not_in_saved_observation(self):
        self.lua.execute('''before=s.visible_feedback;o:request(81,s)
assert(s.visible_feedback.request_action==-1)
o:tick(s);assert(before.request_action==-1)
assert(s.visible_feedback.request_action==81)
''')
        self.assertEqual(features({'visible_feedback':dict(self.lua.globals().s.visible_feedback.items())}),
                         list(self.lua.globals().E.features(self.lua.globals().s).values()))

    def test_visible_start_unknown_strength_and_round_reset(self):
        self.lua.execute('''o:request(81,s)
s.fighter_perception.p1={status='special',recognized=true,started=true,family='shoryuken',strength='unknown',posture='air'}
o:tick(s)
assert(s.visible_feedback.result=='started' and s.visible_feedback.actual_strength=='unknown')
s.fighter_perception.p2={status='dizzy',recognized=true,started=false}
s.fighter_perception.p1.started=false;o:tick(s)
assert(s.visible_feedback.p2_dizzy=='present')
o:reset(s);assert(s.visible_feedback.request_action==-1 and s.visible_feedback.result=='none')
assert(not s.visible_feedback.started_since_request)
''')

    def test_unrecognized_visual_state_stays_unknown(self):
        self.lua.execute('''s.fighter_perception.p2.recognized=false;s.fighter_perception.p2.status='unknown'
o:request(6,s);for i=1,12 do o:tick(s) end
assert(s.visible_feedback.p2_dizzy=='unknown' and s.visible_feedback.result=='unconfirmed')
s.fighter_perception.p1={status='attack',recognized=true,started=true,family='normal_punch',strength='unknown',posture='standing'}
o:tick(s);assert(s.visible_feedback.result=='unknown')
''')


if __name__=='__main__':unittest.main()
