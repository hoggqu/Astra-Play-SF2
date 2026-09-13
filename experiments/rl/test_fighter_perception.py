"""Visible sprite alias invariance, observation history and encoder parity."""
import unittest
from pathlib import Path
from . import fighter_perception as py
try:
    from lupa import LuaRuntime
except ImportError:
    LuaRuntime=None
HERE=Path(__file__).resolve().parent


@unittest.skipIf(LuaRuntime is None,'optional lupa is unavailable')
class FighterPerceptionTests(unittest.TestCase):
    def setUp(self):
        self.lua=LuaRuntime(unpack_returned_tuples=True)
        self.f=self.lua.execute((HERE/'fighter_perception.lua').read_text())
        self.mapping=self.lua.execute((HERE/'fighter_animation_map.lua').read_text())
        self.o=self.f.new(self.mapping)

    def convert(self,v):
        if isinstance(v,dict):return self.lua.table_from({k:self.convert(x) for k,x in v.items()})
        return v

    def state(self,anim=382306,op_anim=407552,x=100,y=40,char=4,op_char=3):
        # Fixtures explicitly provide decoded pictures, never a runtime RAM fallback.
        def actor(c,a,px,py,face,wins):
            pose=self.mapping[c].anims[a] or 0
            picture=dict(known=pose!=0,position_known=pose!=0,pose_id=pose,
                         grounded='unknown',facing=face,x=px,y=py)
            if pose:picture['meta']=dict(self.mapping[c].poses[pose])
            return dict(char=c,anim=a,x=px,y=py,wins=99,facing_render=99,
                        visible_wins=wins,visible_fighter=picture)
        return self.convert({'p1':actor(char,anim,x,y,'right',0),
                             'p2':actor(op_char,op_anim,250,40,'left',1)})

    def flatten(self,s):
        return {'fighter_perception':{'interface':s.fighter_perception.interface,
                'p1':dict(s.fighter_perception.p1),'p2':dict(s.fighter_perception.p2)}}

    def test_actual_sprite_aliases_all_metadata_and_entire_history_are_identical(self):
        aliases=0
        for char,character in self.mapping.items():
            groups={}
            for address,pose in character.anims.items():groups.setdefault(pose,[]).append(address)
            for addresses in groups.values():
                if len(addresses)<2:continue
                aliases+=1
                a,b=addresses[:2]
                left=self.f.new(self.mapping);right=self.f.new(self.mapping)
                s1=self.state(anim=a,char=char);s2=self.state(anim=b,char=char)
                left.reset(left,s1);right.reset(right,s2)
                self.assertEqual(self.flatten(s1),self.flatten(s2))
                for x in (101,101,103):
                    s1=self.state(anim=a,char=char,x=x);s2=self.state(anim=b,char=char,x=x)
                    left.tick(left,s1);right.tick(right,s2)
                    self.assertEqual(self.flatten(s1),self.flatten(s2))
        self.assertGreater(aliases,500)

    def test_special_strength_and_variant_collapse(self):
        states=[]
        for a in (390174,390346,390518):
            s=self.state(anim=a);self.o.reset(self.o,s);states.append(dict(s.fighter_perception.p1))
        self.assertEqual(states[0],states[1]);self.assertEqual(states[0],states[2])
        self.assertEqual(states[0]['strength'],'unknown');self.assertEqual(states[0]['variant'],0)

    def test_observed_pose_age_does_not_change_for_invisible_animation_pointer(self):
        s=self.state(anim=390174);self.o.reset(self.o,s)
        s=self.state(anim=390346);self.o.tick(self.o,s)
        self.assertEqual(s.fighter_perception.p1.pose_age,1)
        self.assertEqual(s.fighter_perception.p1.pose_changes,0)

    def test_positions_elapsed_facing_and_visible_win_icons(self):
        s=self.state();self.o.reset(self.o,s);before=self.flatten(s)
        t=self.state(x=108);self.o.tick(self.o,t)
        self.assertEqual(t.fighter_perception.p1.dx,8)
        self.assertEqual(t.fighter_perception.p1.facing,'right')
        self.assertEqual(t.fighter_perception.p2.facing,'left')
        self.assertEqual(t.fighter_perception.p2.wins,1)
        self.assertEqual(self.flatten(s),before)

    def test_unknown_animation_remains_explicit(self):
        s=self.state(anim=1234567);self.o.reset(self.o,s)
        p=s.fighter_perception.p1
        self.assertFalse(p.recognized);self.assertEqual(p.pose_id,0);self.assertEqual(p.status,'unknown')
        self.assertEqual(p.pose_age,-1)

    def test_pure_python_lua_feature_parity(self):
        s=self.state();self.o.reset(self.o,s)
        for a in (383790,383814,390174,390222,388926,389026,388594):
            s=self.state(anim=a);self.o.tick(self.o,s)
            actual=list(self.f.features(s).values());expected=py.features(self.flatten(s))
            self.assertEqual(len(actual),872);self.assertEqual(actual,expected)

    def test_visible_short_attack_retained_after_return_to_nonattack(self):
        s=self.state();self.o.reset(self.o,s)
        s=self.state(anim=390174);self.o.tick(self.o,s)
        self.assertTrue(s.fighter_perception.p1.started)
        attack_id=s.fighter_perception.p1.move_id
        s=self.state(anim=388926);self.o.tick(self.o,s)
        self.assertEqual(s.fighter_perception.p1.last_attack_move,attack_id)
        self.assertEqual(s.fighter_perception.p1.last_attack_age,1)

    def test_no_raw_state_timer_intent_or_future_animation_access(self):
        source=(HERE/'fighter_perception.lua').read_text().split('\n',1)[1]
        self.assertNotRegex(source,r'\bp\.a\b')
        for value in ('read_u16','read_u32','read_i16','0x123','0x124','0x60','0x5e','remaining','emu.pause','set_value'):
            self.assertNotIn(value,source)

if __name__=='__main__':unittest.main()
