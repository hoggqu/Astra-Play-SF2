"""Current-picture matcher tests; synthetic tiles need no ROM or emulator."""
import unittest
from pathlib import Path
try:
    from lupa import LuaRuntime
except ImportError:
    LuaRuntime=None
HERE=Path(__file__).resolve().parent

@unittest.skipIf(LuaRuntime is None,'optional lupa is unavailable')
class FighterRenderTests(unittest.TestCase):
    def setUp(self):
        self.lua=LuaRuntime(unpack_returned_tuples=True)
        self.module=self.lua.execute((HERE/'fighter_render.lua').read_text())
        self.mapping=self.lua.execute('''return {[4]={sprites={[1]=100},poses={[1]={status='attack',move_id=33,variant=1,family='normal_punch',strength='light',posture='stand',start=true,facing_bias=0}}}}''')
        self.buffer_api=self.lua.execute('''return {composition=function(mem,pointer) return pointer end,
        pattern=function(comp,flip) local out={} for i=0,3 do out[#out+1]={x=i*16,y=0,code=100+i,flipx=(flip&1)~=0,flipy=(flip&2)~=0} end return out end}''')
        self.renderer=self.module.new(self.mapping,self.buffer_api)

    def picture(self,x=100,missing=()):
        tiles=[dict(x=x+i*16,y=100,code=100+i,flipx=False,flipy=False,palette=3)
               for i in range(4) if i not in missing and x+i*16<384 and x+i*16+16>0]
        return self.lua.table_from(dict(known=True,sprites=self.lua.table_from([self.lua.table_from(t) for t in tiles])))

    def resolve(self,picture):return self.renderer.resolve(self.renderer,None,4,picture)

    def test_complete_current_picture_yields_screen_bbox(self):
        r=self.resolve(self.picture())
        self.assertTrue(r.known);self.assertEqual((r.x,r.y),(132,116))
        self.assertEqual(r.grounded,'unknown');self.assertEqual(r.facing,'right')

    def test_missing_visible_tile_fails_closed(self):
        self.assertFalse(self.resolve(self.picture(missing=(0,))).known)

    def test_clipped_reference_anchor_still_matches_visible_body(self):
        r=self.resolve(self.picture(-16))
        self.assertTrue(r.known);self.assertEqual(r.matched_tiles,3)
        self.assertEqual((r.bbox.left,r.bbox.right),(0,48))

    def test_same_picture_conflicting_labels_collapse(self):
        self.mapping[4].sprites[2]=101
        other=dict(self.mapping[4].poses[1]);other.update(move_id=34,family='normal_kick',strength='heavy')
        self.mapping[4].poses[2]=self.lua.table_from(other)
        r=self.resolve(self.picture())
        self.assertTrue(r.known);self.assertEqual(r.pose_id,1)
        self.assertEqual(r.meta.move_id,0);self.assertEqual(r.meta.variant,0)
        self.assertEqual(r.meta.family,'unknown');self.assertEqual(r.meta.strength,'unknown')

    def test_two_identical_characters_at_different_positions_are_ambiguous(self):
        a=self.picture(0);b=self.picture(200)
        for tile in b.sprites.values():a.sprites[len(a.sprites)+1]=tile
        self.assertFalse(self.resolve(a).known)

    def test_unavailable_current_buffer_fails_closed(self):
        self.assertFalse(self.resolve(self.lua.table_from(dict(known=False))).known)

if __name__=='__main__':unittest.main()
