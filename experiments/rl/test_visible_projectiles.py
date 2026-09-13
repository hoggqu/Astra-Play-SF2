"""Execute actual Lua using synthetic current display buffers; no emulator needed."""
from pathlib import Path
import unittest
from lupa.lua54 import LuaRuntime
from .visible_projectiles import features, FEATURE_COUNT

HERE = Path(__file__).resolve().parent

def plain(value):
    if not hasattr(value, 'items'):
        return value
    pairs = dict(value.items())
    if pairs and all(type(k) is int for k in pairs) and set(pairs) == set(range(1, len(pairs)+1)):
        return [plain(pairs[i]) for i in range(1, len(pairs)+1)]
    return {k: plain(v) for k, v in pairs.items()}

class VisibleProjectilesTests(unittest.TestCase):
    def setUp(self):
        self.lua = LuaRuntime()
        self.b = self.lua.eval('dofile')(str(HERE/'visible_sprite_buffer.lua'))
        self.p = self.lua.eval('dofile')(str(HERE/'visible_projectiles.lua'))
    def table(self, value):
        return self.lua.table_from(value, recursive=True)
    def observed(self, *objects, known=True):
        return self.table({'known': known, 'objects': list(objects)})
    def obj(self, x, y=140, kind='hadouken'):
        return {'x': x, 'y': y, 'kind': kind}
    def test_exact_oam_clipping_terminator_and_block_flip(self):
        words = [64,16,0x120,0x1123, 0,16,0x999,0, 100,100,0,0xff00,100,100,0x888,0]
        output = plain(self.b.expand(self.table(words)))
        self.assertEqual([(t['x'],t['y'],t['code']) for t in output['sprites']], [(0,0,0x121),(16,0,0x120),(0,16,0x131),(16,16,0x130)])
        self.assertTrue(all(t['flipx'] and t['palette']==3 for t in output['sprites']))
    def test_composition_nonzero_count_consumes_blank_grid_entries(self):
        words = {0x100: 2,0x102:3,0x104:0,0x106:8,0x108:16,0x10a:0x111,0x10c:0,0x10e:0x112,
                 0x200:4,0x204:0,0x206:0,0x208:16,0x20a:0,0x20c:32,0x20e:0}
        mem = self.lua.table();mem['read_u16']=lambda _,at: words.get(at,0);mem['read_i16']=mem['read_u16'];mem['read_u32']=lambda _,at:0x200
        comp=self.b.composition(mem,0x100)
        self.assertEqual([x['dx'] for x in comp.pieces.values()],[0,32])
        self.assertEqual([x['x'] for x in self.b.pattern(comp,0).values()],[-8,24])
        self.assertEqual([x['x'] for x in self.b.pattern(comp,1).values()],[-8,-40])
    def test_current_draw_only_blink_and_hidden_fields_independent(self):
        observer = self.p.new();s=self.table({'p1':{'hp':144,'x':999},'projectiles':[{'hp':999,'status':257}]})
        observer.reset(observer,s,self.observed(self.obj(100)))
        self.assertTrue(s.visible_projectiles.slots[1].present)
        old=plain(s.visible_projectiles)
        observer.tick(observer,s,self.observed())
        self.assertFalse(s.visible_projectiles.slots[1].present)
        self.assertTrue(old['slots'][0]['present'])
        observer.tick(observer,s,self.observed(self.obj(110)))
        self.assertEqual(s.visible_projectiles.slots[1].vx,5)
    def test_owner_requires_exclusive_visible_compatible_cast(self):
        observer=self.p.new();s=self.table({'fighter_perception':{'p1':{'visible_x':100,'visible_y':180,'position_known':True}}})
        observer.reset(observer,s,self.observed(self.obj(120)))
        self.assertEqual(s.visible_projectiles.slots[1].owner,'unknown')
        s.fighter_perception.p1.visual_family='hadouken'
        observer.reset(observer,s,self.observed(self.obj(120)))
        self.assertEqual(s.visible_projectiles.slots[1].owner,'p1')
        s.fighter_perception.p2=self.table({'visible_x':130,'visible_y':180,'position_known':True,'visual_family':'hadouken'})
        observer.reset(observer,s,self.observed(self.obj(120)))
        self.assertEqual(s.visible_projectiles.slots[1].owner,'unknown')
    def test_both_directions_all_five_families_and_python_parity(self):
        observer=self.p.new();s=self.table({});kinds=['hadouken','sonic_boom','tiger_shot','yoga_fire','yoga_flame']
        observer.reset(observer,s,self.observed(*[self.obj(40+i*60,100,k) for i,k in enumerate(kinds)]))
        observer.tick(observer,s,self.observed(*[self.obj(40+i*60+(5 if i%2 else -5),100,k) for i,k in enumerate(kinds)]))
        actual=list(self.p.features(s).values());self.assertEqual(len(actual),FEATURE_COUNT);self.assertEqual(actual,features(plain(s)))
        self.assertEqual([s.visible_projectiles.slots[i].vx for i in range(1,6)],[-5,5,-5,5,-5])
    def test_unknown_reader_does_not_fallback_to_ram(self):
        observer=self.p.new();s=self.table({'projectiles':[{'hp':144,'status':257,'x':100}]})
        observer.reset(observer,s,self.observed(known=False))
        self.assertFalse(s.visible_projectiles.scan_known)
        self.assertFalse(any(x.present for x in s.visible_projectiles.slots.values()))
    def test_palette_distinguishes_nonprojectile_tile_and_clusters(self):
        lookup=self.table({0x111*32+3:'hadouken'})
        buffer=self.table({'sprites':[{'x':10,'y':100,'code':0x111,'palette':3},{'x':26,'y':100,'code':0x111,'palette':3},{'x':200,'y':100,'code':0x111,'palette':2}]})
        groups=plain(self.p.detect(buffer,lookup));self.assertEqual(len(groups),1);self.assertEqual(groups[0]['x'],26)
    def test_shared_reader_is_memoized_per_snapshot_only(self):
        self.lua.execute("reads=0; manager={machine={devices={[':']={items={['0/m_buffered_obj']=1}}}}};emu={item=function()return {size=2,count=1024,read=function(self,i)reads=reads+1;return i==3 and 0xff00 or 0 end}end}")
        s=self.table({});self.b.read(None,s);self.b.read(None,s)
        self.assertEqual(self.lua.globals().reads,1024)
        self.b.read(None,self.table({}));self.assertEqual(self.lua.globals().reads,2048)

if __name__ == '__main__':
    unittest.main()
