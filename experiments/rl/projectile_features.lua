-- Candidate-only current-state observation. Read game RAM; never write it.
local P={interface='sf2_projectiles6_owner_reciprocal_v1'}
local kinds={[0]=true,[1]=true,[3]=true,[4]=true}
local function clip(x) return math.max(-1,math.min(1,x)) end
local function integer(x) return type(x)=='number' and x==math.floor(x) end
function P.encode(s,objects,links)
 local selected={};local rejected=0
 for _,o in ipairs(objects) do
  local owner=o.owner==0x83c6 and 1 or (o.owner==0x86c6 and 2 or nil)
  if o.status==0x0101 and o.hp==256 and kinds[o.type] and owner
    and integer(o.slot) and o.slot>=0 and o.slot<8 and o.base==0xff938a+0xc0*o.slot
    and links[owner]==o.base%0x10000 and integer(o.x) and integer(o.y) then
   local prev=selected[owner];local dx,dy=math.abs(o.x-s.p1.x),math.abs(o.y-s.p1.y)
   if not prev or dx<prev.dx or (dx==prev.dx and (dy<prev.dy or (dy==prev.dy and o.slot<prev.o.slot))) then
    selected[owner]={o=o,dx=dx,dy=dy}
   end
  else rejected=rejected+1 end
 end
 local out={}
 for owner=1,2 do
  local row=selected[owner]
  out[#out+1]=row and 1 or 0
  out[#out+1]=row and clip((row.o.x-s.p1.x)/512) or 0
  out[#out+1]=row and clip((row.o.y-s.p1.y)/256) or 0
 end
 return out,{rejected=rejected,own_slot=selected[1] and selected[1].o.slot or false,cpu_slot=selected[2] and selected[2].o.slot or false}
end
function P.read(mem,s)
 local objects={}
 for slot=0,7 do
  local b=0xff938a+0xc0*slot
  objects[#objects+1]={slot=slot,base=b,status=mem:read_u16(b),hp=mem:read_i16(b+0x2a),
   type=mem:read_u8(b+0x20),owner=mem:read_u16(b+0x26),x=mem:read_i16(b+6),y=mem:read_i16(b+10)}
 end
 return P.encode(s,objects,{mem:read_u16(0xff83c6+0x1d4),mem:read_u16(0xff86c6+0x1d4)})
end
return P
