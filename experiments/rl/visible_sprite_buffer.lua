-- Read-only access to the exact CPS1 sprite buffer currently rendered by MAME.
-- Coordinates use the sf2 World 910522 normal 384x224 viewport.
if package.loaded['astra_sf2_visible_sprite_buffer_v1'] then return package.loaded['astra_sf2_visible_sprite_buffer_v1'] end
local cache=setmetatable({}, {__mode='k'})
local B={INTERFACE='sf2_visible_sprite_buffer_v1',WIDTH=384,HEIGHT=224}
local item
local function expand(words)
 local out={known=true,sprites={},width=384,height=224}
 for i=1,#words-3,4 do
  local x,y,code,attr=words[i],words[i+1],words[i+2],words[i+3]
  if (attr & 0xff00)==0xff00 then break end
  local nx=((attr>>8)&15)+1;local ny=((attr>>12)&15)+1
  local fx=(attr&32)~=0;local fy=(attr&64)~=0
  for row=0,ny-1 do for col=0,nx-1 do
   local sx=((x+col*16)&511)-64;local sy=((y+row*16)&511)-16
   local tile=(code&0xfff0)+((code+(fx and nx-1-col or col))&15)+16*(fy and ny-1-row or row)
   if sx<384 and sx+16>0 and sy<224 and sy+16>0 then
    out.sprites[#out.sprites+1]={x=sx,y=sy,code=tile,palette=attr&31,flipx=fx,flipy=fy,order=(i-1)//4}
   end
  end end
 end
 return out
end
B.expand=expand
function B.read(mem,s)
 if s and cache[s] then return cache[s] end
 local ok,result=pcall(function()
  if not item then
   local index=manager.machine.devices[':'].items['0/m_buffered_obj']
   assert(index,'current CPS1 sprite buffer unavailable');item=emu.item(index)
  end
  assert(item.size==2 and item.count==1024,'unexpected CPS1 buffer layout')
  local words={};for i=0,1023 do words[#words+1]=item:read(i) end
  return expand(words)
 end)
 if not ok then result={known=false,sprites={},width=384,height=224} end
 if s then cache[s]=result end
 return result
end
function B.composition(mem,pointer)
 if type(pointer)~='number' or pointer<0 or pointer>0xffff0 then return nil end
 local flag=mem:read_u16(pointer);local count=flag&0x7fff
 if count>256 then return nil end
 local attr=mem:read_u16(pointer+2);if (flag&0x8000)==0 and (attr&0xff00)~=0 then count=1 end
 local result={pieces={},count=count,layout=mem:read_u16(pointer+4),x=mem:read_i16(pointer+6),y=mem:read_i16(pointer+8)}
 local table_address=mem:read_u32(0x7f622+((result.layout&0x7f80)>>5))
 if table_address<0 or table_address>0xfff00 then return nil end
 local offsets=table_address+mem:read_i16(table_address+2*(result.layout&127))
 local cursor=pointer+10;local i=0
 while #result.pieces<count and i<512 do
  i=i+1
  local code=mem:read_u16(cursor);cursor=cursor+2
  local a=attr;if (flag&0x8000)~=0 then a=mem:read_u16(cursor) ~ (attr&0xe0);cursor=cursor+2 end
  if code~=0 then result.pieces[#result.pieces+1]={code=code,palette=a&31,attr=a,dx=mem:read_i16(offsets+4*(i-1)),dy=mem:read_i16(offsets+4*(i-1)+2)} end
 end
 return result
end
-- Static pose pattern relative to rendered actor origin. No actor state needed.
function B.pattern(comp,flipbits)
 local out={};if not comp then return out end
 for _,p in ipairs(comp.pieces) do if p.code~=0 then
  local attr=p.attr ~ ((flipbits or 0)<<5)
  local fx=(attr&32)~=0;local fy=(attr&64)~=0
  local nx=((attr>>8)&15)+1;local ny=((attr>>12)&15)+1
  local x=fx and comp.x-p.dx-16*nx or p.dx-comp.x
  local y=fy and -comp.y-p.dy-16*ny or comp.y+p.dy
  for row=0,ny-1 do for col=0,nx-1 do
   out[#out+1]={x=x+16*col,y=y+16*row,code=(p.code&0xfff0)+((p.code+(fx and nx-1-col or col))&15)+16*(fy and ny-1-row or row),palette=p.palette,flipx=fx,flipy=fy}
  end end
 end end
 return out
end
package.loaded['astra_sf2_visible_sprite_buffer_v1']=B
return B
