-- Current drawn projectile sprites only. No combat object pool, HP or timers.
local P={interface='sf2_visible_projectiles_v1',feature_count=121}
local kinds={'unknown','hadouken','yoga_fire','yoga_flame','sonic_boom','tiger_shot'}
local owners={'unknown','p1','p2'}
local source=debug.getinfo(1,'S').source:sub(2):match('^(.*[/\\])') or ''
local loader=loadfile(source..'visible_sprite_buffer.lua') or loadfile('training/runtime/rl_visible_sprite_buffer.lua')
local B=assert(loader,'Missing current sprite buffer reader')()
local atlas
local function build_atlas(mem)
 local out={}
 -- Static ROM graphics table traversal; these are images, not live move state.
 for kind,spec in pairs({hadouken={0x24402,3},sonic_boom={0x245b8,3},tiger_shot={0x24864,3},yoga_fire={0x22dea,3},yoga_flame={0x23346,4}}) do
  for variant=0,spec[2]-1 do
   local at=spec[1]+mem:read_u16(spec[1]+2*variant);local seen={}
   for _=1,128 do
    if seen[at] or at<0 or at>0xfffe0 then break end;seen[at]=true
    local comp=B.composition(mem,mem:read_u32(at+4))
    for _,tile in ipairs(B.pattern(comp,0)) do
     local key=tile.code*32+tile.palette
     if out[key] and out[key]~=kind then out[key]='unknown' else out[key]=kind end
    end
    if (mem:read_u16(at+2)&0x8000)~=0 then at=mem:read_u32(at+24) else at=at+24 end
   end
  end
 end
 return out
end
P.build_atlas=build_atlas
function P.detect(buffer,lookup)
 local tiles={};for _,t in ipairs(buffer.sprites or {}) do
  local kind=lookup[t.code*32+t.palette]
  if kind then tiles[#tiles+1]={x=t.x,y=t.y,kind=kind} end
 end
 local groups={};local used={}
 for i,t in ipairs(tiles) do if not used[i] then
  local group={kind=t.kind,x=0,y=0,count=0};local queue={i};used[i]=true;local head=1
  while head<=#queue do
   local q=tiles[queue[head]];head=head+1;group.x=group.x+q.x+8;group.y=group.y+q.y+8;group.count=group.count+1
   for j,r in ipairs(tiles) do
    if not used[j] and r.kind==t.kind and math.abs(q.x-r.x)<=20 and math.abs(q.y-r.y)<=20 then used[j]=true;queue[#queue+1]=j end
   end
  end
  group.x=group.x/group.count;group.y=group.y/group.count;groups[#groups+1]=group
 end end
 table.sort(groups,function(a,b) if a.x==b.x then return a.y<b.y end;return a.x<b.x end)
 return groups
end
function P.read(mem,s)
 local buffer=B.read(mem,s)
 if not buffer.known then return {known=false,objects={}} end
 if not atlas then local ok,value=pcall(build_atlas,mem);if not ok then return {known=false,objects={}} end;atlas=value end
 return {known=true,objects=P.detect(buffer,atlas)}
end
local function owner(s,o)
 local candidates={}
 for _,key in ipairs({'p1','p2'}) do
  local fighter=s.fighter_perception and s.fighter_perception[key]
  if fighter and (fighter.visual_family or fighter.family)==o.kind and fighter.position_known and type(fighter.visible_x)=='number' and type(fighter.visible_y)=='number' and math.abs(fighter.visible_x-o.x)<64 and math.abs(fighter.visible_y-o.y)<100 then candidates[#candidates+1]=key end
 end
 return #candidates==1 and candidates[1] or 'unknown'
end
function P.new()
 local self={history={},frame=0}
 local function update(s,raw)
  self.frame=self.frame+1
  local out={interface=P.interface,scan_known=raw and raw.known==true,slots={}}
  local assigned={};local claimed={}
  for _,o in ipairs(raw and raw.objects or {}) do
   local best,dist
   for slot,h in pairs(self.history) do
    local d=math.abs(h.x-o.x)+math.abs(h.y-o.y)
    if not claimed[slot] and h.kind==o.kind and self.frame-h.frame<=12 and d<96 and (not dist or d<dist) then best=slot;dist=d end
   end
   if not best then for slot=1,8 do if not claimed[slot] and (not self.history[slot] or self.frame-self.history[slot].frame>12) then best=slot;break end end end
   if best then
    local old=self.history[best];local dt=old and self.frame-old.frame or 0
    local continuity=old and old.kind==o.kind and dt<=12
    local who=continuity and old.owner or owner(s,o)
    local record={present=true,x=o.x,y=o.y,vx=continuity and (o.x-old.x)/dt or 0,vy=continuity and (o.y-old.y)/dt or 0,motion_known=continuity and true or false,owner=who,kind=o.kind}
    assigned[best]=record;claimed[best]=true;self.history[best]={x=o.x,y=o.y,kind=o.kind,owner=who,frame=self.frame}
   end
  end
  for slot=1,8 do out.slots[slot]=assigned[slot] or {present=false,x=0,y=0,vx=0,vy=0,motion_known=false,owner='unknown',kind='unknown'} end
  s.visible_projectiles=out
 end
 function self:reset(s,raw) self.history={};self.frame=0;update(s,raw) end
 function self:tick(s,raw) update(s,raw) end
 return self
end
function P.features(s)
 local v=assert(s.visible_projectiles);assert(v.interface==P.interface);assert(#v.slots==8 and type(v.scan_known)=='boolean');local out={}
 local function clamp(x,n) return math.max(-1,math.min(1,x/n)) end
 for _,p in ipairs(v.slots) do
  assert(type(p.present)=='boolean' and type(p.motion_known)=='boolean')
  local valid_owner=false;for _,o in ipairs(owners) do if o==p.owner then valid_owner=true end end;assert(valid_owner,'Invalid projectile owner')
  local valid_kind=false;for _,k in ipairs(kinds) do if k==p.kind then valid_kind=true end end;assert(valid_kind,'Invalid projectile kind')
  for _,key in ipairs({'x','y','vx','vy'}) do local x=p[key];assert(type(x)=='number' and x==x and math.abs(x)<math.huge,'Invalid projectile coordinate') end
  out[#out+1]=p.present and 1 or 0
  out[#out+1]=clamp(p.x,384);out[#out+1]=clamp(p.y,224)
  out[#out+1]=clamp(p.vx,16);out[#out+1]=clamp(p.vy,16)
  out[#out+1]=p.motion_known and 1 or 0
  for _,o in ipairs(owners) do out[#out+1]=p.owner==o and 1 or 0 end
  for _,k in ipairs(kinds) do out[#out+1]=p.kind==k and 1 or 0 end
 end
 out[#out+1]=v.scan_known and 1 or 0;assert(#out==121);return out
end
return P
