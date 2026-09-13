-- Current visible fighter poses and observed history only. Never reads p.a.
local F={interface='sf2_visible_fighters_v1',feature_count=872,actor_feature_count=436}
local statuses={'unknown','idle','walk','crouch','jump','attack','special','throw','guard','hit','falling','down','getup','dizzy','victory','defeat'}
local faces={'unknown','left','right'}
local default_map
local function clamp(x,n) return math.max(-1,math.min(1,x/n)) end
local renderer,buffer_module
function F.read(memory,s)
 default_map=default_map or assert(loadfile('training/runtime/rl_fighter_animation_map.lua'))()
 buffer_module=buffer_module or assert(loadfile('training/runtime/rl_visible_sprite_buffer.lua'))()
 if not renderer then renderer=assert(loadfile('training/runtime/rl_fighter_render.lua'))().new(default_map,buffer_module) end
 local buffer=s.visible_sprite_buffer or buffer_module.read(memory,s)
 s.visible_sprite_buffer=buffer
 for _,key in ipairs({'p1','p2'}) do
  s[key].visible_fighter=renderer:resolve(memory,s[key].char,buffer)
 end
 return s
end

local function hot(out,value,values)
 local found=false
 for _,v in ipairs(values) do local equal=v==value;out[#out+1]=equal and 1 or 0;found=found or equal end
 assert(found,'Unknown fighter enum')
end
local function int(v,lo,hi)
 assert(type(v)=='number' and v%1==0 and v>=lo and v<=hi,'Invalid fighter integer')
 return v
end
function F.features(s)
 local data=assert(s.fighter_perception,'Missing fighter perception')
 assert(data.interface==F.interface,'Wrong fighter perception interface')
 local out={}
 for _,key in ipairs({'p1','p2'}) do
  local p=assert(data[key]);hot(out,p.status,statuses)
  for _,spec in ipairs({{'move_id',64},{'variant',16},{'pose_id',256}}) do
   local value=int(p[spec[1]],0,spec[2]);for i=0,spec[2] do out[#out+1]=value==i and 1 or 0 end
  end
  for _,field in ipairs({'status_age','action_age','pose_age','pose_changes'}) do
   local age=int(p[field],-1,2147483647);out[#out+1]=age<0 and -1 or math.min(age,120)/120
  end
  for _,field in ipairs({'dx','dy'}) do
   local value=p[field];assert(type(value)=='number' and value==value and math.abs(value)<math.huge,'Invalid observed movement')
   out[#out+1]=clamp(value,16)
  end
  assert(p.grounded=='unknown' or type(p.grounded)=='boolean');out[#out+1]=p.grounded=='unknown' and -1 or p.grounded and 1 or 0
  assert(type(p.recognized)=='boolean');out[#out+1]=p.recognized and 1 or 0
  hot(out,p.facing,faces);hot(out,p.wins,{-1,0,1,2})
  local last=int(p.last_attack_move,0,64);for i=0,64 do out[#out+1]=last==i and 1 or 0 end
  local age=int(p.last_attack_age,-1,2147483647);out[#out+1]=age<0 and -1 or math.min(age,120)/120
 end
 assert(#out==F.feature_count);return out
end
local attack={attack=true,special=true,throw=true}
local clear={idle=true,walk=true,crouch=true,guard=true,hit=true,falling=true,down=true,getup=true,dizzy=true,victory=true,defeat=true}
function F.new(mapping)
 if not mapping then
  default_map=default_map or assert(loadfile('training/runtime/rl_fighter_animation_map.lua'))()
  mapping=default_map
 end
 local o={frame=0,history={}}
 function o:observe(p,prior,reset)
  local picture=p.visible_fighter or {known=false,position_known=false,pose_id=0,grounded='unknown',facing='unknown'}
  local character=mapping[p.char];local pose=picture.known and picture.pose_id or 0
  local meta=picture.known and (picture.meta or character and character.poses[pose]) or nil
  local status=meta and meta.status or 'unknown'
  local position_known=picture.position_known==true
  local x,y=position_known and picture.x or 0,position_known and picture.y or 0
  local move=meta and meta.move_id or 0
  local previous=prior and prior.output
  local dx=not reset and position_known and previous.position_known and x-prior.x or 0
  local dy=not reset and position_known and previous.position_known and y-prior.y or 0
  local family=meta and meta.family or 'unknown'
  local started=false
  if not reset and attack[status] then
   if status=='attack' then started=meta.start and pose~=previous.pose_id
   elseif not prior.active then started=true
   elseif move~=0 and prior.active_move~=0 and move~=prior.active_move then started=true
   end
  end
  if not reset and status=='jump' and previous.status~='jump' and dy<0 then started=true end
  if reset then prior={active=false,last_attack_move=0,last_attack_frame=nil} end
  if attack[status] then
   prior.active=true;prior.active_move=move
   if started then prior.last_attack_move=move;prior.last_attack_frame=self.frame end
  elseif clear[status] then prior.active=false;prior.active_move=0 end
  local function age(field,same)
   if reset then return 0 end
   return same and previous[field]+1 or 0
  end
  local facing=picture.facing or 'unknown'
  local wins=(p.visible_wins==0 or p.visible_wins==1 or p.visible_wins==2) and p.visible_wins or -1
  local v={status=status,move_id=move,variant=meta and meta.variant or 0,pose_id=pose,
   status_age=age('status_age',previous and status==previous.status),
   action_age=move==0 and -1 or age('action_age',previous and move==previous.move_id and not started),
   pose_age=pose==0 and -1 or age('pose_age',previous and pose==previous.pose_id),
   pose_changes=reset and 0 or (move~=0 and move==previous.move_id and not started and previous.pose_changes+(pose~=previous.pose_id and 1 or 0) or 0),
   dx=dx,dy=dy,grounded=picture.grounded,recognized=pose~=0,facing=facing,wins=wins,
   visible_x=position_known and x or false,visible_y=position_known and y or false,position_known=position_known,
   last_attack_move=prior.last_attack_move,last_attack_age=prior.last_attack_frame and self.frame-prior.last_attack_frame or -1,
   family=family,visual_family=family,strength=meta and meta.strength or 'unknown',posture=meta and meta.posture or 'unknown',started=started,
   move_name=move>0 and string.format('char_%02d_event_%02x',p.char,(move-1)*2) or 'unknown'}
  prior.output=v;prior.x=x;prior.y=y;prior.char=p.char
  return prior,v
 end
 function o:reset(s)
  self.frame=0;self.history={}
  local data={interface=F.interface}
  for _,key in ipairs({'p1','p2'}) do self.history[key],data[key]=self:observe(s[key],nil,true) end
  s.fighter_perception=data;return data
 end
 function o:tick(s)
  assert(self.history.p1,'Reset fighter observer before tick');self.frame=self.frame+1
  local data={interface=F.interface}
  for _,key in ipairs({'p1','p2'}) do
   local prior=self.history[key];assert(prior.char==s[key].char,'Character changed without reset')
   self.history[key],data[key]=self:observe(s[key],prior,false)
  end
  s.fighter_perception=data;return data
 end
 return o
end
return F
