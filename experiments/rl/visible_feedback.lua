-- Observation only: never masks inputs, waits for recovery, alters reward or writes RAM.
local F={interface='ken_visible_feedback_v1',feature_count=114}
local default_map
local results={'none','pending','started','other','unconfirmed','unknown'}
local families={'unknown','normal_punch','normal_kick','hadouken','shoryuken','tatsumaki','throw','jump','locomotion'}
local strengths={'unknown','light','medium','heavy'}
local dizzy_states={'unknown','absent','present'}
local function onehot(out,value,values)
 for _,v in ipairs(values) do out[#out+1]=value==v and 1 or 0 end
end
function F.features(s)
 local v=assert(s.visible_feedback,'Missing visible feedback snapshot');local out={}
 assert(v.interface==F.interface,'Wrong visible feedback interface')
 assert(type(v.request_action)=='number' and v.request_action%1==0 and v.request_action>=-1 and v.request_action<85,'Invalid request action')
 local function enum(value,values)
  for _,allowed in ipairs(values) do if value==allowed then return end end
  error('Invalid visible feedback label')
 end
 enum(v.result,results);enum(v.actual_family,families);enum(v.actual_strength,strengths)
 enum(v.p1_dizzy,dizzy_states);enum(v.p2_dizzy,dizzy_states)
 for _,age in ipairs({v.request_age,v.start_age}) do
  assert(type(age)=='number' and age%1==0 and age>=-1 and age<math.huge,'Invalid feedback age')
 end
 assert(type(v.started_since_request)=='boolean','Invalid start flag')
 for action=-1,84 do out[#out+1]=v.request_action==action and 1 or 0 end
 onehot(out,v.result,results);onehot(out,v.actual_family,families);onehot(out,v.actual_strength,strengths)
 out[#out+1]=v.request_age<0 and -1 or math.min(v.request_age,120)/120
 out[#out+1]=v.start_age<0 and -1 or math.min(v.start_age,120)/120
 onehot(out,v.p1_dizzy,dizzy_states);onehot(out,v.p2_dizzy,dizzy_states)
 out[#out+1]=v.started_since_request and 1 or 0
 assert(#out==F.feature_count);return out
end
function F.new(actions,mapping)
 assert(actions.count==85 and actions.interface=='ken_actions85_full_v1','Wrong action interface')
 if not mapping then
  default_map=default_map or assert(loadfile('training/runtime/rl_visible_animation_map.lua'))()
  mapping=default_map
 end
 local animations,dizzy={},{ }
 for _,row in ipairs(mapping.sequences) do
  for i=0,row[2]-1 do
   local address=row[1]+24*i
   assert(not animations[address],'Overlapping animation map')
   animations[address]={family=row[3],strength=row[4],posture=row[5],start=i==0 and row[6],sequence=row[1]}
  end
 end
 for char,seq in pairs(mapping.dizzy) do
  dizzy[char]={};for _,address in ipairs(seq) do dizzy[char][address]=true end
 end
 local function dizzy_state(p)
  if not p or not dizzy[p.char] or type(p.anim)~='number' or p.anim<=0 then return 'unknown' end
  return dizzy[p.char][p.anim] and 'present' or 'absent'
 end
 local function classify(p) return p and p.char==4 and animations[p.anim] or nil end
 local o={}
 function o:snapshot(s)
  s.visible_feedback={interface=F.interface,request_action=self.request_action,result=self.result,
   actual_family=self.actual_family,actual_strength=self.actual_strength,
   request_age=self.request_frame and self.frame-self.request_frame or -1,
   start_age=self.start_frame and self.frame-self.start_frame or -1,
   p1_dizzy=dizzy_state(s.p1),p2_dizzy=dizzy_state(s.p2),started_since_request=self.started_since_request}
  return s.visible_feedback
 end
 function o:reset(s)
  self.frame=0;self.request_action=-1;self.result='none';self.request_frame=nil
  self.actual_family='unknown';self.actual_strength='unknown';self.start_frame=nil
  self.started_since_request=false;self.previous={a=s.p1.a,anim=s.p1.anim,y=s.p1.y,x=s.p1.x,char=s.p1.char}
  self:snapshot(s)
 end
 function o:request(action,s)
  actions.descriptor(action) -- validates; deliberately never changes s / saved observations
  self.request_action=action;self.request_frame=self.frame;self.result='pending';self.started_since_request=false
 end
 function o:record_start(family,strength,posture)
  self.actual_family=family;self.actual_strength=strength;self.start_frame=self.frame
  if self.request_action<0 then return end
  self.started_since_request=true
  local d=actions.descriptor(self.request_action)
  if family=='unknown' then self.result='unknown'
  elseif d.target=='contextual' then
   -- A raw button may legally yield a normal or throw; never call a throw failure.
   if d.button~='' then
    -- A contextual input can complete a previously begun motion. Its visible
    -- attack is useful feedback, but is not proof this request caused it.
    if family=='normal_punch' or family=='normal_kick' then
     local expected=d.button:sub(2,2)=='P' and 'normal_punch' or 'normal_kick'
     self.result=family==expected and strength==d.strength and 'started' or 'unknown'
    elseif family=='throw' then
     self.result=d.strength~='light' and (d.direction=='F' or d.direction=='B') and 'started' or 'unknown'
    else self.result='unknown' end
   elseif family=='jump' or family=='locomotion' then self.result='started'
   else self.result='unknown' end
  elseif d.target=='jump_attack' then
   self.result=(posture=='air' and (family=='normal_punch' or family=='normal_kick')) and 'started' or 'other'
  else self.result=family==d.target and 'started' or 'other' end
 end
 function o:tick(s)
  assert(self.previous,'Reset observer before tick');self.frame=self.frame+1
  local p,prev=s.p1,self.previous;local current,before=classify(p),classify(prev)
  if current and current.start and p.anim~=prev.anim then
   -- Special/throw sequences can loop while one move remains active. Ordinary
   -- chainable attacks can genuinely restart without leaving a10.
   local normal=current.family=='normal_punch' or current.family=='normal_kick'
   if normal or not before or before.family~=current.family then
    self:record_start(current.family,current.strength,current.posture)
   end
  elseif not current and (p.a==10 or p.a==12) and prev.a~=p.a then
   -- Coarse a10/a12 can precede the visible startup by one frame. Do not
   -- invent an extra move start before the first recognized animation.
   if self.request_action>=0 then self.result='unknown' end
  elseif not current and p.a~=10 and p.a~=12 and prev.y==40 and p.y>40 and p.a==4 then
   self:record_start('jump','unknown','air')
  elseif not current and p.a==0 and prev.a==0 and p.y==40 and p.x~=prev.x then
   -- Displacement can include pushback; report no movement success from it.
  end
  if self.result=='pending' and self.frame-self.request_frame>=12 then self.result='unconfirmed' end
  self.previous={a=p.a,anim=p.anim,y=p.y,x=p.x,char=p.char};self:snapshot(s)
 end
 function o:features(s) return F.features(s) end
 return o
end
return F
