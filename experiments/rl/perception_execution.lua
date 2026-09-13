-- Execution feedback from resolved visible fighter evidence only.
-- No raw action byte, animation pointer, hidden strength or future phase reads.
local F={interface='ken_screen_execution_feedback_v2',feature_count=114}
local results={'none','pending','started','other','unconfirmed','unknown'}
local families={'unknown','normal_punch','normal_kick','hadouken','shoryuken','tatsumaki','throw','jump','locomotion'}
local strengths={'unknown','light','medium','heavy'}
local dizziness={'unknown','absent','present'}
local function onehot(out,value,values)
 local known=false
 for _,allowed in ipairs(values) do out[#out+1]=value==allowed and 1 or 0;known=known or value==allowed end
 assert(known,'Invalid execution feedback enum')
end
function F.features(s)
 local v=assert(s.visible_feedback);assert(v.interface==F.interface)
 assert(type(v.request_action)=='number' and v.request_action%1==0 and v.request_action>=-1 and v.request_action<85)
 local out={};for i=-1,84 do out[#out+1]=v.request_action==i and 1 or 0 end
 onehot(out,v.result,results);onehot(out,v.actual_family,families);onehot(out,v.actual_strength,strengths)
 for _,age in ipairs({v.request_age,v.start_age}) do
  assert(type(age)=='number' and age%1==0 and age>=-1 and age<math.huge)
  out[#out+1]=age<0 and -1 or math.min(age,120)/120
 end
 onehot(out,v.p1_dizzy,dizziness);onehot(out,v.p2_dizzy,dizziness)
 assert(type(v.started_since_request)=='boolean');out[#out+1]=v.started_since_request and 1 or 0
 assert(#out==114);return out
end
local function dizzy(p)
 if p.status=='dizzy' then return 'present' end
 return p.recognized and 'absent' or 'unknown'
end
function F.new(actions)
 assert(actions.count==85 and actions.interface=='ken_actions85_full_v1')
 local o={}
 function o:snapshot(s)
  local p=assert(s.fighter_perception)
  s.visible_feedback={interface=F.interface,request_action=self.request_action,result=self.result,
   actual_family=self.actual_family,actual_strength=self.actual_strength,
   request_age=self.request_frame and self.frame-self.request_frame or -1,
   start_age=self.start_frame and self.frame-self.start_frame or -1,
   p1_dizzy=dizzy(p.p1),p2_dizzy=dizzy(p.p2),started_since_request=self.started_since_request}
 end
 function o:reset(s)
  self.frame=0;self.request_action=-1;self.result='none';self.request_frame=nil
  self.actual_family='unknown';self.actual_strength='unknown';self.start_frame=nil
  self.started_since_request=false;self:snapshot(s)
 end
 function o:request(action,s)
  actions.descriptor(action)
  self.request_action=action;self.request_frame=self.frame;self.result='pending';self.started_since_request=false
 end
 function o:record_start(family,strength,posture)
  self.actual_family=family;self.actual_strength=strength;self.start_frame=self.frame
  if self.request_action<0 then return end
  self.started_since_request=true
  local d=actions.descriptor(self.request_action)
  if family=='unknown' then self.result='unknown'
  elseif d.target=='contextual' then
   if d.button~='' then
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
  assert(self.frame,'Reset execution observer before tick');self.frame=self.frame+1
  local p=assert(s.fighter_perception).p1
  if p.started then self:record_start(p.family,p.strength,p.posture) end
  if self.result=='pending' and self.frame-self.request_frame>=12 then self.result='unconfirmed' end
  self:snapshot(s)
 end
 return o
end
return F
