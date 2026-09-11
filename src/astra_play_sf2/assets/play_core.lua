-- Continuous single-match state machine. No emulator, filesystem or pause API.
local Core = {}
Core.__index = Core

local function clone(v)
 if type(v)~='table' then return v end
 local out={};for k,x in pairs(v) do out[k]=clone(x) end;return out
end
local function counts(s) return {s.p1.wins,s.p2.wins} end
local function same(a,b) return a[1]==b[1] and a[2]==b[2] end
local function actors(s,opponent)
 return s.p1.char==4 and s.p2.char==opponent
end
local function defeated(p,timer)
 -- 2194/2251: defeated Ken can retain a2/a12 while the visible sprite is
 -- the same grounded KO animation. This exception is specific to Ken/this ROM.
 -- Hardest13 Zangief: a2 persists in grounded final KO animation328980.
 -- V3 ordinary throw KO: Bison likewise retains a2 in grounded281952.
 -- This only admits the pose; negative live/display HP, native winner pip,
 -- grounded winner action16 and the 360-frame maturity gates still apply.
 local ko_pose=p.a==0 or (p.char==4 and p.anim==388594) or (p.char==6 and p.a==2 and p.anim==328980) or (p.char==8 and p.a==2 and p.anim==281952)
 return p.y==40 and ((p.hp<0 and p.displayed_hp<0 and ko_pose) or (timer==0 and p.a==18))
end
local function settled_winner(winner,loser,timer,stop)
 -- hardest12 diagnostic Honda R1: native time winner at frame3089, then
 -- late KO at3103. Honda is finally grounded in animation462306 at3301,
 -- but displayed/latched HP remain2 while live HP is-1. Old predicate
 -- required negative displayed HP or time-loss action18 and missed both.
 -- This is result recognition only; policy/input handling is unchanged.
 local function time_hp(p)
  local hp=p.timeout_hp
  return type(hp)=='number' and hp>=0 and hp<=144 and hp%1==0 and p.displayed_hp==hp
 end
 -- Hardest13 Ryu probe: defeated Ken has hp-1 but latched/display8,
 -- grounded final KO388594; Ryu has native pip1 and latched/display20.
 -- Bison zero-HP time loss: 6223 verifies Ken a12/388594, hp-1, display/latched0, CPU nativepip1/latched28.
 -- Cross14 level6 m11 R2: stop6943, native pip2 at6973, Bison hp-1
 -- at6989, grounded a0/281952 from7155. Earliest mature state7303
 -- keeps char8 and display/latched1 while Ken has34. Only this observed
 -- Bison pose is added; actor/pip/time/HP/360-frame gates remain intact.
 local late_verified_ko=stop.timer==0 and timer==0
  and ((loser.char==1 and loser.a==0 and loser.anim==462306) or (loser.char==4 and (loser.a==0 or loser.a==12) and loser.anim==388594) or (loser.char==8 and loser.a==0 and loser.anim==281952) or (loser.char==0 and loser.a==0 and loser.anim==493744) or (loser.char==3 and loser.a==0 and loser.anim==407552)) and loser.y==40 and loser.hp<0
  and time_hp(winner) and time_hp(loser) and winner.timeout_hp>loser.timeout_hp
 return winner.y==40 and winner.a==16 and (defeated(loser,timer) or late_verified_ko)
end
local function settled_time_draw(s,stop)
 -- Time-result HP is latched separately from live HP. In the verified ROM,
 -- mature displayed HP agrees with timeout_hp even when live HP later changes.
 -- boss06-screen3/sagat-standshot-s0-l105: both a18, grounded, timeout/display
 -- HP5 and unchanged pips at stop+555..560 frames, followed by another round.
 local a,b=s.p1,s.p2
 local hp=a.timeout_hp
 return stop.timer==0 and s.timer==0 and a.a==18 and b.a==18
  and a.y==40 and b.y==40 and type(hp)=='number' and hp>=0 and hp<=144 and hp%1==0
  and b.timeout_hp==hp and a.displayed_hp==hp and b.displayed_hp==hp
end
-- Coin21-l3-05: equal positive time HP was observed before Ken's late KO.
-- Restrict the new pose alternative to the observed Ken/Chun-Li pairing.
local function settled_late_ken_time_draw(s,stop,latch)
 if not latch then return false end
 local a,b=s.p1,s.p2;local hp=latch.hp
 return stop.timer==0 and s.timer==0 and a.char==4 and b.char==5
  and a.y==40 and b.y==40 and a.a==0 and a.anim==388594 and a.hp<0
  and b.a==18 and b.hp==hp
  and a.timeout_hp==hp and b.timeout_hp==hp
  and a.displayed_hp==hp and b.displayed_hp==hp
end
function Core.opening(s,opponent,score)
 return actors(s,opponent) and s.timer==99 and same(counts(s),score)
  and s.p1.hp==144 and s.p2.hp==144 and s.p1.y==40 and s.p2.y==40
  and s.p1.x>0 and s.p2.x>0 and s.p1.anim>0 and s.p2.anim>0
end
function Core.new(options,opening)
 assert(type(options.choose)=='function','Missing frozen policy')
 assert(options.timeout_guard==nil or type(options.timeout_guard)=='boolean','timeout_guard must be boolean')
 assert(Core.opening(opening,options.opponent,{0,0}),'Need a visible full-health Ken R1, timer99, zero score')
 local self=setmetatable({mode=assert(options.mode),opponent=options.opponent,
  choose=options.choose,frame=0,phase='fighting',round=1,rounds={},score={0,0},
  timeout_guard=options.timeout_guard==true,timeout_guard_open=false,
  max_frames=options.max_frames or 36000,lead=options.lead or 0,
  max_rounds=options.max_rounds or 10,settle_frames=360,opening_frames=90,
  round_opening=clone(opening),previous_enemy=clone(opening.p2),
  previous_action=opening.p1.a,previous_side=opening.p1.x<opening.p2.x,
  last_dy=0,q={},qi=1,remain=0,hadclock=true},Core)
 assert(self.lead>=0 and self.lead<=120 and self.lead%1==0)
 return self
end
function Core:fail(reason,effects)
 self.phase='invalid';effects.input='';effects.terminal={valid=false,reason=reason}
 return effects
end
function Core:begin_round(s,effects)
 self.round=self.round+1
 if self.round>self.max_rounds then return self:fail('round limit exceeded',effects) end
 self.round_opening=clone(self.candidate or s);self.candidate=nil;self.ready_frames=0
 self.phase='fighting';self.previous_enemy=clone(s.p2)
 self.previous_action=s.p1.a;self.previous_side=s.p1.x<s.p2.x
 self.last_dy=0;self.enemy_rebound=false;self.enemy_attack_frame=nil
 self.timeout_guard_open=false;self.timeout_guard_side=nil
 self.time_draw_latch=nil;self.time_draw_ko_seen=false
 self.q={};self.qi=1;self.remain=0;self.hadclock=true
 effects.events[#effects.events+1]={kind='round_start',round=self.round,state=clone(self.round_opening)}
 return effects
end
-- Record chronology, not an outcome: both fighters must still be alive at
-- a newly observed positive equal latch. A prior KO disqualifies this evidence.
function Core:observe_equal_time_latch(s)
 local a,b=s.p1,s.p2
 if a.hp<0 or b.hp<0 then self.time_draw_ko_seen=true;return end
 if self.time_draw_latch or self.time_draw_ko_seen then return end
 local stop=self.round_stop;local hp=a.timeout_hp
 if stop.timer==0 and s.timer==0 and a.char==4 and b.char==5
  and stop.p1.timeout_hp==0 and stop.p2.timeout_hp==0
  and same(counts(s),self.score)
  and type(hp)=='number' and hp>0 and hp<=144 and hp%1==0
  and b.timeout_hp==hp and a.hp==hp and b.hp==hp then
  self.time_draw_latch={hp=hp,frame=self.frame}
 end
end
function Core:guard_timeout(s,effects)
 if not self.timeout_guard_open then return effects end
 local a,b=s.p1,s.p2
 -- This is an input-only option. Never use it to infer the round's result.
 -- Once native result evidence appears, this round's window cannot reopen.
 if s.timer~=0 or a.hp<0 or b.hp<0 or not same(counts(s),self.score)
  or a.a==16 or a.a==18 or b.a==16 or b.a==18 then
  self.timeout_guard_open=false;effects.input='';return effects
 end
 -- Preserve the last known side at exact overlap; otherwise follow current x.
 if a.x~=b.x then self.timeout_guard_side=a.x<b.x end
 local back=self.timeout_guard_side and 'L' or 'R'
 effects.input=b.y>50 and back or 'D '..back
 return effects
end
function Core:drive(s,effects)
 local a,b=clone(s.p1),clone(s.p2)
 if (b.a==10 or b.a==12) and b.a~=self.previous_enemy.a then self.enemy_attack_frame=self.frame end
 b.attack_age=self.enemy_attack_frame and (self.frame-self.enemy_attack_frame) or 999
 local dy=b.y-self.previous_enemy.y
 if b.y<=40 then self.enemy_rebound=false
 elseif dy>0 and self.last_dy<0 and self.previous_enemy.y>50 then self.enemy_rebound=true end
 b.rebound=self.enemy_rebound or false
 if dy~=0 then self.last_dy=dy end
 b.vy=b.y<=40 and 0 or self.last_dy;self.previous_enemy=b
 local side=a.x<b.x
 if ((a.a==14 or a.a==20) and a.a~=self.previous_action) or side~=self.previous_side then
  self.q={};self.qi=1;self.remain=0;effects.input=''
 end
 self.previous_action=a.a;self.previous_side=side
 if self.remain<=0 then
  if self.qi>#self.q then
   self.q=self.choose(a,b,self.mode);self.qi=1
   assert(type(self.q)=='table' and #self.q>0,'Policy returned no input sequence')
  end
  local step=self.q[self.qi];self.qi=self.qi+1
  assert(type(step[1])=='number' and step[1]>0 and step[1]%1==0,'Invalid policy duration')
  self.remain=step[1];effects.input=step[2] or ''
 end
 self.remain=self.remain-1
 return effects
end
function Core:tick(s)
 local effects={events={}}
 if self.phase=='complete' or self.phase=='invalid' then return effects end
 self.frame=self.frame+1
 if self.frame>self.max_frames then return self:fail('match frame watchdog exceeded',effects) end
 if not actors(s,self.opponent) then return self:fail('characters changed before match completion',effects) end
 if s.timer<0 then return self:fail('invalid game timer',effects) end
 local current=counts(s)
 if current[1]<self.score[1] or current[2]<self.score[2] or current[1]>2 or current[2]>2 then
  return self:fail('invalid win counter transition',effects)
 end
 if self.phase=='fighting' then
  if s.timer>0 then self.hadclock=true end
  if s.p1.hp<0 or s.p2.hp<0 or (self.hadclock and s.timer==0) then
   self.phase='settling';self.terminal_frame=self.frame;self.round_stop=clone(s)
   self.time_draw_latch=nil;self.time_draw_ko_seen=s.p1.hp<0 or s.p2.hp<0
   effects.input='';effects.events[1]={kind='round_stop',round=self.round,state=clone(s)}
   self.timeout_guard_open=self.timeout_guard and s.timer==0
   self.timeout_guard_side=self.previous_side
   self:guard_timeout(s,effects)
   return effects
  end
  if not same(current,self.score) then return self:fail('win counter changed without a round ending',effects) end
  if self.lead>0 then self.lead=self.lead-1;effects.input='';return effects end
  return self:drive(s,effects)
 end
 if self.phase=='settling' then
  -- A clock of zero only starts settlement; damage and win counters can change later.
  if Core.opening(s,self.opponent,current) then
   return self:fail('new round arrived before previous result was resolved',effects)
  end
  self:observe_equal_time_latch(s)
  self:guard_timeout(s,effects)
  if self.frame-self.terminal_frame<self.settle_frames then return effects end
  local da,db=current[1]-self.score[1],current[2]-self.score[2]
  if da>1 or db>1 then return self:fail('skipped win counter',effects) end
  local outcome
  if da==1 and db==1 then return self:fail('both win counters increased; result requires review',effects) end
  if da==1 and db==0 and settled_winner(s.p1,s.p2,s.timer,self.round_stop) then outcome='win'
  elseif da==0 and db==1 and settled_winner(s.p2,s.p1,s.timer,self.round_stop) then outcome='loss'
  elseif da==0 and db==0 and s.p1.hp<0 and s.p2.hp<0
   and s.p1.displayed_hp<0 and s.p2.displayed_hp<0
   and s.p1.y==40 and s.p2.y==40
   and (s.p1.a==0 or (s.p1.char==4 and s.p1.a==12 and s.p1.anim==388594))
   and s.p2.a==0 then outcome='draw'
  elseif da==0 and db==0 and settled_time_draw(s,self.round_stop) then outcome='draw'
  elseif da==0 and db==0 and settled_late_ken_time_draw(s,self.round_stop,self.time_draw_latch) then outcome='draw' end
  if not outcome then
   if self.frame-self.terminal_frame>1800 then return self:fail('unresolved settlement',effects) end
   return effects
  end
  local row={round=self.round,outcome=outcome,opening=self.round_opening,
   stop=self.round_stop,settled=clone(s),score=clone(current),frame=self.frame}
  self.rounds[#self.rounds+1]=row;self.score=current
  self.timeout_guard_open=false
  effects.events[#effects.events+1]={kind='round_result',round=self.round,result=row}
  if current[1]==2 or current[2]==2 then
   if current[1]==2 and current[2]==2 then return self:fail('both sides reached match-winning score',effects) end
   self.phase='complete';effects.input=''
   effects.terminal={valid=true,result=current[1]==2 and 'ken_win' or 'cpu_win',score=clone(current)}
  else
   self.phase='between';self.between_frame=self.frame;self.ready_frames=0
  end
  return effects
 end
 if self.phase=='between' then
  if not same(current,self.score) then return self:fail('win counter changed between rounds',effects) end
  if self.frame-self.between_frame>2400 then return self:fail('next round did not appear',effects) end
  if Core.opening(s,self.opponent,self.score) then
   if not self.candidate then self.candidate=clone(s) end
   self.ready_frames=self.ready_frames+1
   if self.ready_frames>=self.opening_frames then
    self:begin_round(s,effects)
    if self.phase=='fighting' then return self:drive(s,effects) end
   end
  else
   -- Never wait out actual combat for an old 90-frame timer or accept initialization zeros.
   if self.candidate and (s.p1.hp<144 or s.p2.hp<144 or (s.timer>0 and s.timer<99)) then
    self:begin_round(s,effects)
    if self.phase=='fighting' then return self:drive(s,effects) end
   else self.candidate=nil;self.ready_frames=0 end
  end
  return effects
 end
 return self:fail('unknown controller phase',effects)
end
return Core
