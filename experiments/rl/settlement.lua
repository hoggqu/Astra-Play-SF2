-- Experimental result-only adapter v4 candidate. Stage as rl_settlement.lua in NEW runs.
-- Original Core rules remain; this adds pose-independent TIME-result evidence.
-- No game RAM/input/state edits. Native callers must deduplicate frame callbacks.
return function(Core)
 assert(not Core.rl_settlement_adapter,'Do not stack settlement protocol versions')
 Core.rl_settlement_adapter='native-time-pip-and-confirmed-draw-v4'
 local observe,tick,begin=Core.observe_equal_time_latch,Core.tick,Core.begin_round
 local function ko(h) return type(h)=='number' and h==-1 end
 local function health(h) return type(h)=='number' and h>=0 and h<=144 and h%1==0 end
 function Core:observe_equal_time_latch(s)
  observe(self,s)
  if self.time_draw_latch or self.time_draw_ko_seen then return end
  local a,b,stop=s.p1,s.p2,self.round_stop
  -- Retain the separately confirmed zero-HP Ken/Chun-Li time draw. A missing
  -- pip by itself is NEVER a draw; original final-pose/maturity gates apply.
  if stop.timer==0 and s.timer==0 and a.char==4 and b.char==5
   and stop.p1.hp==0 and stop.p2.hp==0
   and stop.p1.displayed_hp==0 and stop.p2.displayed_hp==0
   and a.hp==0 and b.hp==0 and a.a==18
   and a.timeout_hp==0 and b.timeout_hp==0
   and a.displayed_hp==0 and b.displayed_hp==0
   and a.wins==self.score[1] and b.wins==self.score[2] then
   self.time_draw_latch={hp=0,frame=self.frame}
  end
 end
 function Core:begin_round(s,effects)
  self.rl_time_award=nil
  self.rl_draw_evidence=nil;self.rl_draw_disqualified=nil
  self.rl_double_ko=nil;self.rl_double_ko_disqualified=nil
  return begin(self,s,effects)
 end
 function Core:tick(s)
  local double=self.rl_double_ko
  if self.phase=='settling' and double and double.mature and not self.rl_double_ko_disqualified
   and self.frame+1<=self.max_frames and Core.opening(s,self.opponent,self.score) then
   self.frame=self.frame+1
   local current={s.p1.wins,s.p2.wins}
   local row={round=self.round,outcome='draw',opening=self.round_opening,
    stop=self.round_stop,settled=double.mature.state,settled_frame=double.mature.frame,
    confirmation=s,confirmation_frame=self.frame,score=current,frame=self.frame,
    recognition='rl-native-double-ko-next-round-v4',double_ko=double.first}
   self.rounds[#self.rounds+1]=row;self.score=current;self.timeout_guard_open=false
   self.phase='between';self.between_frame=self.frame;self.ready_frames=1;self.candidate=s
   return {input='',events={{kind='round_result',round=self.round,result=row}}}
  end
  local draw=self.rl_draw_evidence
  -- Positive native confirmation, BEFORE asking the old Core to process the
  -- next round. Never revive an invalid Core or overwrite an earlier result.
  if self.phase=='settling' and draw and draw.mature and not self.rl_draw_disqualified
   and self.round_stop.timer==0 and self.frame+1<=self.max_frames
   and Core.opening(s,self.opponent,self.score) then
   self.frame=self.frame+1
   local current={s.p1.wins,s.p2.wins}
   local row={round=self.round,outcome='draw',opening=self.round_opening,
    stop=self.round_stop,settled=draw.mature.state,settled_frame=draw.mature.frame,
    confirmation=s,confirmation_frame=self.frame,score=current,frame=self.frame,
    recognition='rl-native-equal-time-next-round-v3',equal_time=draw.first}
   self.rounds[#self.rounds+1]=row;self.score=current;self.timeout_guard_open=false
   self.phase='between';self.between_frame=self.frame;self.ready_frames=1;self.candidate=s
   return {input='',events={{kind='round_result',round=self.round,result=row}}}
  end
  local effects=tick(self,s)
  -- A non-time double KO needs positive chronology and a native next round,
  -- never an animation whitelist or an inference from absent winner pips.
  if self.phase=='settling' and self.round_stop.timer>0 then
   local a,b,stop=s.p1,s.p2,self.round_stop
   if a.wins~=self.score[1] or b.wins~=self.score[2] then self.rl_double_ko_disqualified=true end
   local initializing=double and double.mature and a.anim==0 and b.anim==0
    and (a.hp==0 or a.hp==144) and (b.hp==0 or b.hp==144)
    and a.displayed_hp==a.hp and b.displayed_hp==b.hp
   local function display(h) return type(h)=='number' and h>=-1 and h<=144 and h%1==0 end
   if not double and not self.rl_double_ko_disqualified and ko(stop.p1.hp) and ko(stop.p2.hp)
    and ko(a.hp) and ko(b.hp) and s.timer==stop.timer and display(a.displayed_hp) and display(b.displayed_hp) then
    double={first={frame=self.frame,state=s},display={a.displayed_hp,b.displayed_hp}}
    self.rl_double_ko=double
   end
   if double and not self.rl_double_ko_disqualified and not initializing then
    if not (ko(a.hp) and ko(b.hp) and s.timer==stop.timer and display(a.displayed_hp) and display(b.displayed_hp)
     and a.displayed_hp<=double.display[1] and b.displayed_hp<=double.display[2]) then
     self.rl_double_ko_disqualified=true
    else
     double.display={a.displayed_hp,b.displayed_hp}
     if ko(a.displayed_hp) and ko(b.displayed_hp) and self.frame-self.terminal_frame>=self.settle_frames
      and a.y==40 and b.y==40 and a.x>0 and b.x>0 and a.anim>0 and b.anim>0 then
      double.mature={frame=self.frame,state=s}
     end
    end
   end
  end
  -- Never override an original result, failure or a new-round transition.
  if self.phase~='settling' or self.round_stop.timer~=0 or s.timer~=0 then return effects end
  local a,b=s.p1,s.p2
  local da,db=a.wins-self.score[1],b.wins-self.score[2]
  if da~=0 or db~=0 then self.rl_draw_disqualified=true end
  if not self.rl_draw_disqualified then
   local equal=health(a.timeout_hp) and a.timeout_hp==b.timeout_hp
    and a.displayed_hp==a.timeout_hp and b.displayed_hp==b.timeout_hp
   local initializing=draw and draw.mature and a.anim==0 and b.anim==0
    and a.timeout_hp==0 and b.timeout_hp==0
    and (a.hp==0 or a.hp==144) and (b.hp==0 or b.hp==144)
    and a.displayed_hp==a.hp and b.displayed_hp==b.hp
   if draw and not (equal and a.timeout_hp==draw.hp) and not initializing then
    self.rl_draw_disqualified=true
   end
   if not draw and equal and a.hp==a.timeout_hp and b.hp==b.timeout_hp then
    draw={hp=a.timeout_hp,first={frame=self.frame,state=s}}
    self.rl_draw_evidence=draw
   end
   if draw and not self.rl_draw_disqualified and equal and a.timeout_hp==draw.hp
    and self.frame-self.terminal_frame>=self.settle_frames
    and a.y==40 and b.y==40 and a.x>0 and b.x>0 and a.anim>0 and b.anim>0 then
    draw.mature={frame=self.frame,state=s}
   end
  end
  local outcome=(da==1 and db==0) and 'win' or (da==0 and db==1) and 'loss' or nil
  local award=self.rl_time_award
  if award then
   if outcome~=award.outcome or a.wins~=award.score[1] or b.wins~=award.score[2]
    or a.timeout_hp~=award.hp[1] or b.timeout_hp~=award.hp[2]
    or a.displayed_hp~=award.hp[1] or b.displayed_hp~=award.hp[2] then
    return self:fail('native time-result pip or locked/display HP changed after award',effects)
   end
  elseif outcome and health(a.timeout_hp) and health(b.timeout_hp)
   and a.displayed_hp==a.timeout_hp and b.displayed_hp==b.timeout_hp
   -- Positive chronology: both LIVE values still equal the actual awarded
   -- time-HP values. Later live HP may change; never manufacture those values.
   and a.hp==a.timeout_hp and b.hp==b.timeout_hp
   and ((outcome=='win' and a.timeout_hp>b.timeout_hp)
    or (outcome=='loss' and b.timeout_hp>a.timeout_hp)) then
   award={outcome=outcome,frame=self.frame,score={a.wins,b.wins},
    hp={a.timeout_hp,b.timeout_hp},state=s}
   self.rl_time_award=award
  end
  if not award or self.frame-award.frame<360
   or self.frame-self.terminal_frame<self.settle_frames
   or a.y~=40 or b.y~=40 then return effects end
  local current={a.wins,b.wins}
  local row={round=self.round,outcome=award.outcome,opening=self.round_opening,
   stop=self.round_stop,settled=s,score=current,frame=self.frame,
   recognition='rl-native-time-pip-v2',time_award=award,
   stable_award_frames=self.frame-award.frame}
  self.rounds[#self.rounds+1]=row;self.score=current;self.timeout_guard_open=false
  effects.input='';effects.events[#effects.events+1]={kind='round_result',round=self.round,result=row}
  if current[1]==2 or current[2]==2 then
   self.phase='complete'
   effects.terminal={valid=true,result=current[1]==2 and 'ken_win' or 'cpu_win',score=current}
  else self.phase='between';self.between_frame=self.frame;self.ready_frames=0 end
  return effects
 end
 return Core
end
