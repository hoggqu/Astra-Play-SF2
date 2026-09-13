-- Experimental result-only adapter v6 candidate. Stage as rl_settlement.lua in NEW runs.
-- Original Core rules remain; this adds pose-independent TIME-result evidence.
-- No game RAM/input/state edits. Native callers must deduplicate frame callbacks.
return function(Core)
 assert(not Core.rl_settlement_adapter,'Do not stack settlement protocol versions')
 Core.rl_settlement_adapter='native-pip-ko-time-and-confirmed-draw-v6'
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
  self.rl_time_award=nil;self.rl_time_previous=nil;self.rl_ko_award=nil;self.rl_ko_disqualified=nil
  self.rl_draw_evidence=nil;self.rl_draw_disqualified=nil
  self.rl_double_ko=nil;self.rl_double_ko_disqualified=nil
  return begin(self,s,effects)
 end
 function Core:tick(s)
  local previous=self.rl_time_previous
  -- Callers supply one tick per native frame. Retain the actual preceding
  -- snapshot, including its old score, without rewriting any observed field.
  self.rl_time_previous={frame=self.frame+1,state=s}
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
  -- A non-TIME KO is established by its terminal HP chronology and the
  -- native score, not a character-specific loser animation whitelist.
  -- The old Core always runs first; its results/failures remain authoritative.
  if self.phase=='settling' and self.round_stop.timer>0 and not self.rl_ko_disqualified then
   local a,b,stop=s.p1,s.p2,self.round_stop
   local side=ko(stop.p2.hp) and health(stop.p1.hp) and 1
    or ko(stop.p1.hp) and health(stop.p2.hp) and 2 or nil
   if side then
    local winner=side==1 and a or b;local loser=side==1 and b or a
    local startwinner=side==1 and stop.p1 or stop.p2
    local da,db=a.wins-self.score[1],b.wins-self.score[2]
    local awarded=(side==1 and da==1 and db==0) or (side==2 and da==0 and db==1)
    local oldscore=da==0 and db==0
    local evidence=self.rl_ko_award
    local display=type(loser.displayed_hp)=='number' and loser.displayed_hp>=-1
     and loser.displayed_hp<=144 and loser.displayed_hp%1==0
    if not evidence then
     evidence={first_frame=self.frame,previous_frame=self.frame-1,
      last_display=(side==1 and stop.p2 or stop.p1).displayed_hp}
     self.rl_ko_award=evidence
    end
    local valid=self.frame==evidence.previous_frame+1 and s.timer==stop.timer
     and a.char==stop.p1.char and b.char==stop.p2.char
     and ko(loser.hp) and winner.hp==startwinner.hp and health(winner.hp)
     and winner.displayed_hp==winner.hp and display
     and type(evidence.last_display)=='number' and loser.displayed_hp<=evidence.last_display
     and (awarded or (oldscore and not evidence.award_frame))
    if not valid then self.rl_ko_disqualified=true;return effects end
    evidence.previous_frame=self.frame;evidence.last_display=loser.displayed_hp
    if awarded and not evidence.award_frame then
     evidence.award_frame=self.frame;evidence.award_state=s
    end
    if evidence.award_frame and self.frame-evidence.award_frame>=360
     and self.frame-self.terminal_frame>=self.settle_frames
     and a.y==40 and b.y==40 and a.x>0 and b.x>0 and a.anim>0 and b.anim>0
     and winner.a==16 and loser.displayed_hp==-1 then
     local current={a.wins,b.wins}
     local row={round=self.round,outcome=side==1 and 'win' or 'loss',opening=self.round_opening,
      stop=self.round_stop,settled=s,score=current,frame=self.frame,
      recognition='rl-native-nontime-pip-ko-v6',ko_award=evidence,
      stable_award_frames=self.frame-evidence.award_frame}
     self.rounds[#self.rounds+1]=row;self.score=current;self.timeout_guard_open=false
     effects.input='';effects.events[#effects.events+1]={kind='round_result',round=self.round,result=row}
     if current[1]==2 or current[2]==2 then
      self.phase='complete'
      effects.terminal={valid=true,result=current[1]==2 and 'ken_win' or 'cpu_win',score=current}
     else self.phase='between';self.between_frame=self.frame;self.ready_frames=0 end
     return effects
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
  local adjacent=false
  if outcome and previous and previous.frame==self.frame-1 then
   local p=previous.state
   local loser=outcome=='win' and b or a
   local winner=outcome=='win' and a or b
   adjacent=p.timer==0 and p.p1.char==a.char and p.p2.char==b.char
    and p.p1.wins==self.score[1] and p.p2.wins==self.score[2]
    and health(p.p1.hp) and health(p.p2.hp)
    and p.p1.displayed_hp==p.p1.hp and p.p2.displayed_hp==p.p2.hp
    and p.p1.hp==a.timeout_hp and p.p2.hp==b.timeout_hp
    and loser.hp==-1 and winner.hp==winner.timeout_hp
  end
  local award=self.rl_time_award
  if award then
   if outcome~=award.outcome or a.wins~=award.score[1] or b.wins~=award.score[2]
    or a.timeout_hp~=award.hp[1] or b.timeout_hp~=award.hp[2]
    or a.displayed_hp~=award.hp[1] or b.displayed_hp~=award.hp[2] then
    return self:fail('native time-result pip or locked/display HP changed after award',effects)
   end
  elseif outcome and health(a.timeout_hp) and health(b.timeout_hp)
   and a.displayed_hp==a.timeout_hp and b.displayed_hp==b.timeout_hp
   -- TIME can assign the pip, latch HP and force the loser to -1 on the same
   -- native frame. In that case require the immediately preceding raw live/
   -- display HP and old pips to establish the chronology; never backfill RAM.
   and ((a.hp==a.timeout_hp and b.hp==b.timeout_hp) or adjacent)
   and ((outcome=='win' and a.timeout_hp>b.timeout_hp)
    or (outcome=='loss' and b.timeout_hp>a.timeout_hp)) then
   award={outcome=outcome,frame=self.frame,score={a.wins,b.wins},
    hp={a.timeout_hp,b.timeout_hp},state=s,
    chronology=adjacent and 'adjacent-native-time-ko-v5' or 'same-frame-live-v2',
    previous=adjacent and previous or nil}
   self.rl_time_award=award
  end
  if not award or self.frame-award.frame<360
   or self.frame-self.terminal_frame<self.settle_frames
   or a.y~=40 or b.y~=40 then return effects end
  local current={a.wins,b.wins}
  local row={round=self.round,outcome=award.outcome,opening=self.round_opening,
   stop=self.round_stop,settled=s,score=current,frame=self.frame,
   recognition=award.chronology=='adjacent-native-time-ko-v5' and 'rl-native-time-pip-adjacent-ko-v5' or 'rl-native-time-pip-v2',time_award=award,
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
