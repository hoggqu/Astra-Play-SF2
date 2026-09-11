-- Experimental result-only adapter v2. Stage as rl_settlement.lua in NEW runs.
-- Original Core rules remain; this adds pose-independent TIME-result evidence.
-- No game RAM/input/state edits. Native callers must deduplicate frame callbacks.
return function(Core)
 assert(not Core.rl_settlement_adapter,'Do not stack settlement protocol versions')
 Core.rl_settlement_adapter='native-time-pip-v2'
 local observe,tick,begin=Core.observe_equal_time_latch,Core.tick,Core.begin_round
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
  return begin(self,s,effects)
 end
 function Core:tick(s)
  local effects=tick(self,s)
  -- Never override an original result, failure or a new-round transition.
  if self.phase~='settling' or self.round_stop.timer~=0 or s.timer~=0 then return effects end
  local a,b=s.p1,s.p2
  local da,db=a.wins-self.score[1],b.wins-self.score[2]
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
