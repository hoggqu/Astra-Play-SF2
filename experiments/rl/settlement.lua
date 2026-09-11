-- Experimental native-result recognition only. No input, RAM writes or policy.
-- Keep packaged play_core.lua frozen; raw snapshots remain unchanged in evidence.
return function(Core)
 if Core.rl_settlement_adapter then return Core end
 Core.rl_settlement_adapter=true
 local observe, tick, begin=Core.observe_equal_time_latch,Core.tick,Core.begin_round
 function Core:observe_equal_time_latch(s)
  observe(self,s)
  if self.time_draw_latch or self.time_draw_ko_seen then return end
  local a,b,stop=s.p1,s.p2,self.round_stop
  -- Native settlement-001: both already have zero live/display HP at clock
  -- expiry; Ken remains alive at a18 before Chun-Li's late spinning-kick KO.
  -- The existing mature Ken/Chun-Li draw pose and unchanged-pip gates still apply.
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
  self.rl_time_loss_latch=nil
  return begin(self,s,effects)
 end
 function Core:tick(s)
  local a,b=s.p1,s.p2
  if self.phase=='settling' and self.round_stop.timer==0 and s.timer==0
   and a.char==4 and b.char==3 and a.a==18 and a.hp>=0 and b.hp>=0
   and a.timeout_hp==a.hp and b.timeout_hp==b.hp and b.hp>a.hp
   and a.displayed_hp==a.hp and b.displayed_hp==b.hp
   and a.wins==self.score[1] and b.wins==self.score[2]+1 then
   self.rl_time_loss_latch={ken_hp=a.hp,cpu_hp=b.hp,frame=self.frame+1}
  end
  local effects=tick(self,s)
  if self.phase~='settling' or self.frame-self.terminal_frame<self.settle_frames then return effects end
  local latch=self.rl_time_loss_latch
  -- Native settlement-002: Guile was awarded the time-result pip BEFORE Ken's
  -- late flash-kick KO; Ken then keeps action8 in the verified final KO sprite.
  if not latch or self.round_stop.timer~=0 or s.timer~=0 or a.char~=4 or b.char~=3
   or a.wins~=self.score[1] or b.wins~=self.score[2]+1
   or a.y~=40 or a.a~=8 or a.anim~=388594 or a.hp>=0
   or b.y~=40 or b.a~=16 or b.hp~=latch.cpu_hp
   or a.timeout_hp~=latch.ken_hp or a.displayed_hp~=latch.ken_hp
   or b.timeout_hp~=latch.cpu_hp or b.displayed_hp~=latch.cpu_hp then return effects end
  local current={a.wins,b.wins}
  local row={round=self.round,outcome='loss',opening=self.round_opening,
   stop=self.round_stop,settled=s,score=current,frame=self.frame,
   recognition='rl-native-guile-late-ken-action8',time_loss_latch=latch}
  self.rounds[#self.rounds+1]=row;self.score=current;self.timeout_guard_open=false
  effects.input='';effects.events[#effects.events+1]={kind='round_result',round=self.round,result=row}
  if current[2]==2 then
   self.phase='complete';effects.terminal={valid=true,result='cpu_win',score=current}
  else self.phase='between';self.between_frame=self.frame;self.ready_frames=0 end
  return effects
 end
 return Core
end
