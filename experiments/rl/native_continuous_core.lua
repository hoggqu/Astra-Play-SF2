-- Native-time driving; caller filters zero-time callbacks and applies deferred inputs in frame_done.
-- No hit/side cancellation: the training action macro always lasts 12 frames.
local Core=assert(loadfile('training/runtime/play_core.lua'))()
local Actions=assert(loadfile('training/runtime/rl_actions.lua'))()
function Core:rl_prime(s)
 self.rl_round=self.round;self.rl_elapsed=0
 self.rl_sequence=self.choose(s.p1,s.p2,self.mode,s,true)
 assert(#self.rl_sequence==Actions.frames)
 return self.rl_sequence[1][2]
end
function Core:drive(s,effects)
 if self.rl_round~=self.round then effects.input='';effects.rl_deferred_input=self:rl_prime(s);return effects end
 self.rl_elapsed=self.rl_elapsed+1
 if self.rl_elapsed==Actions.frames then
  self.rl_sequence=self.choose(s.p1,s.p2,self.mode,s,false);self.rl_elapsed=0
  assert(#self.rl_sequence==Actions.frames)
  effects.input='';effects.rl_deferred_input=self.rl_sequence[1][2];return effects
 end
 effects.input=self.rl_sequence[self.rl_elapsed+1][2]
 return effects
end
return Core
