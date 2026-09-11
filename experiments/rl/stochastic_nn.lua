-- Independent policy RNG; never touches math.random or the emulator/game RNG.
return function(Base)
 local S={features=Base.features,modulus=2147483647,multiplier=48271}
 function S.next_random(state)
  assert(type(state)=='number' and state%1==0 and state>=1 and state<S.modulus,'Invalid policy seed/state')
  local next_state=(state*S.multiplier)%S.modulus
  return next_state,(next_state-1)/(S.modulus-1)
 end
 function S.sample(logits,u)
  assert(type(u)=='number' and u>=0 and u<1,'Invalid sampling uniform')
  local maximum=math.max(table.unpack(logits));local weights,total={},0
  for i,x in ipairs(logits) do
   assert(x==x and math.abs(x)<math.huge,'Nonfinite logit')
   weights[i]=math.exp(x-maximum);total=total+weights[i]
  end
  local acc=0;local action=#weights-1
  for i,w in ipairs(weights) do acc=acc+w/total;if u<acc then action=i-1;break end end
  return action,logits[action+1]-maximum-math.log(total)
 end
 function S.new(model,actions)
  assert(model.selection=='categorical_softmax','Stochastic policy requires explicit selection identity')
  assert(model.policy_prng=='park_miller_48271_v1','Unknown policy PRNG')
  S.next_random(model.policy_seed) -- Validate seed without consuming a draw.
  -- Base.predict computes logits plus an argmax. Its private compatibility copy
  -- satisfies the original assertion; that argmax is discarded, never deployed.
  local logits_model={};for k,v in pairs(model) do logits_model[k]=v end
  logits_model.selection='deterministic_argmax'
  local p={history={},rng_state=model.policy_seed,draws=0}
  function p:choose(s,reset)
   local f=S.features(s)
   if reset then self.history={f,f,f,f}
   else table.remove(self.history,1);self.history[#self.history+1]=f end
   assert(#self.history==4,'History was not initialized')
   local obs={};for _,row in ipairs(self.history) do for _,x in ipairs(row) do obs[#obs+1]=x end end
   local _,logits=Base.predict(logits_model,obs)
   local u;self.rng_state,u=S.next_random(self.rng_state);self.draws=self.draws+1
   local action,logprob=S.sample(logits,u)
   local forward=s.p1.x<s.p2.x and 'R' or 'L';local seq={}
   for frame=0,actions.frames-1 do seq[#seq+1]={1,actions.keys(action,frame,s,forward)} end
   return seq,string.format('rl_action_%d;policy_draw=%d;policy_state=%d;policy_logprob=%.17g',action,self.draws,self.rng_state,logprob)
  end
  return p
 end
 return S
end
