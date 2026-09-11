-- Eleven frozen learned branches; no game writes or external fallback.
local Base=assert(loadfile('training/runtime/rl_bundle_base_nn.lua'))()
local N={features=Base.features}
local allowed={[0]=true,[1]=true,[2]=true,[3]=true,[5]=true,[6]=true,[7]=true,[8]=true,[9]=true,[10]=true,[11]=true}
function N.identity(model,op)
 assert(model.schema=='astra.rl-opponent-bundle-policy.v1' and model.policy_kind=='ppo_opponent_bundle')
 assert(model.action_interface=='ken_actions16_pulsed_normals_v2' and model.actions==16 and model.observations==344 and model.decision_frames==12)
 assert(model.selection=='deterministic_argmax' and allowed[op],'Unknown opponent route')
 local id=assert(model.route[tostring(op)]);local branch=assert(model.branches[id])
 assert(branch.opponent==op and branch.actor.model_sha256==branch.source_model_sha256,'Branch identity mismatch')
 return id,branch
end
function N.predict(model,obs)
 assert(#obs==344)
 local op=nil
 for i=0,11 do
  local value=obs[3*86+11+i]
  assert(value==0 or value==1,'Invalid opponent one-hot')
  if value==1 then assert(op==nil,'Ambiguous opponent one-hot');op=i end
 end
 local id,branch=N.identity(model,op)
 local action,logits=Base.predict(branch.actor,obs)
 return action,logits,id,branch.source_model_sha256
end
function N.new(model,actions)
 assert(actions.count==16 and actions.frames==12 and actions.interface=='ken_actions16_pulsed_normals_v2','Invalid action interface')
 for op in pairs(allowed) do N.identity(model,op) end
 local p={history={}}
 function p:choose(s,reset)
  local f=N.features(s)
  if reset then self.history={f,f,f,f} else table.remove(self.history,1);self.history[#self.history+1]=f end
  assert(#self.history==4)
  local obs={};for _,row in ipairs(self.history) do for _,x in ipairs(row) do obs[#obs+1]=x end end
  local action,_,id,hash=N.predict(model,obs)
  assert(model.branches[id].opponent==s.p2.char,'State route mismatch')
  local forward=s.p1.x<s.p2.x and 'R' or 'L';local seq={}
  for frame=0,actions.frames-1 do seq[#seq+1]={1,actions.keys(action,frame,s,forward)} end
  return seq,'rl_bundle_'..id..'_'..hash..'_action_'..action
 end
 return p
end
return N
