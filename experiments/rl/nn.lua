-- Pure neural inference: same clipped 4 x 86 features and 15 actions as env.py.
local N={}
local function append(out,x) out[#out+1]=math.max(-1,math.min(1,x)) end
function N.features(s)
 local a,b=s.p1,s.p2;local out={}
 for _,v in ipairs({a.hp/144,b.hp/144,s.timer/99,(b.x-a.x)/512,a.x/1024,b.x/1024,
   (a.y-40)/256,(b.y-40)/256,a.y==40 and 1 or 0,b.y==40 and 1 or 0}) do append(out,v) end
 for i=0,11 do append(out,b.char==i and 1 or 0) end
 for _,p in ipairs({a,b}) do for i=0,31 do append(out,p.a==i and 1 or 0) end end
 assert(#out==86);return out
end
local function tanh(x)
 local e=math.exp(-2*math.abs(x));local y=(1-e)/(1+e)
 return x<0 and -y or y
end
function N.predict(model,obs)
 assert(model.schema=='astra.rl-policy.v1' and model.observations==344 and model.actions==15)
 assert(#obs==344 and model.activation=='tanh' and model.selection=='deterministic_argmax')
 local x=obs
 for index,layer in ipairs(model.layers) do
  local out={}
  assert(#layer.weight==#layer.bias)
  for i,row in ipairs(layer.weight) do
   assert(#row==#x,'Layer input mismatch')
   local value=layer.bias[i]
   for j,w in ipairs(row) do value=value+w*x[j] end
   out[i]=index<#model.layers and tanh(value) or value
  end
  x=out
 end
 assert(#x==15)
 local best=1
 for i=2,#x do if x[i]>x[best] then best=i end end
 return best-1,x
end
function N.new(model,actions)
 local p={history={}}
 function p:choose(s,reset)
  local f=N.features(s)
  if reset then self.history={f,f,f,f}
  else table.remove(self.history,1);self.history[#self.history+1]=f end
  assert(#self.history==4,'History was not initialized')
  local obs={}
  for _,row in ipairs(self.history) do for _,x in ipairs(row) do obs[#obs+1]=x end end
  local action=N.predict(model,obs)
  local forward=s.p1.x<s.p2.x and 'R' or 'L';local seq={}
  for frame=0,actions.frames-1 do seq[#seq+1]={1,actions.keys(action,frame,s,forward)} end
  return seq,'rl_action_'..action
 end
 return p
end
return N
