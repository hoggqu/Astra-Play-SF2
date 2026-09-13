-- Current screen-derived fields and observed history, never internal a-state.
local E=assert(loadfile('training/runtime/rl_perception_execution.lua'))()
local F=assert(loadfile('training/runtime/rl_fighter_perception.lua'))()
local P=assert(loadfile('training/runtime/rl_visible_projectiles.lua'))()
local N={interface='sf2_screen_perception_v2',frame_features=1129,observations=4516}
local function add(out,x)
 assert(type(x)=='number' and x==x and math.abs(x)<math.huge,'Nonfinite perception')
 out[#out+1]=math.max(-1,math.min(1,x))
end
local function hp(p)
 local x=p.displayed_hp
 return type(x)=='number' and x==x and x>=-1 and x<=144 and math.max(0,x)/144 or -1
end
function N.features(s)
 local a,b=s.p1,s.p2;local out={};local va,vb=s.fighter_perception.p1,s.fighter_perception.p2
 local function position(p)
  if p.position_known~=true then return nil end
  local x,y=p.visible_x,p.visible_y
  assert(type(x)=='number' and x==x and math.abs(x)<math.huge)
  assert(type(y)=='number' and y==y and math.abs(y)<math.huge)
  return {x,y}
 end
 local function grounded(p)
  if p.grounded=='unknown' then return -1 end
  assert(type(p.grounded)=='boolean');return p.grounded and 1 or 0
 end
 local pa,pb=position(va),position(vb)
 for _,v in ipairs({hp(a),hp(b),s.timer/99,pa and pb and (pb[1]-pa[1])/384 or -1,
  pa and pa[1]/384 or -1,pb and pb[1]/384 or -1,pa and pa[2]/224 or -1,pb and pb[2]/224 or -1,
  grounded(va),grounded(vb)}) do add(out,v) end
 for i=0,11 do add(out,b.char==i and 1 or 0) end
 for _,block in ipairs({E.features(s),F.features(s),P.features(s)}) do
  for _,v in ipairs(block) do add(out,v) end
 end
 assert(#out==N.frame_features);return out
end
return N
