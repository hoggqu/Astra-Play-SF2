-- Ken World Warrior input coverage. Decisions remain exactly 12 native frames.
-- No game-state gates, automatic retries, facing correction or action masks.
-- IDs 0..15 preserve ken_actions16_pulsed_normals_v2 waveforms exactly.
local A={frames=12,count=85,interface='ken_actions85_full_v1',specs={}}
local directions={'N','F','B','D','DF','DB','U','UF','UB'}
local buttons={'','LP','MP','HP','LK','MK','HK'}
local strengths={['']='none',LP='light',MP='medium',HP='heavy',LK='light',MK='medium',HK='heavy'}
local function add(name,direction,button,target,kind)
 A.specs[#A.specs+1]={id=#A.specs,name=name,direction=direction,button=button,
  target=target or 'contextual',kind=kind or 'raw',strength=assert(strengths[button])}
end
add('neutral','N','');add('forward','F','');add('back','B','')
add('crouch_guard','DB','');add('jump_forward','UF','');add('jump_back','UB','')
add('jab','N','LP');add('fierce','N','HP');add('crouch_jab','D','LP')
add('sweep','D','HK');add('medium_kick','N','MK');add('heavy_kick','N','HK')
add('fireball','F','LP','hadouken','macro')
add('uppercut','F','LP','shoryuken','macro')
add('jump_heavy_kick','UF','HK','jump_attack','macro')
add('medium_uppercut','F','MP','shoryuken','macro')
for _,direction in ipairs(directions) do
 for _,button in ipairs(buttons) do
  add('raw_'..direction:lower()..'_'..(button=='' and 'none' or button:lower()),direction,button)
 end
end
add('hadouken_mp','F','MP','hadouken','macro')
add('hadouken_hp','F','HP','hadouken','macro')
add('shoryuken_hp','F','HP','shoryuken','macro')
add('tatsumaki_lk','B','LK','tatsumaki','macro')
add('tatsumaki_mk','B','MK','tatsumaki','macro')
add('tatsumaki_hk','B','HK','tatsumaki','macro')
assert(#A.specs==A.count)
function A.descriptor(action)
 assert(type(action)=='number' and action%1==0 and action>=0 and action<A.count,'Invalid action')
 local copy={};for k,v in pairs(A.specs[action+1]) do copy[k]=v end;return copy
end
function A.keys(action,frame,s,forward)
 local spec=A.descriptor(action)
 assert(type(frame)=='number' and frame%1==0 and frame>=0 and frame<A.frames,'Invalid action frame')
 assert(forward=='L' or forward=='R','Invalid forward direction')
 local back=forward=='R' and 'L' or 'R'
 local relative={N='',F=forward,B=back,D='D',DF='D '..forward,DB='D '..back,
  U='U',UF='U '..forward,UB='U '..back}
 local direction=relative[spec.direction];local button=spec.button
 if spec.kind=='raw' then
  if frame==11 or button=='' then return direction end
  return direction=='' and button or direction..' '..button
 end
 if spec.target=='hadouken' or spec.target=='tatsumaki' then
  if frame<3 then return 'D' elseif frame<6 then return 'D '..direction
  elseif frame<8 then return direction..' '..button else return '' end
 elseif spec.target=='shoryuken' then
  if frame<2 then return forward elseif frame<4 then return 'D'
  elseif frame<6 then return 'D '..forward..' '..button else return '' end
 elseif spec.target=='jump_attack' then
  return frame<3 and 'U '..forward or (frame<9 and button or '')
 end
 error('Unsupported action target')
end
return A
