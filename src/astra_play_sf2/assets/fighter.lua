-- Uses read-only fighter position, health and visible animation state.
if train_busy and train_busy() then error('Cannot reload modules during a training match') end
-- All control writes go only to the normal P1 arcade input fields.
if play_busy and play_busy() then error('Cannot reload fighting policy during continuous play') end
fighter_continuous_guard_version=1
fighter_train_guard_version=1
mem=manager.machine.devices[':maincpu'].spaces['program']
function fighter(i)
 local b=0xff83c6+i*0x300
 return {x=mem:read_i16(b+6),y=mem:read_i16(b+10),hp=mem:read_i16(b+42),a=mem:read_u8(b+3),char=mem:read_u8(b+0x291),anim=mem:read_u32(b+0x1a)}
end
function state()
 local a,b=fighter(0),fighter(1)
 print(string.format('Ken hp=%d x=%d y=%d a=%d | CPU char=%d hp=%d x=%d y=%d a=%d',a.hp,a.x,a.y,a.a,b.char,b.hp,b.x,b.y,b.a))
end
bot=nil; botframe=botframe or 0
function fight(frames,mode)
 guard_training_action()
 if job or bot or advance or loadwatch or savewatch then error('controller busy') end
 local a,b=fighter(0),fighter(1)
 if not manager.machine.paused then error('Observe and pause the game before starting a fight') end
 if a.char~=4 or a.hp<0 or b.hp<0 then error('Not a live Ken round; inspect the screenshot') end
 bot={left=frames,mode=mode or 'balanced',opponent=b.char,previous_enemy=b,previous_action=a.a,previous_side=a.x<b.x,last_dy=0,q={},qi=1,remain=0,actions={},hadclock=false}
 apply_training_speed();emu.unpause()
end
-- Pure selector extension. Loading this file installs no globals/callbacks.
-- In a COPIED training fighter, bind this returned function locally before
-- function choose(), then call it at the top of choose with choose as argument.
-- A nil first result means continue the unmodified old choose body.
local function c19_extension(a,b,mode,choose_baseline)
 if mode=='c22v4_blanka_air_mp_fast' then
  local seq,reason=choose_baseline(a,b,'c19_blanka_low_lead_hold')
  if b.char==2 and reason=='dp' and a.a==8 and a.anim==389226 and b.anim==437078
   and seq[3] and (seq[3][2]=='D L MP' or seq[3][2]=='D R MP') then
   local f=a.x<b.x and 'R' or 'L';local back=f=='R' and 'L' or 'R'
   assert(#seq==4 and seq[1][1]==2 and seq[1][2]==f
    and seq[2][1]==2 and seq[2][2]=='D'
    and seq[3][1]==2 and seq[3][2]=='D '..f..' MP'
    and seq[4][1]==12 and seq[4][2]==back,
    'Blanka air MP fast requires original 2/2/2/back12 macro')
   return {{1,seq[1][2]},{1,seq[2][2]},{seq[3][1],seq[3][2]},{seq[4][1],seq[4][2]}},
    'current Blanka437078 guarded air MP directions one frame each'
  end
  return seq,reason
 end
 if mode=='c22v4_dhalsim_far_ground_start_guard' then
  local seq,reason=choose_baseline(a,b,'c19_dhalsim_confirm_guard')
  local startup=b.anim==303076 or b.anim==303100 or b.anim==303124 or b.anim==303148
  if b.char==7 and reason=='jump distant visible attack startup' and startup and math.abs(a.x-b.x)>=240 then
   return {{2,'D '..(a.x<b.x and 'L' or 'R')}},'guard confirmed far Dhalsim303076 startup before jump'
  end
  return seq,reason
 end
 if mode=='c22v3_dhalsim_startup_family_guard' then
  local seq,reason=choose_baseline(a,b,'c19_dhalsim_confirm_guard')
  local d=math.abs(a.x-b.x)
  local startup=b.anim==308076 or b.anim==308100 or b.anim==308124 or b.anim==308148
   or b.anim==308172 or b.anim==308196 or b.anim==308220 or b.anim==308244
  if b.char==7 and reason=='jump distant visible attack startup' and startup and d>=160 and d<240 then
   return {{2,'D '..(a.x<b.x and 'L' or 'R')}},'guard current midrange Dhalsim308076 animation family before jump'
  end
  return seq,reason
 end
 if mode=='c22v3_bison_ground_throw' then
  local seq,reason=choose_baseline(a,b,'c22_bison_timeout_guard')
  if b.char==8 and reason=='dp' and b.y==40 and math.abs(a.x-b.x)<48 then
   local f=a.x<b.x and 'R' or 'L';local back=f=='R' and 'L' or 'R'
   return {{3,f..' HP'},{4,back}},'ground ordinary HP throw with original timeout baseline'
  end
  return seq,reason
 end
 if mode=='c22v3_bison_close_throw' then
  local seq,reason=choose_baseline(a,b,'c22_bison_timeout_guard')
  if b.char==8 and reason=='dp' and b.y<=50 and math.abs(a.x-b.x)<48 then
   local f=a.x<b.x and 'R' or 'L';local back=f=='R' and 'L' or 'R'
   return {{3,f..' HP'},{4,back}},'close ordinary HP throw with original timeout baseline'
  end
  return seq,reason
 end
 if mode=='c22v3_bison_fire2' then
  local seq,reason=choose_baseline(a,b,'c22_bison_timeout_guard')
  if b.char==8 and reason=='three frame Bison LP fire with original range190 and guard56' then
   local out={};for i,step in ipairs(seq) do out[i]={step[1],step[2]} end
   out[1][1]=2;out[2][1]=2;out[3][1]=2
   return out,'two frame Bison LP fire with original range190 and guard56'
  end
  return seq,reason
 end
 if mode=='c22v3_bison_descent_fullrange' then
  local seq,reason=choose_baseline(a,b,'c22_bison_timeout_guard')
  if b.char==8 and reason=='dp' and b.anim==284620 and (b.vy or 0)<0
   and b.y>50 and b.y<=120 and math.abs(a.x-b.x)<110 then
   return {{2,a.x<b.x and 'L' or 'R'}},'wait current low Bison284620 through original air MP range'
  end
  return seq,reason
 end
 -- Independent input-timing experiment. No move-frame-data claim.
 if mode=='c22v3_bison_air_mp_fast' then
  local seq,reason=choose_baseline(a,b,'c22_bison_timeout_guard')
  if b.char==8 and reason=='dp' and b.y>50 then
   local out={};for i,step in ipairs(seq) do out[i]={step[1],step[2]} end
   -- Existing eligible air DP is exactly 2/2/2/12, with MP on step three.
   out[1][1]=1;out[2][1]=1
   return out,'Bison original air MP with one-frame first two directions'
  end
  return seq,reason
 end
 if mode=='c22_bison_timeout_guard' then return choose_baseline(a,b,'c22_bison_landing_near') end
 if mode=='c22_chunli_timeout_guard' then return choose_baseline(a,b,'c19_chunli_air_lp') end
 -- Independent experiment; no anti-air success or frame-data claim.
 if mode=='c22v2_bison_280676_chp' then
  local seq,reason=choose_baseline(a,b,'c22_bison_landing_near')
  if b.char==8 and reason=='dp' and b.anim==280676 and b.y>50
   and (b.vy or 0)<0 and math.abs(a.x-b.x)<56 then
   return {{2,'D'},{4,'D HP'},{12,a.x<b.x and 'L' or 'R'}},
    'experimental crouching HP only current Bison280676 descent within56'
  end
  return seq,reason
 end
 if mode=='c22_chunli_guarded_descent' then
  local seq,reason=choose_baseline(a,b,'c19_chunli_air_lp')
  if b.char==5 and reason=='light air DP with original ground policy'
   and a.a==8 and a.anim==389226 and b.anim==362444 then
   return {{2,a.x<b.x and 'L' or 'R'}},'wait current guarded ChunLi362444 descent before LP DP'
  end
  return seq,reason
 end
 -- Independent hypothesis. Delegate ORIGINAL near; do not stack animation wait.
 if mode=='c22v2_bison_280676_lp' then
  local seq,reason=choose_baseline(a,b,'c22_bison_landing_near')
  if b.char==8 and reason=='dp' and b.anim==280676 and b.y>50
   and (b.vy or 0)<0 and math.abs(a.x-b.x)<56 then
   local out={};for i,step in ipairs(seq) do out[i]={step[1],step[2]} end
   out[3][2]=out[3][2]:gsub(' MP$',' LP')
   return out,'light DP only current Bison280676 descent within56'
  end
  return seq,reason
 end
 -- Hypothesis only: add to a new branch's c19_extension after Root review.
 if mode=='c22v2_bison_280676_guard' then
  local seq,reason=choose_baseline(a,b,'c22_bison_landing_near')
  if b.char==8 and reason=='dp' and b.anim==280676 and b.y>50
   and (b.vy or 0)<0 and math.abs(a.x-b.x)<56 then
   return {{2,a.x<b.x and 'L' or 'R'}},'stand guard current Bison280676 descent within56 before MP DP'
  end
  return seq,reason
 end
 if mode=='c22_honda_guarded_descent' then
  local seq,reason=choose_baseline(a,b,'c21_honda_fire190_guard')
  if reason=='light air DP with original ground policy' and b.char==1
   and a.a==8 and a.anim==389226 and b.anim==470706 and (b.vy or 0)<=-5 then
   return {{2,a.x<b.x and 'L' or 'R'}},'wait current Honda470706 fast descent while guarding'
  end
  return seq,reason
 end
 if mode=='c22_bison_landing_near' then
  local seq,reason=choose_baseline(a,b,'c21_bison_near_low_wait')
  if b.char==8 and reason=='sweep' and b.anim==284620 and b.y==40 and math.abs(a.x-b.x)<90 then
   return {{2,'D '..(a.x<b.x and 'L' or 'R')}},'wait currently grounded near Bison284620 before sweep'
  end
  return seq,reason
 end
 if mode=='c22_bison_landing_wait' then
  local seq,reason=choose_baseline(a,b,'c21_bison_near_low_wait')
  if b.char==8 and reason=='sweep' and b.anim==284620 and b.y==40 then
   return {{2,'D '..(a.x<b.x and 'L' or 'R')}},'wait currently grounded Bison284620 before sweep'
  end
  return seq,reason
 end
 if mode=='c21_bison_near_low_wait' then
  local seq,reason=choose_baseline(a,b,'c19_bison_fire3')
  if b.char==8 and reason=='dp' and b.anim==284620 and (b.vy or 0)<0
   and b.y>50 and b.y<=120 and math.abs(a.x-b.x)<90 then
   return {{2,a.x<b.x and 'L' or 'R'}},'wait current near low Bison284620 before remaining MP DP'
  end
  return seq,reason
 end
 if mode=='c21_honda_timeout_guard' then return choose_baseline(a,b,'c19_honda_air_lp') end
 if mode=='c21_honda_fire190_guard' then return choose_baseline(a,b,'c21_honda_trailing_fire190') end
 if mode=='c21_honda_trailing_fire190' then
  local seq,reason=choose_baseline(a,b,'c19_honda_air_lp')
  if reason=='guard and wait' and b.char==1 and (a.a==0 or a.a==2) and (b.a==0 or b.a==2)
   and a.y<=45 and b.y<=50 and a.hp>=0 and a.hp<b.hp and math.abs(a.x-b.x)>=190 then
   local f=a.x<b.x and 'R' or 'L';local back=f=='R' and 'L' or 'R'
   return {{4,'D'},{4,'D '..f},{4,f..' LP'},{16,back}},'Honda trailing neutral ground wait to LP fire at 190'
  end
  return seq,reason
 end
 if mode=='c19_guile_fire3' then
  local seq,reason=choose_baseline(a,b,'guile_pressure')
  if b.char==3 and reason=='fire' then
   for i=1,3 do seq[i][1]=3 end
   return seq,'Guile original LP fire with three-frame direction inputs'
  end
  return seq,reason
 end
 if mode=='c19_dhalsim_confirm260' then
  local seq,reason=choose_baseline(a,b,'c19_dhalsim_confirm_guard')
  local d=math.abs(a.x-b.x)
  if b.char==7 and reason=='jump distant visible attack startup' and b.attack_age==0 and d>=240 and d<260 then
   return {{2,'D '..(a.x<b.x and 'L' or 'R')}},'confirm newly visible Dhalsim attack out to260'
  end
  return seq,reason
 end
 if mode=='c19_blanka_low_lead_hold' then
  local seq,reason=choose_baseline(a,b,'c14_blanka_fire220_sweep95')
  if b.char==2 and reason=='fire' and b.hp<=48 and a.hp>b.hp then
   return {{2,'D '..(a.x<b.x and 'L' or 'R')}},'protect current lead against low HP Blanka before fire'
  end
  return seq,reason
 end
 if mode=='c19_guile_fire_fast' then
  local seq,reason=choose_baseline(a,b,'guile_pressure')
  if b.char==3 and reason=='fire' then
   for i=1,3 do seq[i][1]=2 end
   return seq,'Guile original LP fire with two-frame direction inputs'
  end
  return seq,reason
 end
 if mode=='c19_dhalsim_tell_wait' then
  local seq,reason=choose_baseline(a,b,'c19_dhalsim_confirm_guard')
  local d=math.abs(a.x-b.x)
  if b.char==7 and reason=='jump distant visible attack startup' and b.anim==308076 and d>=160 and d<240 then
   return {{2,'D '..(a.x<b.x and 'L' or 'R')}},'wait two frames at current Dhalsim308076 before jump'
  end
  return seq,reason
 end
 if mode=='c19_blanka_low_fire_wait' then
  local seq,reason=choose_baseline(a,b,'c14_blanka_fire220_sweep95')
  if b.char==2 and reason=='fire' and b.a==2 and b.anim==432098 and b.hp<=48 and a.hp>b.hp then
   return {{2,'D '..(a.x<b.x and 'L' or 'R')}},'wait current Blanka432098 low HP before fire'
  end
  return seq,reason
 end
 if mode=='c19_vega_low_descent_wait' or mode=='c19_vega_air_lp' then
  local seq,reason=choose_baseline(a,b,'c14_vega_far_hold')
  if b.char==11 and mode=='c19_vega_low_descent_wait' and reason=='earlier descending approach distance' and b.anim==231664 then
   return {{2,'D '..(a.x<b.x and 'L' or 'R')}},'wait currently visible low Vega231664 before extended DP'
  end
  if b.char==11 and mode=='c19_vega_air_lp' and b.y>50 and (reason=='dp' or reason=='earlier descending approach distance' or reason=='earlier currently observed rebound descent') then
   seq[3][2]='D '..(a.x<b.x and 'R' or 'L')..' LP'
   return seq,'light air DP with original Vega approach and ground policy'
  end
  return seq,reason
 end
 if mode=='c19_bison_low_descent_wait' then
  local seq,reason=choose_baseline(a,b,'c19_bison_drop_guard56')
  if b.char==8 and reason=='dp' and b.anim==284620 and (b.vy or 0)<0 and b.y>50 and b.y<=120 then
   return {{2,a.x<b.x and 'L' or 'R'}},'wait current low Bison284620 descent before remaining MP DP'
  end
  return seq,reason
 end
 if mode=='c19_bison_fire3' then
  local seq,reason=choose_baseline(a,b,'c19_bison_drop_guard56')
  if b.char==8 and reason=='fire' then
   seq[1][1]=3;seq[2][1]=3;seq[3][1]=3
   return seq,'three frame Bison LP fire with original range190 and guard56'
  end
  return seq,reason
 end
 if mode=='c19_bison_drop_guard56' then
  local seq,reason=choose_baseline(a,b,'h13_bison_guile_fire190')
  if b.char==8 and reason=='dp' and b.y>50 and (b.vy or 0)<0 and b.anim==284620 and math.abs(a.x-b.x)<56 then
   return {{2,a.x<b.x and 'L' or 'R'}},'stand guard current Bison descent within56 before MP DP'
  end
  return seq,reason
 end
 if mode=='c19_bison_low_air_lp' then
  local seq,reason=choose_baseline(a,b,'c19_bison_close_drop_guard')
  if b.char==8 and b.hp<=48 and b.y>50 and reason=='dp' then
   seq[3][2]='D '..(a.x<b.x and 'R' or 'L')..' LP'
   return seq,'light air DP against low HP Bison with original close guard'
  end
  return seq,reason
 end
 if mode=='c19_ryu_fire_fast' then
  local seq,reason=choose_baseline(a,b,'guile_pressure')
  if b.char==0 and reason=='fire' then
   for i=1,3 do seq[i][1]=2 end
   return seq,'Ryu original LP fire with two-frame direction inputs'
  end
  return seq,reason
 end
 if mode=='c19_chunli_air_lp' or mode=='c19_honda_air_lp' then
  local seq,reason=choose_baseline(a,b,mode=='c19_honda_air_lp' and 'vega_sweep_guard' or 'vega_drop')
  local target=mode=='c19_honda_air_lp' and 1 or 5
  if b.char==target and b.y>50 and reason=='dp' then
   seq[3][2]='D '..(a.x<b.x and 'R' or 'L')..' LP'
   return seq,'light air DP with original ground policy'
  end
  return seq,reason
 end
 if mode=='c19_bison_close_drop_guard' then
  local seq,reason=choose_baseline(a,b,'h13_bison_guile_fire190')
  if b.char==8 and reason=='dp' and b.y>50 and (b.vy or 0)<0 and b.anim==284620 and math.abs(a.x-b.x)<40 then
   return {{2,a.x<b.x and 'L' or 'R'}},'stand guard current close Bison descent before MP DP'
  end
  return seq,reason
 end
 if mode=='c19_zangief_fire_fast' or mode=='c19_zangief_fire_hp' or mode=='c19_zangief_sweep95' then
  local seq,reason=choose_baseline(a,b,'guile_pressure')
  if b.char==6 and reason=='fire' and mode=='c19_zangief_fire_fast' then
   for i=1,3 do seq[i][1]=2 end
   return seq,'Zangief original LP fire with two-frame direction inputs'
  end
  if b.char==6 and reason=='fire' and mode=='c19_zangief_fire_hp' then
   seq[3][2]=(a.x<b.x and 'R' or 'L')..' HP'
   return seq,'Zangief original fire range with heavy projectile'
  end
  if b.char==6 and reason=='sweep' and mode=='c19_zangief_sweep95' and math.abs(a.x-b.x)>=95 then
   return {{2,'D '..(a.x<b.x and 'L' or 'R')}},'guard outside Zangief close sweep range95'
  end
  return seq,reason
 end
 if mode=='c19_zangief_fire190' or mode=='c19_zangief_fire220' or mode=='c19_zangief_sweep_guard' then
  local seq,reason=choose_baseline(a,b,'guile_pressure')
  local d=math.abs(a.x-b.x)
  local back=a.x<b.x and 'L' or 'R'
  if b.char==6 and mode=='c19_zangief_sweep_guard' and reason=='sweep' and (b.a==10 or b.a==12) then
   return {{2,'D '..back}},'guard current Zangief ground attack before sweep'
  end
  local minimum=mode=='c19_zangief_fire190' and 190 or 220
  if b.char==6 and mode~='c19_zangief_sweep_guard' and reason=='fire' and d<minimum then
   return {{2,'D '..back}},'wait outside Zangief short fire recovery range'
  end
  return seq,reason
 end
 if mode=='c19_bison_fire240_guard' or mode=='c19_bison_guard_pair' then
  local seq,reason=choose_baseline(a,b,mode=='c19_bison_guard_pair' and 'c19_bison_high_drop_guard' or 'h13_bison_fire190_air_lp')
  if b.char==8 and reason=='fire' and b.a==0 and math.abs(a.x-b.x)<240 then
   return {{2,'D '..(a.x<b.x and 'L' or 'R')}},'wait outside standing Bison near fire recovery range240'
  end
  return seq,reason
 end
 if mode=='c19_bison_high_drop_guard' or mode=='c19_bison_drop_family_guard' then
  local seq,reason=choose_baseline(a,b,'h13_bison_fire190_air_lp')
  local visible_drop=b.anim==280652 or (mode=='c19_bison_drop_family_guard' and b.anim==280676)
  if b.char==8 and reason=='far-fire policy with light air DP' and visible_drop and b.y>130 and math.abs(a.x-b.x)<50 then
   return {{2,a.x<b.x and 'L' or 'R'}},'guard close high currently visible Bison descent before light DP'
  end
  return seq,reason
 end
 if mode=='c19_bison_fire190_air_hp' then
  local seq,reason=choose_baseline(a,b,'h13_bison_guile_fire190')
  if b.char==8 and b.y>50 and reason=='dp' then
   seq[3][2]='D '..(a.x<b.x and 'R' or 'L')..' HP'
   return seq,'far-fire policy with heavy air DP'
  end
  return seq,reason
 end
 if mode=='c19_dhalsim_confirm_guard' then
  local seq,reason=choose_baseline(a,b,'c19_dhalsim_limb_guard')
  local d=math.abs(a.x-b.x)
  if b.char==7 and reason=='jump distant visible attack startup' and d>=160 and d<240 and b.attack_age==0 then
   return {{2,'D '..(a.x<b.x and 'L' or 'R')}},'confirm newly visible attack animation before midrange jump'
  end
  return seq,reason
 end
 if mode=='c19_dhalsim_sweep_guard' then
  local seq,reason=choose_baseline(a,b,'h13_lp_approach_guard')
  if b.char==7 and reason=='sweep' and (b.a==10 or b.a==12) then
   return {{2,'D '..(a.x<b.x and 'L' or 'R')}},'guard visible Dhalsim attack before midrange sweep'
  end
  return seq,reason
 end
 if mode=='c19_dhalsim_limb_guard' then
  local seq,reason=choose_baseline(a,b,'ryu_close')
  local d=math.abs(a.x-b.x)
  local limb=b.anim==303916 or b.anim==303940 or b.anim==303964 or b.anim==303988 or b.anim==304012
  if b.char==7 and reason=='jump distant visible attack startup' and d>=160 and d<240 and limb then
   return {{2,'D '..(a.x<b.x and 'L' or 'R')}},'guard currently visible midrange Dhalsim limb before jump'
  end
  return seq,reason
 end
 if mode=='c19_sagat_low_wait' or mode=='c19_sagat_dp2' then
  local seq,reason=choose_baseline(a,b,'h13_sagat_far_guard')
  if b.char==9 and mode=='c19_sagat_low_wait' and reason=='sweep' and b.hp<=48 then
   return {{2,'D '..(a.x<b.x and 'L' or 'R')}},'wait low Sagat HP midrange before sweep'
  end
  if b.char==9 and mode=='c19_sagat_dp2' and reason=='dp' then
   for i=1,3 do seq[i][1]=2 end
   return seq,'shorter Sagat DP direction input'
  end
  return seq,reason
 end
 if mode=='c19_dhalsim_dp2' then
  local seq,reason=choose_baseline(a,b,'h13_lp_approach_guard')
  if b.char==7 and reason=='dp' then
   for i=1,3 do seq[i][1]=2 end
   return seq,'shorter Dhalsim DP direction input'
  end
  return seq,reason
 end
 if mode=='c19_balrog_baseline' then
  return choose_baseline(a,b,'h13_balrog_finish_guard')
 end
 if mode=='c19_balrog_lp_air' then
  return choose_baseline(a,b,'h13_balrog_lp_air')
 end
 if mode=='c19_balrog_sweep95' or mode=='c19_balrog_late_air_guard' then
  local seq,reason=choose_baseline(a,b,'h13_balrog_finish_guard')
  local d=math.abs(a.x-b.x);local back=a.x<b.x and 'L' or 'R'
  if b.char==10 and mode=='c19_balrog_sweep95' and reason=='sweep' and d>=95 then
   return {{2,'D '..back}},'avoid far sweep recovery before approach'
  end
  if b.char==10 and mode=='c19_balrog_late_air_guard' and reason=='dp' and b.y>50 and b.y<100 and d<40 and (b.vy or 0)<0 then
   return {{2,back}},'block already close low descending attack'
  end
  return seq,reason
 end
 if mode~='c19_balrog_visible_dash_guard' then return nil end
 -- Compute the current finish_guard decision first: preserve its recovery,
 -- low-HP guard, close DP and all airborne behavior exactly.
 local seq,reason=choose_baseline(a,b,'h13_balrog_finish_guard')
 local d=math.abs(a.x-b.x)
 local visible_dash=b.anim==256006 or b.anim==256030
 local ready=a.y<=45 and a.a~=10 and a.a~=12 and a.a~=14 and a.a~=20
 if reason=='sweep' and ready and b.char==10 and b.y<=50
  and d>=60 and d<108 and (b.a==10 or b.a==12) and visible_dash then
  local back=a.x<b.x and 'L' or 'R'
  return {{2,'D '..back}},'guard observed Balrog dash before sweep'
 end
 return seq,reason
end

function choose(a,b,mode)
 local seq,reason=c19_extension(a,b,mode,choose);if seq then return seq,reason end
 if mode=='c14_vega_high_wait' then
  if b.y>=130 then return choose(a,b,'c14_vega_far_wait') end
  return choose(a,b,'h13_vega180_sweep90')
 end
 if mode=='c14_vega_wall_wait' then
  if a.x<=400 or a.x>=880 then return choose(a,b,'c14_vega_far_wait') end
  return choose(a,b,'h13_vega180_sweep90')
 end
 if mode=='c14_vega_far_hold' or mode=='c14_vega_far_hp' then
  local seq,reason=choose(a,b,'h13_vega180_sweep90')
  if reason=='earlier descending approach distance' and not b.rebound then
   local f=a.x<b.x and 'R' or 'L';local back=f=='R' and 'L' or 'R'
   if mode=='c14_vega_far_hold' then return {{2,'D '..back}},'hold position before closer ordinary descent' end
   seq[3][2]='D '..f..' HP';return seq,'heavy DP only ordinary far descent extension'
  end
  return seq,reason
 end
 if mode=='c14_vega_lead_wait' then
  if a.hp>b.hp then return choose(a,b,'c14_vega_far_wait') end
  return choose(a,b,'h13_vega180_sweep90')
 end
 if mode=='c14_blanka_fire220_sweep100' or mode=='c14_blanka_fire220_sweep95' then
  local seq,reason=choose(a,b,'c14_blanka_fire220_guard')
  local limit=mode=='c14_blanka_fire220_sweep100' and 100 or 95
  if reason=='sweep' and math.abs(a.x-b.x)>=limit then local back=a.x<b.x and 'L' or 'R';return {{2,'D '..back}},'wait inside tightened sweep range' end
  return seq,reason
 end
 if mode=='c14_blanka_fire220_guard' then
  local d=math.abs(a.x-b.x);local back=a.x<b.x and 'L' or 'R'
  local ready=a.y<=45 and a.a~=10 and a.a~=12 and a.a~=14 and a.a~=20
  if ready and b.y<=50 and d>=60 and d<108 and (b.a==10 or b.a==12) then return {{2,'D '..back}},'guard visible attack before fire220 sweep' end
  return choose(a,b,'c14_blanka_fire220')
 end
 if mode=='c14_blanka_fire220_behind' then
  local seq,reason=choose(a,b,'c14_blanka_fire220')
  if reason=='fire' and a.hp>=b.hp then local back=a.x<b.x and 'L' or 'R';return {{4,'D '..back}},'retain original range guard while not behind' end
  return seq,reason
 end
 if mode=='c14_blanka_lead_guard' then
  local seq,reason=choose(a,b,'c14_blanka_fire190')
  if reason=='fire' and a.hp<=48 and a.hp>b.hp then local back=a.x<b.x and 'L' or 'R';return {{2,'D '..back}},'protect low health lead instead of fire' end
  return seq,reason
 end
 if mode=='c14_blanka_fire220' then
  local seq,reason=choose(a,b,'guile_pressure')
  if reason=='fire' and math.abs(a.x-b.x)<220 then local back=a.x<b.x and 'L' or 'R';return {{2,'D '..back}},'wait outside short fire220 range' end
  return seq,reason
 end
 if mode=='c14_vega_guile' then return choose(a,b,'guile_pressure') end
 if mode=='c14_vega_far_wait' then
  local seq,reason=choose(a,b,'h13_vega180_sweep90')
  if reason=='earlier descending approach distance' and not b.rebound then
   local back=a.x<b.x and 'L' or 'R';return {{2,back}},'guard far ordinary descent until closer'
  end
  return seq,reason
 end
 if mode=='c14_vega_far_lp' then
  local seq,reason=choose(a,b,'h13_vega180_sweep90')
  if reason=='earlier descending approach distance' and not b.rebound then
   local f=a.x<b.x and 'R' or 'L';seq[3][2]='D '..f..' LP';return seq,'light DP only in ordinary far descent extension'
  end
  return seq,reason
 end
 if mode=='c14_blanka_fire190_guard' then return choose(a,b,'h13_bison_guile_fire190') end
 if mode=='c14_blanka_fire190' then
  local seq,reason=choose(a,b,'guile_pressure')
  if reason=='fire' and math.abs(a.x-b.x)<190 then local back=a.x<b.x and 'L' or 'R';return {{2,'D '..back}},'wait outside short fire190 range' end
  return seq,reason
 end
 if mode=='h13_bison_fire190_air_lp' or mode=='h13_bison_fire190_hp' then
  local seq,reason=choose(a,b,'h13_bison_guile_fire190')
  local f=a.x<b.x and 'R' or 'L'
  if mode=='h13_bison_fire190_air_lp' and b.y>50 and reason=='dp' then seq[3][2]='D '..f..' LP';return seq,'far-fire policy with light air DP' end
  if mode=='h13_bison_fire190_hp' and reason=='fire' then seq[3][2]=f..' HP';return seq,'heavy fire at existing190 distance' end
  return seq,reason
 end
 if mode=='h13_bison_guile_air_lp' or mode=='h13_bison_guile_air_hp' then
  local seq,reason=choose(a,b,'h13_bison_guile_sweep_guard')
  if b.y>50 and reason=='dp' then
   local f=a.x<b.x and 'R' or 'L';seq[3][2]='D '..f..(mode=='h13_bison_guile_air_lp' and ' LP' or ' HP')
   return seq,'air DP strength trial with ground sweep guard'
  end
  return seq,reason
 end
 if mode=='h13_bison_guile_fire190' then
  local seq,reason=choose(a,b,'h13_bison_guile_sweep_guard')
  if reason=='fire' and math.abs(a.x-b.x)<190 then local back=a.x<b.x and 'L' or 'R';return {{2,'D '..back}},'wait beyond near fire range190' end
  return seq,reason
 end
 if mode=='h13_bison_guile_fire130' then
  local d=math.abs(a.x-b.x);local f=a.x<b.x and 'R' or 'L';local back=f=='R' and 'L' or 'R'
  local ready=a.y<=45 and a.a~=10 and a.a~=12 and a.a~=14 and a.a~=20
  if ready and b.y<=50 and d>=130 and d<155 and b.a~=10 and b.a~=12 then return {{4,'D'},{4,'D '..f},{4,f..' LP'},{16,back}},'LP fire from range130' end
  return choose(a,b,'h13_bison_guile_sweep_guard')
 end
 if mode=='h13_bison_jump_intercept' or mode=='h13_bison_jump_early' then
  local d=math.abs(a.x-b.x);local back=a.x<b.x and 'L' or 'R'
  if a.a==12 or a.a==14 or a.a==20 then return choose(a,b,'h13_bison_guile_both_guard') end
  if a.y>45 then
   if d<105 and math.abs(a.y-b.y)<115 then return {{3,'HK'},{3,back}},'air kick in current observed range' end
   return {{2,back}},'wait in air for current approach'
  end
  if a.a~=10 and b.y>(mode=='h13_bison_jump_early' and 50 or 80) and (b.vy or 0)>0 and d<180 then return {{6,'U'}},'neutral jump toward observed rising opponent' end
  return choose(a,b,'h13_bison_guile_both_guard')
 end
 if mode=='h13_bison_guile_both_guard' then
  local seq,reason=choose(a,b,'h13_bison_guile_sweep_guard')
  if b.y>50 and reason=='dp' then local back=a.x<b.x and 'L' or 'R';return {{2,back}},'block descending approach with ground sweep guard' end
  return seq,reason
 end
 if mode=='h13_bison_guile_close_throw' then
  local d=math.abs(a.x-b.x);local f=a.x<b.x and 'R' or 'L';local back=f=='R' and 'L' or 'R'
  local ready=a.y<=45 and a.a~=10 and a.a~=12 and a.a~=14 and a.a~=20
  if ready and b.y<=50 and d<48 then return {{3,f..' HP'},{4,back}},'close ordinary HP throw instead of LP DP' end
  return choose(a,b,'h13_bison_guile_sweep_guard')
 end
 if mode=='h13_bison_guile_sweep_guard' then
  local d=math.abs(a.x-b.x);local back=a.x<b.x and 'L' or 'R'
  local ready=a.y<=45 and a.a~=10 and a.a~=12 and a.a~=14 and a.a~=20
  if ready and b.y<=50 and d>=60 and d<108 and (b.a==10 or b.a==12) then return {{2,'D '..back}},'guard visible attack before guile sweep' end
  return choose(a,b,'guile_pressure')
 end
 if mode=='h13_bison_guile_air_guard' then
  local seq,reason=choose(a,b,'guile_pressure')
  if b.y>50 and reason=='dp' then local back=a.x<b.x and 'L' or 'R';return {{2,back}},'block descending approach instead of air DP' end
  return seq,reason
 end
 if mode=='h13_bison_lp150_counter125' then
  local d=math.abs(a.x-b.x);local back=a.x<b.x and 'L' or 'R'
  local ready=a.y<=45 and a.a~=10 and a.a~=12 and a.a~=14 and a.a~=20
  if ready and b.hp<=48 and b.y<=50 and d>=125 and d<185 and (b.a==10 or b.a==12) then return {{2,'D '..back}},'guard active low HP attack outside counter125; retain LP150' end
  return choose(a,b,'h13_bison_lp150')
 end
 if mode=='h13_bison_lp150_near_dp1' then
  local seq,reason=choose(a,b,'h13_bison_lp150')
  if b.hp>48 and math.abs(a.x-b.x)<40 and (reason=='first descending LP with height150 gate' or reason=='short light DP') then
   for i=1,3 do seq[i][1]=1 end
   return seq,'near-head existing descending LP directions 2 to 1'
  end
  return seq,reason
 end
 if mode=='h13_bison_bait150' then
  local d=math.abs(a.x-b.x);local back=a.x<b.x and 'L' or 'R'
  local ready=a.y<=45 and a.a~=10 and a.a~=12 and a.a~=14 and a.a~=20
  if ready and b.hp>48 and b.y<=50 and d>=150 then return {{2,'D '..back}},'wait outside effective ground bait range' end
  return choose(a,b,'h13_bison_lp150')
 end
 if mode=='h13_vega180_sweep90' then
  local d=math.abs(a.x-b.x);local back=a.x<b.x and 'L' or 'R'
  local ready=a.y<=45 and a.a~=10 and a.a~=12 and a.a~=14 and a.a~=20
  if ready and b.y<=50 and d>=90 and d<108 and (b.a==10 or b.a==12) then return {{2,'D '..back}},'guard visible ground attack only at far sweep range' end
  return choose(a,b,'h13_vega125_rebound180')
 end
 if mode=='h13_vega180_sweep' then
  local d=math.abs(a.x-b.x);local back=a.x<b.x and 'L' or 'R'
  local ready=a.y<=45 and a.a~=10 and a.a~=12 and a.a~=14 and a.a~=20
  if ready and b.y<=50 and d>=60 and d<108 and (b.a==10 or b.a==12) then return {{2,'D '..back}},'guard visible ground attack before sweep with rebound180' end
  return choose(a,b,'h13_vega125_rebound180')
 end
 if mode=='h13_vega125_lp' then
  local f=a.x<b.x and 'R' or 'L';local back=f=='R' and 'L' or 'R'
  local ready=a.y<=45 and a.a~=10 and a.a~=12 and a.a~=14 and a.a~=20;local d=math.abs(a.x-b.x)
 if ready and b.y>50 and b.y<150 and (b.vy or 0)<0 and d<125 and not b.rebound then
  return {{2,f},{2,'D'},{2,'D '..f..' LP'},{12,back}},'non-rebound descending LP at existing125 range'
 end
 return choose(a,b,'h13_vega125')
 end
 if mode=='h13_vega125_rebound180' then
  local f=a.x<b.x and 'R' or 'L';local back=f=='R' and 'L' or 'R'
  local ready=a.y<=45 and a.a~=10 and a.a~=12 and a.a~=14 and a.a~=20;local d=math.abs(a.x-b.x)
 if ready and b.rebound and b.y>=150 and b.y<180 and (b.vy or 0)<0 and d<125 then
  return {{2,f},{2,'D'},{2,'D '..f..' MP'},{12,back}},'earlier currently observed rebound descent'
 end
 return choose(a,b,'h13_vega125')
 end
 if mode=='h13_bison_lp150' then
  local f=a.x<b.x and 'R' or 'L';local back=f=='R' and 'L' or 'R'
  local ready=a.y<=45 and a.a~=10 and a.a~=12 and a.a~=14 and a.a~=20;local d=math.abs(a.x-b.x)
 if ready and b.hp>48 and b.y>=140 and b.y<150 and (b.vy or 0)<0 and d<110 and not b.rebound then
  return {{2,f},{2,'D'},{2,'D '..f..' LP'},{12,back}},'first descending LP with height150 gate'
 end
 return choose(a,b,'bison_air_lp')
 end
 if mode=='h13_bison_counter125' then
  local f=a.x<b.x and 'R' or 'L';local back=f=='R' and 'L' or 'R'
  local ready=a.y<=45 and a.a~=10 and a.a~=12 and a.a~=14 and a.a~=20;local d=math.abs(a.x-b.x)
 if ready and b.hp<=48 and b.y<=50 and d>=125 and d<185 and (b.a==10 or b.a==12) then
  return {{2,'D '..back}},'guard active low HP attack outside counter125'
 end
 return choose(a,b,'bison_air_lp')
 end
 if mode=='h13_honda_escape' then
  local held=a.anim==381682 or a.anim==381706 or a.anim==381730 or a.anim==381754 or a.anim==381802
  if b.char==1 and b.a==10 and b.y<=50 and a.y>45 and math.abs(a.x-b.x)<80 and held then return {{1,'L LP'},{1,'R HP'}},'alternate ordinary inputs in observed Honda hold' end
  return choose(a,b,'vega_drop')
 end
 if mode=='h13_zangief_space190' then
  local d=math.abs(a.x-b.x);local back=a.x<b.x and 'L' or 'R'
  local ready=a.y<=45 and a.a~=10 and a.a~=12 and a.a~=14 and a.a~=20
  if ready and b.y<=50 and d>=108 and d<190 then return {{2,'D '..back}},'wait outside the original ground fire range' end
  return choose(a,b,'h13_balanced_air')
 end
 if mode=='h13_zangief_dp2' then
  local seq,reason=choose(a,b,'h13_balanced_air')
  if reason=='dp' then for i=1,3 do seq[i][1]=2 end;return seq,'DP direction timing 4 to 2; fire timing unchanged' end
  return seq,reason
 end
 if mode=='h13_bison_both_lp' then return choose(a,b,b.hp<=48 and 'h13_bison_low_lp' or 'bison_air_lp') end
 if mode=='h13_balrog_finish_guard' then
  local d=math.abs(a.x-b.x);local back=a.x<b.x and 'L' or 'R'
  local ready=a.y<=45 and a.a~=10 and a.a~=12 and a.a~=14 and a.a~=20
  if ready and b.y<=50 and b.hp<=16 and d>=60 and d<108 and (b.a==10 or b.a==12) then return {{2,'D '..back}},'guard visible near-finish attack' end
  return choose(a,b,'vega_drop')
 end
 if mode=='h13_balrog_lp_air' or mode=='h13_bison_lp_full' then
  local d=math.abs(a.x-b.x);local f=a.x<b.x and 'R' or 'L';local back=f=='R' and 'L' or 'R'
  local ready=a.y<=45 and a.a~=10 and a.a~=12 and a.a~=14 and a.a~=20
  if ready and b.y>50 and b.y<(mode=='h13_balrog_lp_air' and 150 or 140) and (b.vy or 0)<0 and d<110 and not (mode=='h13_bison_lp_full' and b.rebound and b.y>90 and d<150) then return {{2,f},{2,'D'},{2,'D '..f..' LP'},{12,back}},'shorter air DP commitment' end
  return choose(a,b,mode=='h13_balrog_lp_air' and 'vega_drop' or 'bison_guard_bait')
 end
 if mode=='h13_sagat_far_guard' or mode=='h13_sagat_low_guard' then
  local d=math.abs(a.x-b.x);local back=a.x<b.x and 'L' or 'R'
  local ready=a.y<=45 and a.a~=10 and a.a~=12 and a.a~=14 and a.a~=20
  local shot=b.anim==269332 or b.anim==269356 or b.anim==269408 or b.anim==269432
  if ready and b.y<=50 then
   if mode=='h13_sagat_far_guard' and d>260 and (b.a==10 or b.a==12) and not shot then return {{2,'D '..back}},'guard ongoing far attack instead of approach' end
   if mode=='h13_sagat_low_guard' and b.hp<=48 and d>=60 and d<108 then return {{2,'D '..back}},'guard low HP midrange instead of sweep' end
  end
  return choose(a,b,'sagat_far_sweepguard')
 end
 if mode=='h13_bison_low_lp' then
  local d=math.abs(a.x-b.x);local f=a.x<b.x and 'R' or 'L';local back=f=='R' and 'L' or 'R'
  local ready=a.y<=45 and a.a~=10 and a.a~=12 and a.a~=14 and a.a~=20
  if ready and b.hp<=48 and b.y<=50 and d<185 and ((b.a==10 or b.a==12) or d<65) then
   return {{2,f},{2,'D'},{2,'D '..f..' LP'},{12,back}},'low HP ground LP counter'
  end
  return choose(a,b,'bison_guard_mix')
 end
 -- Hardest13: independently selectable, observed-state-only candidates.
 if mode=='h13_lp_approach_guard' or mode=='h13_lp_sweep_guard' or mode=='h13_fast_guard' then
  local d=math.abs(a.x-b.x);local back=a.x<b.x and 'L' or 'R'
  local ready=a.y<=45 and a.a~=10 and a.a~=12 and a.a~=14 and a.a~=20
  local zone=mode=='h13_lp_approach_guard' and d>=108 or mode=='h13_lp_sweep_guard' and d>=60 and d<108 or mode=='h13_fast_guard' and d>=60
  if ready and b.y<=50 and zone and (b.a==10 or b.a==12) then return {{2,'D '..back}},'guard observed ground attack before commitment' end
  return choose(a,b,mode=='h13_fast_guard' and 'fast_lp' or 'lp_safe')
 end
 if mode=='h13_balanced_air' or mode=='h13_balanced_safe' then
  local d=math.abs(a.x-b.x);local back=a.x<b.x and 'L' or 'R'
  local ready=a.y<=45 and a.a~=10 and a.a~=12 and a.a~=14 and a.a~=20
  if ready and mode=='h13_balanced_air' and b.y>50 and d>=110 then return {{2,back}},'guard distant airborne approach' end
  if ready and mode=='h13_balanced_safe' and b.y<=50 and d>=108 and (b.a==10 or b.a==12) then return {{2,'D '..back}},'wait visible ground attack before fire' end
  return choose(a,b,'balanced')
 end
 if mode=='h13_ryu190' then
  local d=math.abs(a.x-b.x);local back=a.x<b.x and 'L' or 'R'
  local ready=a.y<=45 and a.a~=10 and a.a~=12 and a.a~=14 and a.a~=20
  if ready and b.y<=50 and d>=160 and d<190 and (b.attack_age or 999)<12 then return {{2,'D '..back}},'wait for safer visible jump range' end
  return choose(a,b,'ryu_close')
 end
 if mode=='h13_vega_sweep' then return choose(a,b,'vega_sweep_guard') end
 if mode=='h13_vega_jab' then return choose(a,b,'vega_close_jab') end
 if mode=='h13_vega125' then
  local d=math.abs(a.x-b.x);local f=a.x<b.x and 'R' or 'L';local back=f=='R' and 'L' or 'R'
  local ready=a.y<=45 and a.a~=10 and a.a~=12 and a.a~=14 and a.a~=20
  if ready and b.y>50 and b.y<150 and (b.vy or 0)<0 and d>=110 and d<125 then return {{2,f},{2,'D'},{2,'D '..f..' MP'},{12,back}},'earlier descending approach distance' end
  return choose(a,b,'vega_drop')
 end
 -- Timeout guarding is implemented by continuous Core; combat policy is unchanged.
 if mode=='vega_timeout_guard' then return choose(a,b,'vega_drop') end
 -- Sagat08: preserve jump190/HK125, only guard an already visible attack before the low-HP sweep.
 if mode=='sagat_far_sweepguard' then
  local d=math.abs(a.x-b.x);local back=a.x<b.x and 'L' or 'R'
  local ready=a.y<=45 and a.a~=10 and a.a~=12 and a.a~=14 and a.a~=20
  if ready and b.y<=50 and b.hp<=48 and d>=60 and d<108 and (b.a==10 or b.a==12) then return {{2,'D '..back}},'guard current attack before low HP sweep' end
  return choose(a,b,'sagat_jump_far')
 end
 if mode=='sagat_standshot' then
  local d=math.abs(a.x-b.x);local back=a.x<b.x and 'L' or 'R'
  local ready=a.y<=45 and a.a~=10 and a.a~=12 and a.a~=14 and a.a~=20
  if ready and b.y<=50 and b.anim==269380 and d>=108 then return {{2,back}},'standing guard current high shot animation' end
  return choose(a,b,'sagat_lowshot')
 end
 local high_only=mode=='sagat_high_only'
 local late_kick=mode=='sagat_air_late'
 local jump_min=mode=='sagat_jump_205' and 205 or mode=='sagat_jump_220' and 220 or 190
 local far_jump=mode=='sagat_jump_far' or mode=='sagat_jump_205' or mode=='sagat_jump_220'
 local early_kick=mode=='sagat_air_early' or far_jump
 if high_only or early_kick or late_kick then mode='sagat_lowshot' end
 if mode=='sagat_guard_both' then
  local d=math.abs(a.x-b.x);local back=a.x<b.x and 'L' or 'R'
  local ready=a.y<=45 and a.a~=10 and a.a~=12 and a.a~=14 and a.a~=20
  local shot=b.anim==269332 or b.anim==269356 or b.anim==269408 or b.anim==269432
  if ready and b.y<=50 and d>260 and not shot and (b.a==10 or b.a==12) then return {{2,'D '..back}},'guard ongoing far attack' end
  return choose(a,b,'sagat_sweepguard')
 end
 if mode=='bison_air_lp' or mode=='bison_air_early' or mode=='bison_close_lp' then
  local d=math.abs(a.x-b.x);local f=a.x<b.x and 'R' or 'L';local back=f=='R' and 'L' or 'R'
  local ready=a.y<=45 and a.a~=10 and a.a~=12 and a.a~=14 and a.a~=20
  if ready then
   local air=b.hp>48 and b.y>50 and (b.vy or 0)<0 and d<110 and not (b.rebound and b.y>90)
   local early=mode=='bison_air_early' and air and b.y>=140 and b.y<180
   local lp=(mode=='bison_air_lp' and air and b.y<140) or (mode=='bison_close_lp' and b.hp<=48 and b.y<=50 and d<65)
   if early or lp then return {{2,f},{2,'D'},{2,'D '..f..(lp and ' LP' or ' MP')},{12,back}},early and 'earlier first descent' or 'short light DP' end
  end
  return choose(a,b,'bison_guard_mix')
 end
 -- Boss06 candidates: independent changes; delegate unchanged baseline elsewhere.
 if mode=='vega_close_jab' or mode=='vega_early_drop' or mode=='vega_sweep_guard' then
  local d=math.abs(a.x-b.x);local f=a.x<b.x and 'R' or 'L';local back=f=='R' and 'L' or 'R'
  local ready=a.y<=45 and a.a~=10 and a.a~=12 and a.a~=14 and a.a~=20
  if ready then
   if mode=='vega_close_jab' and b.y<=50 and d<60 then return {{3,'D LP'},{3,'D '..back}},'close jab instead of DP' end
   if mode=='vega_early_drop' and b.y>=150 and b.y<180 and (b.vy or 0)<0 and d<110 then return {{2,f},{2,'D'},{2,'D '..f..' MP'},{12,back}},'earlier visible descent DP' end
   if mode=='vega_sweep_guard' and b.y<=50 and d>=60 and d<108 and (b.a==10 or b.a==12) then return {{2,'D '..back}},'guard active attack before sweep' end
  end
  return choose(a,b,'vega_drop')
 end
 if mode=='bison_bait_range' or mode=='bison_air_tight' or mode=='bison_patient_space' then
  local d=math.abs(a.x-b.x);local back=a.x<b.x and 'L' or 'R'
  local ready=a.y<=45 and a.a~=10 and a.a~=12 and a.a~=14 and a.a~=20
  if ready then
   if mode=='bison_bait_range' and b.hp>48 and b.y<=50 and d>=60 and d<120 then return {{2,'D '..back}},'guard inside medium bait range' end
   if mode=='bison_air_tight' and b.hp>48 and b.y>50 and b.y<140 and (b.vy or 0)<0 and d>=80 and d<110 and not (b.rebound and b.y>90) then return {{2,back}},'wait for closer first descent' end
   if mode=='bison_patient_space' and b.hp<=48 and b.y<=50 and b.a==14 then return {{2,d<90 and back or 'D '..back}},'space during current Bison hit state' end
  end
  return choose(a,b,'bison_guard_mix')
 end
 if mode=='sagat_sweepguard' or mode=='sagat_far_guard' then
  local d=math.abs(a.x-b.x);local back=a.x<b.x and 'L' or 'R'
  local ready=a.y<=45 and a.a~=10 and a.a~=12 and a.a~=14 and a.a~=20
  local shot=b.anim==269332 or b.anim==269356 or b.anim==269408 or b.anim==269432
  if ready and b.y<=50 and (b.a==10 or b.a==12) then
   if mode=='sagat_sweepguard' and b.hp<=48 and d>=60 and d<108 then return {{2,'D '..back}},'guard active attack before low HP sweep' end
   if mode=='sagat_far_guard' and d>260 and not shot then return {{2,'D '..back}},'guard ongoing far attack' end
  end
  return choose(a,b,'sagat_lowshot')
 end
 local quick_sagat=mode=='sagat_quickdp'
 if quick_sagat then
  if b.hp<=48 and a.y<=45 and b.y<=50 and math.abs(a.x-b.x)<108 then return choose(a,b,'fast_lp') end
  mode='sagat_lowshot'
 end
 if mode=='bison_patient_mix' then return choose(a,b,b.hp<=48 and 'bison_patient_counter' or 'bison_guard_bait') end
 if mode=='bison_patient_counter' then
  if a.y<=45 and b.y<=50 and b.a==14 then
   local back=a.x<b.x and 'L' or 'R';return {{2,'D '..back}},'wait current Bison hit state before counter'
  end
  return choose(a,b,'bison_counter')
 end
 if mode=='bison_guard_mix' then return choose(a,b,b.hp<=48 and 'bison_counter' or 'bison_guard_bait') end
 if mode=='bison_mix' then return choose(a,b,b.hp<=48 and 'bison_counter' or 'bison_bait') end
 if mode=='sagat_mix' then return choose(a,b,b.hp<=48 and 'lp_safe' or 'sagat_shot') end
 if mode=='sagat_range' then
  local close=b.hp<=48 and a.y<=45 and b.y<=50 and math.abs(a.x-b.x)<108
  return choose(a,b,close and 'lp_safe' or 'sagat_shot')
 end
 if mode=='sagat_lowshot' and b.hp<=48 and a.y<=45 and b.y<=50 and math.abs(a.x-b.x)<108 then return choose(a,b,'lp_safe') end
 local d=math.abs(a.x-b.x);local f=a.x<b.x and 'R' or 'L';local back=f=='R' and 'L' or 'R'
 local sagat_reaction=mode=='sagat_react' or (mode=='ryu_fast' or mode=='ryu_close') or mode=='sagat_shot' or mode=='sagat_sweep' or mode=='sagat_finish' or mode=='sagat_lowshot'
 local bison_bait=mode=='bison_guard_bait' or mode=='bison_bait' or mode=='bison_bait_hp'
 local function dp(p)
  local n=(quick_sagat or mode=='guile_pressure' or mode=='guile_safe_fire' or mode=='honda_fast' or (mode=='ryu_fast' or mode=='ryu_close') or mode=='fast_lp' or mode=='vega_drop' or mode=='sagat_counter' or mode=='sagat_far' or mode=='bison_counter' or mode=='bison_heavy' or bison_bait or mode=='jump_dp' or mode=='sagat_jump' or mode=='sagat_preempt' or mode=='sagat_heavy') and 2 or 4
  return {{n,f},{n,'D'},{n,'D '..f..' '..p},{12,back}},'dp'
 end
 local function fire() return {{4,'D'},{4,'D '..f},{4,f..((mode=='guile_pressure' or mode=='slowfire' or (mode=='honda_air' or mode=='honda_fast' or mode=='guile_safe_fire')) and ' LP' or ' HP')},{16,back}},'fire' end
 if a.a==12 or a.a==14 or a.a==20 then return {{4,(b.y>50 and back or 'D '..back)}},'recover guard' end
 if (sagat_reaction or bison_bait) and a.y>45 then
  if d<(late_kick and 85 or (early_kick and 125 or 105)) and math.abs(a.y-b.y)<115 then return {{3,'HK'},{3,back}},'air kick when actually in range' end
  return {{2,back}},'wait airborne approach'
 end
 if a.y>45 then return {{3,'HK'},{5,''}},'air kick' end
 if a.a==10 then return {{4,(b.y>50 and back or 'D '..back)}},'recover guard' end
 if bison_bait then
  if b.y>50 then
   if b.rebound and b.y>90 and d<150 then return {{6,'U'}},'jump after observed air rebound' end
   if (b.vy or 0)<0 and b.y<140 and d<110 then return dp(mode=='bison_bait_hp' and 'HP' or 'MP') end
   return {{2,back}},'guard first airborne approach'
  end
  if d<60 then return dp('LP') end
  if mode=='bison_guard_bait' and (b.a==10 or b.a==12) then return {{2,'D '..back}},'guard visible Bison attack instead of bait' end
  return {{4,'D MK'},{8,'D '..back}},'crouching medium kick bait'
 end
 if sagat_reaction then
  if b.y>50 then
   if (b.vy or 0)<0 and b.y<140 and d<105 then return dp('MP') end
   return {{2,back}},'guard airborne Sagat'
  end
  if d<(mode=='ryu_close' and 80 or 65) then return dp('LP') end
  if mode=='sagat_sweep' and d<108 then return {{4,'D HK'},{12,'D '..back}},'sweep after entering mid range' end
  if mode=='sagat_finish' and b.hp<=12 and d>=150 then return fire() end
  -- Current animation only, visually checked in 1041..1064 (sagat-anim-a.tsv).
  -- Do not follow animation links or read the CPU's future move selection.
  local shot_start=b.anim==269332 or b.anim==269356 or (mode=='sagat_lowshot' and not high_only and (b.anim==269408 or b.anim==269432))
  local generic_react=mode=='sagat_react' or (mode=='ryu_fast' or mode=='ryu_close')
  local jump_signal=not generic_react and shot_start or (generic_react and (b.attack_age or 999)<12)
  if d>=(far_jump and jump_min or 160) and jump_signal then return {{6,'U '..f}},'jump distant visible attack startup' end
  if d>260 then return {{4,f}},'approach shot reaction range' end
  return {{2,'D '..back}},'wait outside punch and guard'
 end
 if mode=='jump_late30' then
  if d<50 and b.y<=50 then return {{3,f..' HP'},{5,back}},'close throw after late kick' end
  if b.y>50 then return {{3,back}},'guard air commitment' end
  return {{30,'U '..f},{4,'HK'},{12,back}},'very late jumping roundhouse'
 end
 if mode=='sagat_preempt' or mode=='sagat_heavy' then
  if b.y>50 then
   if (b.vy or 0)<0 and b.y<140 and d<105 then return dp('MP') end
   return {{2,back}},'wait uppercut descent'
  end
  if d<150 then return dp(mode=='sagat_heavy' and 'HP' or 'LP') end
  return {{4,f}},'approach early DP range'
 end
 if mode=='sagat_jump' then
  if b.y>50 then
   if (b.vy or 0)<0 and b.y<150 and d<110 then return dp('MP') end
   return {{2,back}},'wait uppercut descent'
  end
  if d<90 then return dp('LP') end
  if (b.attack_age or 999)<12 then
   return {{18,'U '..f},{4,'HK'},{18,f}},'jump on visible attack startup'
  end
  if d>240 then return {{4,f}},'close long range' end
  return {{2,'D '..back}},'wait shot startup'
 end
 if mode=='jump_dp' or mode=='jump_safe' then
  if b.y>50 then
   if (b.vy or 0)<0 and b.y<150 and d<110 then return dp('MP') end
   return {{3,back}},'wait rising attack'
  end
  if d<(mode=='jump_safe' and 70 or 90) then return dp('LP') end
  if mode=='jump_safe' then return {{20,'U '..f},{4,'HK'},{24,back}},'late jump kick then landing guard' end
  return {{12,'U '..f},{4,'HK'},{24,f},{4,'D HK'},{8,back}},'jump into close DP range'
 end
 if mode=='sagat_counter' or mode=='sagat_far' or mode=='bison_counter' or mode=='bison_heavy' then
  if b.y>50 then
   if (b.vy or 0)<0 and b.y<140 and d<110 then return dp(mode=='bison_heavy' and 'HP' or 'MP') end
   return {{3,back}},'respect rising uppercut'
  end
  if d<((mode=='bison_counter' or mode=='bison_heavy') and 185 or (mode=='sagat_far' and 155 or 125)) then
   if b.a==10 or b.a==12 or d<65 then return dp(mode=='bison_heavy' and 'HP' or 'MP') end
   return {{3,'D '..back}},'wait ground commitment'
  end
  return {{4,f}},'approach counter range'
 end
 if mode=='vega_drop' or mode=='guile_pressure' or mode=='sagat_guard' then
  if b.y>50 then
   if (b.vy or 0)<0 and b.y<150 and d<110 then return dp('MP') end
   return {{2,back}},'wait descending attack'
  end
  if d<60 then return dp('LP') end
  if d<108 and mode~='sagat_guard' then return {{4,'D HK'},{12,'D '..back}},'sweep' end
  if mode=='guile_pressure' and d>=155 and b.a~=10 and b.a~=12 then return fire() end
  return {{4,'D '..back}},'guard and wait'
 end
 if mode=='jump_pressure' then
  if d<50 and b.y<=50 then return {{3,f..' HP'},{4,back}},'close throw attempt' end
  return {{20,'U '..f},{4,'HK'},{18,f}},'jump pressure without DP diversion'
 end
 if mode=='neutral_jump' then
  return {{20,'U'},{4,'HK'},{18,''}},'neutral jump kick'
 end
 if mode=='claw_guard' then
  if b.y>50 then
   if b.y<150 and d<120 then return {{3,'HP'},{5,back}},'standing fierce anti-air' end
   return {{4,back}},'wait wall approach'
  end
  if d<108 then return {{4,'D HK'},{12,'D '..back}},'sweep' end
  return {{4,'D '..back}},'guard outside claw'
 end
 if mode=='claw' and b.y>50 then
  if b.y<145 and d<145 then return dp('HP') end
  return {{4,back}},'wait wall dive height'
 end
 if mode=='patient' and b.a==12 and b.y>45 and d<130 then return {{4,back}},'block rising special' end
 if mode=='lowfire' and b.a==12 and b.y>45 and d<150 then return {{4,back}},'respect rising special' end
 if (mode=='lp_safe' or mode=='fast_lp') and b.y>50 then
  if d<78 then return dp('LP') end
  return {{4,back}},'wait air approach'
 end
 -- Honda baseline: a distant airborne approach must not start another fireball.
 if (mode=='honda_air' or mode=='honda_fast' or mode=='guile_safe_fire') and b.y>50 and d>=110 then return {{4,back}},'wait Honda air approach' end
 -- Guile fire-old-02: two large hits followed fire started during visible a10.
 if mode=='guile_safe_fire' and b.y<=50 and d>=155 and (b.a==10 or b.a==12) then return {{2,'D '..back}},'wait visible Guile attack before fire' end
 if b.y>50 and d<110 then return dp(mode=='lp' and 'LP' or 'HP') end
 if mode=='lp' or mode=='lp_safe' or mode=='fast_lp' or mode=='claw' then
  if d<60 then return dp('LP') end
  if d<108 then return {{4,'D HK'},{12,'D '..back}},'sweep' end
  return {{6,f}},'approach'
 end
 if mode=='throw' then
  if d<50 then return {{3,f..' HP'},{3,f},{3,'D'},{3,'D '..f..' LP'},{12,back}},'throw attempt and follow-up' end
  return {{6,f}},'approach'
 end
 if mode=='lowfire' then
  if d<48 then return {{3,f..' HP'},{3,f},{3,'D'},{3,'D '..f..' LP'},{12,back}},'throw attempt and follow-up' end
  if d<110 then return {{6,'D MK'},{3,'D '..f},{3,f..' HP'},{16,'D '..back}},'low kick fire' end
  return {{6,f}},'approach'
 end
 if mode=='turtle' or mode=='slowfire' or (mode=='honda_air' or mode=='honda_fast' or mode=='guile_safe_fire') then
  if d<75 then return dp('LP') end
  if d<155 then return {{4,'D '..back}},'hold ground guard' end
  return fire()
 end
 if mode=='dp_rush' then
  if d<85 then return dp('HP') end
  return {{8,f}},'approach'
 end
 if mode=='jump' or mode=='patient' or mode=='late_jump' then
  if d>60 then
   if mode=='late_jump' then return {{24,'U '..f},{4,'HK'},{12,f},{4,'D HK'},{8,back}},'late jump sweep' end
   return {{12,'U '..f},{4,'HK'},{24,f},{4,'D HK'},{8,back}},'jump sweep'
  end
  return {{3,f..' HP'},{4,back}},'throw'
 end
 if d<48 then return {{3,f..' HP'},{4,back}},'throw' end
 if d<108 then return {{4,'D HK'},{12,'D '..back}},'sweep' end
 if mode=='sweep' then return {{6,f}},'approach' end
 return fire()
end
if bot_subscription then bot_subscription:unsubscribe() end
bot_subscription=emu.add_machine_frame_notifier(function()
 if not bot then return end
 botframe=botframe+1;bot.left=bot.left-1
 local a,b=fighter(0),fighter(1)
 if (b.a==10 or b.a==12) and b.a~=bot.previous_enemy.a then bot.enemy_attack_frame=botframe end
 b.attack_age=bot.enemy_attack_frame and (botframe-bot.enemy_attack_frame) or 999
 local dy=b.y-bot.previous_enemy.y
 if b.y<=40 then bot.enemy_rebound=false
 elseif dy>0 and bot.last_dy<0 and bot.previous_enemy.y>50 then bot.enemy_rebound=true end
 b.rebound=bot.enemy_rebound or false
 if dy~=0 then bot.last_dy=dy end
 b.vy=b.y<=40 and 0 or bot.last_dy
 bot.previous_enemy=b
 local timer=mem:read_u8(0xff8ace)
 if timer>0 then bot.hadclock=true end
 if bot.left<=0 or a.char~=4 or b.char~=bot.opponent or a.hp<0 or b.hp<0 or (bot.hadclock and timer==0) then
  release();state();for k,n in pairs(bot.actions) do print(k,n) end
  local log=io.open('training/results.log','a');log:write(string.format('frame=%d mode=%s Ken=%d opponent=%d CPU=%d timer=%d\n',botframe,bot.mode,a.hp,b.char,b.hp,timer));log:close()
  bot=nil;emu.pause();shot();return
 end
 local side=a.x<b.x
 if ((a.a==14 or a.a==20) and a.a~=bot.previous_action) or side~=bot.previous_side then
  bot.q={};bot.qi=1;bot.remain=0;release()
  bot.actions['interrupt on hit or side change']=(bot.actions['interrupt on hit or side change'] or 0)+1
 end
 bot.previous_action=a.a;bot.previous_side=side
 if bot.remain<=0 then
  if bot.qi>#bot.q then
   local q,name=choose(a,b,bot.mode);bot.q=q;bot.qi=1
   bot.actions[name]=(bot.actions[name] or 0)+1
  end
  local step=bot.q[bot.qi];bot.qi=bot.qi+1;bot.remain=step[1]
  release();for k in string.gmatch(step[2] or '', '%S+') do keys[k]:set_value(1) end
 end
 bot.remain=bot.remain-1
end)
state();print('fighter controller ready')
-- Advance transition only until fresh full-health fighters appear.
function next_round(maxframes)
 guard_training_action()
 if bot or job or advance or loadwatch or savewatch then error('controller busy') end
 advance={left=maxframes or 2400,settle=nil}
 apply_training_speed();emu.unpause()
end
if advance_subscription then advance_subscription:unsubscribe() end
advance_subscription=emu.add_machine_frame_notifier(function()
 if not advance then return end
 advance.left=advance.left-1
 local a,b=fighter(0),fighter(1)
 if a.hp==144 and b.hp==144 and a.char==4 and not advance.settle then advance.settle=90 end
 if advance.settle then advance.settle=advance.settle-1 end
 if advance.left<=0 or advance.settle==0 then advance=nil;release();emu.pause();shot();state() end
end)
-- Loading a save state can resume emulation; pause on the first restored frame.
loadwatch=nil
if postloadsub then postloadsub:unsubscribe() end
postloadsub=emu.add_machine_post_load_notifier(function()
 if loadwatch then loadwatch.ready=true end
end)
if loadsub then loadsub:unsubscribe() end
loadsub=emu.add_machine_frame_notifier(function()
 if not loadwatch then return end
 loadwatch.left=loadwatch.left-1
 if loadwatch.ready or loadwatch.left<=0 then
  local ready=loadwatch.ready
  last_restore={confirmed=ready,path=loadwatch.path,token=loadwatch.token}
  loadwatch=nil;emu.pause();release();state();shot()
  print(ready and 'RESTORED AND PAUSED' or 'RESTORE NOT CONFIRMED: frame limit reached')
 end
end)
function restore(path,token)
 guard_training_action()
 if formal_active then error('Formal no-load run: restore is disabled; training must be separate') end
 if bot or job or advance or loadwatch or savewatch then error('controller busy') end
 local file,err=io.open(path,'rb');if not file then error(err) end;file:close()
 release();last_restore=nil;loadwatch={ready=false,left=120,path=path,token=token}
 apply_training_speed();manager.machine:load(path)
end

-- Saving also resumes emulation. Wait for the file, then pause on the next frame.
savewatch=nil
if savesub then savesub:unsubscribe() end
savesub=emu.add_machine_frame_notifier(function()
 if not savewatch then return end
 savewatch.left=savewatch.left-1
 local file=io.open(savewatch.path,'rb')
 if file or savewatch.left<=0 then
  if file then file:close() else print('CHECKPOINT FAILED: '..savewatch.path) end
  savewatch=nil;release();emu.pause();shot()
 end
end)
function checkpoint(path)
 guard_training_action()
 if bot or job or advance or loadwatch or savewatch then error('controller busy') end
 if not manager.machine.paused then error('Pause before saving a checkpoint') end
 local old=io.open(path,'rb');if old then old:close();error('Checkpoint exists; choose a new filename') end
 release();savewatch={path=path,left=120}
 apply_training_speed();manager.machine:save(path)
end

function stop()
 if train_busy and train_busy() then return train_abort() end
 if play_busy and play_busy() then return play_abort() end
 bot=nil;job=nil;advance=nil;loadwatch=nil;savewatch=nil;release()
 normal_speed()
 emu.pause();shot()
end
