-- Training-only RPC. Synchronous stepping and save-state resets are NOT formal play.
local m=manager.machine
local Core=assert(loadfile('training/runtime/play_core.lua'))()
local IO=assert(loadfile('training/runtime/status_io.lua'))()
local Speed=assert(loadfile('training/runtime/speed.lua'))()
local Actions=assert(loadfile('training/runtime/rl_actions.lua'))()
local level=7-astra_difficulty_bits
local modes={[0]='c19_ryu_fire_fast',[1]='c22_honda_guarded_descent',[2]='c19_blanka_low_lead_hold',[3]='c19_guile_fire3',[5]='c22_chunli_timeout_guard',[6]='c19_zangief_fire_fast',[7]='c22v4_dhalsim_far_ground_start_guard',[8]='c22v3_bison_ground_throw',[9]='c19_sagat_low_wait',[10]='c19_balrog_sweep95',[11]='c19_vega_low_descent_wait'}
local function json(v)
 local t=type(v)
 if t=='nil' then return 'null' end
 if t=='number' or t=='boolean' then return tostring(v) end
 if t=='string' then return '"'..v:gsub('[%z\1-\31\\"]',function(c) return string.format('\\u%04x',c:byte()) end)..'"' end
 local out={}
 for k,x in pairs(v) do out[#out+1]=json(tostring(k))..':'..json(x) end
 return '{'..table.concat(out,',')..'}'
end
local function snapshot()
 local s={p1=fighter(0),p2=fighter(1)}
 local raw=mem:read_u8(0xff8ace);local hi,lo=raw>>4,raw&15
 s.timer=hi<=9 and lo<=9 and hi*10+lo or -1
 for i=0,1 do
  local p=s[i==0 and 'p1' or 'p2'];local base=0xff83c6+i*0x300
  p.wins=mem:read_u8(base+0x290)
  p.displayed_hp=mem:read_i16(base+0x1bc);p.timeout_hp=mem:read_i16(base+0x164)
 end
 s.effective_difficulty=astra_difficulty.check(level).effective_difficulty
 return s
end
local pending,core,baseline,loaded,frames,episode
local held=''
local function input(keys_text)
 held=keys_text or ''
 release()
 for k in (keys_text or ''):gmatch('%S+') do assert(keys[k],k):set_value(1) end
end
local function answer(extra)
 local id=assert(pending).id
 release();emu.pause()
 local result=extra or {};result.id=id;result.state=snapshot()
 result.frames=frames or 0;result.episode=episode or 0;result.training_only=true
 if core then
  result.phase=core.phase
  if #core.rounds>0 then result.outcome=core.rounds[1].outcome;result.done=true;result.native_round=core.rounds[1] end
 end
 pending=nil
 IO.publish(string.format('training/rl-reply-%08d.json',id),json(result)..'\n')
end
local function fail(err)
 release();emu.pause()
 if pending then
  local id=pending.id;pending=nil
  IO.publish(string.format('training/rl-reply-%08d.json',id),json({id=id,error=tostring(err)})..'\n')
 end
 IO.publish('training/rl-error.json',json({error=tostring(err)})..'\n')
end
-- Only this controller can own the copied runtime after attaching.
assert(m.paused and not (job or bot or advance or loadwatch or savewatch or play_busy()))
play_bridge_generation=play_bridge_generation+1
train_busy=function() return true end
rl_load_subscription=emu.add_machine_post_load_notifier(function() if pending and pending.op=='reset' then loaded=true end end)
rl_frame_subscription=emu.add_machine_frame_notifier(function()
 if not pending then return end
 local ok,err=xpcall(function()
  Speed.check(m.video,'fast')
  if pending.op=='reset' then
   if not loaded then return end
   pending.refresh=pending.refresh-1
   if pending.refresh>0 then return end
   local s=snapshot();frames=0;episode=(episode or 0)+1
   assert(Core.opening(s,s.p2.char,{0,0}),'Reset did not restore full-health Ken R1')
   core=Core.new({mode=assert(modes[s.p2.char]),opponent=s.p2.char,choose=baseline and choose or function() return {{1,''}} end,
    timeout_guard=baseline and (s.p2.char==1 or s.p2.char==5 or s.p2.char==8 or s.p2.char==11),lead=baseline and s.p2.char==8 and 2 or 0},s)
   answer({reset_confirmed=true});return
  end
  frames=frames+1;pending.elapsed=pending.elapsed+1
  local s=snapshot();local effects=core:tick(s)
  if effects.terminal and not effects.terminal.valid then error(effects.terminal.reason) end
  if #core.rounds>0 then answer();return end
  if core.phase~='fighting' then
   -- Continue native result maturation without giving the learner empty steps.
   input(baseline and effects.input or '')
   return
  end
  if baseline and effects.input~=nil then input(effects.input) end
  if pending.elapsed>=Actions.frames then answer({done=false});return end
  if not baseline then input(Actions.keys(pending.action,pending.elapsed,s,pending.forward)) end
 end,debug.traceback)
 if not ok then fail(err) end
end)
rl_rpc_subscription=emu.register_frame_done(function()
 if pending then return end
 local f=io.open('training/rl-request.txt','rb');if not f then return end
 local text=f:read('*a');f:close();assert(os.remove('training/rl-request.txt'))
 local id,op,arg,flag=text:match('^(%d+) (%a+) (%d+) (%d+)\n$')
 local ok,err=xpcall(function()
  assert(id,'Malformed RL request');id=tonumber(id);arg=tonumber(arg);flag=tonumber(flag)
  pending={id=id,op=op,elapsed=0}
  assert(m.paused,'RPC requires paused training boundary')
  assert(not (job or bot or advance or loadwatch or savewatch or play_busy()),'Foreign controller active')
  Speed.apply(m.video,'fast')
  if op=='reset' then
   assert(arg<=12,'Invalid reset parameters')
   local checkpoints=assert(loadfile('training/runtime/rl_checkpoint.lua'))()
   if type(checkpoints)=='string' then checkpoints={checkpoints} end
   local path=assert(checkpoints[math.floor(flag/2)+1],'Invalid checkpoint index')
   baseline=flag%2==1;core=nil;held='';loaded=false;pending.refresh=2+arg
   release();m:load(path);emu.unpause()
  elseif op=='step' then
   assert(core and #core.rounds==0,'Reset before stepping a terminal episode')
   assert(arg<Actions.count and flag==0,'Invalid action')
   pending.action=arg;pending.forward=fighter(0).x<fighter(1).x and 'R' or 'L'
   if baseline then input(held) else input(Actions.keys(arg,0,snapshot(),pending.forward)) end
   emu.unpause()
  else error('Unknown RL operation') end
 end,debug.traceback)
 if not ok then fail(err) end
end)
