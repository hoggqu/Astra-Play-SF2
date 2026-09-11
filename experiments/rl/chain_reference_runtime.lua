-- Training-only independent forced-input native deployment reference. One initial load.
local m=manager.machine
local Core=assert(loadfile('training/runtime/rl_native_core.lua'))()
local IO=assert(loadfile('training/runtime/status_io.lua'))()
local Speed=assert(loadfile('training/runtime/speed.lua'))()
local Actions=assert(loadfile('training/runtime/rl_actions.lua'))()
local NN=assert(loadfile('training/runtime/rl_nn.lua'))()
local checkpoints=assert(loadfile('training/runtime/rl_checkpoint.lua'))()
if type(checkpoints)=='string' then checkpoints={checkpoints} end
local expected_opponents=assert(loadfile('training/runtime/rl_batch_checkpoints.lua'))()
local level=7-astra_difficulty_bits
local function json(v)
 local t=type(v)
 if t=='nil' then return 'null' elseif t=='number' or t=='boolean' then return tostring(v)
 elseif t=='string' then return '"'..v:gsub('[%z\1-\31\\"]',function(c) return string.format('\\u%04x',c:byte()) end)..'"' end
 local out={}
 if #v>0 then for _,x in ipairs(v) do out[#out+1]=json(x) end;return '['..table.concat(out,',')..']' end
 for k,x in pairs(v) do out[#out+1]=json(tostring(k))..':'..json(x) end
 return '{'..table.concat(out,',')..'}'
end
local function snapshot()
 local s={p1=fighter(0),p2=fighter(1)}
 local raw=mem:read_u8(0xff8ace);local hi,lo=raw>>4,raw&15
 s.timer=hi<=9 and lo<=9 and hi*10+lo or -1
 for i=0,1 do
  local p=s[i==0 and 'p1' or 'p2'];local base=0xff83c6+i*0x300
  p.wins=mem:read_u8(base+0x290);p.displayed_hp=mem:read_i16(base+0x1bc);p.timeout_hp=mem:read_i16(base+0x164)
 end
 s.emulated_seconds=m.time:as_double();s.native_frame_period=m.screens[':screen'].frame_period
 s.effective_difficulty=astra_difficulty.check(level).effective_difficulty
 return s
end
local function input(text)
 release();for k in (text or ''):gmatch('%S+') do assert(k~='C' and k~='S' and keys[k]):set_value(1) end
end
local function hp(s,key) return math.max(0,math.min(144,s[key].hp)) end
local function reward(before,after,outcome)
 return .25*((hp(before,'p2')-hp(after,'p2'))-(hp(before,'p1')-hp(after,'p1')))/144
  + (outcome=='win' and 1 or outcome=='loss' and -1 or 0)
end
local pending,core,loaded,last_time,frames,opening,decisions,index
local function fail(err)
 release();emu.pause()
 local id=pending and pending.id or -1;pending=nil
 IO.publish(string.format('training/rl-batch-reply-%08d.json',id),json({id=id,error=tostring(err)})..'\n')
end
local function choose(a,b,mode,s,reset_history)
 local planned=assert(pending.actions[index],'Reference action plan exhausted')
 assert(planned.frame==frames and planned.round==core.round,'Reference action boundary differs')
 index=index+1
 decisions[#decisions+1]={frame=frames,round=core.round,action=planned.action,state=s,reset_history=reset_history}
 local seq={};local forward=a.x<b.x and 'R' or 'L'
 for frame=0,Actions.frames-1 do seq[#seq+1]={1,Actions.keys(planned.action,frame,s,forward)} end
 return seq
end
assert(m.paused and not (job or bot or advance or loadwatch or savewatch or play_busy()))
play_bridge_generation=play_bridge_generation+1;train_busy=function() return true end
rl_batch_load_subscription=emu.add_machine_post_load_notifier(function() if pending then loaded=true end end)
rl_batch_frame_subscription=emu.add_machine_frame_notifier(function()
 if not pending then return end
 local now=m.time:as_double();if now==last_time then return end
 local delta=last_time and now-last_time or nil;last_time=now
 local ok,err=xpcall(function()
  if delta and not pending.resetting then assert(math.abs(delta-m.screens[':screen'].frame_period)<1e-7) end
  Speed.check(m.video,'fast')
  if pending.resetting then
   if not loaded then return end
   pending.refresh=pending.refresh-1;if pending.refresh>0 then return end
   opening=snapshot();frames=0;decisions={};index=1
   local opponent=assert(expected_opponents[pending.reset.checkpoint+1])
   assert(Core.opening(opening,opponent,{0,0}) and opening.p1.char==4)
   core=Core.new({mode='rl_chain_reference',opponent=opponent,choose=choose,timeout_guard=false,lead=0},opening)
   pending.resetting=false;pending.deferred=core:rl_prime(opening);return
  end
  frames=frames+1;local s=snapshot();local effects=core:tick(s)
  assert(not m.paused,'Reference paused during play')
  if effects.terminal and not effects.terminal.valid then error(effects.terminal.reason) end
  pending.trace[#pending.trace+1]={frame=frames,state=s,phase=core.phase,round=core.round,events=effects.events}
  if core.phase=='complete' then
   assert(frames==pending.frames,'Reference match ended at a different frame')
   assert(index==#pending.actions+1,'Reference did not consume complete action plan')
   release();emu.pause()
   local result={id=pending.id,training_only=true,round_chain=true,state=s,observation={},episode_start=true,
    transitions={},episodes={},trace=pending.trace,rounds=core.rounds,decisions=decisions,
    opening=opening,initial_loads=1,in_play_pauses=0,chain_metrics={}}
   pending=nil
   IO.publish(string.format('training/rl-batch-reply-%08d.json',result.id),json(result)..'\n');return
  end
  assert(frames<=pending.frames,'Reference failed to end at planned frame')
  if effects.input~=nil then input(effects.input) end
  pending.deferred=effects.rl_deferred_input
 end,debug.traceback)
 if not ok then fail(err) end
end)
rl_batch_rpc_subscription=emu.register_frame_done(function()
 if pending then
  if pending.deferred~=nil then local text=pending.deferred;pending.deferred=nil;input(text) end
  return
 end
 local f=io.open('training/rl-batch-request.lua','rb');if not f then return end
 local text=f:read('*a');f:close();assert(os.remove('training/rl-batch-request.lua'))
 local ok,err=xpcall(function()
  pending=assert(load(text,'chain-reference','t',{}))()
  assert(m.paused and pending.op=='reference_match' and #pending.actions>0 and pending.frames<=80000)
  assert(pending.reset.lead>=0 and pending.reset.lead<=12)
  Speed.apply(m.video,'fast')
  loaded=false;pending.resetting=true;pending.refresh=2+pending.reset.lead;pending.trace={}
  release();m:load(assert(checkpoints[pending.reset.checkpoint+1]));emu.unpause()
 end,debug.traceback)
 if not ok then fail(err) end
end)
