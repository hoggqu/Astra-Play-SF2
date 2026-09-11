-- Optional training-only batch sampler. One policy version per rollout request.
local m=manager.machine
local Core=assert(loadfile('training/runtime/play_core.lua'))()
Core=assert(loadfile('training/runtime/rl_settlement.lua'))()(Core)
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
local pending,core,model,history,obs,state,loaded,episode_start
local frames,episode,episode_return,episode_steps=0,0,0,0
local current_checkpoint,current_lead
local last_native_time
-- Preserve unknown native settlements without guessing an outcome or emitting
-- a transition. This is memory-only until an error, outside the action policy.
local last_observed,settlement_trace
local episode_actions
local function stack(s,reset)
 local f=NN.features(s)
 if reset then history={f,f,f,f} else table.remove(history,1);history[#history+1]=f end
 local out={};for _,row in ipairs(history) do for _,v in ipairs(row) do out[#out+1]=v end end
 return out
end
local function answer(extra)
 release();emu.pause()
 local result=extra or {};result.id=pending.id;result.training_only=true
 result.observation=obs;result.state=state;result.episode_start=episode_start
 result.transitions=pending.transitions;result.episodes=pending.episodes
 result.model_sha256=model and model.model_sha256 or false
 result.checkpoint=current_checkpoint;result.lead=current_lead
 result.partial_episode=episode_steps>0 and {episode=episode,checkpoint=current_checkpoint,lead=current_lead,
  opponent=state.p2.char,steps=episode_steps,frames=frames,['return']=episode_return,final_state=state} or false
 pending=nil
 IO.publish(string.format('training/rl-batch-reply-%08d.json',result.id),json(result)..'\n')
end
local function fail(err)
 release();emu.pause()
 if core and settlement_trace then
  pcall(IO.publish,'training/rl-batch-unresolved-settlement.json',json({error=tostring(err),
   episode=episode,checkpoint=current_checkpoint,lead=current_lead,frame=frames,
   current_state=last_observed,round_stop=core.round_stop,terminal_frame=core.terminal_frame,
   score=core.score,phase=core.phase,time_draw_latch=core.time_draw_latch,
   time_draw_ko_seen=core.time_draw_ko_seen,trace=settlement_trace})..'\n')
 end
 if state and episode_steps>0 then
  pcall(IO.append,'training/rl-batch-partials.jsonl',json({episode=episode,checkpoint=current_checkpoint,
   lead=current_lead,opponent=state.p2.char,steps=episode_steps,frames=frames,['return']=episode_return,
   last_complete_state=state,reason='controller_error',error=tostring(err)})..'\n')
 end
 if pending then
  local id=pending.id;pending=nil
  IO.publish(string.format('training/rl-batch-reply-%08d.json',id),json({id=id,error=tostring(err)})..'\n')
 end
 IO.publish('training/rl-batch-error.json',json({error=tostring(err)})..'\n')
end
local function reset(choice)
 assert(type(choice)=='table' and choice.lead>=0 and choice.lead<=12 and choice.lead%1==0)
 local path=assert(checkpoints[choice.checkpoint+1],'Bad checkpoint index')
 current_checkpoint=choice.checkpoint;current_lead=choice.lead
 loaded=false;pending.refresh=2+choice.lead;pending.resetting=true
 release();m:load(path)
end
local function make_native_core(s)
 local Native=assert(loadfile('training/runtime/rl_native_core.lua'))()
 pending.native_action_index=#pending.transitions+1
 return Native.new({mode='rl_native_harness',opponent=s.p2.char,timeout_guard=false,lead=0,
  choose=function(a,b,mode,observation,reset_history)
   local index=pending.native_action_index;pending.native_action_index=index+1
   local action=pending.actions[index] or 0;local seq={};local forward=a.x<b.x and 'R' or 'L'
   for frame=0,11 do seq[#seq+1]={1,Actions.keys(action,frame,observation,forward)} end
   return seq
  end},s)
end
local function decide()
 local index=#pending.transitions+1
 local action,logprob
 if pending.actions then action=assert(pending.actions[index]);logprob=0
 else
  assert(model,'A model is required before stochastic sampling')
  local _,logits=NN.predict(model,obs)
  local maximum=math.max(table.unpack(logits));local total=0;local weights={}
  for i,x in ipairs(logits) do weights[i]=math.exp(x-maximum);total=total+weights[i] end
  local u=assert(pending.uniforms[index]);assert(u>=0 and u<1)
  local acc=0;action=#weights-1
  for i,w in ipairs(weights) do acc=acc+w/total;if u<acc then action=i-1;break end end
  logprob=logits[action+1]-maximum-math.log(total)
 end
 assert(action>=0 and action<Actions.count and action%1==0)
 pending.action=action;pending.logprob=logprob;pending.before=state;pending.before_obs=obs
 episode_actions[#episode_actions+1]=action
 pending.before_start=episode_start;pending.elapsed=0
 pending.forward=state.p1.x<state.p2.x and 'R' or 'L'
 if not pending.native_reference then release();pending.start_input=true end
end
local function continue_or_answer()
 if pending.op=='reset' or #pending.transitions>=pending.count then answer()
 else
  decide()
  if pending.native_needs_prime then pending.native_needs_prime=nil;pending.native_deferred=core:rl_prime(state) end
 end
end
assert(m.paused and not (job or bot or advance or loadwatch or savewatch or play_busy()))
play_bridge_generation=play_bridge_generation+1
train_busy=function() return true end
rl_batch_load_subscription=emu.add_machine_post_load_notifier(function() if pending and pending.resetting then loaded=true end end)
rl_batch_frame_subscription=emu.add_machine_frame_notifier(function()
 if not pending or pending.start_input then return end
 local now=m.time:as_double()
 if now==last_native_time then return end
 local native_delta=last_native_time and now-last_native_time or nil
 last_native_time=now
 local ok,err=xpcall(function()
  if native_delta and not pending.resetting then
   assert(math.abs(native_delta-m.screens[':screen'].frame_period)<1e-7,'Skipped native sampling frame')
  end
  Speed.check(m.video,'fast')
  if pending.resetting then
   if not loaded then return end
   pending.refresh=pending.refresh-1;if pending.refresh>0 then return end
   state=snapshot();frames=0;episode=episode+1;episode_return=0;episode_steps=0
   last_observed=state;settlement_trace=nil
   episode_actions={}
   local opponent=assert(expected_opponents[current_checkpoint+1],'Missing checkpoint actor metadata')
   assert(Core.opening(state,opponent,{0,0}) and state.p1.char==4 and state.p2.char==opponent,
    'Batch reset did not restore expected full-health Ken R1 opponent')
   if pending.native_reference then core=make_native_core(state);pending.native_needs_prime=true
   else core=Core.new({mode='rl_batch',opponent=state.p2.char,choose=function() return {{1,''}} end,timeout_guard=false,lead=0},state) end
   obs=stack(state,true);episode_start=true;pending.resetting=false
   continue_or_answer();return
  end
  frames=frames+1;pending.elapsed=pending.elapsed+1
  local s=snapshot();last_observed=s
  local effects=core:tick(s)
  if core.round_stop then
   settlement_trace=settlement_trace or {}
   settlement_trace[#settlement_trace+1]={frame=frames,state=s}
  end
  if pending.native_reference then
   if effects.input~=nil then input(effects.input) end
   pending.native_deferred=effects.rl_deferred_input
  end
  if effects.terminal and not effects.terminal.valid then error(effects.terminal.reason) end
  local done=#core.rounds>0
  if not done and core.phase~='fighting' then input('');return end
  if done or pending.elapsed>=Actions.frames then
   local outcome=done and core.rounds[1].outcome or nil
   -- A confirmed draw may be accepted at the NEXT round's opening. Its raw
   -- settled snapshot belongs to the previous round; new 144 HP is not damage.
   local reward_state=done and core.rounds[1].settled or s
   local r=reward(pending.before,reward_state,outcome)
   pending.transitions[#pending.transitions+1]={observation=pending.before_obs,action=pending.action,reward=r,
    done=done,episode_start=pending.before_start,logprob=pending.logprob,state=s,frames=frames}
   episode_return=episode_return+r;episode_steps=episode_steps+1
   state=s;obs=stack(s,false);episode_start=false
   if done then
    local row={episode=episode,outcome=outcome,opponent=s.p2.char,checkpoint=current_checkpoint,lead=current_lead,
     frames=frames,steps=episode_steps,['return']=episode_return,final_state=reward_state,
     confirmation_state=core.rounds[1].confirmation,native_round=core.rounds[1]}
    if core.rounds[1].confirmation then
     IO.publish(string.format('training/rl-draw-action-plan-%05d.json',episode),json({episode=episode,
      checkpoint=current_checkpoint,lead=current_lead,actions=episode_actions,
      settlement_trace=settlement_trace,native_round=core.rounds[1]})..'\n')
    end
    pending.episodes[#pending.episodes+1]=row
    -- Persist mature results before the next state load, even if this batch later fails.
    IO.append('training/rl-batch-episodes.jsonl',json(row)..'\n')
    pending.resets_used=pending.resets_used+1
    reset(assert(pending.resets[pending.resets_used],'Reset plan exhausted'));return
   end
   continue_or_answer();return
  end
  if not pending.native_reference then input(Actions.keys(pending.action,pending.elapsed,s,pending.forward)) end
 end,debug.traceback)
 if not ok then fail(err) end
end)
rl_batch_rpc_subscription=emu.register_frame_done(function()
 if pending then
  if pending.native_deferred~=nil then local text=pending.native_deferred;pending.native_deferred=nil;input(text) end
  if pending.start_input then
   pending.start_input=nil;input(Actions.keys(pending.action,0,state,pending.forward))
  end
  return
 end
 local f=io.open('training/rl-batch-request.lua','rb');if not f then return end
 local text=f:read('*a');f:close();assert(os.remove('training/rl-batch-request.lua'))
 local ok,err=xpcall(function()
  local request=assert(load(text,'batch-request','t',{}))()
  assert(type(request)=='table' and type(request.id)=='number','Bad batch request')
  pending=request;pending.transitions={};pending.episodes={};pending.resets_used=0
  assert(m.paused,'Batch requires paused training boundary')
  assert(not (job or bot or advance or loadwatch or savewatch or play_busy()),'Foreign controller active')
  Speed.apply(m.video,'fast')
  if pending.model then model=pending.model;pending.model=nil end
  if pending.op=='reset_rollout' then
   assert(pending.count>=1 and pending.count<=256 and pending.resets and #pending.resets>=pending.count)
   pending.op='rollout';reset(pending.reset)
  elseif pending.op=='reset' then reset(pending.reset)
  elseif pending.op=='rollout' then
   assert(core and #core.rounds==0,'Reset before batch sampling')
   assert(pending.count>=1 and pending.count<=256 and pending.count%1==0,'Invalid batch size')
   assert(pending.resets and #pending.resets>=pending.count,'Missing bounded reset schedule')
   decide()
   if pending.native_reference then
    assert(pending.actions and episode_steps==0,'Native harness requires forced actions at fresh reset')
    core=make_native_core(state);input(core:rl_prime(state))
   end
  else error('Unknown batch operation') end
  if pending.start_input then pending.start_input=nil;input(Actions.keys(pending.action,0,state,pending.forward)) end
  emu.unpause()
 end,debug.traceback)
 if not ok then fail(err) end
end)
