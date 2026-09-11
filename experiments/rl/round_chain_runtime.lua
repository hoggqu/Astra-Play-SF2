-- Candidate round-chain training only. One native Core per whole opponent match.
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
local pending,core,model,history,obs,state,loaded,episode_start
local frames,episode,episode_return,episode_steps=0,0,0,0
local current_checkpoint,current_lead,last_native_time,last_observed,settlement_trace
local episode_actions,recorded_rounds,active_action,episode_first_frame,round_closed,match_actions,match_opening
local metrics={loads=0,natural_rounds=0,fighting_frames=0,settling_frames=0,between_frames=0,matches=0}
local function stack(s,reset)
 local f=NN.features(s)
 if reset then history={f,f,f,f} else table.remove(history,1);history[#history+1]=f end
 local out={};for _,row in ipairs(history) do for _,v in ipairs(row) do out[#out+1]=v end end
 return out
end
local function partial()
 return not round_closed and episode_steps>0 and {episode=episode,checkpoint=current_checkpoint,lead=current_lead,
  opponent=state.p2.char,round=core.round,steps=episode_steps,frames=frames-episode_first_frame,
  ['return']=episode_return,final_state=state} or false
end
local function answer()
 assert(not active_action,'Cannot pause halfway through an action')
 release();emu.pause()
 local result={id=pending.id,training_only=true,round_chain=true,observation=obs,state=state,
  episode_start=episode_start,transitions=pending.transitions,episodes=pending.episodes,
  model_sha256=model and model.model_sha256 or false,checkpoint=current_checkpoint,lead=current_lead,
  partial_episode=partial(),chain_metrics=metrics,trace=pending.trace}
 pending=nil
 IO.publish(string.format('training/rl-batch-reply-%08d.json',result.id),json(result)..'\n')
end
local function fail(err)
 release();emu.pause()
 if core and settlement_trace then
  pcall(IO.publish,'training/rl-batch-unresolved-settlement.json',json({error=tostring(err),
   episode=episode,checkpoint=current_checkpoint,lead=current_lead,frame=frames,
   current_state=last_observed,round_stop=core.round_stop,terminal_frame=core.terminal_frame,
   score=core.score,phase=core.phase,trace=settlement_trace,
   opening=match_opening,match_actions=match_actions,rounds=core.rounds})..'\n')
 end
 if state and episode_steps>0 then
  local row=partial();row.reason='controller_error';row.error=tostring(err)
  pcall(IO.append,'training/rl-batch-partials.jsonl',json(row)..'\n')
 end
 if pending then
  local id=pending.id;pending=nil
  IO.publish(string.format('training/rl-batch-reply-%08d.json',id),json({id=id,error=tostring(err)})..'\n')
 end
 IO.publish('training/rl-batch-error.json',json({error=tostring(err)})..'\n')
end
local function reset(choice)
 assert(not active_action,'Cannot load during an active action')
 assert(type(choice)=='table' and choice.lead>=0 and choice.lead<=12 and choice.lead%1==0)
 local path=assert(checkpoints[choice.checkpoint+1],'Bad checkpoint index')
 current_checkpoint=choice.checkpoint;current_lead=choice.lead
 loaded=false;pending.refresh=2+choice.lead;pending.resetting=true
 metrics.loads=metrics.loads+1;release();m:load(path)
end
local function begin_episode(s)
 assert(not active_action,'Previous round has an unfinished action')
 round_closed=false;episode=episode+1;episode_return=0;episode_steps=0;episode_actions={}
 episode_first_frame=frames;state=s;obs=stack(s,true);episode_start=true;settlement_trace=nil
end
local function finish(s,row)
 local a=assert(active_action,'Missing terminal/decision action')
 local outcome=row and row.outcome or nil
 local r=reward(a.before,row and assert(row.settled) or s,outcome)
 pending.transitions[#pending.transitions+1]={observation=a.observation,action=a.action,reward=r,
  done=row~=nil,episode_start=a.episode_start,logprob=a.logprob,state=s,frames=frames,
  round=core.round,episode=episode}
 episode_return=episode_return+r;episode_steps=episode_steps+1;active_action=nil
 state=s;episode_start=false
 if row then
  round_closed=true
  local result={episode=episode,round=row.round,outcome=outcome,opponent=s.p2.char,
   checkpoint=current_checkpoint,lead=current_lead,frames=frames-episode_first_frame,
   match_frame=frames,steps=episode_steps,['return']=episode_return,final_state=row.settled,
   confirmation_state=row.confirmation,native_round=row}
  pending.episodes[#pending.episodes+1]=result
  IO.append('training/rl-batch-episodes.jsonl',json(result)..'\n')
  if row.confirmation then
   IO.publish(string.format('training/rl-draw-action-plan-%05d.json',episode),json({episode=episode,
    checkpoint=current_checkpoint,lead=current_lead,actions=episode_actions,
    settlement_trace=settlement_trace,native_round=row,round_chain=true})..'\n')
  end
 else obs=stack(s,false) end
end
local function select_action()
 assert(not active_action)
 local index=#pending.transitions+1
 local action,logprob
 if pending.actions then action=assert(pending.actions[index]);logprob=0
 else
  assert(model,'Model required')
  local _,logits=NN.predict(model,obs)
  local maximum=math.max(table.unpack(logits));local total=0;local weights={}
  for i,x in ipairs(logits) do weights[i]=math.exp(x-maximum);total=total+weights[i] end
  local u=assert(pending.uniforms[index]);assert(u>=0 and u<1)
  local acc=0;action=#weights-1
  for i,w in ipairs(weights) do acc=acc+w/total;if u<acc then action=i-1;break end end
  logprob=logits[action+1]-maximum-math.log(total)
 end
 assert(action>=0 and action<Actions.count and action%1==0)
 active_action={action=action,logprob=logprob,before=state,observation=obs,episode_start=episode_start}
 episode_actions[#episode_actions+1]=action
 match_actions[#match_actions+1]={action=action,frame=frames,round=core.round}
 local seq={};local forward=state.p1.x<state.p2.x and 'R' or 'L'
 for frame=0,Actions.frames-1 do seq[#seq+1]={1,Actions.keys(action,frame,state,forward)} end
 return seq
end
local function neutral_sequence()
 local seq={};for i=1,Actions.frames do seq[i]={1,''} end;return seq
end
local function choose(a,b,mode,s,reset_history)
 if reset_history then
  begin_episode(s);metrics.natural_rounds=metrics.natural_rounds+1
 else finish(s,nil) end
 if #pending.transitions>=pending.count then pending.boundary=true;return neutral_sequence() end
 return select_action()
end
local function start_action()
 core.rl_round=core.round;core.rl_elapsed=0;core.rl_sequence=select_action()
 pending.deferred=core.rl_sequence[1][2]
end
assert(m.paused and not (job or bot or advance or loadwatch or savewatch or play_busy()))
play_bridge_generation=play_bridge_generation+1
train_busy=function() return true end
rl_batch_load_subscription=emu.add_machine_post_load_notifier(function() if pending and pending.resetting then loaded=true end end)
rl_batch_frame_subscription=emu.add_machine_frame_notifier(function()
 if not pending then return end
 local now=m.time:as_double();if now==last_native_time then return end
 local delta=last_native_time and now-last_native_time or nil;last_native_time=now
 local ok,err=xpcall(function()
  if delta and not pending.resetting then
   assert(math.abs(delta-m.screens[':screen'].frame_period)<1e-7,'Skipped native sampling frame')
  end
  Speed.check(m.video,'fast')
  if pending.resetting then
   if not loaded then return end
   pending.refresh=pending.refresh-1;if pending.refresh>0 then return end
   state=snapshot();frames=0;recorded_rounds=0;last_observed=state;match_actions={};match_opening=state
   local opponent=assert(expected_opponents[current_checkpoint+1])
   assert(Core.opening(state,opponent,{0,0}) and state.p1.char==4,'Unexpected checkpoint opening')
   core=Core.new({mode='rl_round_chain',opponent=opponent,timeout_guard=false,lead=0,choose=choose},state)
   begin_episode(state);pending.resetting=false
   if pending.op=='reset' or #pending.transitions>=pending.count then answer() else start_action() end
   return
  end
  frames=frames+1;local s=snapshot();last_observed=s
  local phase=core.phase;local key=phase..'_frames';if metrics[key] then metrics[key]=metrics[key]+1 end
  local effects=core:tick(s)
  if core.round_stop and phase~='between' then
   settlement_trace=settlement_trace or {};settlement_trace[#settlement_trace+1]={frame=frames,state=s}
  end
  if effects.terminal and not effects.terminal.valid then error(effects.terminal.reason) end
  if #core.rounds>recorded_rounds then
   assert(#core.rounds==recorded_rounds+1,'Multiple unconsumed round results')
   finish(s,core.rounds[#core.rounds]);recorded_rounds=#core.rounds
  end
  if pending.capture_trace then
   pending.trace=pending.trace or {}
   pending.trace[#pending.trace+1]={frame=frames,state=s,phase=core.phase,round=core.round,
    input=effects.input,deferred_input=effects.rl_deferred_input,events=effects.events,loads=metrics.loads}
  end
  if core.phase=='complete' then
   assert(not active_action and effects.terminal and effects.terminal.valid)
   metrics.matches=metrics.matches+1
   IO.publish(string.format('training/rl-chain-match-%05d.json',metrics.matches),json({
    checkpoint=current_checkpoint,lead=current_lead,opening=match_opening,frames=frames,
    rounds=core.rounds,actions=match_actions,round_chain=true,formal_clear=false})..'\n')
   pending.resets_used=pending.resets_used+1
   reset(assert(pending.resets[pending.resets_used],'Reset plan exhausted'));return
  end
  if pending.boundary then pending.deferred=nil;answer();return end
  if effects.input~=nil then input(effects.input) end
  pending.deferred=effects.rl_deferred_input
  if core.phase~='fighting' then input('') end
 end,debug.traceback)
 if not ok then fail(err) end
end)
rl_batch_rpc_subscription=emu.register_frame_done(function()
 if pending then
  if pending.deferred~=nil and (not pending.deferred_after or m.time:as_double()>pending.deferred_after) then
   local text=pending.deferred;pending.deferred=nil;pending.deferred_after=nil;input(text)
  end
  return
 end
 local f=io.open('training/rl-batch-request.lua','rb');if not f then return end
 local text=f:read('*a');f:close();assert(os.remove('training/rl-batch-request.lua'))
 local ok,err=xpcall(function()
  local request=assert(load(text,'batch-request','t',{}))()
  assert(type(request)=='table' and type(request.id)=='number')
  pending=request;pending.transitions={};pending.episodes={};pending.resets_used=0
  assert(m.paused and not (job or bot or advance or loadwatch or savewatch or play_busy()))
  Speed.apply(m.video,'fast')
  if pending.model then model=pending.model;pending.model=nil end
  if pending.op=='reset' then reset(pending.reset)
  else
   assert(pending.count>=1 and pending.count<=256 and pending.count%1==0)
   assert(pending.resets and #pending.resets>=pending.count)
   if pending.op=='reset_rollout' then reset(pending.reset)
   else
    assert(pending.op=='rollout' and core and core.phase=='fighting' and not active_action)
    start_action()
    pending.deferred_after=m.time:as_double()
   end
  end
  emu.unpause()
 end,debug.traceback)
 if not ok then fail(err) end
end)
