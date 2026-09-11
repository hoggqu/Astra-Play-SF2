-- Frame-timed controller: regular arcade inputs only, no game-memory changes.
if train_busy and train_busy() then error('Cannot reload modules during a training match') end
if play_busy and play_busy() then error('Cannot reload training controls during continuous play') end
training_continuous_guard_version=1
training_train_guard_version=1
function guard_training_action()
 if train_busy and train_busy() then error('Training match is running; use train_abort() to stop') end
 if play_busy and play_busy() then error('Continuous play is locked; use play_abort() to stop explicitly') end
end
local machine = manager.machine
local ports = machine.ioport.ports
keys = {
 R=ports[':IN1'].fields['P1 Right'], L=ports[':IN1'].fields['P1 Left'],
 D=ports[':IN1'].fields['P1 Down'], U=ports[':IN1'].fields['P1 Up'],
 LP=ports[':IN1'].fields['P1 Jab Punch'], MP=ports[':IN1'].fields['P1 Strong Punch'], HP=ports[':IN1'].fields['P1 Fierce Punch'],
 LK=ports[':IN2'].fields['P1 Short Kick'], MK=ports[':IN2'].fields['P1 Forward Kick'], HK=ports[':IN2'].fields['P1 Roundhouse Kick'],
 C=ports[':IN0'].fields['Coin 1'], S=ports[':IN0'].fields['1 Player Start']
}
job = nil
shots = shots or 0
training_fast = training_fast or false
function apply_training_speed()
 machine.video.throttled=not training_fast
 machine.video.throttle_rate=1
end
function normal_speed()
 training_fast=false
 machine.video.throttled=true
 machine.video.throttle_rate=1
end
function speed(mode)
 guard_training_action()
 if job or bot or advance or loadwatch or savewatch then error('Wait for the current action before changing speed') end
 if mode~='normal' and mode~='fast' then error('Speed must be normal or fast') end
 training_fast=mode=='fast';apply_training_speed()
end
function release() for _,f in pairs(keys) do f:clear_value() end end
function shot()
 local p
 repeat
  shots=shots+1
  p=string.format('training/%03d.png',shots)
  local existing=io.open(p,'rb')
  if existing then existing:close() else break end
 until false
 machine.screens[':screen']:snapshot(p)
 machine.screens[':screen']:snapshot('training/current.png')
 if write_observation then write_observation(p) end
 if formal_active then formal_event('screenshot',p) end
 print('SHOT '..p)
end
function act(seq)
 guard_training_action()
 if job or bot or advance or loadwatch or savewatch then error('Previous action still running') end
 job={seq=seq,i=1,left=seq[1][1],total=0}
 release()
 for k in string.gmatch(seq[1][2] or '', '%S+') do assert(keys[k],k):set_value(1) end
 apply_training_speed();emu.unpause()
end
if input_subscription then input_subscription:unsubscribe() end
input_subscription=emu.add_machine_frame_notifier(function()
 if not job then return end
 job.left=job.left-1;job.total=job.total+1
 if job.left<=0 then
  release();job.i=job.i+1
  if job.i>#job.seq then
   print('DONE frames='..job.total); job=nil; emu.pause(); shot();return
  end
  local s=job.seq[job.i];job.left=s[1]
  for k in string.gmatch(s[2] or '', '%S+') do assert(keys[k],k):set_value(1) end
 end
end)
function repeatseq(seq,n)
 local out={}
 for j=1,n do for _,s in ipairs(seq) do out[#out+1]={s[1],s[2]} end end
 return out
end
function dp(forward,punch,n)
 local seq={{2,forward},{2,'D'},{2,'D '..forward..' '..(punch or 'LP')},{30,''}}
 act(repeatseq(seq,n or 1))
end
function fire(forward,punch,n)
 act(repeatseq({{3,'D'},{3,'D '..forward},{2,forward..' '..(punch or 'HP')},{40,''}},n or 1))
end
emu.pause();shot();print('Controller ready')
