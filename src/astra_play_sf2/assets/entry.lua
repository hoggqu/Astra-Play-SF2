-- Native new-game readiness, sf2 World 910522 only. Read-only RAM, neutral input.
-- Task slots are 32 bytes at FF0000; a zero first byte means no active task.
-- ROM: 64E2 creates attract (20); 6604 credit screen (40); 6E8A game (60/80).
-- Continue and endings retain game tasks; 9A28/B0B8 recreate attract on exit.
local machine=manager.machine
local space=machine.devices[':maincpu'].spaces['program']
local E={result=nil}
local waiting=nil
local fields={'task_boot','task_attract','task_credit','task_input','task_game','mode','playback','ending','fade_busy'}
function E.read()
 return {task_boot=space:read_u8(0xff0000),task_attract=space:read_u8(0xff0020),
  task_credit=space:read_u8(0xff0040),task_input=space:read_u8(0xff0060),task_game=space:read_u8(0xff0080),
  mode=space:read_u16(0xff8000),playback=space:read_u8(0xff82da),ending=space:read_u8(0xff831b),fade_busy=space:read_u8(0xffdd59)}
end
function E.ready(kind,s)
 if s.task_boot~=0 or s.task_input~=0 or s.task_game~=0 or s.playback~=1 then return false end
 if kind=='coin' then return s.task_attract~=0 and s.task_credit==0 end
 -- Mode 4 dispatches to 6D0E, but 6D0A first waits in 2532 for the fade to end.
 if kind=='start' then return s.task_attract==0 and s.task_credit~=0 and s.mode==4 and s.fade_busy==0 end
 return false
end
local function quote(s)
 return '"'..tostring(s):gsub('\\','\\\\'):gsub('"','\\"'):gsub('\n','\\n'):gsub('\r','\\r')..'"'
end
function E.json()
 local r=E.result
 if not r then return 'null' end
 local state={}
 for _,k in ipairs(fields) do state[#state+1]=quote(k)..':'..r.state[k] end
 return string.format('{"kind":%s,"status":%s,"frames":%d,"max_frames":%d,"stable_frames":%d,"required_stable_frames":2,"error":%s,"state":{%s}}',
  quote(r.kind),quote(r.status),r.frames,r.max_frames,r.stable_frames,quote(r.error or ''),table.concat(state,','))
end
function entry_busy() return waiting~=nil end
local function finish(status,err)
 local r=waiting;r.status=status;r.error=err;E.result=r;waiting=nil
 release();emu.pause();observe()
end
local function begin(kind,max_frames)
 guard_training_action()
 assert(not job and not bot and not advance and machine.paused,'Entry wait requires an idle controller')
 assert(type(max_frames)=='number' and max_frames%1==0 and max_frames>=2 and max_frames<=9000,'Invalid entry wait budget')
 session_check();astra_difficulty.check(7-astra_difficulty_bits)
 waiting={kind=kind,status='waiting',frames=0,max_frames=max_frames,stable_frames=0,state=E.read()}
 E.result=waiting;release();apply_training_speed();emu.unpause()
end
function wait_coin_ready(max_frames) begin('coin',max_frames) end
function wait_start_ready(max_frames) begin('start',max_frames) end
function E.require_ready(kind)
 assert(E.result and E.result.kind==kind and E.result.status=='ready' and E.ready(kind,E.read()),'Native '..kind..' readiness missing')
end
astra_entry_subscription=emu.add_machine_frame_notifier(function()
 if not waiting then return end
 waiting.frames=waiting.frames+1;waiting.state=E.read()
 local ok,err=pcall(function() session_check();astra_difficulty.check(7-astra_difficulty_bits) end)
 if not ok then finish('error',tostring(err));return end
 waiting.stable_frames=E.ready(waiting.kind,waiting.state) and (waiting.stable_frames+1) or 0
 if waiting.stable_frames>=2 then finish('ready')
 elseif waiting.frames>=waiting.max_frames then finish('timeout','Native '..waiting.kind..' readiness timed out') end
end)
astra_entry=E
return E
