-- Session lifecycle auditing, independent of the policy. Never writes game RAM.
assert(not formal_active and manager.machine.paused)
local counts={resets=0,loads=0,saves=0}
local prefix=nil
formal_active=false
local violation=false
local function persist()
 local body=string.format('{"resets":%d,"loads":%d,"saves":%d,"active":%s,"violation":%s}\n',
  counts.resets,counts.loads,counts.saves,tostring(formal_active),tostring(violation))
 local f=assert(io.open('training/lifecycle.json','w'));f:write(body);f:close()
 if prefix then f=assert(io.open(prefix..'-lifecycle.json','w'));f:write(body);f:close() end
end
local function quote(s)
 return '"'..tostring(s):gsub('\\','\\\\'):gsub('"','\\"'):gsub('\n','\\n'):gsub('\r','\\r')..'"'
end
function formal_event(kind,detail)
 if not prefix then return end
 local f=assert(io.open(prefix..'-events.jsonl','a'))
 f:write(string.format('{"event":%s,"detail":%s,"emulated_seconds":%.6f}\n',quote(kind),quote(detail or ''),manager.machine.time:as_double()));f:close()
end
function formal_audit(src)
 if formal_active then formal_event('command',src) end
end
function session_check()
 assert(not violation and counts.resets==0 and counts.loads==0 and counts.saves==0,'Native session lifecycle violation')
end
function session_begin(path)
 guard_training_action();session_check()
 assert(not formal_active and not job and not advance and manager.machine.paused)
 assert((manager.machine.ioport.ports[':DSWB']:read()&7)==astra_difficulty_bits,'DIP update not observed')
 assert(path:match('^training/[%w_/%-]+$'))
 prefix=path;formal_active=true;persist();formal_event('begin','New ordinary coin; no continue/reset/load/save');observe()
end
function session_end()
 guard_training_action();session_check();assert(formal_active)
 formal_event('end','No visual certification asserted');formal_active=false;persist();observe()
 -- Do not rewrite the preceding attempt during subsequent idle lifecycle events.
 prefix=nil
end
function session_difficulty(level)
 guard_training_action();session_check();assert(not formal_active and manager.machine.paused)
 assert(type(level)=='number' and level%1==0 and level>=3 and level<=7)
 astra_difficulty_bits=7-level;astra_difficulty_label=tostring(level)
 manager.machine.ioport.ports[':DSWB'].fields['Difficulty'].user_value=astra_difficulty_bits
 -- I/O reads reflect the previous native frame while paused. Allow the field
 -- update to latch outside any attempt, then the caller checks the observation.
 act({{2,''}})
end
local function changed(kind)
 counts[kind]=counts[kind]+1;violation=true;persist();formal_event('violation',kind)
 stop()
end
astra_reset_sub=emu.add_machine_reset_notifier(function() changed('resets') end)
astra_load_sub=emu.add_machine_post_load_notifier(function() changed('loads') end)
astra_save_sub=emu.add_machine_pre_save_notifier(function() changed('saves') end)
persist()
