-- Read-only diagnostics paired with each native screenshot. No winner inference.
if train_busy and train_busy() then error('Cannot reload modules during a training match') end
if play_busy and play_busy() then error('Cannot reload observation module during continuous play') end
function write_observation(snapshot)
 local m=manager.machine
 if m.system.name~='sf2' then error('Observation map supports sf2 World 910522 only') end
 local a,b=fighter(0),fighter(1)
 local timer=mem:read_u8(0xff8ace)
 local hi,lo=timer>>4,timer&15
 local seconds=(hi<=9 and lo<=9) and (hi*10+lo) or -1
 local d=astra_difficulty.read()
 local busy=job~=nil or bot~=nil or advance~=nil or loadwatch~=nil or savewatch~=nil or (play_busy and play_busy()) or (train_busy and train_busy()) or false
 local function player(i,p)
  local base=0xff83c6+i*0x300
  return string.format('{"character":%d,"hp":%d,"displayed_hp":%d,"timeout_hp":%d,"x":%d,"y":%d,"action":%d,"animation":%d,"round_wins":%d}',
   p.char,p.hp,mem:read_i16(base+0x1bc),mem:read_i16(base+0x164),p.x,p.y,p.a,p.anim,mem:read_u8(base+0x290))
 end
 local f=assert(io.open('training/status.json','w'))
 f:write(string.format('{\n  "rom":"sf2",\n  "screenshot":"%03d.png",\n  "emulated_seconds":%.6f,\n  "paused":%s,\n  "throttled":%s,\n  "throttle_rate":%.3f,\n  "speed_factor":%d,\n  "controller_busy":%s,\n  "difficulty_bits":%d,\n  "difficulty_mirror":%d,\n  "effective_difficulty":%d,\n  "ai_rank":%d,\n  "ai_parameter_a":%d,\n  "ai_parameter_b":%d,\n  "ai_index":%d,\n  "timer_bcd":%d,\n  "timer_seconds":%d,\n  "p1":%s,\n  "p2":%s\n}\n',
  shots,m.time:as_double(),tostring(m.paused),tostring(m.video.throttled),m.video.throttle_rate,m.video.speed_factor,
  tostring(busy),d.difficulty_bits,d.difficulty_mirror,d.effective_difficulty,d.ai_rank,d.ai_parameter_a,d.ai_parameter_b,d.ai_index,timer,seconds,player(0,a),player(1,b)))
 f:close()
 -- Keep the same metadata next to the numbered evidence, even after current moves on.
 local src=assert(io.open('training/status.json','r'));local body=src:read('*a');src:close()
 local dest=assert(io.open(snapshot:gsub('%.png$','.json'),'w'));dest:write(body);dest:close()
end

function observe()
 shot()
end
