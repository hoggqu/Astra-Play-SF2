-- Executed by -autoboot_script, without an interactive Lua console.
local ok,err=xpcall(function()
 local m=manager.machine
 assert(emu.app_version():match('^0%.288'), 'MAME 0.288 required')
 assert(m.system.name=='sf2','sf2 World 910522 required')
 assert(m.devices[':maincpu'].spaces['program'] and m.screens[':screen'])
 assert(m.ioport.ports[':DSWB'].fields['Difficulty'])
 train_busy=function() return false end
 assert(loadfile('training/runtime/settings.lua'))()
 m.ioport.ports[':DSWB'].fields['Difficulty'].user_value=astra_difficulty_bits
 for _,name in ipairs({'control','fighter','observe','play','bridge','session'}) do
  assert(loadfile('training/runtime/'..name..'.lua'))()
 end
 observe()
 local f=assert(io.open('training/ready.json','w'))
 f:write('{"ready":true,"mame_version":"0.288","rom":"sf2"}\n');f:close()
end,debug.traceback)
if not ok then
 local f=assert(io.open('training/bootstrap-error.txt','w'));f:write(tostring(err));f:close()
 print('ASTRA BOOTSTRAP ERROR: '..tostring(err));emu.pause()
end
