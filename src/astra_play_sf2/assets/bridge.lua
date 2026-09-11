-- Local file inbox avoids the interactive console's line-editing errors.
if train_busy and train_busy() then error('Cannot reload modules during a training match') end
if play_busy and play_busy() then error('Cannot reload inbox during continuous play') end
play_guarded_bridge_version=1
train_guarded_bridge_version=1
local base='training/'
local ticks=0
-- register_frame_done returns no subscription in MAME 0.288. Retire old
-- callbacks by generation, and avoid the pre-upgrade legacy inbox entirely.
play_bridge_generation=(play_bridge_generation or 0)+1
local generation=play_bridge_generation
local start_match,continuous_busy,continuous_request
if play_take_bridge_handler then
 start_match,continuous_busy,continuous_request=play_take_bridge_handler()
else
 continuous_busy=play_busy;continuous_request=play_request
end
local start_training,training_busy,training_request
if train_take_bridge_handler then
 start_training,training_busy,training_request=train_take_bridge_handler()
else
 training_busy=train_busy;training_request=train_request
end
local function parse_training(src)
 local text=src:match('^%s*(.-)%s*$')
 local prefix,id,mode,speed,lead=text:match("^train_match%('(training/[%w_/%-]+)',(%d+),'([%w_]+)','(%a+)',(%d+)%)$")
 if prefix then return prefix,tonumber(id),mode,speed,tonumber(lead) end
end
local function parse_start(src)
 local text=src:match('^%s*(.-)%s*$')
 local prefix,id,flag,speed=text:match("^play_match%('(training/[%w_/%-]+)',(%d+),{training_validation=(%a+),speed='(%a+)'}%)$")
 if not prefix then prefix,id,flag=text:match("^play_match%('(training/[%w_/%-]+)',(%d+),{training_validation=(%a+)}%)$") end
 if prefix and (flag=='true' or flag=='false') then
  return prefix,tonumber(id),flag=='true',speed or 'normal'
 end
end
if bridge_subscription then bridge_subscription:unsubscribe() end
bridge_subscription=emu.register_frame_done(function()
 if generation~=play_bridge_generation then return end
 ticks=ticks+1
 if ticks%4~=0 then return end
 local f=io.open(base..'request-v2.lua','r')
 if not f then return end
 local src=f:read('*a');f:close();os.remove(base..'request-v2.lua')
 if formal_audit then formal_audit(src) end
 local err
 if continuous_busy and continuous_busy() then
  local ok,accepted,result=pcall(continuous_request,src)
  if not ok then err=accepted elseif not accepted then err=result end
 elseif training_busy and training_busy() then
  local ok,accepted,result=pcall(training_request,src)
  if not ok then err=accepted elseif not accepted then err=result end
 else
  local tp,ti,tm,ts,tl=parse_training(src)
  local prefix,opponent,validation,speed=parse_start(src)
  if tp then
   if not start_training then err='Load train.lua followed by bridge.lua first'
   else local ok,result=pcall(start_training,tp,ti,tm,ts,tl);if not ok then err=result end end
  elseif src:match('^%s*train_match%s*%(') then
   err='Training start must be one complete canonical train_match request'
  elseif prefix then
   if not start_match then err='Reload play.lua followed by bridge.lua before continuous play'
   else
    local ok,result=pcall(start_match,prefix,opponent,{training_validation=validation,speed=speed})
    if not ok then err=result end
   end
  elseif src:match('^%s*play_match%s*%(') then
   err='Continuous start must be one complete canonical play_match request, with no extra Lua'
  else
   local code;code,err=load(src,'local-training-request','t',_ENV)
   if code then local ok,result=pcall(code);if not ok then err=result end end
  end
 end
 local out=io.open(base..'request-v2-status.txt','w');out:write(err and tostring(err) or 'accepted');out:close()
 if err then print('REQUEST ERROR: '..tostring(err)) end
end)
print('Local request bridge ready')
