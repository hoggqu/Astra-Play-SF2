-- Shared speed settings for entry, neutral actions and continuous matches.
local S={}
function S.settings(mode)
 local rates={normal=1,['2x']=2,['4x']=4,fast=1}
 assert(rates[mode],'Speed must be normal, 2x, 4x or fast')
 return {throttled=mode~='fast',throttle_rate=rates[mode],speed_factor=1000}
end
function S.apply(video,mode)
 local s=S.settings(mode)
 assert(video.speed_factor==s.speed_factor,'Native base speed factor changed')
 video.throttle_rate=s.throttle_rate;video.throttled=s.throttled
end
function S.check(video,mode)
 local s=S.settings(mode)
 assert(video.throttled==s.throttled and video.throttle_rate==s.throttle_rate and video.speed_factor==s.speed_factor,
  'selected game speed changed')
end
return S
