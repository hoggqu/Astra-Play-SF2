-- Input-interface experiment only: faster action 12, all other macros unchanged.
local Base=assert(loadfile('training/runtime/rl_actions_base.lua'))()
local A={frames=Base.frames,count=Base.count}
function A.keys(action,frame,s,forward)
 local original=Base.keys(action,frame,s,forward) -- Retain all original validation.
 if action~=12 then return original end
 if frame<2 then return 'D'
 elseif frame<4 then return 'D '..forward
 elseif frame<6 then return forward..' LP'
 else return '' end
end
return A
