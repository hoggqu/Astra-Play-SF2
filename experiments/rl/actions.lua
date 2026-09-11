-- 12-frame decisions; special-move directions use the side at decision time.
local A={frames=12,count=15}
function A.keys(action,frame,s,forward)
 assert(type(action)=='number' and action%1==0 and action>=0 and action<A.count,'Invalid action')
 assert(frame>=0 and frame<A.frames,'Invalid action frame')
 local back=forward=='R' and 'L' or 'R'
 local basic={'',forward,back,'D '..back,'U '..forward,'U '..back,'LP','HP','D LP','D HK','MK','HK'}
 if action<12 then return basic[action+1] end
 if action==12 then
  if frame<3 then return 'D' elseif frame<6 then return 'D '..forward elseif frame<8 then return forward..' LP' else return '' end
 elseif action==13 then
  if frame<2 then return forward elseif frame<4 then return 'D' elseif frame<6 then return 'D '..forward..' LP' else return '' end
 else return frame<3 and 'U '..forward or (frame<9 and 'HK' or '') end
end
return A
