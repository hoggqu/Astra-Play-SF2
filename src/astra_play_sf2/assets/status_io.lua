-- Single-writer status transport. Never delete or replace a file being polled.
-- Progress is append-only; terminal JSON is published once to a fresh name.
local S={}
local function put(path,mode,body)
 local f=assert(io.open(path,mode))
 local ok,err=pcall(function() assert(f:write(body)) end)
 local closed,close_err=f:close()
 if not ok then error(err) end
 assert(closed,close_err)
end
function S.append(path,body)
 put(path,'ab',body)
end
function S.publish(path,body)
 -- Paths belong to one fresh match prefix and one CLI-owned writer.
 local old=io.open(path,'rb')
 if old then old:close();error('Terminal status already exists: '..path) end
 put(path..'.tmp','wb',body)
 assert(os.rename(path..'.tmp',path))
end
return S
