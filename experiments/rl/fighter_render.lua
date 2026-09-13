-- Resolve fighter pictures from the CURRENT latched CPS1 sprite list, not RAM pose.
local R={interface='sf2_rendered_fighters_v1'}
local function flag(v) return v and 1 or 0 end
local function tile_key(p) return p.code..':'..flag(p.flipx)..':'..flag(p.flipy) end
local function positioned(p) return p.x..':'..p.y..':'..tile_key(p) end
local function copy(t) local o={};for k,v in pairs(t) do o[k]=v end;return o end
function R.new(mapping,buffer_module)
 local o={compiled={}}
 function o:compile(mem,char)
  if self.compiled[char] then return self.compiled[char] end
  local character=mapping[char];if not character then return nil end
  local patterns,frequency={},{}
  for pose,pointer in pairs(character.sprites) do
   local comp=buffer_module.composition(mem,pointer)
   for flip=0,3 do
    local pieces=buffer_module.pattern(comp,flip)
    if #pieces>=3 then
     local row={pose=pose,flip=flip,pieces=pieces,meta=character.poses[pose]}
     patterns[#patterns+1]=row
     for _,piece in ipairs(pieces) do local k=tile_key(piece);frequency[k]=(frequency[k] or 0)+1 end
    end
   end
  end
  local lookup={}
  for _,row in ipairs(patterns) do
   -- Every reference tile may anchor a clipped sprite. A single rare anchor
   -- can disappear beyond the screen edge even while the body is visible.
   for _,piece in ipairs(row.pieces) do
    local key=tile_key(piece);lookup[key]=lookup[key] or {}
    lookup[key][#lookup[key]+1]={pattern=row,anchor=piece}
   end
  end
  local compiled={lookup=lookup};self.compiled[char]=compiled;return compiled
 end
 function o:resolve(mem,char,buffer)
  local unknown={known=false,position_known=false,pose_id=0,grounded='unknown',facing='unknown'}
  if not buffer or not buffer.known then return unknown end
  local compiled=self:compile(mem,char);if not compiled then return unknown end
  local present={};for _,p in ipairs(buffer.sprites) do present[positioned(p)]=true end
  local best_count=0;local candidates={};local checked={}
  for _,anchor in ipairs(buffer.sprites) do
   for _,entry in ipairs(compiled.lookup[tile_key(anchor)] or {}) do
    local pattern=entry.pattern
    local ox=anchor.x-entry.anchor.x;local oy=anchor.y-entry.anchor.y
    local check=pattern.pose..':'..pattern.flip..':'..ox..':'..oy
    if not checked[check] then
     checked[check]=true;local matched=0;local valid=true;local x1,y1,x2,y2=384,224,0,0
     for _,p in ipairs(pattern.pieces) do
      local x,y=ox+p.x,oy+p.y
      if x<384 and x+16>0 and y<224 and y+16>0 then
       local key=x..':'..y..':'..tile_key(p)
       if not present[key] then valid=false;break end
       matched=matched+1;x1=math.min(x1,math.max(0,x));y1=math.min(y1,math.max(0,y))
       x2=math.max(x2,math.min(384,x+16));y2=math.max(y2,math.min(224,y+16))
      end
     end
     if valid and matched>=3 and matched>=best_count then
      if matched>best_count then candidates={};best_count=matched end
      candidates[#candidates+1]={pose=pattern.pose,flip=pattern.flip,meta=pattern.meta,x1=x1,y1=y1,x2=x2,y2=y2}
     end
    end
   end
  end
  if #candidates==0 then return unknown end
  local first=candidates[1];local meta=copy(first.meta);local pose=first.pose
  local same_position=true;local same_facing=true;local flip=first.flip
  for _,c in ipairs(candidates) do
   pose=math.min(pose,c.pose)
   same_position=same_position and c.x1==first.x1 and c.x2==first.x2 and c.y1==first.y1 and c.y2==first.y2
   same_facing=same_facing and (c.flip&1)==(flip&1)
   for key,value in pairs(meta) do
    if c.meta[key]~=value then
     if key=='start' then meta[key]=false elseif key=='move_id' or key=='variant' then meta[key]=0 else meta[key]='unknown' end
    end
   end
  end
  if not same_position then return unknown end
  if meta.move_id==0 then meta.variant=0 end
  -- Picture flip is reported only for a directional reference pose whose
  -- orientation bias is established from all aliased current sprite records.
  local bias=meta.facing_bias
  local facing='unknown'
  if same_facing and (bias==0 or bias==1) then facing=((flip&1)~bias)==1 and 'left' or 'right' end
  return {known=true,position_known=true,pose_id=pose,meta=meta,x=(first.x1+first.x2)/2,y=first.y2,
   bbox={left=first.x1,top=first.y1,right=first.x2,bottom=first.y2},grounded='unknown',facing=facing,
   matched_tiles=best_count,candidate_aliases=#candidates}
 end
 return o
end
return R
