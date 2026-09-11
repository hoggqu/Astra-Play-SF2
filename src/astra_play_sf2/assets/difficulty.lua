-- sf2 World 910522 only. Read the game's decoded DIP, never patch game RAM.
local D={}
function D.read()
 local m=manager.machine
 local s=m.devices[':maincpu'].spaces['program']
 return {difficulty_bits=m.ioport.ports[':DSWB']:read()&7,
  difficulty_mirror=s:read_u8(0xff808b)&7,effective_difficulty=s:read_u16(0xff82c6),
  ai_rank=s:read_u16(0xff8a04),ai_parameter_a=s:read_u16(0xff8a06),
  ai_parameter_b=s:read_u16(0xff8a08),ai_index=s:read_u16(0xff89ce)}
end
function D.check(level,s)
 s=s or D.read()
 assert(type(level)=='number' and level%1==0 and level>=3 and level<=7,'Invalid requested difficulty')
 assert(s.difficulty_bits==7-level and s.difficulty_mirror==level and s.effective_difficulty==level,
  string.format('Difficulty mismatch: requested=%d DIP=%s mirror=%s internal=%s; configure before boot',
   level,tostring(s.difficulty_bits),tostring(s.difficulty_mirror),tostring(s.effective_difficulty)))
 return s
end
-- loadfile modules share globals; MAME's autoboot chunk has its own environment.
astra_difficulty=D
return D
