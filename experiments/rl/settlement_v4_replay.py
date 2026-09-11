"""Replay a saved settlement trajectory into a reconstructed controller context."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from lupa.lua54 import LuaRuntime

HERE=Path(__file__).resolve().parent
ASSETS=HERE.parents[1]/'src/astra_play_sf2/assets'


def replay(path,adapter=HERE/'settlement.lua'):
    document=json.loads(Path(path).read_text());original=deepcopy(document)
    lua=LuaRuntime(unpack_returned_tuples=True)
    def table(x):
        if isinstance(x,dict):return lua.table_from({k:table(v) for k,v in x.items()})
        if isinstance(x,list):return lua.table_from([table(v) for v in x])
        return x
    module=lua.execute((ASSETS/'play_core.lua').read_text());module=lua.execute(adapter.read_text())(module)
    # Reconstruct only controller history. Every replayed native state is raw.
    opening=deepcopy(document['trace'][0]['state'])
    for p in ('p1','p2'):opening[p].update(hp=144,displayed_hp=144,wins=0,y=40,anim=1)
    opening['timer']=99
    opts=table(dict(mode='offline-settlement-replay',opponent=document['current_state']['p2']['char']))
    opts.choose=lua.eval('function() return {{12,""}} end')
    core=module.new(opts,table(opening));core.score=table(document['score']);core.round=3
    core.frame=document['terminal_frame']-1;core.round_opening=table(document['trace'][0]['state'])
    rows=[r for r in document['trace'] if r['frame']>=document['terminal_frame']]
    if [r['frame'] for r in rows]!=list(range(rows[0]['frame'],rows[-1]['frame']+1)):raise RuntimeError('Missing native frames')
    for row in rows:core.tick(core,table(row['state']))
    if document!=original:raise RuntimeError('Raw evidence changed')
    result={'phase':core.phase,'frames':len(rows),'adapter_sha256':hashlib.sha256(adapter.read_bytes()).hexdigest()}
    if len(core.rounds):
        row=core.rounds[1]
        result.update(outcome=row.outcome,recognition=row.recognition,settled_frame=row.settled_frame,
                      confirmation_frame=row.confirmation_frame,settled_hp=[row.settled.p1.hp,row.settled.p2.hp],
                      confirmation_hp=[row.confirmation.p1.hp,row.confirmation.p2.hp])
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('trace',type=Path)
    p.add_argument('--previous-adapter',type=Path,help='Optional archived pre-v4 adapter for explicit failure comparison')
    a=p.parse_args();new=replay(a.trace)
    if new.get('recognition')!='rl-native-double-ko-next-round-v4':raise RuntimeError('Expected confirmed double-KO not reproduced')
    result={'training_only':True,'native_replay':False,'current':new}
    if a.previous_adapter:
        old=replay(a.trace,a.previous_adapter)
        if old['phase']!='invalid':raise RuntimeError('Archived adapter did not reproduce the failure')
        result['old']=old
    print(json.dumps(result,indent=2))
