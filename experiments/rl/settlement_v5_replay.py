"""Offline replay of a raw TIME-award trajectory; preserve the original invalid run."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from lupa.lua54 import LuaRuntime
from astra_play_sf2.config import atomic_json
from astra_play_sf2.runner import sha256
HERE=Path(__file__).resolve().parent
ASSETS=HERE.parents[1]/'src/astra_play_sf2/assets'


def replay(path,adapter):
    data=json.loads(path.read_text());original=deepcopy(data);lua=LuaRuntime(unpack_returned_tuples=True)
    module=lua.execute((ASSETS/'play_core.lua').read_text());module=lua.execute(adapter.read_text())(module)
    opts=lua.table_from(dict(opponent=data['opening']['p2']['char'],mode='offline-time-replay',choose=lua.eval('function()return {{12,""}}end')))
    core=module.new(opts,lua.table_from(data['opening'],recursive=True));core.score=lua.table_from(data['score'])
    core.frame=data['terminal_frame']-1
    rows=data['trace'];expected=list(range(data['terminal_frame'],data['frame']+1))
    if [r['frame'] for r in rows]!=expected:raise RuntimeError('Incomplete native trace')
    result={'source_sha256':sha256(path),'adapter_sha256':sha256(adapter),'observed_frames':len(rows)}
    for row in rows:
        core.tick(core,lua.table_from(row['state'],recursive=True))
        if len(core.rounds) and 'outcome' not in result:
            r=core.rounds[1];award=r.time_award
            result.update(outcome=r.outcome,recognition=r.recognition,settled_frame=r.frame,
                settled_hp=[r.settled.p1.hp,r.settled.p2.hp],score=list(r.score.values()),
                award_frame=award.frame,award_hp=list(award.hp.values()),stable_award_frames=r.stable_award_frames,
                previous_frame=award.previous.frame if award.previous else None,
                previous_hp=[award.previous.state.p1.hp,award.previous.state.p2.hp] if award.previous else None)
    result['phase']=core.phase
    if data!=original:raise RuntimeError('Raw trace was changed')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--trace',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();old=replay(args.trace,HERE/'settlement.lua');new=replay(args.trace,HERE/'settlement_v5.lua')
    if old['phase']!='invalid' or new.get('outcome')!='loss' or new.get('stable_award_frames')!=360 or new.get('recognition')!='rl-native-time-pip-adjacent-ko-v5':
        raise RuntimeError('Expected original failure/new mature loss was not reproduced')
    if args.output.exists():raise FileExistsError(args.output)
    report={'schema':'astra.rl-time-adjacent-replay.v1','native_replay':False,'training_only':True,'original':old,'candidate':new}
    atomic_json(args.output,report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
