"""Replay native diagnostic traces through the visible observer (no emulator)."""
import argparse
import json
import hashlib
from pathlib import Path
from . import visible_feedback
HERE = Path(__file__).resolve().parent


def replay(response):
    from lupa import LuaRuntime
    lua=LuaRuntime(unpack_returned_tuples=True)
    module=lua.execute((HERE/'visible_feedback.lua').read_text())
    actions=lua.execute((HERE/'full_actions.lua').read_text())
    mapping=lua.execute((HERE/'visible_animation_map.lua').read_text())
    observer=module.new(actions,mapping)
    def convert(value):
        if isinstance(value,dict):return lua.table_from({k:convert(v) for k,v in value.items()})
        if isinstance(value,list):return lua.table_from([convert(v) for v in value])
        return value
    opening=convert(response['opening']);observer.reset(observer,opening)
    decisions={d['frame']:d['action'] for d in response['decisions']}
    if 0 in decisions:observer.request(observer,decisions[0],opening)
    starts=[];dizzy=[];last_time=response['opening']['emulated_seconds'];period=response['opening']['native_frame_period']
    for index,row in enumerate(response['trace'],1):
        if row['frame']!=index:raise ValueError('Missing native trace frame')
        state=row['state'];now=state['emulated_seconds']
        if abs(now-last_time-period)>1e-7:raise ValueError('Non-native cadence')
        last_time=now;s=convert(state);observer.tick(observer,s)
        v=dict(s.visible_feedback)
        if list(module.features(s).values())!=visible_feedback.features({'visible_feedback':v}):raise ValueError('Lua/Python feature mismatch')
        if v['start_age']==0:starts.append({'frame':index,**v})
        if v['p2_dizzy']=='present':dizzy.append(index)
        if index in decisions:observer.request(observer,decisions[index],s)
    return {'frames':len(response['trace']),'starts':starts,'opponent_dizzy_frames':dizzy,'feature_parity':'PASS'}


def audit(actions_run,throws_run,button_run,dizzy_run):
    results={}
    for kind,folder,pattern in [('actions',actions_run,'action-*.json'),('throws',throws_run,'context-*.json'),('buttons',button_run,'case-*.json'),('dizzy',dizzy_run,'sample-*/telemetry.json')]:
        paths=sorted(Path(folder).glob(pattern))
        if not paths:raise ValueError('No '+kind+' traces')
        results[kind]=[{ 'file':str(p.relative_to(folder)),**replay(json.loads(p.read_text()))} for p in paths]
    expected={12:'hadouken',13:'shoryuken',15:'shoryuken',79:'hadouken',80:'hadouken',81:'shoryuken',82:'tatsumaki',83:'tatsumaki',84:'tatsumaki'}
    for action,family in expected.items():
        r=next(r for r in results['actions'] if r['file']==f'action-{action:03d}.json')
        starts=[s for s in r['starts'] if s['actual_family']==family]
        if len(starts)!=1:raise ValueError(f'Expected exactly one {family} start for action {action}')
        if starts[0]['actual_strength']!='unknown':raise ValueError('Special strength leaked from identical pose')
    for action in (4,5,58,65,72):
        r=next(r for r in results['actions'] if r['file']==f'action-{action:03d}.json')
        if sum(s['actual_family']=='jump' for s in r['starts'])!=1:raise ValueError('Missing native jump start')
    for row,count in zip(results['buttons'],(1,12,1,4)):
        if len(row['starts'])!=count:raise ValueError('Incorrect held/repeated normal starts: '+row['file'])
    throw_success=sum(any(s['actual_family']=='throw' for s in r['starts']) for r in results['throws'])
    if throw_success<8:raise ValueError('Missing eight native throw confirmations')
    if not any(r['opponent_dizzy_frames'] for r in results['dizzy']):raise ValueError('Missing observed native dizziness')
    return {'schema':'astra.rl-visible-feedback-audit.v1','status':'PASS','formal_clear':False,
            'source_sha256':{name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in ('visible_feedback.lua','visible_animation_map.lua','visible_feedback.py','full_actions.lua')},
            'special_variants_confirmed':9,'throw_contexts_confirmed':throw_success,
            'frames_replayed':sum(r['frames'] for rows in results.values() for r in rows),
            'results':results,'visual_review_required':'Native animation mapping audit does not replace human screenshot inspection.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('actions-run','throws-run','button-run','dizzy-run','output'):p.add_argument('--'+name,type=Path,required=True)
    args=vars(p.parse_args());out=args.pop('output');result=audit(**args)
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='results'}))
if __name__=='__main__':main()
