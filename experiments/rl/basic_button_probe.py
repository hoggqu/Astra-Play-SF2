"""Four bounded train-only LP/MK hold-vs-release probes in one owned MAME."""
import argparse
import _thread
from collections import Counter
import importlib
import json
from pathlib import Path
import shutil
import sys
import threading
import time
from astra_play_sf2.config import atomic_json,load_config
from astra_play_sf2.runner import sha256
from .projectile_probe import select_train
HERE=Path(__file__).resolve().parent


def actions_source(original):
    anchor='function A.keys(action,frame,s,forward)'
    if original.count(anchor)!=1:raise ValueError('Unexpected action signature')
    text=original.replace(anchor,'function A.keys(action,frame,s,forward,pulse)')
    anchor=" local back=forward=='R' and 'L' or 'R'"
    if text.count(anchor)!=1:raise ValueError('Unexpected basic action anchor')
    return text.replace(anchor," if pulse and (action==6 or action==10) and frame==11 then return '' end\n"+anchor)


def runtime_source(original):
    def replace(old,new):
        nonlocal original
        if original.count(old)!=1:raise ValueError('Missing/ambiguous runtime anchor: '+old)
        original=original.replace(old,new)
    replace('Actions.keys(planned.action,frame,s,forward)','Actions.keys(planned.action,frame,s,forward,pending.pulse)')
    replace(' return s\nend'," s.native_ports={m.ioport.ports[':IN1']:read(),m.ioport.ports[':IN2']:read()}\n return s\nend")
    replace("  if core.phase=='complete' then","  if frames==pending.frames then")
    replace('opening=opening,initial_loads=1,in_play_pauses=0,chain_metrics={}',
            'opening=opening,initial_loads=1,in_play_pauses=0,chain_metrics={},partial=true')
    return original


def summarize(response,action,pulse,neutral=48):
    frames=neutral+144+48
    rows=response['trace'];opening=response['opening'];previous=opening;transitions=[];damage=[];starts=[]
    counts=Counter();port_changes=[];last_input='';period=opening['native_frame_period']
    for index,row in enumerate(rows,1):
        s=row['state'];p=s['p1'];before=previous['p1'];counts[p['a']]+=1
        if row['frame']!=index or abs(s['emulated_seconds']-previous['emulated_seconds']-period)>1e-7:raise RuntimeError('Native frame gap')
        if (p['a'],p['anim'])!=(before['a'],before['anim']):transitions.append({'frame':index,'a':p['a'],'anim':p['anim'],'hp':p['hp'],'consumed':row['consumed_input']})
        if p['a'] in (10,12) and before['a'] not in (10,12):starts.append({'frame':index,'a':p['a'],'anim':p['anim']})
        if p['hp']<before['hp']:damage.append({'frame':index,'before':before['hp'],'after':p['hp'],'a':p['a'],'anim':p['anim']})
        if row['consumed_input']!=last_input or s['native_ports']!=previous['native_ports']:
            port_changes.append({'frame':index,'consumed':row['consumed_input'],'ports':s['native_ports']});last_input=row['consumed_input']
        previous=s
    if len(rows)!=frames or response['initial_loads']!=1 or response['in_play_pauses']!=0:raise RuntimeError('Wrong bounded probe lifecycle')
    if len(response['decisions'])!=frames//12+1 or [d['frame'] for d in response['decisions']]!=list(range(0,frames+1,12)):raise RuntimeError('Missing requested decisions')
    if neutral>=120 and rows[neutral-1]['state']['timer']>=99:raise RuntimeError('First press was not confirmed after active timer started')
    startup={6:383790,10:385966}[action]
    animation_starts=[r['frame'] for r in transitions if r['a']==10 and r['anim']==startup]
    return {'startup_animation':startup,'startup_animation_frames':animation_starts,'startup_animation_count':len(animation_starts),
            'timer_before_first_press':rows[neutral-1]['state']['timer'],'frames':len(rows),'action':action,'pulse':pulse,'requested_repetitions':12,'leading_neutral_frames':neutral,'trailing_neutral_frames':48,'unexecuted_stop_boundary_neutral_decisions':1,
            'attack_state_entries':starts,'attack_state_entry_count':len(starts),'state_counts':dict(counts),
            'damage_events':damage,'action_animation_transitions':transitions,'input_port_transitions':port_changes,
            'classification':'a10/a12 attack-state entry; a0 neutral, a14/a20 hit-state by existing policy conventions; raw animations retained for restarts within same state'}


def run(source,dataset,output,neutral=48):
    if neutral not in (48,120):raise ValueError('Bounded probe permits neutral48 or120 only')
    frames=neutral+144+48
    source=source.resolve();dataset=dataset.resolve();output=output.resolve();sample=select_train(dataset,(3,),1)[0]
    output.mkdir(parents=True,exist_ok=False);code=output/'code';package=code/source.name
    shutil.copytree(source.parent,code,ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.pyo'))
    (package/'actions.lua').write_text(actions_source((source/'actions.lua').read_text()))
    (package/'batch_runtime.lua').write_text(runtime_source((HERE/'chain_reference_runtime.lua').read_text()))
    sys.path[:0]=[str(code/'src'),str(code)];batch=importlib.import_module(source.name+'.batch_env')
    record={'schema':'astra.rl-basic-button-probe.v1','status':'running','training_only':True,'formal_clear':False,'included_in_win_rates':False,
            'checkpoint_id':sample['id'],'checkpoint_sha256':sample['sha256'],'opponent':3,'lead':0,'holdout_opened':False,
            'cases':[],'source_actions_sha256':sha256(source/'actions.lua'),'probe_actions_sha256':sha256(package/'actions.lua'),
            'probe_runtime_sha256':sha256(package/'batch_runtime.lua')}
    started=time.monotonic();env=None;timer=threading.Timer(300,_thread.interrupt_main);timer.daemon=True;timer.start()
    def save():atomic_json(output/'result.json',record)
    try:
        save();env=batch.BatchEnv(load_config(),output/'native',3,checkpoints=[sample]);record['runtime_sha256']=env.manifest['runtime_sha256']
        for index,(action,pulse) in enumerate(((6,False),(6,True),(10,False),(10,True)),1):
            plan=[{'frame':i*12,'round':1,'action':0 if i<neutral//12 or i>=neutral//12+12 else action} for i in range(frames//12+1)]
            response=env.batch_rpc({'op':'reference_match','reset':{'checkpoint':0,'lead':0},'actions':plan,'frames':frames,'pulse':pulse})
            path=output/f'case-{index:02d}.json';atomic_json(path,response)
            analysis=summarize(response,action,pulse,neutral);analysis.update(telemetry_sha256=sha256(path),status='partial')
            record['cases'].append(analysis);save()
        if any(json.loads((output/f'case-{i:02d}.json').read_text())['opening']!=json.loads((output/'case-01.json').read_text())['opening'] for i in range(2,5)):
            raise RuntimeError('Starting snapshot differs across cases')
        record['status']='complete'
    except BaseException as error:record.update(status='invalid',error=f'{type(error).__name__}: {error}');raise
    finally:
        timer.cancel()
        if env:env.close()
        record['wall_seconds']=time.monotonic()-started;save()
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','dataset','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--neutral',type=int,choices=(48,120),default=48)
    result=run(**vars(p.parse_args()));print(json.dumps({'status':result['status'],'wall_seconds':result['wall_seconds'],
        'cases':[{k:r[k] for k in ('action','pulse','startup_animation_frames','startup_animation_count','timer_before_first_press','damage_events')} for r in result['cases']]},indent=2))
if __name__=='__main__':main()
