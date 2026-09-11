"""Bounded read-only projectile telemetry; train states only, never a win-rate run."""
import argparse
from collections import Counter
import importlib
import json
from pathlib import Path
import shutil
import sys
import time
from astra_play_sf2.config import atomic_json, load_config
from astra_play_sf2.runner import sha256

HERE = Path(__file__).resolve().parent


def select_train(dataset, opponents=(3,9), per_opponent=2):
    if not opponents or len(set(opponents))!=len(opponents) or any(o not in (0,3,7,9) for o in opponents):
        raise ValueError('Choose distinct projectile-capable opponent IDs 0,3,7,9')
    path=Path(dataset).resolve();data=json.loads(path.read_text())
    if data.get('status')!='complete' or data.get('difficulty')!=3:
        raise ValueError('Need a completed Normal training collection')
    selected=[]
    for opponent in opponents:
        rows=[r for r in data['openings'] if r['split']=='train' and r['opponent']==opponent]
        if len(rows)<per_opponent:raise ValueError('Missing requested train coverage')
        for original in rows[:per_opponent]:
            row=dict(original);p=(path.parent/row['path']).resolve()
            if not p.is_relative_to(path.parent) or row.get('status')!='accepted' or row['difficulty']!=3:
                raise ValueError('Invalid train checkpoint metadata')
            if sha256(p)!=row['sha256']:raise ValueError('Train checkpoint hash mismatch')
            row['path']=str(p);selected.append(row)
    return selected  # Dev/holdout paths are never opened or loaded.


def probe_runtime(original):
    def replace(old,new):
        nonlocal original
        if original.count(old)!=1:raise ValueError('Missing/ambiguous reference anchor: '+old[:70])
        original=original.replace(old,new)
    replace(' return s\nend', ''' s.projectile_pointers={mem:read_u32(0xff83c6+0x1d4),mem:read_u32(0xff86c6+0x1d4)}
 s.projectile_pointer_words={
  {mem:read_u16(0xff83c6+0x1d4),mem:read_u16(0xff83c6+0x1d6)},
  {mem:read_u16(0xff86c6+0x1d4),mem:read_u16(0xff86c6+0x1d6)}}
 s.projectiles={}
 for i=0,7 do
  local base=0xff938a+i*0xc0
  s.projectiles[#s.projectiles+1]={slot=i,base=base,status=mem:read_u16(base),
   x=mem:read_i16(base+6),y=mem:read_i16(base+10),type=mem:read_u8(base+0x20),
   hp=mem:read_i16(base+0x2a),owner_pointer=mem:read_u32(base+0x26),
   owner_words={mem:read_u16(base+0x26),mem:read_u16(base+0x28)}}
 end
 s.native_ports={m.ioport.ports[':IN1']:read(),m.ioport.ports[':IN2']:read()}
 return s
end''')
    start=original.index('local function choose(');end=original.index('\nassert(m.paused',start)
    original=original[:start]+'''local policy
local function choose(a,b,mode,s,reset_history)
 local seq,reason=policy:choose(s,reset_history)
 local action=assert(tonumber(reason:match('rl_action_(%d+)')))
 decisions[#decisions+1]={frame=frames,round=core.round,action=action,state=s,reset_history=reset_history}
 return seq
end
'''+original[end:]
    replace("if core.phase=='complete' then", "if core.phase=='complete' or frames>=pending.frames then")
    replace("   assert(frames==pending.frames,'Reference match ended at a different frame')\n",'')
    replace("   assert(index==#pending.actions+1,'Reference did not consume complete action plan')\n",'')
    replace('opening=opening,initial_loads=1,in_play_pauses=0,chain_metrics={}',
            "opening=opening,initial_loads=1,in_play_pauses=0,chain_metrics={},observed_frames=frames,completed_match=core.phase=='complete'")
    replace("pending.op=='reference_match' and #pending.actions>0 and pending.frames<=80000",
            "pending.op=='reference_match' and pending.model and pending.frames>=1 and pending.frames<=3600")
    replace("  Speed.apply(m.video,'fast')", "  policy=NN.new(pending.model,Actions)\n  Speed.apply(m.video,'fast')")
    return original


def summarize(response):
    rows=response['trace'];previous=response['opening'];period=previous['native_frame_period']
    intervals=[];active={};association=Counter();heights=Counter();types=Counter();hp_values=Counter();deaths=[]
    for number,row in enumerate(rows,1):
        s=row['state']
        if row['frame']!=number or abs(s['emulated_seconds']-previous['emulated_seconds']-period)>1e-7:
            raise RuntimeError('Incomplete native frame cadence')
        for player,words in zip(s['projectile_pointers'],s['projectile_pointer_words']):
            if player!=(words[0]<<16|words[1]):raise RuntimeError('Player u16/u32 reads disagree')
        for obj in s['projectiles']:
            slot=obj['slot'];owner={0x83c6:0,0x86c6:1}.get(obj['owner_pointer']>>16)
            if obj['owner_pointer']!=(obj['owner_words'][0]<<16|obj['owner_words'][1]):raise RuntimeError('Owner u16/u32 reads disagree')
            if obj['status']==257:
                if slot not in active:
                    active[slot]={'slot':slot,'first_frame':number,'owner_index':owner,'type':obj['type'],'x_first':obj['x'],'ys':set()}
                active[slot].update(last_frame=number,x_last=obj['x'],last_hp=obj['hp'])
                active[slot]['ys'].add(obj['y']);hp_values[str(obj['hp'])]+=1
                if obj['hp']>0:
                    association['positive_hp_object_frames']+=1
                    association['owner_recognized']+=owner is not None
                    association['owner_slot_lowword_matches']+=owner is not None and s['projectile_pointers'][owner]>>16==obj['base']&65535
                    heights[str((owner,obj['y']))]+=1;types[str((owner,obj['type']))]+=1
            elif slot in active:
                finished=active.pop(slot);finished['ys']=sorted(finished['ys']);finished.update(inactive_frame=number,status_after=obj['status']);intervals.append(finished)
            old=previous['projectiles'][slot]
            if old['hp']>0 and obj['hp']<0 and obj['status']:
                deaths.append({'frame':number,'slot':slot,'hp_after':obj['hp'],'owner_index':owner,'x':obj['x'],'y':obj['y'],
                    'player_hp_before':[previous[p]['hp'] for p in ('p1','p2')],'player_hp_after':[s[p]['hp'] for p in ('p1','p2')]})
        previous=s
    for item in active.values():item['ys']=sorted(item['ys']);item['truncated_at_budget']=True;intervals.append(item)
    return {'native_frames':len(rows),'native_period':period,'association':dict(association),'positive_hp_heights':dict(heights),
            'positive_hp_types':dict(types),'active_status_hp_values':dict(hp_values),'intervals':intervals,'death_events':deaths,
            'pointer_word_checks':True,'limitations':['Positive HP/status is a candidate observation, not a proven collision predicate.',
             'Intervals and object-frame counts are not independent shots or win-rate samples.',
             'Owner low-word association and adjacent-word reads do not certify a full pointer schema.']}


def run(source, model, dataset, output, frames=3600, opponents=(3,9)):
    source=Path(source).resolve();model=Path(model).resolve();dataset=Path(dataset).resolve();output=Path(output).resolve()
    if not 1<=frames<=3600:raise ValueError('Budget must be 1..3600 native frames per sample')
    selected=select_train(dataset,opponents);output.mkdir(parents=True,exist_ok=False)
    record={'schema':'astra.rl-projectile-probe-suite.v1','status':'running','training_only':True,'formal_clear':False,
            'included_in_win_rates':False,'frame_budget_per_sample':frames,'samples_requested':len(selected),'opponents':list(opponents),'model_sha256':sha256(model),
            'dataset_sha256':sha256(dataset),'holdout_opened':False,'samples':[],'source':str(source)}
    environment=None;started=time.monotonic();code=output/'code';package=code/source.name
    def save():atomic_json(output/'result.json',record)
    try:
        save();shutil.copytree(source.parent,code,ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.pyo'))
        if not (code/'src/astra_play_sf2').is_dir():
            from .round_chain_trial import freeze_production
            freeze_production(source,code)
        reference=HERE/'chain_reference_runtime.lua';runtime=package/'batch_runtime.lua'
        runtime.write_text(probe_runtime(reference.read_text()))
        record.update(tool_sha256=sha256(Path(__file__)),base_reference_sha256=sha256(reference),probe_runtime_sha256=sha256(runtime),
                      source_files_sha256={p.name:sha256(p) for p in source.iterdir() if p.is_file()})
        sys.path[:0]=[str(code/'src'),str(code)]
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        export=importlib.import_module(source.name+'.export');batch=importlib.import_module(source.name+'.batch_env')
        payload=export.export_policy(model)
        for index,sample in enumerate(selected,1):
            folder=output/f'sample-{index:02d}'
            row={'checkpoint_id':sample['id'],'checkpoint_sha256':sample['sha256'],'opponent':sample['opponent'],'split':'train','status':'running'}
            record['samples'].append(row);save()
            environment=batch.BatchEnv(load_config(),folder,3,checkpoints=[sample])
            response=environment.batch_rpc({'op':'reference_match','reset':{'checkpoint':0,'lead':0},'frames':frames,'model':payload})
            atomic_json(folder/'telemetry.json',response)
            if response['initial_loads']!=1 or response['in_play_pauses']!=0 or not 1<=response['observed_frames']<=frames:
                raise RuntimeError('Probe lifecycle or budget mismatch')
            analysis=summarize(response);atomic_json(folder/'coverage.json',analysis)
            row.update(status='complete' if response['completed_match'] else 'partial',observed_frames=response['observed_frames'],
                reason='match_complete' if response['completed_match'] else 'frame_budget_reached',initial_loads=response['initial_loads'],
                in_play_pauses=response['in_play_pauses'],telemetry_sha256=sha256(folder/'telemetry.json'),coverage=analysis)
            environment.close();environment=None;save()
        if sha256(model)!=record['model_sha256']:raise RuntimeError('Frozen model changed')
        if any(sha256(source/name)!=digest for name,digest in record['source_files_sha256'].items()):raise RuntimeError('Frozen source changed')
        record['status']='complete'
    except BaseException as error:
        record.update(status='invalid',error=f'{type(error).__name__}: {error}');raise
    finally:
        if environment is not None:environment.close()
        record['wall_seconds']=time.monotonic()-started;save()
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','model','dataset','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--frames',type=int,default=3600)
    p.add_argument('--opponents',type=int,nargs='+',choices=(0,3,7,9),default=[3,9],help='Two train samples per requested opponent; defaults to Guile and Sagat')
    a=p.parse_args();result=run(**vars(a));print(json.dumps({'status':result['status'],'samples':len(result['samples']),'wall_seconds':result['wall_seconds']}))

if __name__=='__main__':main()
