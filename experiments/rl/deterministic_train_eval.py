"""Twelve bounded train-state argmax matches; diagnostic, never formal reliability."""
import argparse
from collections import Counter
import importlib
import json
import math
from pathlib import Path
import shutil
import sys
import threading
import _thread
import time
from astra_play_sf2.config import atomic_json,load_config
from astra_play_sf2.runner import sha256

HERE=Path(__file__).resolve().parent
OPPONENTS=(3,0,6)
ALL_OPPONENTS=(0,1,2,3,5,6,7,8,9,10,11)


def select_train(dataset):
    return select_states(dataset,'train',OPPONENTS,2)


def select_states(dataset,split,opponents,count):
    if split not in ('train','dev'):raise ValueError('Holdout is reserved; select train or dev')
    if count<1 or not opponents or len(set(opponents))!=len(opponents) or any(o not in ALL_OPPONENTS for o in opponents):
        raise ValueError('Invalid state selection')
    path=Path(dataset).resolve();data=json.loads(path.read_text());selected=[]
    if data.get('status')!='complete' or data.get('difficulty')!=3:raise ValueError('Need completed Normal dataset')
    for opponent in opponents:
        rows=[r for r in data['openings'] if r['split']==split and r['opponent']==opponent]
        if len(rows)<count:raise ValueError('Insufficient selected states per opponent')
        for original in rows[:count]:
            row=dict(original);file=(path.parent/row['path']).resolve()
            if not file.is_relative_to(path.parent) or row.get('status')!='accepted' or row['difficulty']!=3:
                raise ValueError('Invalid checkpoint metadata')
            if sha256(file)!=row['sha256']:raise ValueError('Checkpoint hash mismatch')
            row['path']=str(file);selected.append(row)
    if len({r['sha256'] for r in selected})!=len(selected):raise ValueError('Duplicate selected states')
    return selected


def runtime(original):
    def replace(old,new):
        nonlocal original
        if original.count(old)!=1:raise ValueError('Ambiguous reference anchor: '+old[:60])
        original=original.replace(old,new)
    start=original.index('local function choose(');end=original.index('\nassert(m.paused',start)
    original=original[:start]+'''local policy
local function choose(a,b,mode,s,reset_history)
 local seq,reason=policy:choose(s,reset_history)
 local action=assert(tonumber(reason:match('rl_action_(%d+)')))
 decisions[#decisions+1]={frame=frames,round=core.round,action=action,state=s,reset_history=reset_history}
 return seq
end
'''+original[end:]
    replace('if pending then loaded=true end end)', 'if pending then loaded=true;pending.native_loads=pending.native_loads+1 end end)')
    replace("   assert(frames==pending.frames,'Reference match ended at a different frame')\n",'')
    replace("   assert(index==#pending.actions+1,'Reference did not consume complete action plan')\n",'')
    replace('opening=opening,initial_loads=1,in_play_pauses=0,chain_metrics={}',
            'opening=opening,initial_loads=pending.native_loads,in_play_pauses=0,chain_metrics={},terminal=effects.terminal,observed_frames=frames')
    replace("pending.op=='reference_match' and #pending.actions>0 and pending.frames<=80000",
            "pending.op=='reference_match' and pending.model and pending.frames==36000")
    replace("  Speed.apply(m.video,'fast')", "  policy=NN.new(pending.model,Actions);pending.native_loads=0\n  Speed.apply(m.video,'fast')")
    return original


def audit(response,opponent):
    rows=response['trace'];period=response['opening']['native_frame_period'];previous=response['opening']['emulated_seconds']
    if not isinstance(period,(int,float)) or not math.isfinite(period) or period<=0:raise RuntimeError('Invalid native period')
    if response['initial_loads']!=1 or response['in_play_pauses']!=0:raise RuntimeError('Invalid diagnostic lifecycle')
    if response['opening']['p1']['char']!=4 or response['opening']['p2']['char']!=opponent:raise RuntimeError('Wrong opening actors')
    for i,row in enumerate(rows,1):
        state=row['state']
        if row['frame']!=i or state['effective_difficulty']!=3 or state['native_frame_period']!=period or not math.isfinite(state['emulated_seconds']) or abs(state['emulated_seconds']-previous-period)>1e-7:
            raise RuntimeError('Missing native frame or wrong difficulty')
        previous=state['emulated_seconds']
    terminal=response.get('terminal',{})
    if not rows or rows[-1]['phase']!='complete' or not terminal.get('valid'):raise RuntimeError('No mature whole-match result')
    score=[sum(r['outcome']=='win' for r in response['rounds']),sum(r['outcome']=='loss' for r in response['rounds'])]
    result=terminal.get('result')
    if result=='ken_win' and score[0]==2 and score[1]<2:outcome='win'
    elif result=='cpu_win' and score[1]==2 and score[0]<2:outcome='loss'
    else:raise RuntimeError('Whole-match/round score disagreement')
    if terminal.get('score')!=score or [rows[-1]['state'][p]['wins'] for p in ('p1','p2')]!=score:
        raise RuntimeError('Terminal native pips disagree')
    decisions=response['decisions'];openings={1:0};stops={}
    for row in rows:
        for event in row['events']:
            if event.get('kind')=='round_start':
                if event['round'] in openings:raise RuntimeError('Duplicate round start')
                openings[event['round']]=row['frame']
            elif event.get('kind')=='round_stop':
                if event['round'] in stops:raise RuntimeError('Duplicate round stop')
                stops[event['round']]=row['frame']
    if not decisions or set(openings)!=set(stops) or {d['round'] for d in decisions}!=set(openings):
        raise RuntimeError('Missing native round/decision coverage')
    for number,first in openings.items():
        actual=[d for d in decisions if d['round']==number]
        if [d['frame'] for d in actual]!=list(range(first,stops[number],12)) or not actual or not actual[0]['reset_history']:
            raise RuntimeError('Wrong decision coverage or round history')
        if any(d['reset_history'] for d in actual[1:]):raise RuntimeError('Unexpected history reset')
    return {'ok':True,'outcome':outcome,'score':score,'frames':len(rows),'decisions':len(decisions),
            'round_outcomes':[r['outcome'] for r in response['rounds']]}


def run(source,model,dataset,output,max_seconds=900,dev_all=False):
    source=Path(source).resolve();model=Path(model).resolve();dataset=Path(dataset).resolve();output=Path(output).resolve()
    if not 1<=max_seconds<=900:raise ValueError('Suite budget must be 1..900 seconds')
    opponents=ALL_OPPONENTS if dev_all else OPPONENTS
    selected=select_states(dataset,'dev',opponents,1) if dev_all else select_train(dataset)
    output.mkdir(parents=True,exist_ok=False)
    record={'schema':'astra.rl-train-argmax-diagnostic.v1','status':'running','formal_clear':False,'included_in_formal_win_rates':False,
            'selection':'deterministic_argmax','split':'dev' if dev_all else 'train','holdout_opened':False,
            'matches_requested':len(selected)*2,'suite_budget_seconds':max_seconds,
            'model_sha256':sha256(model),'dataset_sha256':sha256(dataset),'cases':[],'source':str(source)}
    environment=None;started=time.monotonic();expired=threading.Event()
    def expire():expired.set();_thread.interrupt_main()
    timer=threading.Timer(max_seconds,expire);timer.daemon=True;timer.start()
    def save():atomic_json(output/'result.json',record)
    try:
        save();code=output/'code';shutil.copytree(source.parent,code,ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.pyo'))
        if not (code/'src/astra_play_sf2').is_dir():
            from .round_chain_trial import freeze_production
            freeze_production(source,code)
        package=code/source.name;reference=HERE/'chain_reference_runtime.lua'
        (package/'batch_runtime.lua').write_text(runtime(reference.read_text()))
        record.update(tool_sha256=sha256(Path(__file__)),reference_sha256=sha256(reference),runtime_sha256=sha256(package/'batch_runtime.lua'),
            source_files_sha256={p.name:sha256(p) for p in source.iterdir() if p.is_file()})
        sys.path[:0]=[str(code/'src'),str(code)]
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        export=importlib.import_module(source.name+'.export');batch=importlib.import_module(source.name+'.batch_env')
        payload=export.export_policy(model);environment=batch.BatchEnv(load_config(),output/'environment',3,checkpoints=selected)
        record['mame_pid']=environment.process.pid;save()
        for checkpoint,sample in enumerate(selected):
            for lead in (0,12):
                index=len(record['cases'])+1;case={'id':f'case-{index:02d}','opponent':sample['opponent'],'checkpoint_id':sample['id'],
                    'checkpoint_sha256':sample['sha256'],'lead':lead,'status':'running'}
                record['cases'].append(case);save()
                response=environment.batch_rpc({'op':'reference_match','reset':{'checkpoint':checkpoint,'lead':lead},'frames':36000,'model':payload})
                path=output/(case['id']+'.json');atomic_json(path,response)
                case.update(audit=audit(response,sample['opponent']),status='complete',raw_sha256=sha256(path));save()
        if sha256(model)!=record['model_sha256']:raise RuntimeError('Model changed during evaluation')
        if any(sha256(source/name)!=digest for name,digest in record['source_files_sha256'].items()):raise RuntimeError('Source changed during evaluation')
        record['by_opponent']={}
        for opponent in opponents:
            cases=[x for x in record['cases'] if x['opponent']==opponent]
            record['by_opponent'][str(opponent)]={'matches':dict(Counter(x['audit']['outcome'] for x in cases)),
                'rounds':dict(Counter(r for x in cases for r in x['audit']['round_outcomes'])),
                'by_round':{str(n):dict(Counter(x['audit']['round_outcomes'][n-1] for x in cases if len(x['audit']['round_outcomes'])>=n)) for n in range(1,1+max(len(x['audit']['round_outcomes']) for x in cases))}}
        record['status']='complete'
    except BaseException as error:
        record.update(status='invalid',error='Suite wall-time budget exhausted' if expired.is_set() else f'{type(error).__name__}: {error}')
        if record['cases'] and record['cases'][-1]['status']=='running':
            record['cases'][-1].update(status='invalid',error=record['error'])
        raise
    finally:
        timer.cancel()
        if environment is not None:environment.close()
        record['wall_seconds']=time.monotonic()-started;save()
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source','model','dataset','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--max-seconds',type=int,default=900)
    p.add_argument('--dev-all',action='store_true',help='Model selection: one dev state for each of all 11 opponents, leads 0/12; never holdout or formal clear')
    a=p.parse_args();r=run(**vars(a));print(json.dumps({'status':r['status'],'by_opponent':r['by_opponent'],'wall_seconds':r['wall_seconds']}))

if __name__=='__main__':main()
