"""Dev-only native branch parity gate. Explicit bundle identity; never formal play."""
import argparse,importlib,json,re,shutil,sys,threading,_thread,time
from pathlib import Path
from astra_play_sf2.config import atomic_json,load_config
from astra_play_sf2.runner import sha256
from .deterministic_train_eval import select_states,audit,runtime as reference_runtime,ALL_OPPONENTS
HERE=Path(__file__).resolve().parent
MASKS={'R':(0,1),'L':(0,2),'D':(0,4),'U':(0,8),'LP':(0,16),'MP':(0,32),'HP':(0,64),'LK':(1,1),'MK':(1,2),'HK':(1,4)}

def runtime(source):
    text=reference_runtime(source)
    old=" local action=assert(tonumber(reason:match('rl_action_(%d+)')))"
    new=" local bid,hash,aid=reason:match('^rl_bundle_(op%d%d)_([a-f0-9]+)_action_(%d+)$');assert(bid and #hash==64)\n local action=assert(tonumber(aid));assert(action>=0 and action<16)"
    if text.count(old)!=1:raise ValueError('Ambiguous action reason anchor')
    text=text.replace(old,new).replace('action=action,state=s,reset_history=reset_history}', 'action=action,state=s,reset_history=reset_history,branch_id=bid,branch_source_model_sha256=hash}')
    old=' return s\nend'
    if text.count(old)!=1:raise ValueError('Ambiguous snapshot anchor')
    return text.replace(old," s.native_ports={m.ioport.ports[':IN1']:read(),m.ioport.ports[':IN2']:read()}\n return s\nend")

def without_ports(value):
    if isinstance(value,dict):return {k:without_ports(v) for k,v in value.items() if k!='native_ports'}
    if isinstance(value,list):return [without_ports(x) for x in value]
    return value

def check_ports(row):
    ports=row['state']['native_ports'];expected=[127,7]
    for key in row['consumed_input'].split():
        if key not in MASKS:raise RuntimeError('Unexpected diagnostic input')
        port,mask=MASKS[key];expected[port]&=~mask
    if [ports[0]&127,ports[1]&7]!=expected:
        raise RuntimeError('Actual P1 native port bits disagree with consumed held input at frame '+str(row['frame']))

def compare(actual,old,op,bid,model_hash):
    checked=audit(actual,op);baseline=audit(old,op)
    if checked!=baseline:raise RuntimeError('Mature round/match audit differs')
    for key in ('opening','state','terminal','rounds','observed_frames','initial_loads','in_play_pauses'):
        if without_ports(actual[key])!=old[key]:raise RuntimeError('Raw whole-match field differs: '+key)
    if len(actual['trace'])!=len(old['trace']):raise RuntimeError('Native frame count differs')
    for a,b in zip(actual['trace'],old['trace']):
        if without_ports(a)!=b:raise RuntimeError('Native state/input/event differs at frame '+str(a['frame']))
        check_ports(a)
    if len(actual['decisions'])!=len(old['decisions']):raise RuntimeError('Decision count differs')
    for a,b in zip(actual['decisions'],old['decisions']):
        if a.get('branch_id')!=bid or a.get('branch_source_model_sha256')!=model_hash:raise RuntimeError('Decision used wrong frozen branch')
        clean=without_ports({k:v for k,v in a.items() if k not in ('branch_id','branch_source_model_sha256')})
        if clean!=b:raise RuntimeError('Native decision/action differs at frame '+str(a['frame']))
    return {'ok':True,'native_state_frames':len(actual['trace']),'held_input_frames':len(actual['trace']),
            'actual_port_frames_checked':len(actual['trace']),'old_actual_ports_compared':False,
            'decisions':len(actual['decisions']),'branch_id':bid,'branch_source_model_sha256':model_hash,'match':checked}

def run(source,bundle,dataset,references,output,max_seconds=900):
    source=source.resolve();bundle=bundle.resolve();dataset=dataset.resolve();output=output.resolve();references=references.resolve()
    if not 1<=max_seconds<=900:raise ValueError('Budget1..900 seconds')
    selected=select_states(dataset,'dev',ALL_OPPONENTS,1)
    output.mkdir(parents=True,exist_ok=False)
    record={'schema':'astra.rl-opponent-bundle-native-dev-gate.v1','status':'running','policy_kind':'ppo_opponent_bundle',
        'bundle_sha256':sha256(bundle),'dataset_sha256':sha256(dataset),'references_sha256':sha256(references),
        'training':False,'formal':False,'holdout_used':False,'matches_requested':22,'cases':[],
        'old_actual_ports_available':False,'port_audit':'New actual IN1/IN2 P1 bits checked against consumed held input; old raw records held text only.'}
    env=None;started=time.monotonic();timer=threading.Timer(max_seconds,_thread.interrupt_main);timer.daemon=True;timer.start()
    def save():atomic_json(output/'result.json',record)
    try:
        save();sys.path[:0]=[str(source.parent/'src'),str(source.parent)]
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        support=importlib.import_module(source.name+'.support');export=importlib.import_module(source.name+'.export')
        support.capture_interface(bundle,source);payload=export.export_policy(bundle)
        if payload.get('schema')!='astra.rl-opponent-bundle-policy.v1' or payload.get('policy_kind')!='ppo_opponent_bundle':raise RuntimeError('Bundle payload required')
        record['bundle_route_sha256']=payload['route_sha256'];record['branch_model_sha256']={k:b['source_model_sha256'] for k,b in payload['branches'].items()}
        source_pins={str(p):sha256(p) for p in source.parent.rglob('*') if p.is_file() and '__pycache__'not in p.parts}
        config=json.loads(references.read_text());models={}
        for row in config['candidates'].values():
            path=Path(row['dev']).resolve();d=json.loads(path.read_text())
            if d['status']!='complete' or d['model_sha256']!=row['model_sha256'] or d.get('split')!='dev' or len(d['cases'])!=22:raise RuntimeError('Reference dev is not complete')
            models[row['model_sha256']]=(path,d)
        plans=[]
        for sample in selected:
            bid=payload['route'][str(sample['opponent'])];branch_hash=payload['branches'][bid]['source_model_sha256']
            path,reference=models[branch_hash];source_pins[str(path)]=sha256(path)
            for lead in (0,12):
                matches=[c for c in reference['cases'] if c['opponent']==sample['opponent'] and c['checkpoint_sha256']==sample['sha256'] and c['lead']==lead]
                if len(matches)!=1 or matches[0]['status']!='complete':raise RuntimeError('Reference state/lead missing or duplicated')
                case=matches[0];raw=path.parent/(case['id']+'.json')
                if sha256(raw)!=case['raw_sha256'] or audit(json.loads(raw.read_text()),sample['opponent'])!=case['audit']:raise RuntimeError('Reference native audit/hash differs')
                source_pins[str(raw)]=case['raw_sha256'];plans.append({'sample':sample,'lead':lead,'branch_id':bid,'branch_hash':branch_hash,'reference':str(raw),'reference_sha256':case['raw_sha256']})
        source_pins[str(bundle)]=record['bundle_sha256'];source_pins[str(dataset)]=record['dataset_sha256'];source_pins[str(references)]=record['references_sha256']
        for s in selected:source_pins[s['path']]=s['sha256']
        record['pins']=source_pins;record['plan']=plans;record['tool_sha256']=sha256(Path(__file__));save()
        code=output/'code';shutil.copytree(source.parent,code,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        package=code/source.name;(package/'batch_runtime.lua').write_text(runtime((HERE/'chain_reference_runtime.lua').read_text()))
        p=package/'batch_env.py';text=p.read_text();anchor="            (runtime / 'rl_nn.lua').write_bytes((HERE / 'nn.lua').read_bytes())"
        if text.count(anchor)!=1:raise RuntimeError('Ambiguous private batch staging anchor')
        text=text.replace(anchor,anchor+"\n            (runtime / 'rl_bundle_base_nn.lua').write_bytes((HERE / 'bundle_base_nn.lua').read_bytes())")
        p.write_text(text)
        for name in list(sys.modules):
            if name==source.name or name.startswith(source.name+'.'):sys.modules.pop(name)
        sys.path[:0]=[str(code/'src'),str(code)]
        batch=importlib.import_module(source.name+'.batch_env');env=batch.BatchEnv(load_config(),output/'environment',3,checkpoints=selected)
        record['mame_pid']=env.process.pid
        record['diagnostic_runtime_sha256']={p.name:sha256(p) for p in (env.run/'training/runtime').glob('*') if p.is_file()};save()
        for index,plan in enumerate(plans):
            case={'id':f'case-{index+1:02d}','status':'running',**{k:v for k,v in plan.items() if k!='sample'}};record['cases'].append(case);save()
            actual=env.batch_rpc({'op':'reference_match','reset':{'checkpoint':index//2,'lead':plan['lead']},'frames':36000,'model':payload})
            raw=output/(case['id']+'.json');atomic_json(raw,actual);case['raw_sha256']=sha256(raw);save()
            old=json.loads(Path(plan['reference']).read_text());case['audit']=compare(actual,old,plan['sample']['opponent'],plan['branch_id'],plan['branch_hash']);case['status']='complete';save()
        for p,h in source_pins.items():
            if sha256(Path(p))!=h:raise RuntimeError('Pinned source/model/reference/state changed')
        record['status']='complete';record['verified_branches']=sorted({c['branch_id'] for c in record['cases']})
        if len(record['verified_branches'])!=11:raise RuntimeError('Not all branches covered')
    except BaseException as error:
        record.update(status='invalid',error=f'{type(error).__name__}: {error}')
        if record['cases'] and record['cases'][-1]['status']=='running':record['cases'][-1].update(status='invalid',error=record['error'])
        raise
    finally:
        timer.cancel()
        if env is not None:env.close()
        record['wall_seconds']=time.monotonic()-started;save()
    return record

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source','bundle','dataset','references','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--max-seconds',type=int,default=900)
    print(json.dumps(run(**vars(p.parse_args())),indent=2))
if __name__=='__main__':main()
