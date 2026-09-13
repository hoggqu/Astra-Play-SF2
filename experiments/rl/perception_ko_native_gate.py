"""Replay a retained KO failure and verify native next-round continuation."""
import argparse
import importlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from astra_play_sf2.config import atomic_json, load_config
from astra_play_sf2.runner import sha256
from .perception_native_gate import reference_runtime
from .pulsed_native_gate import patch_snapshot
from .versioned_campaign import source_pins, require_pins


def run(source, failure, dataset, output, snapshot_frame=1830, block=64):
    source,failure,dataset,output=map(lambda x:Path(x).resolve(),(source,failure,dataset,output))
    identity,pins=source_pins(source.parent,source.name)
    original=json.loads(failure.read_text());output.mkdir(parents=True,exist_ok=False)
    result={'status':'running','training_only':True,'formal_clear':False,
            'source_build_sha256':pins['build.json'],'failure_sha256':sha256(failure)}
    atomic_json(output/'result.json',result);started=time.monotonic();env=None
    def unload():
        for name in list(sys.modules):
            if name==source.name or name.startswith(source.name+'.'):del sys.modules[name]
    try:
        for role in ('batch','native'):
            root=output/'sources'/role
            shutil.copytree(source.parent,root,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
            package=root/source.name
            if role=='batch':
                body=patch_snapshot((package/'batch_runtime.lua').read_text())
                anchor='partial_episode=partial(),chain_metrics=metrics,trace=pending.trace}'
                assert body.count(anchor)==1
                body=body.replace(anchor,'partial_episode=partial(),chain_metrics=metrics,trace=pending.trace,planned_actions=match_actions}')
            else:body=reference_runtime()
            anchor='frames=frames+1;'
            assert body.count(anchor)==1
            body=body.replace(anchor,anchor+f"if frames=={int(snapshot_frame)} then m.screens[':screen']:snapshot('training/ko-{int(snapshot_frame)}.png') end;")
            (package/'batch_runtime.lua').write_text(body)
            sys.path.insert(0,str(root));unload()
            try:
                module=importlib.import_module(source.name+'.batch_env')
                loader=importlib.import_module(source.name+'.dataset')
                groups,difficulty=loader.load_dataset(dataset)
                options={'checkpoint':original['checkpoint'],'lead':original['lead']}
                env=module.BatchEnv(load_config(),output/role,difficulty,checkpoints=groups['train'])
                if role=='batch':
                    actions=[a['action'] for a in original['match_actions']]+[0]*16
                    trace=[];transitions=[];episodes=[]
                    for offset in range(0,len(actions),block):
                        chunk=actions[offset:offset+block]
                        answer=env.batch_rpc({'op':'reset_rollout' if offset==0 else 'rollout',
                            'reset':options,'actions':chunk,'count':len(chunk),'resets':[options]*len(chunk),'capture_trace':1})
                        atomic_json(output/f'batch-{offset:03d}.json',answer)
                        trace.extend(answer['trace']);transitions.extend(answer['transitions']);episodes.extend(answer['episodes'])
                    plan=answer['planned_actions'];frames=trace[-1]['frame']
                    assert plan[:len(original['match_actions'])]==original['match_actions']
                    plan=plan+[{'action':0,'round':plan[-1]['round'],'frame':frames}]
                else:
                    reference=env.batch_rpc({'op':'reference_match','reset':options,'actions':plan,'frames':frames})
                    atomic_json(output/'reference.json',reference)
            finally:
                if env:env.close();env=None
                sys.path.remove(str(root));unload()
        assert reference['initial_loads']==1 and reference['in_play_pauses']==0
        assert len(reference['trace'])==len(trace)
        for a,b in zip(trace,reference['trace']):
            if any(a[k]!=b[k] for k in ('frame','state','phase','round','events')):
                atomic_json(output/'mismatch.json',{'batch':a,'native':b})
                raise RuntimeError('Batch/native KO trajectory mismatch')
        np.testing.assert_array_equal(np.asarray([t['observation'] for t in transitions],dtype=np.float32),
                                      np.asarray([d['observation'] for d in reference['decisions'][:-1]],dtype=np.float32))
        rounds=reference['rounds'];assert len(rounds)==1 and rounds[0]['outcome']=='win'
        assert rounds[0]['score']==[1,0] and rounds[0]['frame']>=original['terminal_frame']+360
        after=[d for d in reference['decisions'] if d['round']==2]
        assert len(after)>=2 and after[0]['reset_history'] and after[1]['frame']-after[0]['frame']==12
        # All pre-result raw game snapshots must reproduce the retained failure.
        for old in original['trace']:
            if old['frame']>rounds[0]['frame']:break
            current=trace[old['frame']-1]['state']
            for key in ('p1','p2','timer','emulated_seconds'):
                assert current[key]==old['state'][key],(old['frame'],key)
        result.update(status='complete',frames=frames,decisions=len(transitions),
                      native_result=rounds[0],next_round_decisions=len(after),
                      all_observations_equal=True,all_native_states_ports_equal=True,
                      initial_loads=1,in_play_pauses=0)
        require_pins(source.parent,pins)
    except BaseException as error:
        result.update(status='invalid',error=f'{type(error).__name__}: {error}');raise
    finally:
        result['wall_seconds']=time.monotonic()-started;atomic_json(output/'result.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('source','failure','dataset','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--snapshot-frame',type=int,default=1830)
    p.add_argument('--block',type=int,choices=(1,16,32,64,128,256),default=64)
    r=run(**vars(p.parse_args()));print(json.dumps({k:v for k,v in r.items() if k!='native_result'},indent=2))
