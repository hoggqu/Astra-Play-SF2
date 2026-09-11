"""Bounded training-only TIME transition replay, then natural R2 first inputs."""
import argparse
import importlib
import json
from pathlib import Path
import shutil
import sys
from astra_play_sf2.config import atomic_json,load_config
from astra_play_sf2.runner import sha256
HERE=Path(__file__).resolve().parent


def run(source,trace,output):
    source=source.resolve();trace=trace.resolve();output=output.resolve()
    raw=json.loads(trace.read_text());worker=trace.parent.parent;manifest=json.loads((worker/'manifest.json').read_text())
    cp=raw['checkpoint'];sample=dict(manifest['checkpoints'][cp]);state=trace.parent/f'rl-start-{cp:03d}.sta'
    if sha256(state)!=sample['sha256']:raise RuntimeError('Original checkpoint changed')
    sample.update(path=str(state),difficulty=3)
    output.mkdir(parents=True,exist_ok=False);package=output/'source'/source.name
    shutil.copytree(source,package,ignore=shutil.ignore_patterns('__pycache__'))
    (package/'settlement.lua').write_bytes((HERE/'settlement_v5.lua').read_bytes())
    runtime=(HERE/'chain_reference_runtime.lua').read_text()
    runtime=runtime.replace(" local planned=assert(pending.actions[index],'Reference action plan exhausted')", " local planned=pending.actions[index] or {frame=frames,round=core.round,action=0}")
    runtime=runtime.replace("  if core.phase=='complete' then", "  if frames==pending.frames then")
    runtime=runtime.replace("   assert(index==#pending.actions+1,'Reference did not consume complete action plan')\n",'')
    runtime=runtime.replace('opening=opening,initial_loads=1,in_play_pauses=0,chain_metrics={}',
                            'opening=opening,initial_loads=1,in_play_pauses=0,chain_metrics={},partial=true')
    (package/'batch_runtime.lua').write_text(runtime)
    sys.path.insert(0,str(package.parent));module=importlib.import_module(source.name+'.batch_env')
    env=None;result={'schema':'astra.rl-time-adjacent-native-probe.v1','status':'invalid','training_only':True,
        'partial':True,'formal_clear':False,'source_trace_sha256':sha256(trace),'checkpoint_sha256':sha256(state),
        'original_checkpoint_index':cp,'lead':raw['lead']}
    try:
        env=module.BatchEnv(load_config(),output/'native',3,checkpoints=[sample]);result['runtime_sha256']=env.manifest['runtime_sha256']
        response=env.batch_rpc({'op':'reference_match','reset':{'checkpoint':0,'lead':raw['lead']},'frames':4050,'actions':raw['match_actions']})
        atomic_json(output/'evidence.json',response)
        if response['opening']!=raw['opening']:raise RuntimeError('Reference load phase differs from original')
        if response['initial_loads']!=1 or response['in_play_pauses']!=0:raise RuntimeError('In-play lifecycle changed')
        byframe={row['frame']:row for row in response['trace']}
        if sorted(byframe)!=list(range(1,4051)):raise RuntimeError('Missing native frame coverage')
        for row in raw['trace']:
            if row['frame']<=3434 and row['state']!=byframe[row['frame']]['state']:
                atomic_json(output/'mismatch.json',{'original':row,'replay':byframe[row['frame']]});raise RuntimeError('Original settlement state differs')
        period=response['opening']['native_frame_period'];last=response['opening']['emulated_seconds']
        for row in response['trace']:
            if abs(row['state']['emulated_seconds']-last-period)>1e-7:raise RuntimeError('Native frame timing gap')
            last=row['state']['emulated_seconds']
        rounds=response['rounds']
        if len(rounds)!=1 or rounds[0]['outcome']!='loss' or rounds[0]['recognition']!='rl-native-time-pip-adjacent-ko-v5' or rounds[0]['frame']!=3434:
            raise RuntimeError('Expected actual mature TIME loss missing')
        r2=[d for d in response['decisions'] if d['round']==2]
        if len(r2)<2 or not r2[0]['reset_history'] or r2[1]['frame']-r2[0]['frame']!=12:
            raise RuntimeError('Missing natural R2 history reset/first macro cadence')
        first=r2[0]['frame']
        if any(byframe[f]['consumed_input']!='' for f in range(first+1,first+13)):raise RuntimeError('R2 first neutral macro differs')
        result.update(status='complete',frames=4050,initial_loads=1,in_play_pauses=0,
            outcome='loss',score=[0,1],settled_frame=3434,round2_start=first,round2_first_macro_frames=12,
            compared_original_frames=3434-raw['terminal_frame']+1)
    except BaseException as error:
        result['error']=f'{type(error).__name__}: {error}';raise
    finally:
        if env:env.close()
        atomic_json(output/'result.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','trace','output'):p.add_argument('--'+name,type=Path,required=True)
    print(json.dumps(run(**vars(p.parse_args())),indent=2))
if __name__=='__main__':main()
