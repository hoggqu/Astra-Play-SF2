"""One training match from a captured draw start, then independent native replay."""
import argparse
import importlib
import json
import sys
from pathlib import Path
import numpy as np
from astra_play_sf2.config import atomic_json, load_config
from astra_play_sf2.runner import sha256
from .dataset import load_dataset
from .round_chain_audit import audit
from .round_chain_builder import build, PACKAGE

HERE=Path(__file__).resolve().parent


def run(dataset,plan_path,output):
    groups,difficulty=load_dataset(dataset);plan=json.loads(plan_path.read_text())
    if len(plan['actions'])!=255 or plan['native_round'].get('recognition')!='rl-native-equal-time-next-round-v3':
        raise ValueError('Expected real 255-action captured draw')
    parent=json.loads((plan_path.parent.parent/'manifest.json').read_text())
    cp=plan['checkpoint'];options={'checkpoint':cp,'lead':plan['lead']}
    if parent['checkpoints'][cp]['sha256']!=groups['train'][cp]['sha256']:raise ValueError('Checkpoint changed')
    output=output.resolve();output.mkdir(parents=True,exist_ok=False)
    build(output/'source');sys.path.insert(0,str(output/'source'))
    package=output/'source'/PACKAGE
    module=importlib.import_module(PACKAGE+'.batch_env');environment=None
    result={'schema':'astra.rl-round-chain-parity.v1','status':'running','training_only':True,
            'formal_clear':False,'source_plan_sha256':sha256(plan_path),'runs':[]}
    atomic_json(output/'result.json',result)
    try:
        environment=module.BatchEnv(load_config(),output/'chain',difficulty,checkpoints=groups['train'])
        result['runs'].append({'role':'chain','runtime_sha256':environment.manifest['runtime_sha256']})
        traces=[];rng=np.random.default_rng(707)
        first=plan['actions']+[6]
        for block in range(12):
            body={'op':'reset_rollout' if block==0 else 'rollout','reset':options,'count':256,
                  'actions':first if block==0 else rng.integers(0,15,256).tolist(),
                  'resets':[options]*256,'capture_trace':1}
            response=environment.batch_rpc(body)
            atomic_json(output/f'chain-block-{block:03d}.json',response)
            traces.extend(row for row in response['trace'] if row['loads']==1)
            if response['chain_metrics']['matches']:break
        else:raise RuntimeError('Bounded capture did not finish the opponent match')
        captured=json.loads((output/'chain/training/rl-chain-match-00001.json').read_text())
        if captured['rounds'][0]['recognition']!='rl-native-equal-time-next-round-v3':raise RuntimeError('Original draw not reproduced')
        if captured['rounds'][0]['confirmation']!=plan['native_round']['confirmation']:raise RuntimeError('Draw state differs')
        if len(captured['rounds'])<3:raise RuntimeError('Required R2-to-R3 boundary was not exercised')
        atomic_json(output/'captured-match.json',captured)
        environment.close();environment=None
        # Only this completed training diagnostic's private source copy changes.
        # The two staged runtime manifests retain the exact distinct bytes used.
        (package/'batch_runtime.lua').write_bytes((HERE/'chain_reference_runtime.lua').read_bytes())
        environment=module.BatchEnv(load_config(),output/'reference',difficulty,checkpoints=groups['train'])
        result['runs'].append({'role':'native_reference','runtime_sha256':environment.manifest['runtime_sha256']})
        reference=environment.batch_rpc({'op':'reference_match','reset':options,
                  'actions':captured['actions'],'frames':captured['frames']})
        atomic_json(output/'reference-evidence.json',reference)
        if reference['opening']!=captured['opening']:raise RuntimeError('Initial phase mismatch')
        if len(traces)!=len(reference['trace']):raise RuntimeError('Different native frame coverage')
        for a,b in zip(traces,reference['trace']):
            if any(a[k]!=b[k] for k in ('frame','state','phase','round','events')):
                atomic_json(output/'mismatch.json',{'chain':a,'reference':b});raise RuntimeError('Native chain state/event mismatch')
        if reference['rounds']!=captured['rounds']:raise RuntimeError('Different mature round results')
        if reference['initial_loads']!=1 or reference['in_play_pauses']!=0:raise RuntimeError('Reference load/pause violation')
        for expected,actual in zip(captured['actions'],reference['decisions']):
            if any(expected[k]!=actual[k] for k in ('frame','round','action')):raise RuntimeError('Decision cadence mismatch')
        for round_index in range(2,len(captured['rounds'])+1):
            decisions=[r for r in reference['decisions'] if r['round']==round_index]
            if not decisions or not decisions[0]['reset_history']:raise RuntimeError('Missing native round history reset')
            if len(decisions)>1 and decisions[1]['frame']-decisions[0]['frame']!=12:raise RuntimeError('Wrong first-action cadence')
        result['rollout_audit']=audit(output)
        result.update(status='complete',native_frames=len(traces),decisions=len(reference['decisions']),
                      rounds=[r['outcome'] for r in captured['rounds']],draw_confirmation_frame=captured['rounds'][0]['confirmation_frame'],
                      reference_initial_loads=1,reference_in_play_pauses=0)
    except BaseException as error:
        result.update(status='invalid',error=f'{type(error).__name__}: {error}');raise
    finally:
        if environment:environment.close()
        atomic_json(output/'result.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','plan','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();print(json.dumps(run(a.dataset,a.plan,a.output),indent=2))
if __name__=='__main__':main()
