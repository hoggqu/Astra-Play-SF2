"""Replay a captured REAL draw plan through native deployment and batch drivers."""
import argparse
import json
from pathlib import Path
import numpy as np
from astra_play_sf2.config import atomic_json,load_config
from astra_play_sf2.runner import sha256
from .batch_env import BatchEnv
from .dataset import load_dataset


def evaluate(dataset,plan_path,output,in_batch_start=False):
    groups,difficulty=load_dataset(dataset)
    plan=json.loads(plan_path.read_text())
    source_manifest=json.loads((plan_path.parent.parent/'manifest.json').read_text())
    index=plan['checkpoint'];options={'checkpoint':index,'lead':plan['lead']}
    if source_manifest['checkpoints'][index]['sha256']!=groups['train'][index]['sha256']:
        raise ValueError('Source draw/checkpoint identity mismatch')
    native=plan['native_round']
    if native.get('recognition')!='rl-native-equal-time-next-round-v3' or native['outcome']!='draw' or 'confirmation' not in native:
        raise ValueError('Need a captured native next-round-confirmed draw')
    if not 1<=len(plan['actions'])<=256:raise ValueError('Draw plan must fit 256 real decisions')
    actions=plan['actions']+[0]*(256-len(plan['actions']))
    output.mkdir(parents=True,exist_ok=False)
    result={'status':'initializing','training_only':True,'formal_clear':False,
            'scope':'checkpoint training harness through natural R2 opening confirmation; resets thereafter',
            'source_plan_sha256':sha256(plan_path),'checkpoint_sha256':groups['train'][index]['sha256']}
    atomic_json(output/'result.json',result)
    envs=[]
    try:
        for name in ('native-reference','batch'):
            envs.append(BatchEnv(load_config(),output/name,difficulty,checkpoints=groups['train']))
        here=Path(__file__).resolve().parent
        if 'rl-native-equal-time-next-round-v3' not in (here/'settlement.lua').read_text():
            raise RuntimeError('Stage the v3 adapter in the canonical runtime role first')
        result['runtime_sha256']={env.run.name:env.manifest['runtime_sha256'] for env in envs}
        for env in envs:
            for staged,source in [('rl.lua','batch_runtime.lua'),('rl_settlement.lua','settlement.lua'),
                                  ('rl_native_core.lua','native_continuous_core.lua')]:
                if env.manifest['runtime_sha256'][staged]!=sha256(here/source):
                    raise RuntimeError('Staged runtime differs from frozen parity source')
        if in_batch_start:
            chunks=[env.batch_rpc({'op':'reset_rollout','reset':options,'count':256,'actions':actions,'resets':[options]*256,'native_reference':i==0}) for i,env in enumerate(envs)]
        else:
            initial=[env.reset(seed=107,options=options)[0] for env in envs]
            np.testing.assert_array_equal(initial[0],initial[1])
            chunks=[env.batch_rollout(256,actions=actions,resets=[options]*256,native_reference=i==0) for i,env in enumerate(envs)]
        result['initial_phase']='live reset callback' if in_batch_start else 'paused reset RPC'
        for i,chunk in enumerate(chunks):atomic_json(output/f'rollout-{i}.json',chunk)
        for step,(old,new) in enumerate(zip(chunks[0]['transitions'],chunks[1]['transitions'])):
            for key in ('observation','state','reward','done','episode_start','action','frames'):
                if old[key]!=new[key]:raise RuntimeError(f'Draw parity mismatch step {step} field {key}')
        if chunks[0]['episodes']!=chunks[1]['episodes']:
            raise RuntimeError('Mature native round metadata differs')
        draws=[r for r in chunks[0]['episodes'] if r['native_round'].get('recognition')=='rl-native-equal-time-next-round-v3']
        if len(draws)!=1:raise RuntimeError('Expected actual next-round-confirmed draw, not ordinary R1 parity')
        captured=[json.loads(next((env.run/'training').glob('rl-draw-action-plan-*.json')).read_text()) for env in envs]
        if captured[0]['settlement_trace']!=captured[1]['settlement_trace']:
            raise RuntimeError('Per-native-frame settlement/next-round trajectory mismatch')
        if captured[0]['native_round']['confirmation']!=plan['native_round']['confirmation']:
            raise RuntimeError('Replayed draw differs from the original captured native opening')
        result.update(status='complete',decisions=256,draws=1,
                      settlement_native_frames=len(captured[0]['settlement_trace']),
                      settled_frame=draws[0]['native_round']['settled_frame'],
                      confirmation_frame=draws[0]['native_round']['confirmation_frame'])
    except BaseException as error:
        result.update(status='invalid',error=f'{type(error).__name__}: {error}');raise
    finally:
        for env in envs:env.close()
        atomic_json(output/'result.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--in-batch-start',action='store_true')
    args=parser.parse_args();print(json.dumps(evaluate(args.dataset,args.plan,args.output,args.in_batch_start),indent=2))
if __name__=='__main__':main()
