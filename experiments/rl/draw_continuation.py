"""Training-only native draw -> R2 inputs; exactly one initial checkpoint load."""
import argparse
import json
from pathlib import Path
from astra_play_sf2.config import atomic_json,load_config
from astra_play_sf2.runner import sha256
from .batch_env import BatchEnv
from .dataset import load_dataset


def evaluate(dataset,plan_path,output):
    groups,difficulty=load_dataset(dataset);plan=json.loads(plan_path.read_text())
    if len(plan['actions'])!=255:raise ValueError('This bounded diagnostic requires the real 255-action draw')
    if plan['native_round'].get('recognition')!='rl-native-equal-time-next-round-v3':raise ValueError('Not a captured draw')
    original=json.loads((plan_path.parent.parent/'manifest.json').read_text())
    options={'checkpoint':plan['checkpoint'],'lead':plan['lead']}
    if original['checkpoints'][plan['checkpoint']]['sha256']!=groups['train'][plan['checkpoint']]['sha256']:
        raise ValueError('Checkpoint identity changed')
    output.mkdir(parents=True,exist_ok=False);env=None
    result={'schema':'astra.rl-native-draw-continuation.v1','status':'initializing','training_only':True,
            'formal_clear':False,'source_plan_sha256':sha256(plan_path)}
    atomic_json(output/'result.json',result)
    try:
        env=BatchEnv(load_config(),output/'native',difficulty,checkpoints=groups['train'])
        here=Path(__file__).resolve().parent
        if 'TRAINING DIAGNOSTIC ONLY' not in (here/'batch_runtime.lua').read_text():raise ValueError('Stage diagnostic runtime first')
        result['runtime_sha256']=env.manifest['runtime_sha256']
        for staged,source in [('rl.lua','batch_runtime.lua'),('rl_settlement.lua','settlement.lua'),('rl_native_core.lua','native_continuous_core.lua')]:
            if result['runtime_sha256'][staged]!=sha256(here/source):raise RuntimeError('Staged runtime identity mismatch')
        response=env.batch_rpc({'op':'reset_rollout','reset':options,'count':256,'native_reference':1,'continue_after_draw':1,
                                'actions':plan['actions']+[6],'resets':[options]*256})
        evidence=response['continuous_draw'];atomic_json(output/'native-evidence.json',evidence)
        if evidence['load_calls']!=1 or evidence['prior_answers']!=0 or evidence['paused_native_frames']!=0:
            raise RuntimeError('Load/pause occurred during native continuation')
        if evidence['round']['confirmation']!=plan['native_round']['confirmation']:
            raise RuntimeError('Native draw differs from captured source')
        trace=evidence['trace'];start=evidence['round2_start']
        if [row['frame'] for row in trace]!=list(range(1,trace[-1]['frame']+1)):
            raise RuntimeError('Incomplete all-native-frame trace')
        by_frame={row['frame']:row for row in trace}
        starts=[row['frame'] for row in trace if any(e['kind']=='round_start' and e['round']==2 for e in row['events'])]
        if starts!=[start] or by_frame[start].get('deferred_input')!='LP':
            raise RuntimeError('R2 first input does not begin at native deferred phase')
        if any(by_frame[f]['consumed_input']!='LP' for f in range(start+1,start+13)):
            raise RuntimeError('R2 did not consume exactly the first twelve LP frames')
        if by_frame[start+12].get('deferred_input')!='':raise RuntimeError('Next action did not begin at twelve-frame boundary')
        period=trace[0]['state']['native_frame_period']
        if any(abs(b['state']['emulated_seconds']-a['state']['emulated_seconds']-period)>1e-7 for a,b in zip(trace,trace[1:])):
            raise RuntimeError('Native time gap or paused duplicate')
        result.update(status='complete',frames=len(trace),round2_start=start,round2_consumed_action_frames=12,
                      initial_loads=1,paused_native_frames=0,draw_confirmation_frame=evidence['round']['confirmation_frame'])
    except BaseException as error:
        result.update(status='invalid',error=f'{type(error).__name__}: {error}');raise
    finally:
        if env:env.close()
        atomic_json(output/'result.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True);parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(evaluate(args.dataset,args.plan,args.output),indent=2))
if __name__=='__main__':main()
