"""Training diagnostic: replay real actions with the original batch boundaries."""
import argparse,json
from pathlib import Path
from astra_play_sf2.config import atomic_json,load_config
from .batch_env import BatchEnv
from .dataset import load_dataset


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('dataset','plan','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--in-batch-start',action='store_true')
    args=parser.parse_args();plan=json.loads(args.plan.read_text());groups,level=load_dataset(args.dataset)
    episodes=[json.loads(l) for l in (args.plan.parent/'rl-batch-episodes.jsonl').read_text().splitlines()]
    prior=sum(r['steps'] for r in episodes if r['episode']<plan['episode'])
    actions=plan['actions'];segments=[];first=64-prior%64
    while sum(segments)<len(actions):segments.append(min(first if not segments else 64,len(actions)-sum(segments)))
    args.output.mkdir(parents=True,exist_ok=False);env=None
    result={'schema':'astra.rl-draw-boundary-probe.v1','status':'initializing','formal_clear':False,
            'prior_steps':prior,'segments':segments}
    try:
        env=BatchEnv(load_config(),args.output/'batch',level,checkpoints=groups['train'])
        options={'checkpoint':plan['checkpoint'],'lead':plan['lead']}
        if not args.in_batch_start:env.reset(options=options)
        offset=0;found=[]
        for i,n in enumerate(segments):
            chunk=(env.batch_rpc({'op':'reset_rollout','reset':options,'count':n,'actions':actions[offset:offset+n],'resets':[options]*n})
                   if args.in_batch_start and i==0 else env.batch_rollout(n,actions=actions[offset:offset+n],resets=[options]*n))
            atomic_json(args.output/f'block-{i}.json',chunk);found.extend(chunk['episodes']);offset+=n
        result.update(status='complete',episodes=found,matches_original_draw=any(
            r['native_round'].get('confirmation')==plan['native_round']['confirmation'] for r in found))
    except BaseException as error:result.update(status='invalid',error=f'{type(error).__name__}: {error}');raise
    finally:
        if env:env.close()
        atomic_json(args.output/'result.json',result)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
