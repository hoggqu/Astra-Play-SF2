"""Read-only audit of saved chain/native evidence, including PPO round boundaries."""
import argparse
import json
from pathlib import Path
import numpy as np


def features(s):
    a,b=s['p1'],s['p2']
    values=[a['hp']/144,b['hp']/144,s['timer']/99,(b['x']-a['x'])/512,
            a['x']/1024,b['x']/1024,(a['y']-40)/256,(b['y']-40)/256,
            int(a['y']==40),int(b['y']==40)]
    values += [int(b['char']==i) for i in range(12)]
    for p in (a,b):values += [int(p['a']==i) for i in range(32)]
    return np.clip(values,-1,1)


def audit(directory):
    root=Path(directory)
    blocks=[json.loads(p.read_text()) for p in sorted(root.glob('chain-block-*.json'))]
    reference=json.loads((root/'reference-evidence.json').read_text())
    match=json.loads((root/'captured-match.json').read_text())
    count=len(match['rounds'])
    allrows=[r for b in blocks for r in b['transitions']]
    rows=[r for r in allrows if r['episode']<=count]
    episodes=[r for b in blocks for r in b['episodes'] if r['episode']<=count]
    if len(rows)!=len(reference['decisions']):raise RuntimeError('Missing/extra transition')
    if len(episodes)!=count or [r['native_round'] for r in episodes]!=reference['rounds']:
        raise RuntimeError('Mature rewards not emitted exactly once per round')
    for a,b in zip(allrows,allrows[1:]):
        if a['done']!=b['episode_start']:raise RuntimeError('GAE would cross a round boundary')
    if allrows[-1]['done']!=blocks[-1]['episode_start']:raise RuntimeError('Wrong final bootstrap mask')
    history=[];max_error=0.
    for row,decision in zip(rows,reference['decisions']):
        f=features(decision['state'])
        history=[f]*4 if decision['reset_history'] else history[1:]+[f]
        if len(history)!=4:raise RuntimeError('Uninitialized observation history')
        error=float(np.max(np.abs(np.asarray(row['observation'])-np.concatenate(history))))
        if error>1e-12:raise RuntimeError(f'Observation history differs by {error}')
        max_error=max(max_error,error)
        if row['episode_start']!=decision['reset_history']:raise RuntimeError('Wrong round initial observation')
        if row['round']!=decision['round'] or row['action']!=decision['action']:raise RuntimeError('Wrong transition action identity')
        native=match['rounds'][row['round']-1]
        after=native['settled'] if row['done'] else row['state']
        def hp(s,p):return max(0,min(144,s[p]['hp']))
        before=decision['state']
        expected=.25*((hp(before,'p2')-hp(after,'p2'))-(hp(before,'p1')-hp(after,'p1')))/144
        if row['done']:expected += {'win':1,'loss':-1,'draw':0}[native['outcome']]
        if abs(row['reward']-expected)>1e-12:raise RuntimeError('Reward differs from previous-round raw state')
    return {'schema':'astra.rl-round-chain-rollout-audit.v1','status':'complete','transitions':len(rows),
            'round_rewards':count,'observation_max_error':max_error,'gae_round_masks':True,
            'previous_round_terminal_rewards':True}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);a=p.parse_args()
    print(json.dumps(audit(a.directory),indent=2))
