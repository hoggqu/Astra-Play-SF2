"""Categorical evaluation of fixed PPO weights; distinct from argmax evaluation."""
import argparse
from collections import deque
import copy
import json
from pathlib import Path
import re
import signal
from unittest.mock import patch

import numpy as np

from astra_play_sf2.config import atomic_json, load_config
from astra_play_sf2.runner import sha256, seal_run
from . import continuous, native_continuous
from .export import export_policy, lua_literal

HERE = Path(__file__).resolve().parent
MODULUS = 2147483647
MULTIPLIER = 48271
ORIGINAL_EXPORT = export_policy
ORIGINAL_NATIVE_STAGE = native_continuous.stage_native_policy
REASON = re.compile(r'rl_action_(\d+);policy_draw=(\d+);policy_state=(\d+);policy_logprob=([-+\d.eE]+)')


def next_random(state):
    if type(state) is not int or not 1 <= state < MODULUS:
        raise ValueError('policy-seed must be 1..2147483646')
    state = state*MULTIPLIER % MODULUS
    return state, (state-1)/(MODULUS-1)


def stochastic_payload(payload, seed):
    next_random(seed)
    payload = copy.deepcopy(payload)
    if payload['selection'] != 'deterministic_argmax':
        raise ValueError('Expected original PPO export')
    payload.update(selection='categorical_softmax', policy_seed=seed, policy_prng='park_miller_48271_v1')
    return payload


def stage_policy(run, level, payload):
    ORIGINAL_NATIVE_STAGE(run, level, payload)
    runtime = run/'training/runtime'
    (runtime/'rl_stochastic_nn.lua').write_bytes((HERE/'stochastic_nn.lua').read_bytes())
    path = runtime/'play.lua';source = path.read_text(encoding='utf-8')
    source = continuous.replace_once(source, "local NN=assert(loadfile('training/runtime/rl_nn.lua'))()",
        "local BaseNN=assert(loadfile('training/runtime/rl_nn.lua'))()\n"
        "local NN=assert(loadfile('training/runtime/rl_stochastic_nn.lua'))()(BaseNN)")
    source = continuous.replace_once(source, 'policy_kind=\'ppo\',model_sha256=Model.model_sha256,',
        "policy_kind='ppo',model_sha256=Model.model_sha256,selection=Model.selection,policy_seed=Model.policy_seed,policy_prng=Model.policy_prng,")
    source = continuous.replace_once(source, "'training/runtime/rl_nn.lua','training/runtime/rl_actions.lua'",
        "'training/runtime/rl_nn.lua','training/runtime/rl_stochastic_nn.lua','training/runtime/rl_actions.lua'")
    path.write_text(source, encoding='utf-8')
    atomic_json(run/'sampling-protocol.json', {key:payload[key] for key in ('model_sha256','selection','policy_seed','policy_prng')})
    return {p.name:sha256(p) for p in sorted(runtime.iterdir()) if p.is_file()}


def features(state):
    # Float64 matches Lua accumulation; feature definition is unchanged.
    a,b = state['p1'],state['p2']
    values = [a['hp']/144,b['hp']/144,state['timer']/99,(b['x']-a['x'])/512,a['x']/1024,b['x']/1024,
              (a['y']-40)/256,(b['y']-40)/256,float(a['y']==40),float(b['y']==40)]
    values += [float(b['char']==i) for i in range(12)]
    for p in (a,b):values += [float(p['a']==i) for i in range(32)]
    return np.clip(values,-1,1)


def audit_sampling(output, result, payload):
    state = payload['policy_seed'];draws = 0
    layers = [(np.asarray(layer['weight'],dtype=np.float64),np.asarray(layer['bias'],dtype=np.float64)) for layer in payload['layers']]
    for attempt in result['attempts']:
        for relative in attempt['matches']:
            match = json.loads((output/relative).read_text(encoding='utf-8'));summary=match['summary']
            if any(summary.get(key)!=payload[key] for key in ('selection','policy_seed','policy_prng')):
                raise RuntimeError('Stochastic match identity mismatch')
            trace=match['trace'];columns=trace['columns'];timer_i=columns.index('timer');frame_i=columns.index('frame')
            timers={row[frame_i]:row[timer_i] for row in trace['rows']}
            timers[0]=next(e['state']['timer'] for e in match['events'] if e['kind']=='match_start')
            history=deque(maxlen=4);last_round=None
            for decision in trace['decisions']:
                m=REASON.fullmatch(decision.get('reason',''))
                if not m:raise RuntimeError('Missing categorical decision evidence')
                action,draw,logged_state=int(m[1]),int(m[2]),int(m[3]);logprob=float(m[4])
                state,u=next_random(state);draws+=1
                if draw!=draws or logged_state!=state:raise RuntimeError('Policy RNG stream reset or skipped draws')
                f=features({'p1':decision['ken'],'p2':decision['cpu'],'timer':timers[decision['frame']]})
                if decision['round']!=last_round:
                    history.clear();history.extend([f]*4);last_round=decision['round']
                else:history.append(f)
                x=np.concatenate(history)
                for index,(weight,bias) in enumerate(layers):
                    x=weight@x+bias
                    if index<len(layers)-1:x=np.tanh(x)
                weights=np.exp(x-np.max(x));prob=weights/weights.sum()
                selected=min(int(np.searchsorted(np.cumsum(prob),u,side='right')),len(prob)-1)
                if action!=selected or abs(logprob-np.log(prob[action]))>1e-8:
                    raise RuntimeError('Categorical action/logprob differs from fixed network')
    return {'ok':True,'decisions':draws,'last_policy_state':state,'policy_seed':payload['policy_seed'],
            'sequence_continues_across_rounds_matches_and_coins':True}


def evaluate(config, model, output, difficulty=3, attempts=3, speed='fast', policy_seed=42):
    next_random(policy_seed)
    payload=stochastic_payload(ORIGINAL_EXPORT(model),policy_seed)
    with patch.object(continuous,'export_policy',return_value=payload), \
         patch.object(native_continuous,'stage_native_policy',stage_policy):
        result=native_continuous.evaluate(config,model,output,difficulty,attempts,speed,False)
    output=Path(output).resolve()
    result.update(selection=payload['selection'],policy_seed=policy_seed,policy_prng=payload['policy_prng'],
                  evaluation_kind='fixed-weight categorical policy; separate from deterministic argmax',
                  stochastic_source_sha256={p.name:sha256(p) for p in (HERE/'stochastic_nn.lua',Path(__file__))})
    result['sampling_audit']={'ok':False,'reason':'Run did not complete'}
    if result['status']=='complete':
        try:result['sampling_audit']=audit_sampling(output,result,payload)
        except Exception as error:
            result.update(status='invalid',error=f'Stochastic audit: {type(error).__name__}: {error}')
            result['sampling_audit']={'ok':False,'reason':str(error)}
    atomic_json(output/'result.json',result);seal_run(output)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--difficulty',type=int,choices=range(3,8),default=3)
    parser.add_argument('--attempts',type=int,default=3)
    parser.add_argument('--speed',choices=('normal','2x','4x','fast'),default='fast')
    parser.add_argument('--policy-seed',type=int,required=True)
    args=parser.parse_args()
    def stop(_signal,_frame):raise KeyboardInterrupt('SIGTERM')
    signal.signal(signal.SIGTERM,stop)
    result=evaluate(load_config(),**vars(args));print(json.dumps(result,indent=2),flush=True)
    return 2 if result['status']!='complete' else 0 if any(a['outcome']=='rl_gameplay_clear' for a in result['attempts']) else 1


if __name__=='__main__':raise SystemExit(main())
