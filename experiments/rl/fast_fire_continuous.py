"""Fixed-weight input-interface A/B: original vs 2/2/2-frame light fireball."""
import argparse
import json
from pathlib import Path
import signal
from functools import partial
from unittest.mock import patch

from astra_play_sf2.config import atomic_json, load_config
from astra_play_sf2.runner import sha256, seal_run
from . import continuous, native_continuous

HERE=Path(__file__).resolve().parent
ORIGINAL_STAGE=native_continuous.stage_native_policy
INTERFACES={'original':'actions15_fireball_3_3_2_neutral4_v1',
            'fast':'actions15_fireball_2_2_2_neutral6_v1'}


def stage_policy(run,level,payload,variant):
    if variant not in INTERFACES:raise ValueError('Unknown action variant')
    ORIGINAL_STAGE(run,level,payload)
    runtime=run/'training/runtime'
    path=runtime/'play.lua';source=path.read_text(encoding='utf-8')
    source=continuous.replace_once(source,"policy_kind='ppo',model_sha256=Model.model_sha256,",
        "policy_kind='ppo',model_sha256=Model.model_sha256,selection='deterministic_argmax',action_interface='"+INTERFACES[variant]+"',")
    if variant=='fast':
        (runtime/'rl_actions_base.lua').write_bytes((runtime/'rl_actions.lua').read_bytes())
        (runtime/'rl_actions.lua').write_bytes((HERE/'fast_fire_actions.lua').read_bytes())
        source=continuous.replace_once(source,"'training/runtime/rl_nn.lua','training/runtime/rl_actions.lua'",
            "'training/runtime/rl_nn.lua','training/runtime/rl_actions_base.lua','training/runtime/rl_actions.lua'")
    path.write_text(source,encoding='utf-8')
    sources={p.name:sha256(p) for p in sorted(runtime.iterdir()) if p.is_file()}
    atomic_json(run/'action-interface.json',{'variant':variant,'action_interface':INTERFACES[variant],
        'model_sha256':payload['model_sha256'],'selection':'deterministic_argmax',
        'decision_native_frames':12,'modified_action':12 if variant=='fast' else None,
        'action_source_sha256':{key:value for key,value in sources.items() if key in ('rl_actions.lua','rl_actions_base.lua')}})
    return sources


def evaluate(config,model,output,difficulty=3,attempts=3,speed='fast',variant='fast'):
    if variant not in INTERFACES:raise ValueError('Unknown action variant')
    with patch.object(native_continuous,'stage_native_policy',partial(stage_policy,variant=variant)):
        result=native_continuous.evaluate(config,model,output,difficulty,attempts,speed,False)
    output=Path(output).resolve()
    result.update(action_interface=INTERFACES[variant],variant=variant,selection='deterministic_argmax',
        evaluation_kind='fixed-weight motor-input interface A/B; not training or unchanged-policy certification',
        adapter_source_sha256={p.name:sha256(p) for p in (HERE/'fast_fire_actions.lua',Path(__file__))})
    result['action_interface_audit']={'ok':False,'reason':'Run did not complete'}
    if result['status']=='complete':
        try:
            for attempt in result['attempts']:
                for relative in attempt['matches']:
                    summary=json.loads((output/relative).read_text(encoding='utf-8'))['summary']
                    if summary.get('action_interface')!=INTERFACES[variant] or summary.get('selection')!='deterministic_argmax':
                        raise RuntimeError('Match action/selection identity mismatch')
            result['action_interface_audit']={'ok':True,'action_interface':INTERFACES[variant]}
        except Exception as error:
            result.update(status='invalid',error=f'Action interface audit: {type(error).__name__}: {error}')
            result['action_interface_audit']={'ok':False,'reason':str(error)}
    atomic_json(output/'result.json',result);seal_run(output)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--difficulty',type=int,choices=range(3,8),default=3)
    parser.add_argument('--attempts',type=int,default=3)
    parser.add_argument('--speed',choices=('normal','2x','4x','fast'),default='fast')
    parser.add_argument('--variant',choices=tuple(INTERFACES),required=True)
    args=parser.parse_args()
    def stop(_signal,_frame):raise KeyboardInterrupt('SIGTERM')
    signal.signal(signal.SIGTERM,stop)
    result=evaluate(load_config(),**vars(args));print(json.dumps(result,indent=2),flush=True)
    return 2 if result['status']!='complete' else 0 if any(a['outcome']=='rl_gameplay_clear' for a in result['attempts']) else 1


if __name__=='__main__':raise SystemExit(main())
