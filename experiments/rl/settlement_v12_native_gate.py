"""Replay a retained zero-HP winner late KO through the complete native match."""
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


def run(source, failure, dataset, output, snapshot_frame=5432, block=64):
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
            else:
                body=reference_runtime().replace('if frames==pending.frames then', "if core.phase=='complete' then")
            anchor='frames=frames+1;'
            assert body.count(anchor)==1
            body=body.replace(anchor,anchor+f"if frames=={int(snapshot_frame)} then m.screens[':screen']:snapshot('training/ko-{int(snapshot_frame)}.png') end;")
            (package/'batch_runtime.lua').write_text(body)
            sys.path.insert(0,str(root));unload()
            try:
                module=importlib.import_module(source.name+'.batch_env')
                loader=importlib.import_module(source.name+'.dataset')
                groups,difficulty=loader.load_dataset(dataset)
                worker_manifest=json.loads((failure.parent.parent/'manifest.json').read_text())
                cp=original['checkpoint']
                assert worker_manifest['checkpoints'][cp]['sha256']==groups['train'][cp]['sha256']
                options={'checkpoint':cp,'lead':original['lead']}
                env=module.BatchEnv(load_config(),output/role,difficulty,checkpoints=groups['train'])
                if role=='batch':
                    actions=[a['action'] for a in original['match_actions']]
                    trace=[];transitions=[];episodes=[]
                    for offset in range(0,len(actions),block):
                        chunk=actions[offset:offset+block]
                        answer=env.batch_rpc({'op':'reset_rollout' if offset==0 else 'rollout',
                            'reset':options,'actions':chunk,'count':len(chunk),'resets':[options]*len(chunk),'capture_trace':1})
                        atomic_json(output/f'batch-{offset:03d}.json',answer)
                        trace.extend(answer['trace']);transitions.extend(answer['transitions']);episodes.extend(answer['episodes'])
                    plan=original['match_actions'];frames=trace[-1]['frame']
                    assert len(transitions)==len(plan)
                    assert answer['chain_metrics']['matches']==1
                    assert answer['chain_metrics']['loads']==2
                else:
                    reference=env.batch_rpc({'op':'reference_match','reset':options,'actions':plan,'frames':frames})
                    atomic_json(output/'reference.json',reference)
            finally:
                if env:env.close();env=None
                sys.path.remove(str(root));unload()
        assert reference['initial_loads']==1 and reference['in_play_pauses']==0
        assert len(reference['trace'])==len(trace)
        assert [row['frame'] for row in trace]==list(range(1,len(trace)+1))
        period=reference['opening']['native_frame_period']
        last=reference['opening']['emulated_seconds']
        for row in trace:
            assert abs(row['state']['emulated_seconds']-last-period)<1e-7
            last=row['state']['emulated_seconds']
        for a,b in zip(trace,reference['trace']):
            # patch_snapshot adds the actually consumed native input ports to
            # both state objects; batch traces do not expose held-input text.
            if any(a[k]!=b[k] for k in ('frame','state','phase','round','events')):
                atomic_json(output/'mismatch.json',{'batch':a,'native':b})
                raise RuntimeError('Batch/native KO trajectory mismatch')
        np.testing.assert_array_equal(np.asarray([t['observation'] for t in transitions],dtype=np.float32),
                                      np.asarray([d['observation'] for d in reference['decisions']],dtype=np.float32))
        rounds=reference['rounds']
        assert len(rounds)==2 and all(r['outcome']=='win' for r in rounds)
        assert len(original['rounds'])==1
        settled=rounds[-1]
        assert settled['score']==[2,0]
        assert settled['recognition']=='rl-native-zero-winner-time-ko-v12'
        assert settled['frame']==5432 and settled['stable_award_frames']==360
        assert settled['stop']['timer']==0 and settled['settled']['timer']==0
        assert settled['settled']['p1']['hp']==0 and settled['settled']['p2']['hp']==-1
        edge=settled['late_ko']
        assert edge['frame']==5071 and edge['previous']['frame']==5070
        assert edge['previous']['state']['p1']['hp']==0 and edge['previous']['state']['p2']['hp']==9
        assert settled['ko_award']['award_frame']==5072
        assert trace[-1]['phase']=='complete'
        assert [t['done'] for t in transitions].count(True)==2
        assert [e['outcome'] for e in episodes]==['win','win']
        # Reproduce raw snapshots retained from the original first round.
        compared=0
        for old in original['trace']:
            if old['frame']>settled['frame']:break
            current=trace[old['frame']-1]['state']
            for key in ('p1','p2','timer','emulated_seconds'):
                assert current[key]==old['state'][key],(old['frame'],key)
            compared+=1
        result.update(status='complete',frames=frames,decisions=len(transitions),
                      native_result=settled,
                      all_observations_equal=True,all_native_states_ports_equal=True,
                      compared_original_frames=compared,terminal_transitions=2,
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
    p.add_argument('--snapshot-frame',type=int,default=5432)
    p.add_argument('--block',type=int,choices=(1,16,32,64,128,256),default=64)
    r=run(**vars(p.parse_args()));print(json.dumps({k:v for k,v in r.items() if k!='native_result'},indent=2))
