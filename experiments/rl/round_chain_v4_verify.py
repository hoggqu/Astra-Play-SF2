"""Independent fixed-action chain/native replay of a real captured double-KO match."""
import argparse
import importlib
import json
from pathlib import Path
import shutil
import sys
from astra_play_sf2.config import atomic_json,load_config
from astra_play_sf2.runner import sha256
from .dataset import load_dataset
from .round_chain_audit import audit

HERE=Path(__file__).resolve().parent


def run(source,training,original,dataset,output,chain_evidence=None):
    source=source.resolve();training=training.resolve();output=output.resolve()
    failed=json.loads(original.read_text());groups,difficulty=load_dataset(dataset)
    candidates=[]
    for path in (training/'worker-04/training').glob('rl-chain-match-*.json'):
        value=json.loads(path.read_text())
        matches=[r for r in value['rounds'] if r.get('recognition')=='rl-native-double-ko-next-round-v4'
                 and r['confirmation']==failed['current_state'] and r['stop']==failed['round_stop']]
        if matches:candidates.append((path,value,matches[0]))
    if len(candidates)!=1:raise RuntimeError('Expected exactly one captured original double-KO match')
    plan_path,captured,draw=candidates[0]
    before=sum(len(json.loads(p.read_text())['actions']) for p in sorted(plan_path.parent.glob('rl-chain-match-*.json')) if p.name<plan_path.name)
    first_count=64-before%64
    cp=captured['checkpoint'];options={'checkpoint':cp,'lead':captured['lead']}
    manifest=json.loads((plan_path.parent.parent/'manifest.json').read_text())
    if manifest['checkpoints'][cp]['sha256']!=groups['train'][cp]['sha256']:raise RuntimeError('Checkpoint identity changed')
    if len(captured['rounds'])<=draw['round']:raise RuntimeError('Captured match lacks post-draw round continuation')
    output.mkdir(parents=True,exist_ok=False)
    atomic_json(output/'captured-match.json',captured)
    result={'schema':'astra.rl-double-ko-chain-parity.v1','status':'running','training_only':True,
            'formal_clear':False,'first_batch_decisions':first_count,'plan_sha256':sha256(plan_path),'original_trace_sha256':sha256(original),'runs':[]}
    environment=None;trace=[]
    if chain_evidence:
        for p in sorted(chain_evidence.glob('chain-block-*.json')):
            shutil.copy2(p,output/p.name)
            trace.extend(r for r in json.loads(p.read_text())['trace'] if r['loads']==1)
    try:
        for role in (('reference',) if chain_evidence else ('chain','reference')):
            root=output/'sources'/role
            shutil.copytree(source.parent,root,ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.pyo'))
            package=root/source.name
            if role=='reference':(package/'batch_runtime.lua').write_bytes((HERE/'chain_reference_runtime.lua').read_bytes())
            sys.path.insert(0,str(root))
            try:
                module=importlib.import_module(source.name+'.batch_env')
                environment=module.BatchEnv(load_config(),output/role,difficulty,checkpoints=groups['train'])
                result['runs'].append({'role':role,'runtime_sha256':environment.manifest['runtime_sha256']})
                if role=='chain':
                    actions=[r['action'] for r in captured['actions']]
                    offset=0;index=0
                    while offset<len(actions):
                        count=first_count if offset==0 else 64
                        block=actions[offset:offset+count]
                        response=environment.batch_rpc({'op':'reset_rollout' if offset==0 else 'rollout',
                            'reset':options,'count':len(block),'actions':block,'resets':[options]*len(block),'capture_trace':1})
                        atomic_json(output/f'chain-block-{index:03d}.json',response)
                        trace.extend(r for r in response['trace'] if r['loads']==1)
                        offset+=len(block);index+=1
                    replay=json.loads((output/'chain/training/rl-chain-match-00001.json').read_text())
                    if replay!=dict(captured):raise RuntimeError('Forced chain replay differs from original sampled match')
                else:
                    reference=environment.batch_rpc({'op':'reference_match','reset':options,
                        'actions':captured['actions'],'frames':captured['frames']})
                    atomic_json(output/'reference-evidence.json',reference)
            finally:
                if environment:environment.close();environment=None
                sys.path.remove(str(root))
                for name in list(sys.modules):
                    if name==source.name or name.startswith(source.name+'.'):sys.modules.pop(name)
        if len(trace)!=len(reference['trace']):raise RuntimeError('Native trace length differs')
        for a,b in zip(trace,reference['trace']):
            if any(a[k]!=b[k] for k in ('frame','state','phase','round','events')):
                atomic_json(output/'mismatch.json',{'chain':a,'reference':b});raise RuntimeError('Native trajectory differs')
        by_frame={r['frame']:r['state'] for r in reference['trace']}
        for old in failed['trace']:
            if by_frame.get(old['frame'])!=old['state']:raise RuntimeError('Original failure raw trajectory was not reproduced')
        if reference['opening']!=captured['opening'] or reference['rounds']!=captured['rounds']:raise RuntimeError('Native outcome differs')
        after=[r for r in reference['decisions'] if r['round']==draw['round']+1]
        if len(after)<2 or not after[0]['reset_history'] or after[1]['frame']-after[0]['frame']!=12:
            raise RuntimeError('Post-draw initial action/reset cadence missing')
        result.update(status='complete',native_frames=len(trace),original_state_frames=len(failed['trace']),decisions=len(reference['decisions']),
            outcomes=[r['outcome'] for r in captured['rounds']],draw_round=draw['round'],
            confirmation_frame=draw['confirmation_frame'],next_round_start=after[0]['frame'],
            next_round_action_frames=12,initial_loads=reference['initial_loads'],in_play_pauses=reference['in_play_pauses'],
            rollout_audit=audit(output))
    except BaseException as error:
        result.update(status='invalid',error=f'{type(error).__name__}: {error}');raise
    finally:
        if environment:environment.close()
        atomic_json(output/'result.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','training','original','dataset','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--chain-evidence',type=Path)
    a=p.parse_args();print(json.dumps(run(a.source,a.training,a.original,a.dataset,a.output,a.chain_evidence),indent=2))
