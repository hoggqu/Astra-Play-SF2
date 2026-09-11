"""Candidate-only all-action batch/native port parity; never runs formal attempts."""
import argparse
import importlib
import json
from pathlib import Path
import shutil
import sys
import threading
import _thread
from astra_play_sf2.config import atomic_json,load_config
from astra_play_sf2.runner import sha256
from .projectile_probe import select_train
HERE=Path(__file__).resolve().parent


def patch_snapshot(text):
    old=' return s\nend'
    if text.count(old)!=1:raise ValueError('Ambiguous snapshot anchor')
    return text.replace(old," s.native_ports={m.ioport.ports[':IN1']:read(),m.ioport.ports[':IN2']:read()}\n return s\nend")


def reference_runtime(text):
    text=patch_snapshot(text)
    old="  if core.phase=='complete' then"
    if text.count(old)!=1:raise ValueError('Ambiguous reference stop anchor')
    text=text.replace(old,'  if frames==pending.frames then')
    return text


def action_plan():
    # Delay until actual battle; all 16 IDs and consecutive basic buttons.
    plan=[0]*10+list(range(16))+[6,6,6,10,10,6,6,6,10,10]+list(range(16))+[0]*12
    assert len(plan)==64 and set(plan)==set(range(16))
    return plan


def run(training,verification,model,old_model,dataset,output):
    training=training.resolve();verification=verification.resolve();model=model.resolve();output=output.resolve()
    sample=select_train(dataset,(3,),1)[0];output.mkdir(parents=True,exist_ok=False)
    result={'schema':'astra.rl-pulsed-native-gate.v1','status':'invalid','training_only':True,'formal_clear':False,
        'model_sha256':sha256(model),'checkpoint_sha256':sample['sha256'],'lead':0,'actions_covered':list(range(16)),
        'native_frames':768,'runs':[]}
    env=None;roots=[];timer=threading.Timer(600,_thread.interrupt_main);timer.daemon=True;timer.start()
    def unload(name):
        for n in list(sys.modules):
            if n==name or n.startswith(name+'.'):sys.modules.pop(n)
    def save():atomic_json(output/'result.json',result)
    try:
        save()
        # Every source file and the manifest are checked before any native process.
        for role,source in [('training',training),('verification',verification)]:
            manifest=json.loads((source.parent/'build.json').read_text())
            for name,digest in manifest['derived_sha256'].items():
                if sha256(source/name)!=digest:raise RuntimeError('Frozen source mismatch: '+role+'/'+name)
            result[role+'_build_sha256']=sha256(source.parent/'build.json')
        sys.path.insert(0,str(verification.parent));roots.append(str(verification.parent))
        support=importlib.import_module(verification.name+'.support');native=importlib.import_module(verification.name+'.native_continuous')
        snapshot=support.capture_interface(model,verification)
        rejected=False
        try:support.validate_model_file(old_model)
        except ValueError:rejected=True
        if not rejected:raise RuntimeError('Old action-interface ZIP was accepted')
        staged=output/'payload-staged';staged.mkdir()
        hashes=native.stage_native_policy(staged,3,snapshot['payload']);support.check_staged_interface(snapshot,staged,hashes)
        result.update(old_zip_rejected=True,old_zip_sha256=sha256(old_model),payload_runtime_sha256=hashes,action_interface=snapshot['payload']['action_interface'])
        sys.path.remove(roots.pop());unload(verification.name)
        plan=action_plan();traces={};options={'checkpoint':0,'lead':0}
        for role in ('batch','manual_reset','native'):
            source=verification if role=='native' else training
            root=output/'sources'/role;shutil.copytree(source.parent,root,ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.pyo'))
            package=root/source.name
            text=reference_runtime((HERE/'chain_reference_runtime.lua').read_text()) if role=='native' else patch_snapshot((source/'batch_runtime.lua').read_text())
            (package/'batch_runtime.lua').write_text(text)
            sys.path.insert(0,str(root));roots.append(str(root));batch=importlib.import_module(source.name+'.batch_env')
            env=batch.BatchEnv(load_config(),output/role,3,checkpoints=[sample])
            row={'role':role,'runtime_sha256':env.manifest['runtime_sha256']};result['runs'].append(row);save()
            if role=='native':
                planned=[{'round':1,'frame':i*12,'action':a} for i,a in enumerate(plan+[0])]
                response=env.batch_rpc({'op':'reference_match','reset':options,'actions':planned,'frames':768})
                atomic_json(output/'native-evidence.json',response);traces[role]=response['trace']
                if response['initial_loads']!=1 or response['in_play_pauses']!=0:raise RuntimeError('Native reference lifecycle violation')
            else:
                traces[role]=[];offset=0
                if role=='manual_reset':env.batch_rpc({'op':'reset','reset':options})
                for index,count in enumerate((33,31)):
                    response=env.batch_rpc({'op':'reset_rollout' if offset==0 and role=='batch' else 'rollout',
                        'reset':options,'actions':plan[offset:offset+count],'count':count,'resets':[options]*count,'capture_trace':1})
                    atomic_json(output/f'{role}-block-{index:02d}.json',response);traces[role].extend(response['trace']);offset+=count
            env.close();env=None;sys.path.remove(roots.pop());unload(source.name)
        reference=traces['native'];result['comparisons']={}
        for role in ('batch','manual_reset'):
            if len(reference)!=768 or len(traces[role])!=768:raise RuntimeError('Missing full native frame coverage')
            for a,b in zip(traces[role],reference):
                if any(a[k]!=b[k] for k in ('frame','state','phase','round','events')):
                    atomic_json(output/'mismatch.json',{'role':role,'batch':a,'native':b});raise RuntimeError('Native state/port/event mismatch')
            result['comparisons'][role]={'matching_state_frames':768,'matching_port_frames':768,'batch_boundary_after_decisions':33}
        for row in reference:
            if row['round']!=1 or row['phase']!='fighting':raise RuntimeError('Unexpected early round transition in bounded all-action gate')
        # Actual button ports in the final frame of every basic macro must release.
        masks={6:(0,16),7:(0,64),8:(0,16),9:(1,4),10:(1,2),11:(1,4)}
        checks=[]
        for i,a in enumerate(plan):
            if a not in masks:continue
            port,mask=masks[a];before=reference[i*12+10]['state']['native_ports'][port];last=reference[i*12+11]['state']['native_ports'][port]
            if before&mask or not last&mask:raise RuntimeError('Basic macro did not release its actual button on frame11')
            checks.append({'action':a,'macro_start':i*12,'port':port+1,'mask':mask,'pressed_port':before,'released_port':last})
        for role,source in [('training',training),('verification',verification)]:
            if sha256(source.parent/'build.json')!=result[role+'_build_sha256']:raise RuntimeError('Source manifest changed during gate')
            manifest=json.loads((source.parent/'build.json').read_text())
            if any(sha256(source/n)!=h for n,h in manifest['derived_sha256'].items()):raise RuntimeError('Source changed during gate')
        if sha256(model)!=result['model_sha256']:raise RuntimeError('Model changed during gate')
        result.update(status='complete',release_port_checks=checks,native_initial_loads=1,native_in_play_pauses=0)
    except BaseException as error:result['error']=f'{type(error).__name__}: {error}';raise
    finally:
        timer.cancel()
        if env:env.close()
        while roots:sys.path.remove(roots.pop())
        save()
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('training','verification','model','old-model','dataset','output'):p.add_argument('--'+name,type=Path,required=True)
    print(json.dumps(run(**vars(p.parse_args())),indent=2))
if __name__=='__main__':main()
