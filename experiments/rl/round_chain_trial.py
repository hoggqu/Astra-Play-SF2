"""Bounded sequential chain/R1 PPO comparison and one native actions16 check."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from astra_play_sf2.config import atomic_json
from astra_play_sf2.runner import sha256
from .round_chain_builder import build, PACKAGE


def run(source,model,dataset,code,output,seed=211):
    source=source.resolve();model=model.resolve();dataset=dataset.resolve()
    output=output.resolve();output.mkdir(parents=True,exist_ok=False)
    manifest=build(code,source);code=code.resolve()
    if manifest.get('actions')!=16:raise ValueError('This bounded trial requires the frozen actions16 package')
    shutil.copytree(source.parent/'src',code/'src')
    initial_hash=sha256(model)
    result={'schema':'astra.rl-round-chain-trial.v1','status':'running','seed':seed,
            'workers':2,'steps_per_training':2048,'init_model_sha256':initial_hash,'stages':[],
            'code_manifest_sha256':sha256(code/'build.json'),'training_only':True}
    atomic_json(output/'result.json',result)
    try:
        for role,root,package in [('chain',code,PACKAGE),('r1',source.parent,source.name)]:
            if sha256(model)!=initial_hash:raise RuntimeError('Initial model changed')
            target=output/role
            command=[sys.executable,'-m',package+'.batch_train','--dataset',str(dataset),'--output',str(target),
                     '--workers','2','--steps','2048','--block','64','--seed',str(seed),'--init-model',str(model)]
            env=dict(os.environ,PYTHONPATH=str(root/'src')+os.pathsep+str(root))
            with (output/(role+'.log')).open('w') as stream:
                status=subprocess.run(command,env=env,cwd=root,stdout=stream,stderr=subprocess.STDOUT,timeout=600).returncode
            report=json.loads((target/'result.json').read_text())
            result['stages'].append({'role':role,'exit':status,'result':report})
            if status or report['status']!='complete' or report.get('parameters_changed') is not True:
                raise RuntimeError(role+' real PPO update failed')
            if report.get('init_model_sha256')!=initial_hash or report.get('actual_steps')!=2048:raise RuntimeError('Training provenance/budget mismatch')
            atomic_json(output/'result.json',result)
        target=output/'native'
        command=[sys.executable,'-m',PACKAGE+'.native_continuous','--model',str(output/'chain/ppo-batch.zip'),
                 '--output',str(target),'--difficulty','3','--attempts','1','--speed','fast']
        env=dict(os.environ,PYTHONPATH=str(code/'src')+os.pathsep+str(code))
        with (output/'native.log').open('w') as stream:
            status=subprocess.run(command,env=env,cwd=code,stdout=stream,stderr=subprocess.STDOUT,timeout=600).returncode
        report=json.loads((target/'result.json').read_text())
        result['stages'].append({'role':'native','exit':status,'result':report})
        if status not in (0,1) or report['status']!='complete' or not report.get('action_interface_audit',{}).get('ok') or not report.get('native_timing_audit',{}).get('ok'):
            raise RuntimeError('Native chain-trained model validation failed')
        result['status']='complete'
    except BaseException as error:
        result.update(status='invalid',error=f'{type(error).__name__}: {error}');raise
    finally:atomic_json(output/'result.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source','model','dataset','code','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--seed',type=int,default=211);a=p.parse_args()
    print(json.dumps(run(a.source,a.model,a.dataset,a.code,a.output,a.seed),indent=2))
if __name__=='__main__':main()
