"""Bounded sequential chain/R1 PPO comparison and one native actions16 check."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import astra_play_sf2
from astra_play_sf2.config import atomic_json
from astra_play_sf2.runner import sha256
from .round_chain_builder import build, PACKAGE


def freeze_production(source, code, production_source=None):
    """Copy the actual package, including assets, independent of checkout layout."""
    inherited=Path(source).resolve().parent/'src'/'astra_play_sf2'
    if production_source is not None:
        package=Path(production_source).resolve();origin='explicit'
    elif inherited.is_dir():
        package=inherited.resolve();origin='parent_snapshot'
    else:
        package=Path(astra_play_sf2.__file__).resolve().parent;origin='installed_package'
    if not (package/'__init__.py').is_file() or not (package/'assets'/'play_core.lua').is_file():
        raise ValueError('Production source must be the astra_play_sf2 package directory, including assets')
    def hashes(root):
        return {p.relative_to(root).as_posix():sha256(p) for p in sorted(root.rglob('*'))
                if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.pyo')}
    before=hashes(package)
    target=Path(code).resolve()/'src'/'astra_play_sf2'
    shutil.copytree(package,target,ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.pyo'))
    if hashes(package)!=before or hashes(target)!=before:
        raise RuntimeError('Production source changed while freezing the trial')
    record={'schema':'astra.rl-production-snapshot.v1','origin':origin,'source_package':str(package),
            'staged_package':'src/astra_play_sf2','files_sha256':before}
    atomic_json(Path(code)/'production-source.json',record)
    return record


def run(source,model,dataset,code,output,seed=211,production_source=None):
    source=source.resolve();model=model.resolve();dataset=dataset.resolve()
    output=output.resolve();output.mkdir(parents=True,exist_ok=False)
    manifest=build(code,source);code=code.resolve()
    if manifest.get('actions')!=16:raise ValueError('This bounded trial requires the frozen actions16 package')
    production=freeze_production(source,code,production_source)
    initial_hash=sha256(model)
    result={'schema':'astra.rl-round-chain-trial.v1','status':'running','seed':seed,
            'workers':2,'steps_per_training':2048,'init_model_sha256':initial_hash,'stages':[],
            'code_manifest_sha256':sha256(code/'build.json'),'training_only':True,
            'production_source':production,'production_manifest_sha256':sha256(code/'production-source.json')}
    atomic_json(output/'result.json',result)
    try:
        for role,root,package in [('chain',code,PACKAGE),('r1',source.parent,source.name)]:
            if sha256(model)!=initial_hash:raise RuntimeError('Initial model changed')
            target=output/role
            command=[sys.executable,'-m',package+'.batch_train','--dataset',str(dataset),'--output',str(target),
                     '--workers','2','--steps','2048','--block','64','--seed',str(seed),'--init-model',str(model)]
            env=dict(os.environ,PYTHONPATH=str(code/'src')+os.pathsep+str(root))
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
    p.add_argument('--seed',type=int,default=211)
    p.add_argument('--production-source',type=Path,help='astra_play_sf2 package directory including assets; default: parent snapshot or installed package')
    a=p.parse_args()
    print(json.dumps(run(a.source,a.model,a.dataset,a.code,a.output,a.seed,a.production_source),indent=2))
if __name__=='__main__':main()
