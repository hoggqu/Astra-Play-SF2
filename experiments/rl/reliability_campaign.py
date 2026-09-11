"""Bounded train/full-20 evaluation of separate immutable candidates; no pooled success."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from astra_play_sf2.config import atomic_json, config_path
from astra_play_sf2.runner import sha256
from .campaign import stop_owned, read_result
from .dataset import load_dataset
from .native_campaign import freeze_sources, require_sources

HERE = Path(__file__).resolve().parent
INTERFACE = 'ken_actions16_lp_mp_uppercut_v1'


def execution_sources(code, package):
    """Pin this isolated package, manifests and its copied production dependencies."""
    code = Path(code).resolve()
    if not package.isidentifier():
        raise ValueError('Supply one isolated package name, not a dotted module')
    package_path = code/package
    manifest = json.loads((code/'build.json').read_text(encoding='utf-8'))
    if (manifest.get('package') != package or manifest.get('action_interface') != INTERFACE
            or manifest.get('actions') != 16 or manifest.get('observations') != 344):
        raise ValueError('Expected an explicitly identified isolated actions16 build')
    for name, digest in manifest['derived_sha256'].items():
        if Path(name).name != name or sha256(package_path/name) != digest:
            raise ValueError('Generated package differs from its build: '+name)
    production = code/'src/astra_play_sf2'
    if not (production/'__init__.py').is_file():
        raise ValueError('Copy the parent frozen production src into each code root first')
    paths = {package+'/'+name: package_path/name for name in manifest['derived_sha256']}
    paths.update({p.name:p for p in code.glob('*.json')})
    paths.update({'src/'+p.relative_to(code/'src').as_posix():p for p in production.rglob('*')
                  if p.is_file() and p.suffix in ('.py','.lua','.json')})
    return paths


def run_isolated(code, module, arguments, log_path, timeout):
    env = dict(os.environ)
    # Do not retain another candidate's import roots. Third-party packages remain
    # available in the selected Python environment's normal site-packages.
    env['PYTHONPATH'] = str(code/'src')+os.pathsep+str(code)
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        env[name] = '1'
    with Path(log_path).open('wb') as log:
        process = subprocess.Popen([sys.executable,'-m',module,*map(str,arguments)],
            cwd=code,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
            start_new_session=os.name != 'nt')
        try:
            return process.wait(timeout=timeout)
        finally:
            stop_owned(process)


def audit_full_twenty(result, model_hash, code):
    """Exit zero or ten early clears alone never meets this reliability target."""
    if (result.get('schema') != 'astra.rl-continuous.actions16.v1'
            or result.get('status') != 'complete' or result.get('difficulty') != 3
            or result.get('model_sha256') != model_hash
            or result.get('action_interface') != INTERFACE or result.get('actions') != 16
            or result.get('native_timing') is not True
            or result.get('stop_on_first_clear') is not False
            or result.get('attempts_requested') != 20):
        raise RuntimeError('Expected a complete immutable native full-20 actions16 evaluation')
    for key in ('native_timing_audit','action_interface_audit'):
        if result.get(key,{}).get('ok') is not True:
            raise RuntimeError('Missing successful '+key)
    if str(result.get('selection','')).startswith('categorical') and result.get('sampling_audit',{}).get('ok') is not True:
        raise RuntimeError('Categorical deployment requires its sampling audit')
    attempts = result.get('attempts',[])
    if len(attempts) != 20 or [a.get('id') for a in attempts] != [f'l3-{i:03d}' for i in range(1,21)]:
        raise RuntimeError('Exactly twenty distinct ordered natural coin attempts are required')
    seen = set()
    for attempt in attempts:
        if attempt.get('outcome') not in ('loss','rl_gameplay_clear') or attempt.get('audit',{}).get('ok') is not True:
            raise RuntimeError('Invalid or unaudited attempt; never replace it with another coin')
        matches = attempt.get('matches',[])
        if not matches or len(matches) != len(set(matches)) or seen.intersection(matches):
            raise RuntimeError('Missing or reused match evidence')
        seen.update(matches)
        if attempt['outcome'] == 'rl_gameplay_clear' and (attempt.get('match_wins') != 11 or len(matches) != 11):
            raise RuntimeError('Clear requires eleven audited match wins')
    if (result['native_timing_audit'].get('matches') != len(seen)
            or result['action_interface_audit'].get('checked_matches') != len(seen)):
        raise RuntimeError('Full evaluation match audit coverage differs')
    clears = sum(a['outcome']=='rl_gameplay_clear' for a in attempts)
    if code != (0 if clears else 1):
        raise RuntimeError('Evaluation exit code disagrees with its retained outcomes')
    return {'attempts':20,'clears':clears,'losses':20-clears,'clear_rate':clears/20,
            'goal_achieved':clears>=10}


def campaign(dataset, output, init_model, training_code, training_package,
             verification_code, verification_package, initial_training_result=None,
             initial_budget_steps=0, workers=8, cycles=5, steps_per_cycle=409600,
             block=64, seed=129, stage_timeout=7200, stage_runner=run_isolated):
    if type(workers) is not int or workers not in range(1,9):
        raise ValueError('workers must be 1..8')
    if any(type(n) is not int or n<1 for n in (cycles,steps_per_cycle,stage_timeout)):
        raise ValueError('Positive cycle, step and timeout budgets required')
    if (steps_per_cycle%(workers*256) or block not in (1,16,32,64,128,256)
            or type(initial_budget_steps) is not int or initial_budget_steps<0):
        raise ValueError('Invalid completed-update budget or block')
    dataset, model = Path(dataset).resolve(), Path(init_model).resolve()
    training_code, verification_code = Path(training_code).resolve(), Path(verification_code).resolve()
    groups, difficulty = load_dataset(dataset)
    if difficulty != 3: raise ValueError('Reliability target is Normal difficulty 3')
    paths = {}
    for role,root,package in (('train',training_code,training_package),('verify',verification_code,verification_package)):
        paths.update({role+'/'+name:path for name,path in execution_sources(root,package).items()})
    for name in ('reliability_campaign.py','campaign.py','native_campaign.py','dataset.py','__init__.py'):
        paths['orchestrator/'+name]=HERE/name
    import astra_play_sf2
    own_production=Path(astra_play_sf2.__file__).resolve().parent
    for path in own_production.rglob('*'):
        if path.is_file() and path.suffix in ('.py','.lua','.json'):
            paths['orchestrator/production/'+path.relative_to(own_production).as_posix()]=path
    paths['dataset']=dataset
    for index,sample in enumerate(groups.get('train',[])):
        paths[f'training_checkpoint/{index:03d}']=Path(sample['path'])
    if config_path().is_file(): paths['mame_config']=config_path()
    initial_hash=sha256(model);initial_record=None
    if initial_training_result:
        source=Path(initial_training_result).resolve()
        initial_record=read_result(source,'astra.rl-batch-prototype.actions16.v1')
        if initial_record.get('model_sha256')!=initial_hash:
            raise ValueError('Initial completed training result identifies another model')
        paths['initial_training_result']=source
    pins=freeze_sources(paths)
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    result={'schema':'astra.rl-reliability-campaign.v1','status':'running','goal_achieved':False,
        'goal':'one fixed model clears at least 10 of all 20 natural coin attempts',
        'difficulty':3,'action_interface':INTERFACE,'actions':16,'initial_model_sha256':initial_hash,
        'initial_training_result':str(initial_training_result) if initial_training_result else None,
        'initial_training_steps':initial_record.get('actual_steps') if initial_record else None,
        'reported_initial_budget_steps':initial_budget_steps,'cycles_requested':cycles,
        'first_candidate':'evaluate initial completed model before further training',
        'steps_per_training_cycle':steps_per_cycle,'maximum_new_training_steps':(cycles-1)*steps_per_cycle,
        'verification_attempts_per_candidate':20,'required_clears':10,'maximum_attempts':cycles*20,
        'workers':workers,'block':block,'seed':seed,'stage_timeout_seconds':stage_timeout,
        'continuation':'PPO ZIP parameters and optimizer continue; environment/RNG restart per training stage, not exact trajectory recovery',
        'holdout_opened':False,'frozen_v4_certification':False,'sources':pins,'cycles':[],
        'created_utc':datetime.now(timezone.utc).isoformat()}
    def save(): atomic_json(output/'result.json',result)
    def stage(row,name,root,module,args,input_model,expected_hash,folder):
        require_sources(paths,pins)
        if sha256(input_model)!=expected_hash: raise RuntimeError('Input model changed before '+name)
        row['stage']=name
        row.setdefault('commands',{})[name]={'cwd':str(root),'module':module,'arguments':list(map(str,args))}
        save();code=stage_runner(root,module,args,folder/(name+'.log'),stage_timeout)
        row.setdefault('exit_codes',{})[name]=code;save()
        require_sources(paths,pins)
        if sha256(input_model)!=expected_hash: raise RuntimeError('Input model changed during '+name)
        return code
    try:
        save()
        for ordinal in range(1,cycles+1):
            folder=output/f'cycle-{ordinal:03d}';folder.mkdir()
            row={'ordinal':ordinal,'status':'running','model_sha256':sha256(model),'training_steps':0}
            result['cycles'].append(row)
            if ordinal>1:
                train=folder/'train';old_hash=sha256(model)
                args=['--dataset',dataset,'--output',train,'--workers',workers,'--steps',steps_per_cycle,
                      '--block',block,'--seed',seed+ordinal-2,'--init-model',model]
                code=stage(row,'train',training_code,training_package+'.batch_train',args,model,old_hash,folder)
                trained=read_result(train/'result.json','astra.rl-batch-prototype.actions16.v1')
                if (code!=0 or trained.get('actual_steps')!=steps_per_cycle or trained.get('difficulty')!=3
                        or trained.get('benchmark') is not False or trained.get('parity') is not False
                        or trained.get('native_parity') is not False or trained.get('parameters_changed') is not True
                        or trained.get('init_model_sha256')!=old_hash or trained.get('dataset_sha256')!=pins['dataset']
                        or trained.get('action_interface')!=INTERFACE or trained.get('actions')!=16):
                    raise RuntimeError('Training budget, identity, real-update or input provenance mismatch')
                model=train/'ppo-batch.zip'
                if sha256(model)!=trained.get('model_sha256'): raise RuntimeError('Completed model checksum mismatch')
                row.update(model_sha256=trained['model_sha256'],training_steps=trained['actual_steps'],
                    training_result=str((train/'result.json').relative_to(output)),
                    optimizer_initialization=trained.get('optimizer_initialization'),
                    opponent_sampling=trained.get('opponent_sampling'))
            row['model']=str(model);verify=folder/'continuous'
            args=['--model',model,'--output',verify,'--difficulty',3,'--attempts',20,'--speed','fast','--all-attempts']
            code=stage(row,'continuous',verification_code,verification_package+'.native_continuous',args,model,row['model_sha256'],folder)
            verified=read_result(verify/'result.json','astra.rl-continuous.actions16.v1')
            measured=audit_full_twenty(verified,row['model_sha256'],code)
            row.update(measured,status='complete',stage='complete',verification_result=str((verify/'result.json').relative_to(output)),
                       attempts=[{k:a.get(k) for k in ('id','outcome','match_wins','matches','images')} for a in verified['attempts']])
            save();print(json.dumps({'candidate':ordinal,**measured}),flush=True)
            if measured['goal_achieved']:
                result.update(status='complete',goal_achieved=True,success_cycle=ordinal,
                              selected_model=str(model),model_sha256=row['model_sha256']);break
        else: result.update(status='complete',stop_reason='candidate_budget_exhausted')
    except BaseException as error:
        result.update(status='invalid',error=f'{type(error).__name__}: {error}')
        if result['cycles'] and result['cycles'][-1]['status']=='running':
            result['cycles'][-1].update(status='invalid',error=result['error'])
    finally:
        result['completed_new_training_steps']=sum(c['training_steps'] for c in result['cycles'])
        result['completed_candidate_evaluations']=sum(c['status']=='complete' for c in result['cycles'])
        result['wall_seconds']=time.monotonic()-started;save()
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('dataset','output','init-model','training-code','verification-code'):
        parser.add_argument('--'+key,type=Path,required=True)
    for key in ('training-package','verification-package'):parser.add_argument('--'+key,required=True)
    parser.add_argument('--initial-training-result',type=Path)
    parser.add_argument('--initial-budget-steps',type=int,default=0)
    parser.add_argument('--workers',type=int,choices=range(1,9),default=8)
    parser.add_argument('--cycles',type=int,default=5,help='Candidate groups, including initial-model full20')
    parser.add_argument('--steps-per-cycle',type=int,default=409600)
    parser.add_argument('--block',type=int,default=64);parser.add_argument('--seed',type=int,default=129)
    parser.add_argument('--stage-timeout',type=int,default=7200)
    def stopped(_signal,_frame):raise KeyboardInterrupt('Owner cancellation; retained as invalid, never replayed')
    signal.signal(signal.SIGTERM,stopped)
    result=campaign(**vars(parser.parse_args()))
    print(json.dumps(result,indent=2),flush=True)
    return 2 if result['status']!='complete' else 0 if result['goal_achieved'] else 1


if __name__=='__main__':raise SystemExit(main())
