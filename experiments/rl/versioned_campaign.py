"""Sequential PPO/latest-Adam training and audited natural-coin first-clear campaign.

Standalone stdlib runner. Executable policy, architecture and validators come only
from the explicitly pinned generated package; this file does not play the game.
"""
import argparse
import hashlib
import json
import itertools
import math
import os
import re
from pathlib import Path
import signal
import subprocess
import sys
import time
import zipfile


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path); temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def read_complete(path):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if value.get('status') != 'complete':
        raise RuntimeError(f'Incomplete/invalid child result: {path}: {value.get("status")}')
    return value


def full_zip(path):
    with zipfile.ZipFile(path) as archive:
        if not {'data','policy.pth','policy.optimizer.pth'}.issubset(archive.namelist()):
            raise RuntimeError('Need a complete PPO ZIP containing policy, critic and Adam')
        if archive.testzip() is not None: raise RuntimeError('Damaged PPO ZIP')
    return digest(path)


def source_pins(code, package):
    manifest = json.loads((code/'build.json').read_text(encoding='utf-8'))
    if manifest.get('package') != package or not manifest.get('observation_interface'):
        raise ValueError('Package/build identity mismatch')
    files = {'build.json':digest(code/'build.json')}
    for name, expected in manifest['frozen_files_sha256'].items():
        files[name] = expected
    for name, expected in manifest['derived_sha256'].items():
        files[package+'/'+name] = expected
    require_pins(code, files)
    if 'launch.py' not in files: raise ValueError('Frozen launcher missing')
    return manifest, files


def require_pins(root, pins):
    for name, expected in pins.items():
        path = (root/name).resolve()
        if not path.is_relative_to(root) or digest(path) != expected:
            raise RuntimeError('Pinned source changed: '+name)


STOP_PROTOCOL = 'ppo-update-stop-file-v1'


def request_stop(path, reason):
    """First request wins; never replace a cancellation with a later deadline."""
    path=Path(path)
    if not path.exists():
        write_json(path, {'schema':'astra.rl-stop-request.v1','reason':reason})


def read_stop(path):
    path=Path(path)
    if not path.exists():return None
    request=json.loads(path.read_text(encoding='utf-8'))
    if request.get('schema')!='astra.rl-stop-request.v1' or request.get('reason') not in ('max_duration','user_cancelled'):
        raise RuntimeError('Malformed cooperative stop request')
    return request['reason']


def run_child(command, log_path, environment, timeout):
    """Only terminate descendants belonging to this newly owned child process."""
    with Path(log_path).open('xb') as log:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, env=environment, start_new_session=os.name != 'nt',
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name=='nt' else 0)
        started=time.monotonic()
        stop_file=environment.get('ASTRA_RL_STOP_FILE')
        deadline=environment.get('ASTRA_CAMPAIGN_DEADLINE_MONOTONIC')
        try:
            while True:
                now=time.monotonic()
                if stop_file and deadline and now>=float(deadline):request_stop(stop_file,'max_duration')
                remaining=timeout-(now-started)
                if remaining<=0:raise subprocess.TimeoutExpired(command,timeout)
                try:
                    return child.wait(timeout=min(1,remaining))
                except subprocess.TimeoutExpired:
                    pass
                except KeyboardInterrupt:
                    if not stop_file:raise
                    request_stop(stop_file,'user_cancelled')
                    print('Stop requested; waiting for a complete PPO update or the current native evaluation to finish.',flush=True)
        finally:
            if child.poll() is None:
                child.terminate()
                try: child.wait(timeout=55)
                except subprocess.TimeoutExpired:
                    if os.name == 'nt':
                        subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
                    else: os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=5)


def failure_diagnostics(log_path, stage, exit_code):
    """First useful exception within a bounded tail, not a full log/trace scan."""
    log_path=Path(log_path)
    result={'stage':stage,'exit_code':exit_code,'log_path':str(log_path),
            'tail_limit_bytes':1024*1024,'tail_bytes_read':0,'tail_truncated':False}
    try:
        with log_path.open('rb') as stream:
            size=stream.seek(0,os.SEEK_END);start=max(0,size-1024*1024)
            stream.seek(start);raw=stream.read(1024*1024)
        result.update(tail_bytes_read=len(raw),tail_truncated=start>0)
        text=raw.decode('utf-8',errors='replace')
        if start:text=text.partition('\n')[2]  # Do not parse a cut log line.
        pattern=re.compile(r'^\s*((?:[A-Za-z_][\w.]*)(?:Error|Exception)|AssertionError|Exception):\s*(.*)$')
        candidates=[]
        for line in text.splitlines():
            match=pattern.match(line)
            if match:candidates.append((match.group(1),match.group(2)))
        secondary={'EOFError','BrokenPipeError','ConnectionResetError','ConnectionAbortedError'}
        specific=[c for c in candidates if c[0].split('.')[-1] not in secondary]
        prefix_specific=None
        # A trainer may dump its result after the worker traceback. If the tail
        # is truncated, scan log lines from the front with constant memory and
        # a combined read cap of 32 MiB. Never scan game match trace files.
        if start:
            limit=32*1024*1024-len(raw);scanned=0;at_line_start=True
            with log_path.open('rb') as stream:
                while scanned<limit:
                    chunk=stream.readline(min(65536,limit-scanned))
                    if not chunk:break
                    scanned+=len(chunk)
                    match=pattern.match(chunk.decode('utf-8',errors='replace')) if at_line_start else None
                    at_line_start=chunk.endswith(b'\n')
                    if match and match.group(1).split('.')[-1] not in secondary:
                        prefix_specific=(match.group(1),match.group(2));break
            result.update(prefix_scan_bytes=scanned,prefix_scan_limit_bytes=limit,
                          prefix_scan_truncated=prefix_specific is None and scanned<size)
            if prefix_specific:specific=[prefix_specific]
        if specific or candidates:
            name,message=(specific or candidates)[0]
            # Lua RuntimeError often embeds its traceback on one escaped line.
            message=message.split('\\nstack traceback',1)[0][:1000]
            message=re.sub(r"^\{'error': ['\"]",'',message)
            result.update(exception_type=name,message=message,
                          selection='first_specific_exception_in_bounded_prefix' if prefix_specific else 'first_specific_exception_in_tail' if specific else 'first_transport_exception_in_tail')
    except OSError as error:result['log_read_error']=str(error)
    return result


def recover_checkpoint(train_dir, workers, budget, dataset_sha256, init_model_sha256, rollout_steps=None):
    """Validate an independently complete update from an otherwise failed stage.

    Stdlib checks ZIP structure/CRC and the nonempty Adam member. The receiving
    trainer still performs its normal actual-model/optimizer load validation.
    """
    train_dir=Path(train_dir).resolve();result_path=train_dir/'result.json'
    if not result_path.exists():return None
    value=json.loads(result_path.read_text(encoding='utf-8'))
    if not isinstance(value,dict):raise RuntimeError('Malformed recovery training result')
    if rollout_steps is not None and value.get('rollout_steps') != rollout_steps:
        raise RuntimeError('Recovery rollout size mismatch')
    checkpoint=value.get('last_checkpoint')
    if checkpoint is None:return None
    if not isinstance(checkpoint,dict) or checkpoint.get('complete_update') is not True:
        raise RuntimeError('Recovery checkpoint is not a completed PPO update')
    steps=checkpoint.get('steps');updates=checkpoint.get('optimizer_updates')
    if (type(steps) is not int or steps<=0 or steps>budget or steps%(rollout_steps or (workers*256))
        or type(updates) is not int or updates<=0):
        raise RuntimeError('Invalid recovery checkpoint update count')
    if (value.get('dataset_sha256')!=dataset_sha256 or value.get('init_model_sha256')!=init_model_sha256
        or value.get('difficulty')!=3 or type(value.get('completed_update_steps')) is not int
        or value['completed_update_steps']<steps):
        raise RuntimeError('Recovery checkpoint training identity/step mismatch')
    relative=checkpoint.get('path')
    if (not isinstance(relative,str) or not relative or Path(relative).is_absolute()
        or '..' in Path(relative).parts or '\\' in relative or ':' in relative):
        raise RuntimeError('Recovery checkpoint path must be relative and inside train')
    raw=train_dir/relative;path=raw.resolve()
    if raw.is_symlink() or not path.is_relative_to(train_dir) or path.suffix!='.zip' or not path.is_file():
        raise RuntimeError('Recovery checkpoint path outside train, missing, or not a regular ZIP')
    actual=full_zip(path)
    if actual!=checkpoint.get('sha256'):raise RuntimeError('Recovery checkpoint SHA mismatch')
    with zipfile.ZipFile(path) as archive:
        if any(archive.getinfo(name).file_size<=0 for name in ('data','policy.pth','policy.optimizer.pth')):
            raise RuntimeError('Recovery checkpoint has empty model/Adam data')
    return dict(checkpoint,path=str(path),sha256=actual,training_result_sha256=digest(result_path),
                from_incomplete_stage=True,optimizer_load_validation_required=True)


def evaluate_result(value, exit_code, model_hash, attempts, identity, all_attempts=False):
    if value.get('status') != 'complete' or value.get('difficulty') != 3 or value.get('model_sha256') != model_hash:
        raise RuntimeError('Evaluation identity/status mismatch')
    for key in ('action_interface','observation_interface'):
        if value.get(key) != identity.get(key): raise RuntimeError('Evaluation interface mismatch: '+key)
    if value.get('native_timing') is not True:
        raise RuntimeError('Evaluation must use native time')
    for key in ('native_timing_audit','action_interface_audit'):
        if value.get(key,{}).get('ok') is not True: raise RuntimeError('Missing successful '+key)
    rows = value.get('attempts',[])
    if not 1 <= len(rows) <= attempts: raise RuntimeError('Wrong natural-coin attempt count')
    if all_attempts and len(rows)!=attempts:raise RuntimeError('Full evaluation stopped before requested attempt count')
    cleared = False
    for index,row in enumerate(rows):
        if row.get('audit',{}).get('ok') is not True or row.get('outcome') not in ('loss','rl_gameplay_clear'):
            raise RuntimeError('Invalid individual natural-coin attempt')
        if row['outcome'] == 'rl_gameplay_clear':
            if row.get('match_wins') != 11 or len(row.get('matches',[])) != 11 or (not all_attempts and index != len(rows)-1):
                raise RuntimeError('Clear requires eleven audited matches and immediate stop')
            cleared = True
    if exit_code != (0 if cleared else 1) or (not cleared and len(rows) != attempts):
        raise RuntimeError('Evaluation exit code/attempt budget mismatch')
    return cleared


def matchup_totals(folder, attempts):
    """Only count completed audited evaluation matches, never training episodes."""
    totals={};seen=set()
    for attempt in attempts:
        for relative in attempt['matches']:
            path=(folder/relative).resolve()
            if not path.is_relative_to(folder) or path in seen:
                raise RuntimeError('Duplicate/outside evaluation match evidence')
            seen.add(path);match=json.loads(path.read_text(encoding='utf-8'))
            summary=match['summary'];rounds=match['rounds'];opponent=str(summary['opponent'])
            if summary.get('status')!='complete' or summary.get('valid_continuous') is not True:
                raise RuntimeError('Incomplete match cannot enter statistics')
            counts={'round_wins':0,'round_losses':0,'round_draws':0,'match_wins':0,'match_losses':0}
            for row in rounds:
                key={'win':'round_wins','loss':'round_losses','draw':'round_draws'}.get(row['outcome'])
                if key is None: raise RuntimeError('Unknown native round outcome')
                counts[key]+=1
            if [counts['round_wins'],counts['round_losses']]!=summary['score']:
                raise RuntimeError('Native score/statistics disagreement')
            key={'ken_win':'match_wins','cpu_win':'match_losses'}.get(summary['result'])
            if key is None: raise RuntimeError('Unknown native match outcome')
            counts[key]+=1
            target=totals.setdefault(opponent,{k:0 for k in counts})
            for key,value in counts.items():target[key]+=value
    return totals


def campaign(*, code, package, dataset, output, config, cycles=None, steps, workers,
             python=sys.executable, init_model=None, attempts=3, seed=42,
             timeout=86400, block=128, checkpoint_every=20480, first_cycle_steps=None, max_duration=None, stop_on_clear=True, evaluate_on_stop=False,
             runner=run_child, clock=time.monotonic, rollout_steps=None, minibatch_size=64, all_attempts=False):
    if cycles is not None and (type(cycles) is not int or cycles<1):raise ValueError('cycles must be a positive integer')
    if max_duration is not None and (type(max_duration) not in (int,float) or not math.isfinite(max_duration) or max_duration<=0):
        raise ValueError('max-duration must be positive finite seconds')
    if cycles is None and max_duration is None:raise ValueError('Provide cycles or max-duration')
    if type(all_attempts) is not bool:raise ValueError('all-attempts must be boolean')
    if type(stop_on_clear) is not bool or type(evaluate_on_stop) is not bool:raise ValueError('stop-on-clear and evaluate-on-stop must be boolean')
    if any(type(v) is not int or v<1 for v in (steps,timeout)) or type(workers) is not int or workers < 1:
        raise ValueError('Positive budgets and positive integer workers required')
    code,dataset,output,config = (Path(p).resolve() for p in (code,dataset,output,config))
    identity,pins = source_pins(code,package)
    if rollout_steps is None and identity.get('rollout_protocol') == 'exact_global_rollout_per_worker_gae_v1':
        rollout_steps=4096
    if rollout_steps is not None:
        from .rollout_control import validate_sizes
        validate_sizes(rollout_steps,minibatch_size,workers)
    quantum=rollout_steps or (workers*256)
    if steps % quantum or attempts not in (1,2,3) or block not in (1,16,32,64,128,256):
        raise ValueError('Steps must be multiples of the rollout budget; attempts1..3; supported block required')
    if first_cycle_steps is not None and (type(first_cycle_steps) is not int
            or first_cycle_steps < 1 or first_cycle_steps % quantum):
        raise ValueError('first-cycle-steps must be a positive multiple of the rollout budget')
    if type(checkpoint_every) is not int or checkpoint_every < 1 or checkpoint_every % quantum:
        raise ValueError('checkpoint-every must be a positive multiple of the rollout budget')
    data = json.loads(dataset.read_text(encoding='utf-8'))
    if data.get('status') != 'complete' or data.get('difficulty') != 3:
        raise ValueError('Need completed Normal dataset manifest')
    model = Path(init_model).resolve() if init_model else None
    model_hash = full_zip(model) if model else None
    output.mkdir(parents=True, exist_ok=False)
    started=clock();stop_file=output/'stop-request.json'
    environment = os.environ.copy();environment['ASTRA_SF2_CONFIG']=str(config)
    environment['PYTHONPATH']=os.pathsep.join((str(code/'src'),str(code)))
    environment['ASTRA_RL_STOP_FILE']=str(stop_file)
    environment.pop('ASTRA_CAMPAIGN_DEADLINE_MONOTONIC',None)
    if max_duration is not None:environment['ASTRA_CAMPAIGN_DEADLINE_MONOTONIC']=str(started+max_duration)
    record = {'schema':'astra.rl-versioned-campaign.v1','status':'running',
        'goal':'first audited Normal natural-coin eleven-match clear','difficulty':3,
        'package':package,'source_build_sha256':pins['build.json'],'source_pins':pins,
        'dataset_sha256':digest(dataset),'config_sha256':digest(config),
        'runner_sha256':digest(Path(__file__)), 'cycles_requested':cycles,
        'max_duration_seconds':max_duration,'stop_on_clear':stop_on_clear,'evaluate_on_stop':evaluate_on_stop,'any_clear':False,
        'stop_file':str(stop_file),'cooperative_stop_supported':identity.get('cooperative_stop_protocol')==STOP_PROTOCOL,
        'steps_per_cycle':steps,'first_cycle_steps':first_cycle_steps,
        'first_cycle_steps_effective':first_cycle_steps if first_cycle_steps is not None else steps,
        'workers':workers,'rollout_steps':rollout_steps,'minibatch_size':minibatch_size if rollout_steps else None,'decisions_per_update':quantum,'block':block,'checkpoint_every':checkpoint_every,
        'attempts_per_cycle':attempts,'all_attempts':all_attempts,
        'holdout_opened':False,'frozen_v4_certification':False,
        'selection':'latest completed full PPO ZIP and Adam; no eval-best rollback',
        'initial_model_sha256':model_hash,'cycles':[],'evaluation_matchups':{}}
    row=None
    def stop_reason():
        if max_duration is not None and clock()-started>=max_duration:request_stop(stop_file,'max_duration')
        return read_stop(stop_file)
    def finish_stop(reason):
        record.update(status='budget_stopped' if reason=='max_duration' else 'cancelled',stop_reason=reason)
        if row is not None and row['status']=='running':row.update(status='stopped',stop_reason=reason)

    def save():
        if row is not None: write_json(output/f'cycle-{row["ordinal"]:03d}'/'cycle.json',row)
        write_json(output/'result.json',record)
    def check():
        require_pins(code,pins)
        if digest(dataset)!=record['dataset_sha256'] or digest(config)!=record['config_sha256']:
            raise RuntimeError('Dataset/config changed during campaign')
        if digest(Path(__file__)) != record['runner_sha256']: raise RuntimeError('Campaign source changed')
        if model and digest(model)!=model_hash: raise RuntimeError('Stage input model changed')
    def stage(name,args,folder):
        check();command=[str(python),str(code/'launch.py'),name,*map(str,args)]
        record['active_stage']=name
        if row is not None: row['stage']=name;row.setdefault('commands',{})[name]=command
        save();print(json.dumps({'stage':name,'cycle':row['ordinal'] if row else 0}),flush=True)
        status=runner(command,folder/(name+'.log'),environment,timeout)
        check()
        if row is not None: row.setdefault('exit_codes',{})[name]=status
        save()
        if status not in ((0,1) if name=='native_continuous' else (0,)):
            diagnostics=failure_diagnostics(folder/(name+'.log'),name,status)
            if row is not None:row['failure_diagnostics']=diagnostics
            record['failure_diagnostics']=diagnostics
            if name=='batch_train':
                try:
                    recovery=recover_checkpoint(folder/'train',workers,row['training_steps_requested'],
                                                record['dataset_sha256'],row['input_model_sha256'],rollout_steps)
                    if recovery:
                        row['recovery_checkpoint']=recovery;record['recovery_checkpoint']=recovery
                        record.update(latest_model=recovery['path'],latest_model_sha256=recovery['sha256'],
                                      latest_model_from_incomplete_stage=True)
                except (OSError,ValueError,RuntimeError,KeyError,zipfile.BadZipFile) as error:
                    diagnostics['recovery_rejected']=f'{type(error).__name__}: {error}'
            message=f'Invalid child exit: {name}: {status}'
            if diagnostics.get('exception_type'):message+='; '+diagnostics['exception_type']+': '+diagnostics['message']
            message+='; log: '+diagnostics['log_path']
            if diagnostics.get('recovery_rejected'):message+='; recovery rejected: '+diagnostics['recovery_rejected']
            save();raise RuntimeError(message)
        return status
    try:
        save()
        if model is None:
            target=output/'initial'
            stage('initialize',['--output',target,'--seed',seed],output)
            initial=read_complete(target/'result.json');model=target/'ppo-initial.zip';model_hash=full_zip(model)
            if model_hash!=initial.get('model_sha256'): raise RuntimeError('Initial ZIP hash mismatch')
        record.update(latest_model=str(model),latest_model_sha256=model_hash)
        previous_updates=None
        for ordinal in itertools.count(1):
            reason=stop_reason()
            if reason:finish_stop(reason);break
            if cycles is not None and ordinal>cycles:record['status']='exhausted';break
            cycle_steps=first_cycle_steps if ordinal==1 and first_cycle_steps is not None else steps
            folder=output/f'cycle-{ordinal:03d}';folder.mkdir()
            row={'ordinal':ordinal,'status':'running','training_steps_requested':cycle_steps,'input_model':str(model),'input_model_sha256':model_hash}
            record['cycles'].append(row);train=folder/'train'
            stage('batch_train',['--dataset',dataset,'--output',train,'--workers',workers,
                '--steps',cycle_steps,'--block',block,'--checkpoint-every',checkpoint_every,'--seed',seed+ordinal-1,'--init-model',model]+(['--rollout-steps',rollout_steps,'--minibatch-size',minibatch_size] if rollout_steps else []),folder)
            trained=read_complete(train/'result.json')
            actual_steps=trained.get('actual_steps')
            if rollout_steps is not None and (trained.get('rollout_steps')!=rollout_steps or trained.get('minibatch_size')!=minibatch_size or trained.get('effective_ppo',{}).get('batch_size')!=minibatch_size):
                raise RuntimeError('Requested PPO rollout/minibatch was not applied')
            early_stop=trained.get('budget_stop',False)
            if early_stop is not False and early_stop is not True:raise RuntimeError('Invalid training budget-stop flag')
            if early_stop:
                reason=read_stop(stop_file)
                if (identity.get('cooperative_stop_protocol')!=STOP_PROTOCOL
                    or trained.get('cooperative_stop_protocol')!=STOP_PROTOCOL
                    or reason is None or trained.get('stop_reason')!=reason):
                    raise RuntimeError('Unrequested or unsupported training budget stop')
            if (type(actual_steps) is not int or actual_steps<=0 or actual_steps>cycle_steps
                or actual_steps%(rollout_steps or (workers*256)) or (actual_steps!=cycle_steps and not early_stop)
                or trained.get('completed_update_steps')!=actual_steps
                or trained.get('difficulty')!=3 or trained.get('dataset_sha256')!=record['dataset_sha256']
                or trained.get('init_model_sha256')!=model_hash or trained.get('parameters_changed') is not True
                or any(trained.get(k) is not False for k in ('benchmark','parity','native_parity'))):
                raise RuntimeError('Training budget/identity/update mismatch')
            if (trained.get('checkpoint_every_requested') != checkpoint_every
                or trained.get('checkpoint_every_effective') != checkpoint_every):
                raise RuntimeError('Training checkpoint interval/provenance mismatch')
            row['checkpoint_every']=checkpoint_every
            opt=trained.get('optimizer_initialization',{})
            if opt.get('loaded_from_init_model') is not True: raise RuntimeError('Training did not load complete parent ZIP')
            if previous_updates is not None and (opt.get('initial_updates')!=previous_updates or opt.get('initial_state_entries',0)<=0):
                raise RuntimeError('Adam state/update counter was not carried forward')
            checkpoint=trained.get('final_checkpoint',{})
            if checkpoint.get('complete_update') is not True: raise RuntimeError('Final checkpoint is not a completed PPO update')
            if early_stop and checkpoint.get('steps')!=actual_steps:raise RuntimeError('Budget-stop checkpoint step mismatch')
            model=train/'ppo-batch.zip';model_hash=full_zip(model)
            if model_hash!=trained.get('model_sha256') or model_hash!=checkpoint.get('sha256'):
                raise RuntimeError('Trained ZIP hash mismatch')
            previous_updates=checkpoint.get('optimizer_updates')
            if type(previous_updates) is not int or previous_updates<=0: raise RuntimeError('Missing completed optimizer updates')
            record.update(latest_model=str(model),latest_model_sha256=model_hash)
            row.update(model_sha256=model_hash,training_steps=actual_steps,optimizer_updates=previous_updates,
                       training_budget_stopped=early_stop);save()
            reason=stop_reason()
            if reason:
                if reason=='max_duration' and evaluate_on_stop:
                    row['final_evaluation_after_budget']=True
                else:
                    row.update(stage='training_complete',evaluation_skipped_reason=reason)
                    finish_stop(reason);break
            evaluation=folder/'natural-coins'
            status=stage('native_continuous',['--model',model,'--output',evaluation,'--difficulty',3,
                '--attempts',attempts,'--speed','fast']+(['--all-attempts'] if all_attempts else []),folder)
            evaluated=read_complete(evaluation/'result.json')
            clear=evaluate_result(evaluated,status,model_hash,attempts,identity,all_attempts=all_attempts)
            if all_attempts and len(evaluated['attempts'])!=attempts:
                raise RuntimeError('Full evaluation stopped before requested attempt count')
            totals=matchup_totals(evaluation,evaluated['attempts'])
            for opponent,counts in totals.items():
                target=record['evaluation_matchups'].setdefault(opponent,{k:0 for k in counts})
                for key,value in counts.items():target[key]+=value
            row['evaluation_matchups']=totals
            row.update(status='complete',stage='complete',clear=clear,attempts=evaluated['attempts'],
                evaluation_result=str(evaluation/'result.json'));save()
            print(json.dumps({'cycle':ordinal,'steps':actual_steps,'attempts':len(evaluated['attempts']),'clear':clear}),flush=True)
            record['any_clear']=record['any_clear'] or clear
            reason=stop_reason()
            if reason:finish_stop(reason);break
            if clear and stop_on_clear: record['status']='clear';break
        check()
    except KeyboardInterrupt:
        request_stop(stop_file,'user_cancelled');finish_stop('user_cancelled')
    except BaseException as error:
        record.update(status='invalid',error=f'{type(error).__name__}: {error}')
        if row is not None: row['status']='invalid'
        raise
    finally:
        record['wall_seconds']=clock()-started;save()
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('code','dataset','output','config'): parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--package',required=True)
    parser.add_argument('--steps',type=int,required=True)
    parser.add_argument('--cycles',type=int,help='Maximum training/evaluation cycles; provide this or --max-duration')
    parser.add_argument('--max-duration',type=float,help='Wall-clock seconds; soft limit at safe update/evaluation boundaries')
    parser.add_argument('--stop-on-clear',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--evaluate-on-stop',action=argparse.BooleanOptionalAction,default=False,
        help='After a duration stop, evaluate the final model once; may exceed the time budget')
    parser.add_argument('--rollout-steps',type=int)
    parser.add_argument('--minibatch-size',type=int,default=64)
    parser.add_argument('--workers',type=int,required=True,help='Positive number of parallel environments')
    parser.add_argument('--first-cycle-steps',type=int,
        help='Optional first training budget only; later cycles use --steps; multiple of rollout-steps (legacy packages: workers*256)')
    parser.add_argument('--python',default=sys.executable)
    parser.add_argument('--init-model',type=Path)
    parser.add_argument('--attempts',type=int,choices=(1,2,3),default=3)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--timeout',type=int,default=86400)
    parser.add_argument('--block',type=int,choices=(1,16,32,64,128,256),default=128)
    parser.add_argument('--checkpoint-every',type=int,default=20480,
        help='Completed PPO decisions between recoverable full ZIPs; multiple of rollout-steps (legacy packages: workers*256)')
    args=parser.parse_args()
    def stop(_signal,_frame): raise KeyboardInterrupt('SIGTERM')
    signal.signal(signal.SIGTERM,stop)
    try: result=campaign(**vars(args))
    except (Exception,KeyboardInterrupt) as error:
        print(f'Invalid campaign: {type(error).__name__}: {error}',file=sys.stderr);raise SystemExit(2)
    raise SystemExit(130 if result['status']=='cancelled' else 0 if result['status'] in ('clear','budget_stopped') or result.get('any_clear') else 1)


if __name__=='__main__': main()
