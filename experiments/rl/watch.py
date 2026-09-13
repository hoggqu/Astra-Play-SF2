"""Watch one silent, visible PPO game; independent of training and campaign scores."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid
import zipfile


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


def resolve_model(model=None,campaign=None):
    """Resolve an explicitly published checkpoint, never guess by file mtime."""
    if (model is None)==(campaign is None):raise ValueError('Choose one of --model or --campaign')
    if model is not None:return Path(model).expanduser().resolve(),None
    pointer=Path(campaign).expanduser().resolve()
    if pointer.is_dir():
        direct=pointer/'result.json'
        pointer=direct if direct.is_file() else pointer/'run/result.json'
    value=json.loads(pointer.read_text(encoding='utf-8'))
    name=value.get('latest_model');expected=value.get('latest_model_sha256')
    if not isinstance(name,str) or not name or not isinstance(expected,str) or len(expected)!=64:
        raise ValueError('Campaign has no published latest_model and SHA256; use --model for a completed ZIP')
    path=Path(name).expanduser()
    if not path.is_absolute():path=pointer.parent/path
    return path.resolve(),expected


def snapshot_model(source,target,expected=None):
    """Pin a completed ZIP even if training publishes a newer one while watching."""
    before=digest(source)
    if expected is not None and before!=expected:raise ValueError('Published checkpoint SHA256 mismatch')
    shutil.copyfile(source,target)
    after=digest(target)
    if after!=before or digest(source)!=before:raise ValueError('Checkpoint changed during snapshot; retry after publication')
    with zipfile.ZipFile(target) as archive:
        if not {'data','policy.pth','policy.optimizer.pth'}.issubset(archive.namelist()):
            raise ValueError('Expected a complete PPO ZIP with policy, critic and optimizer')
        if archive.testzip() is not None:raise ValueError('Damaged PPO ZIP')
    return after


def prepare_code(code,output):
    from .managed_builder import bootstrap
    if code is None:
        code=output/'code';bootstrap(code)
    code=Path(code).expanduser().resolve()
    manifest=json.loads((code/'build.json').read_text(encoding='utf-8'))
    if manifest.get('package')=='astra_sf2_rl_perception128_timing':
        from .perception128_timing_identity import validate_build
    elif manifest.get('package')=='astra_sf2_rl_managed':
        from .managed_identity import validate_build
    else:
        raise ValueError('Unsupported viewer source package')
    validate_build(manifest,code/manifest['package'])
    return code,digest(code/'build.json')


def viewer_environment(config=None):
    env=os.environ.copy()
    # SSH into WSL often drops DISPLAY although WSLg's X server is available.
    # Discover only the standard existing local socket; preserve explicit choices.
    if (sys.platform.startswith('linux') and not env.get('DISPLAY')
            and Path('/mnt/wslg').is_dir() and Path('/tmp/.X11-unix/X0').is_socket()):
        env['DISPLAY']=':0'
        env.setdefault('SDL_VIDEODRIVER','x11')
    if config is not None:env['ASTRA_SF2_CONFIG']=str(config)
    return env


def watch(*,model=None,campaign=None,code=None,config=None,output=None,
          difficulty=3,speed='normal',attempts=1,runner=subprocess.run):
    if difficulty not in range(3,8) or speed not in ('normal','2x','4x','fast'):
        raise ValueError('Unsupported difficulty or speed')
    if type(attempts) is not int or attempts<1:raise ValueError('Attempts must be positive')
    source,expected=resolve_model(model,campaign)
    if config is not None:
        config=Path(config).expanduser().resolve()
        if not config.is_file():raise ValueError('Configuration file does not exist')
    if output is None:
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        output=Path('.local/rl-watch')/(stamp+'-'+uuid.uuid4().hex[:8])
    output=Path(output).expanduser().resolve();output.mkdir(parents=True,exist_ok=False)
    record={'schema':'astra.rl-watch.v1','purpose':'visual_play_only','training_statistics':False,
            'automatic_verification':False,'status':'preparing','source_model':str(source),
            'difficulty':difficulty,'speed':speed,'attempts_requested':attempts,
            'sound':'none','show_window':True,'always_on_top':False,
            'created_utc':datetime.now(timezone.utc).isoformat()}
    write_json(output/'watch.json',record)
    try:
        frozen=output/'model.zip';record['model_sha256']=snapshot_model(source,frozen,expected)
        code,build_hash=prepare_code(code,output)
        record.update(code=str(code),build_sha256=build_hash)
        command=[sys.executable,str(code/'launch.py'),'native_continuous','--model',str(frozen),
                 '--output',str(output/'play'),'--difficulty',str(difficulty),'--attempts',str(attempts),
                 '--speed',speed,'--show-window','--all-attempts']
        env=viewer_environment(config)
        record['display_environment']={k:env[k] for k in ('DISPLAY','WAYLAND_DISPLAY','SDL_VIDEODRIVER') if k in env}
        record.update(status='playing',command=command);write_json(output/'watch.json',record)
        print('Watch directory: '+str(output),flush=True)
        completed=runner(command,env=env,check=False)
        result_path=output/'play/result.json'
        result=json.loads(result_path.read_text(encoding='utf-8')) if result_path.is_file() else {}
        record.update(child_exit_code=completed.returncode,status='complete' if result.get('status')=='complete' and completed.returncode in (0,1) else 'invalid',
                      attempts=[{'id':a.get('id'),'outcome':a.get('outcome'),'match_wins':a.get('match_wins')} for a in result.get('attempts',[])])
        if record['status']!='complete':record['error']=result.get('error','Play process did not finish cleanly')
    except (Exception,KeyboardInterrupt) as error:
        record.update(status='invalid',error=f'{type(error).__name__}: {error}')
    finally:
        record['finished_utc']=datetime.now(timezone.utc).isoformat();write_json(output/'watch.json',record)
    return record,output


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    selection=parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--model',type=Path,help='A trusted completed PPO ZIP, copied before play')
    selection.add_argument('--campaign',type=Path,help='Campaign JSON or directory with published latest_model and hash')
    parser.add_argument('--code',type=Path,help='Validated perception128 input-timing build; default builds current managed source from this checkout')
    parser.add_argument('--config',type=Path,help='astra-sf2 configuration JSON; default uses normal CLI configuration')
    parser.add_argument('--output',type=Path,help='New independent watch directory; default .local/rl-watch/<unique-id>')
    parser.add_argument('--difficulty',type=int,choices=range(3,8),default=3)
    parser.add_argument('--speed',choices=('normal','2x','4x','fast'),default='normal')
    parser.add_argument('--attempts',type=int,default=1,help='Natural coins; default1, never Continue or load states')
    args=parser.parse_args(argv)
    try:result,output=watch(**vars(args))
    except (ValueError,OSError) as error:parser.exit(2,str(error)+'\n')
    print(json.dumps({'status':result['status'],'attempts':result.get('attempts',[]),
                      'error':result.get('error'),'directory':str(output)},ensure_ascii=False,indent=2))
    return 0 if result['status']=='complete' else 2


if __name__=='__main__':raise SystemExit(main())
