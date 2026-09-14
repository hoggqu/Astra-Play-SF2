"""Bounded two-arm scenario-coverage experiment, independently runnable on each GPU host."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

from . import lr_comparison
from .managed_builder import bootstrap
from .managed_runtime import select_device
from .versioned_campaign import digest, full_zip, run_child, write_json


def diagnostic(code, model, dataset, output, config):
    environment = dict(os.environ, SDL_VIDEODRIVER='dummy', ASTRA_SF2_CONFIG=str(config),
                       OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    command = [sys.executable, '-c',
               "import sys,runpy;root=sys.argv.pop(1);sys.path[:0]=[root+'/src',root];runpy.run_module('experiments.rl.deterministic_train_eval',run_name='__main__')",
               str(Path(__file__).resolve().parents[2]),
               '--source', str(code/'astra_sf2_rl_managed'), '--model', str(model),
               '--dataset', str(dataset), '--output', str(output), '--dev-all']
    status = run_child(command, output.with_suffix('.log'), environment, 1200)
    result = json.loads((output/'result.json').read_text())
    if status != 0 or result['status'] != 'complete':
        raise RuntimeError('Fixed-state diagnostic failed: '+str(output))
    return {'path':str(output), 'by_opponent':result['by_opponent'],
            'formal_clear':False, 'matches_requested':result['matches_requested']}


def run(settings_path, output):
    settings = json.loads(Path(settings_path).read_text())
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    record = {'schema':'astra.rl-scenario-comparison.v1', 'status':'preparing',
              'settings':settings, 'started_unix':started, 'arms':[], 'pid':os.getpid()}
    def save(): write_json(output/'suite.json',record)
    save()
    try:
        if settings['order'] not in (['A','B'], ['B','A']): raise ValueError('Expected both arms once')
        device = select_device(settings['device'])
        if device not in ('cuda','mps'): raise ValueError('This experiment requires an explicit GPU')
        record['device_resolved'] = device
        model = Path(settings['model']).resolve()
        if full_zip(model) != settings['model_sha256']: raise ValueError('Initial model hash mismatch')
        paths = {}
        for label in ('A','B','evaluation'):
            row = settings['datasets'][label]; path = Path(row['path']).resolve()
            if digest(path) != row['sha256']: raise ValueError('Dataset hash mismatch: '+label)
            paths[label] = path
        config = Path(settings['config']).resolve()
        code = output/'diagnostic-code'; bootstrap(code)
        record.update(status='baseline_diagnostic'); save()
        record['baseline_diagnostic'] = diagnostic(code,model,paths['evaluation'],output/'initial-diagnostic',config)
        for label in settings['order']:
            record.update(status='training',active_arm=label); save()
            args = lr_comparison.parser().parse_args([
                '--checkpoint',str(model),'--checkpoint-sha256',settings['model_sha256'],
                '--dataset',str(paths[label]),'--dataset-sha256',settings['datasets'][label]['sha256'],
                '--learning-rate','0.0001','--rounds',str(settings['rounds']),
                '--workers',str(settings['workers']),'--device',device,
                '--steps-per-round','409600','--final-attempts','20',
                '--config',str(config),'--output',str(output/('arm-'+label))])
            result, arm_output = lr_comparison.run(args)
            if result['status'] != 'complete': raise RuntimeError('Arm did not complete: '+label)
            for file in (arm_output/'training/run').glob('cycle-*/train/result.json'):
                trained = json.loads(file.read_text())
                if trained.get('device_resolved') != device: raise RuntimeError('GPU backend changed')
            record.update(status='diagnostic'); save()
            checked = diagnostic(arm_output/'training/code',Path(result['latest_model']),paths['evaluation'],
                                 output/('diagnostic-'+label),config)
            record['arms'].append({'arm':label,'comparison':str(arm_output/'comparison.json'),
                'clears':result['clears'],'attempts':result['attempts'],
                'mean_defeated':result['mean_defeated'],'final_model_sha256':result['final_model_sha256'],
                'fixed_state_diagnostic':checked})
            save()
        record.update(status='complete')
    except BaseException as error:
        record.update(status='invalid',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        record['wall_seconds']=time.time()-started;save()
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); result=run(args.settings,args.output)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
