"""One host, one arm: adaptive baseline A or Sagat reset-probability multiplier B."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time
from . import lr_comparison
from .versioned_campaign import write_json


def arm_arguments(settings, output):
    arm = settings.get('arm')
    if arm not in ('A', 'B') or 'order' in settings:
        raise ValueError('Choose exactly one arm A or B per host; no order/second arm')
    factor = {} if arm == 'A' else {'9': 2}
    values = ['--checkpoint', settings['model'], '--checkpoint-sha256', settings['model_sha256'],
              '--dataset', settings['dataset'], '--dataset-sha256', settings['dataset_sha256'],
              '--learning-rate', '0.0001', '--workers', '12', '--device', settings['device'],
              '--rounds', '2', '--steps-per-round', '409600', '--final-attempts', '40',
              '--opponent-multipliers', json.dumps(factor), '--config', settings['config'],
              '--output', str(Path(output)/('arm-'+arm))]
    return lr_comparison.parser().parse_args(values)


def run(settings_path, output, argument_factory=arm_arguments):
    settings = json.loads(Path(settings_path).read_text())
    output = Path(output).resolve()
    args = argument_factory(settings, output)
    if args.device not in ('mps', 'cuda'):
        raise ValueError('Comparison requires explicit MPS or CUDA')
    output.mkdir(parents=True, exist_ok=False)
    record = dict(schema='astra.rl-single-arm-sampling-comparison.v1', status='training',
                  settings=settings, active_arm=settings['arm'], arms=[], pid=os.getpid(),
                  device_resolved=args.device, started_unix=time.time())
    save = lambda: write_json(output/'suite.json', record)
    save()
    try:
        result, folder = lr_comparison.run(args)  # Exactly ONE invocation; no swapped arm.
        if result['status'] != 'complete':
            raise RuntimeError('Assigned arm incomplete: '+result['status'])
        expected = json.loads(args.opponent_multipliers)
        for file in (folder/'training/run').glob('cycle-*/train/result.json'):
            trained = json.loads(file.read_text())
            if trained['device_resolved'] != args.device:
                raise RuntimeError('Device changed during experiment')
            for field in ('opponent_sampling_initial', 'opponent_sampling'):
                if trained[field]['multipliers'] != expected:
                    raise RuntimeError('Sampling factors differ from assigned arm')
        evaluated = json.loads((folder/'final-evaluation/result.json').read_text())
        failures = Counter()
        for attempt in evaluated['attempts']:
            if attempt['outcome'] != 'rl_gameplay_clear':
                failures[Path(attempt['matches'][-1]).stem.split('-',1)[1]] += 1
        code = json.loads((folder/'training/code/build.json').read_text())
        record['arms'] = [dict(arm=settings['arm'], comparison=str(folder/'comparison.json'),
            clears=result['clears'], attempts=result['attempts'], mean_defeated=result['mean_defeated'],
            failures=dict(failures), final_model_sha256=result['final_model_sha256'],
            settlement_protocol=code['settlement_protocol'])]
        record['status'] = 'complete'
    except BaseException as error:
        record.update(status='invalid', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        record['wall_seconds'] = time.time()-record['started_unix']; save()
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.settings,args.output),ensure_ascii=False,indent=2))


if __name__ == '__main__': main()
