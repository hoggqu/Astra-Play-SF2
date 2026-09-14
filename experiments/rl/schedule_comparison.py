"""Single-arm comparison: local A constant Adam, remote B cosine Adam."""
import argparse
import json
from pathlib import Path
from . import lr_comparison, sampling_comparison


def arm_arguments(settings, output):
    arm = settings.get('arm')
    if arm not in ('A','B') or 'order' in settings:
        raise ValueError('Choose exactly one arm A or B per host; no second arm')
    values = ['--checkpoint',settings['model'],'--checkpoint-sha256',settings['model_sha256'],
              '--dataset',settings['dataset'],'--dataset-sha256',settings['dataset_sha256'],
              '--learning-rate','0.0001','--lr-schedule','constant' if arm=='A' else 'cosine',
              '--workers','12','--device',settings['device'],'--rounds','2',
              '--steps-per-round','409600','--final-attempts','40','--opponent-multipliers','{"9":2}',
              '--config',settings['config'],'--output',str(Path(output)/('arm-'+arm))]
    if arm=='B': values += ['--lr-end','0.00003']
    return lr_comparison.parser().parse_args(values)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--settings',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    print(json.dumps(sampling_comparison.run(args.settings,args.output,arm_arguments),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
