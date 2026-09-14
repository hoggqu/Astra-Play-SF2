"""Fixed-budget LR comparison: identical checkpoint, full per-cycle and final evaluation."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import uuid

from . import autotrain
from .managed_runtime import positive_learning_rate, positive_workers
from .versioned_campaign import digest, full_zip, write_json, run_child, source_pins, require_pins, evaluate_result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--checkpoint-sha256', required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--dataset-sha256', required=True)
    p.add_argument('--learning-rate', type=positive_learning_rate, required=True)
    p.add_argument('--rounds', type=positive_workers, default=10)
    p.add_argument('--workers', type=positive_workers, default=12)
    p.add_argument('--device', choices=('cpu','cuda','mps','auto'), default='cpu')
    p.add_argument('--steps-per-round', type=positive_workers, default=409600)
    p.add_argument('--final-attempts', type=positive_workers, default=20)
    p.add_argument('--opponent-multipliers', help='Optional JSON factors for reset probabilities')
    p.add_argument('--config', type=Path)
    p.add_argument('--output', type=Path)
    return p


def training_args(args, output, checkpoint):
    values = ['--dataset',str(args.dataset.resolve()),'--init-model',str(checkpoint),
              '--learning-rate',str(args.learning_rate),'--rounds',str(args.rounds),
              '--steps-per-round',str(args.steps_per_round),'--workers',str(args.workers),
              '--device',args.device,'--rollout-steps','16384','--minibatch-size','256',
              '--opponent-sampling','adaptive','--seed','42','--attempts','3','--all-attempts',
              '--output',str(output/'training')]
    if args.opponent_multipliers is not None: values += ['--opponent-multipliers',args.opponent_multipliers]
    if args.config: values += ['--config',str(args.config.resolve())]
    return autotrain.parser().parse_args(values)


def run(args):
    if args.steps_per_round % 16384:
        raise ValueError('Comparison steps-per-round must be an exact multiple of 16384')
    if full_zip(args.checkpoint) != args.checkpoint_sha256:
        raise ValueError('Comparison checkpoint hash mismatch')
    if digest(args.dataset) != args.dataset_sha256:
        raise ValueError('Comparison dataset hash mismatch')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = (args.output or Path('.local/rl-runs')/f'lr-ab-{args.learning_rate:g}-{stamp}-{uuid.uuid4().hex[:6]}').resolve()
    output.mkdir(parents=True, exist_ok=False)
    record = dict(schema='astra.rl-lr-comparison.v1',status='preparing',
                  initial_model_sha256=args.checkpoint_sha256,dataset_sha256=args.dataset_sha256,
                  opponent_multipliers=args.opponent_multipliers,
                  learning_rate=args.learning_rate,rounds=args.rounds,workers=args.workers,
                  device_requested=args.device,steps_per_round=args.steps_per_round,
                  training_decisions=args.rounds*args.steps_per_round,rollout_steps=16384,minibatch_size=256,
                  seed=42,cycle_evaluation_attempts=3,final_evaluation_attempts=args.final_attempts,
                  selection='final checkpoint after exact training budget; no best-model selection')
    def save(): write_json(output/'comparison.json',record)
    save()
    try:
        checkpoint=output/'initial-model.zip';shutil.copy2(args.checkpoint,checkpoint)
        if digest(checkpoint)!=args.checkpoint_sha256:raise ValueError('Initial snapshot changed during copy')
        print(f'Comparison output: {output}\nLR={args.learning_rate:g}; {args.rounds} cycles; {record["training_decisions"]} decisions; final {args.final_attempts} coins',flush=True)
        record["status"]="training";save()
        result,training = autotrain.run(training_args(args,output,checkpoint))
        record.update(training=str(training),training_status=result['status'],latest_model=result.get('latest_model'))
        if result['status'] != 'exhausted':
            record['status']=result['status'];save();return record,output
        cycles=result['cycles']
        if len(cycles)!=args.rounds or sum(c.get('training_steps',0) for c in cycles)!=record['training_decisions']:
            raise RuntimeError('Comparison did not complete the exact training budget')
        for cycle in cycles:
            trained=json.loads((training/'run'/f"cycle-{cycle['ordinal']:03d}"/'train/result.json').read_text())
            if trained['effective_ppo']['learning_rate']!=args.learning_rate:
                raise RuntimeError('Effective learning rate differs from comparison arm')
            if len(cycle['attempts'])!=3:raise RuntimeError('Missing full cycle evaluation')
        code=training/'code';manifest=json.loads((code/'build.json').read_text())
        identity,pins=source_pins(code,manifest['package'])
        model=Path(result['latest_model']);model_hash=full_zip(model)
        if model_hash!=result['latest_model_sha256']:raise RuntimeError('Final model hash mismatch')
        evaluation=output/'final-evaluation'
        record.update(status='final_evaluation',final_model_sha256=model_hash,final_evaluation=str(evaluation));save()
        environment=dict(os.environ,**autotrain.training_environment(args.device))
        launch=json.loads((training/'launch.json').read_text())
        environment['ASTRA_SF2_CONFIG']=launch['config']
        environment['PYTHONPATH']=os.pathsep.join((str(code/'src'),str(code)))
        for name in ('ASTRA_RL_STOP_FILE','ASTRA_CAMPAIGN_DEADLINE_MONOTONIC'):environment.pop(name,None)
        command=[sys.executable,str(code/'launch.py'),'native_continuous','--model',str(model),
                 '--output',str(evaluation),'--difficulty','3','--attempts',str(args.final_attempts),
                 '--all-attempts','--speed','fast']
        print(f'Final fixed-model evaluation: {args.final_attempts} natural coins; log: {output / "final-evaluation.log"}',flush=True)
        status=run_child(command,output/'final-evaluation.log',environment,86400)
        evaluated=json.loads((evaluation/'result.json').read_text())
        evaluate_result(evaluated,status,model_hash,args.final_attempts,identity,all_attempts=True)
        require_pins(code,pins)
        if digest(model)!=model_hash:raise RuntimeError('Model changed during final evaluation')
        clears=sum(a['outcome']=='rl_gameplay_clear' for a in evaluated['attempts'])
        record.update(status='complete',clears=clears,attempts=len(evaluated['attempts']),
                      clear_rate=clears/len(evaluated['attempts']),
                      mean_defeated=sum(a['match_wins'] for a in evaluated['attempts'])/len(evaluated['attempts']))
        print(f'Final result: {clears}/{len(evaluated["attempts"])} clears\nTraining report: {training / "report.html"}\nComparison: {output / "comparison.json"}',flush=True)
    except BaseException as error:
        record.update(status='cancelled' if isinstance(error,KeyboardInterrupt) else 'invalid',error=f'{type(error).__name__}: {error}')
        raise
    finally: save()
    return record,output


def main(argv=None):
    p=parser();args=p.parse_args(argv)
    try: result,_=run(args)
    except (ValueError,RuntimeError,OSError) as error:p.exit(2,str(error)+'\n')
    return 0 if result['status']=='complete' else 130 if result['status']=='cancelled' else 2


if __name__=='__main__':raise SystemExit(main())
