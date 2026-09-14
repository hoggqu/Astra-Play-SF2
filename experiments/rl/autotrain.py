"""Run standalone PPO training for a time or round budget, then write an HTML report."""
import argparse
from datetime import datetime, timezone
import json
import math
import os
import signal
from pathlib import Path
import sys
import uuid
from .managed_runtime import positive_workers
from .rollout_control import validate_sizes


def positive_hours(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError('Hours must be positive and finite')
    return number


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True, help='Completed Normal training manifest; only its train split is used')
    p.add_argument('--output', type=Path, help='New output directory; default .local/rl-runs/train-<unique-id>')
    p.add_argument('--hours', type=positive_hours, help='Wall-clock budget in hours; finish current update and final evaluation before exit')
    p.add_argument('--rounds', '--cycles', dest='rounds', type=int, help='Training cycles, each followed by automatic evaluation; not game rounds')
    p.add_argument('--steps-per-round', type=int, default=409600, help='PPO decisions per cycle; rounded up to rollout-steps (default409600)')
    p.add_argument('--rollout-steps', type=int, default=4096, help='Exact total decisions before each PPO update, independent of workers')
    p.add_argument('--minibatch-size', type=int, default=64, help='Samples per gradient step; must divide rollout-steps')
    from .managed_runtime import positive_learning_rate
    p.add_argument('--learning-rate', type=positive_learning_rate, help='Override saved PPO learning rate while preserving Adam; omitted inherits checkpoint')
    p.add_argument('--all-attempts', action='store_true', help='Complete every requested evaluation coin even after a clear')
    p.add_argument('--workers', type=positive_workers, default=8, help='Parallel MAME environments; any positive integer (default8)')
    p.add_argument('--device', choices=('cpu','cuda','mps','auto'), default='cpu', help='Torch update device; MAME/Lua sampling stays on CPU')
    p.add_argument('--opponent-sampling', choices=('adaptive','uniform'), default='adaptive', help='Adaptive weak-opponent practice (default); uniform for a baseline')
    p.add_argument('--attempts', type=int, choices=(1,2,3), default=3, help='Automatic natural coins after each training cycle')
    selection = p.add_mutually_exclusive_group()
    selection.add_argument('--init-model', type=Path, help='Trusted complete compatible PPO ZIP; omitted starts fresh')
    selection.add_argument('--resume', type=Path, help='Previous training output/campaign with a published latest model; always use a new --output')
    p.add_argument('--config', type=Path, help='MAME configuration; default astra-sf2 configuration')
    p.add_argument('--code', type=Path, help='Existing validated managed source; default build from this checkout')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--checkpoint-every', type=int, default=4096)
    p.add_argument('--stop-on-clear', action='store_true', help='Optional early stop at first audited clear; default runs the requested budget')
    return p


def validate_args(args):
    if args.hours is None and args.rounds is None:
        raise ValueError('Provide --hours, --rounds, or both (the first limit reached stops training)')
    if args.rounds is not None and args.rounds < 1:
        raise ValueError('--rounds must be positive')
    if type(args.workers) is not int or args.workers < 1:
        raise ValueError('--workers must be a positive integer')
    validate_sizes(args.rollout_steps,args.minibatch_size,args.workers)
    for name, value in (('--steps-per-round',args.steps_per_round),('--checkpoint-every',args.checkpoint_every)):
        if value < 1:
            raise ValueError(f'{name} must be positive')
    if not args.dataset.expanduser().is_file():
        raise ValueError('Dataset manifest not found; create one with experiments.rl.collect or supply --dataset PATH')
    if args.output is not None and args.output.expanduser().exists():
        raise ValueError('Output already exists; choose a new directory and use --resume to continue')


def training_environment(device, opponent_sampling='adaptive', learning_rate=None):
    environment = {'ASTRA_RL_LEARNING_RATE':str(learning_rate) if learning_rate is not None else '', 'ASTRA_RL_DEVICE':device, 'ASTRA_RL_OPPONENT_SAMPLING':opponent_sampling, 'OMP_NUM_THREADS':'1',
                   'MKL_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1'}
    if sys.platform.startswith('linux') or sys.platform == 'darwin':
        # -video none disables rendering; SDL still needs a non-desktop backend.
        environment['SDL_VIDEODRIVER'] = 'dummy'
    return environment


def effective_budgets(args):
    quantum = args.rollout_steps
    align = lambda value: ((value + quantum - 1) // quantum) * quantum
    return align(args.steps_per_round), align(args.checkpoint_every)


def run(args):
    validate_args(args)
    steps, checkpoint_every = effective_budgets(args)
    from astra_play_sf2.config import config_path
    from .managed_builder import bootstrap
    from .managed_identity import PACKAGE, validate_build
    from .managed_runtime import select_device
    from .versioned_campaign import campaign
    from .watch import resolve_model, digest
    config = (args.config or config_path()).expanduser().resolve()
    if not config.is_file():
        raise ValueError('MAME configuration missing; run astra-sf2 configure --mame PATH --rom-dir PATH first')
    # Fail unavailable devices before creating any MAME worker.
    resolved_device = select_device(args.device)
    model = args.init_model.expanduser().resolve() if args.init_model else None
    if args.resume:
        model, expected = resolve_model(campaign=args.resume)
        if digest(model) != expected: raise ValueError('Resume checkpoint hash does not match published model')
    if model is not None and not model.is_file(): raise ValueError('Initial model does not exist')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = (args.output or Path('.local/rl-runs')/('train-'+stamp+'-'+uuid.uuid4().hex[:8])).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    launch = {'schema':'astra.rl-autotrain.v1','created_utc':datetime.now(timezone.utc).isoformat(),
              'status':'preparing','output':str(output),'dataset':str(args.dataset.expanduser().resolve()),
              'config':str(config),'hours':args.hours,'rounds':args.rounds,'steps_per_round':steps,
              'steps_per_round_requested':args.steps_per_round,'checkpoint_every_requested':args.checkpoint_every,
              'checkpoint_every':checkpoint_every,'decisions_per_update':args.rollout_steps, 'rollout_steps':args.rollout_steps, 'minibatch_size':args.minibatch_size,
              'learning_rate_requested':args.learning_rate,'all_attempts':args.all_attempts,'workers':args.workers,'opponent_sampling':args.opponent_sampling,'device_requested':args.device,'device_resolved':resolved_device,
              'init_model':str(model) if model else None,'stop_on_clear':args.stop_on_clear,
              'training_environment':training_environment(resolved_device, args.opponent_sampling, args.learning_rate),
              'time_limit':'Soft wall-clock budget; setup excluded, completed PPO update and final evaluation may overrun',
              'training_round':'One training decision budget followed by up to attempts natural-coin evaluations'}
    def save():
        (output/'launch.json').write_text(json.dumps(launch,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    save()
    try:
        if args.code:
            code = args.code.expanduser().resolve()
            identity = json.loads((code/'build.json').read_text(encoding='utf-8'))
            validate_build(identity,code/PACKAGE)
        else:
            code = output/'code'; bootstrap(code)
        launch.update(code=str(code),status='running'); save()
        print(f'Output: {output}\nReport after completion: {output / "report.html"}',flush=True)
        print(f'Workers: {args.workers}; Torch: {resolved_device}; decisions/cycle: {steps}; rollout: {args.rollout_steps}; minibatch: {args.minibatch_size}',flush=True)
        print(f'Learning rate: {args.learning_rate if args.learning_rate is not None else "inherit checkpoint / fresh default"}; complete all evaluation coins: {args.all_attempts}',flush=True)
        if (steps, checkpoint_every) != (args.steps_per_round, args.checkpoint_every):
            print(f'Rounded up to complete updates ({args.rollout_steps} decisions): cycle {args.steps_per_round} -> {steps}; checkpoint {args.checkpoint_every} -> {checkpoint_every}',flush=True)
        print(f'Opponent sampling: {args.opponent_sampling}; full checkpoint preserves recent matchup history.',flush=True)
        print('Ctrl+C requests a safe stop; current update/game finishes before exit.',flush=True)
        environment = training_environment(resolved_device, args.opponent_sampling, args.learning_rate)
        previous_environment = {name:os.environ.get(name) for name in environment}
        os.environ.update(environment)
        prior_signals = {}
        def cancel_signal(_number, _frame):
            raise KeyboardInterrupt('Stop requested')
        for number in (signal.SIGINT, signal.SIGTERM):
            prior_signals[number] = signal.getsignal(number)
            signal.signal(number,cancel_signal)
        try:
            result = campaign(code=code,package=PACKAGE,dataset=args.dataset,output=output/'run',config=config,
                              cycles=args.rounds,steps=steps,workers=args.workers,rollout_steps=args.rollout_steps,minibatch_size=args.minibatch_size,
                              init_model=model,attempts=args.attempts,seed=args.seed,python=sys.executable,
                              checkpoint_every=checkpoint_every,max_duration=args.hours*3600 if args.hours else None,
                              stop_on_clear=args.stop_on_clear,evaluate_on_stop=True,all_attempts=args.all_attempts)
        finally:
            for number,handler in prior_signals.items(): signal.signal(number,handler)
            for name,value in previous_environment.items():
                if value is None: os.environ.pop(name,None)
                else: os.environ[name] = value
        launch.update(status=result['status'],latest_model=result.get('latest_model'),latest_model_sha256=result.get('latest_model_sha256'))
    except (Exception,KeyboardInterrupt) as error:
        launch.update(status='invalid',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        launch['finished_utc'] = datetime.now(timezone.utc).isoformat();save()
        if (output/'run/result.json').is_file():
            from .training_report import write_report
            write_report(output,output)
    return result,output


def main(argv=None):
    p = parser();args = p.parse_args(argv)
    try: result,output = run(args)
    except (ValueError,OSError,RuntimeError,ImportError) as error:
        p.exit(2,str(error)+'\nInstall dependencies with: python -m pip install -e . -r experiments/rl/requirements.txt\n')
    print(json.dumps({'status':result['status'],'latest_model':result.get('latest_model'),
                      'report':str(output/'report.html'),'any_clear':result.get('any_clear',False)},ensure_ascii=False,indent=2))
    return 2 if result['status']=='invalid' else 130 if result['status']=='cancelled' else 0


if __name__ == '__main__': raise SystemExit(main())
