"""Bounded multi-process PPO with natural-opening splits and periodic evaluation."""
import argparse
from collections import Counter
from functools import partial
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from .vector import ManagedVec

from astra_play_sf2.config import atomic_json, load_config
from astra_play_sf2.runner import sha256
from .env import EVAL_LEADS, MameEnv
from .dataset import load_dataset



def make_worker(config, folder, difficulty, samples, phase='train'):
    env = MameEnv(config, folder, difficulty, checkpoints=samples)
    env.phase = phase
    return Monitor(env)


def evaluate(env, model, phase, baseline=False):
    env.phase, env.baseline = phase, baseline
    rows = []
    for index in range(len(env.checkpoints)):
        for lead in EVAL_LEADS:
            obs, _ = env.reset(options={'checkpoint': index, 'lead': lead})
            while True:
                action = 0 if baseline else int(model.predict(obs, deterministic=True)[0])
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    rows.append(env.episodes[-1])
                    break
    return {'rounds': dict(Counter(row['outcome'] for row in rows)), 'episodes': len(rows),
            'win_rate': sum(row['outcome']=='win' for row in rows)/len(rows),
            'mean_return': sum(row['return'] for row in rows)/len(rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, choices=range(1, 9), default=4)
    parser.add_argument('--steps', type=int, default=102400)
    parser.add_argument('--eval-every', type=int, default=10240)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--benchmark', action='store_true', help='Random-action throughput only; no PPO or held-out evaluation')
    args = parser.parse_args()
    quantum = args.workers*256
    if args.steps <= 0 or args.steps % quantum or args.eval_every <= 0 or args.eval_every % quantum:
        parser.error('steps and eval-every must be positive multiples of workers*256')
    if args.steps % args.eval_every and not args.benchmark:
        parser.error('steps must be a multiple of eval-every')
    torch.set_num_threads(1)
    groups, difficulty = load_dataset(args.dataset)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = load_config()
    manifest = {'schema': 'astra.rl-scaled.v1', 'training_only': True, 'formal_clear': False,
                'dataset_sha256': sha256(args.dataset), 'difficulty': difficulty,
                'seed': args.seed, 'workers': args.workers, 'requested_steps': args.steps,
                'eval_every': args.eval_every, 'benchmark': args.benchmark,
                'split_ids': {split: [s['id'] for s in rows] for split, rows in groups.items()},
                'status': 'initializing', 'evaluations': [],
                'sources': {p.name: sha256(p) for p in Path(__file__).parent.iterdir() if p.suffix in ('.py', '.lua')}}
    atomic_json(output/'result.json', manifest)
    vector = dev = holdout = None
    started = time.monotonic()
    try:
        vector = ManagedVec([partial(make_worker, config, output/f'worker-{i:02d}', difficulty,
                                      groups['train'], 'benchmark' if args.benchmark else 'train')
                               for i in range(args.workers)], start_method='spawn')
        vector.seed(args.seed)
        if args.benchmark:
            rng = np.random.default_rng(args.seed)
            vector.reset()
            start = time.monotonic()
            for _ in range(args.steps//args.workers):
                vector.step(rng.integers(0, 15, size=args.workers))
            elapsed = time.monotonic()-start
            manifest.update(actual_steps=args.steps, sampling_seconds=elapsed,
                            decisions_per_second=args.steps/elapsed, status='complete')
        else:
            dev = MameEnv(config, output/'dev', difficulty, checkpoints=groups['dev'])
            model = PPO('MlpPolicy', vector, seed=args.seed, device='cpu', n_steps=256,
                        batch_size=64, n_epochs=4, learning_rate=3e-4, gamma=0.99,
                        ent_coef=.01, policy_kwargs={'net_arch': dict(pi=[64,64], vf=[64,64])}, verbose=1)
            manifest['baseline_dev'] = evaluate(dev, model, 'baseline-dev', baseline=True)
            manifest['untrained_dev'] = evaluate(dev, model, 'untrained-dev')
            manifest['status'] = 'training'
            best_score = (-1., float('-inf'))
            train_seconds = 0.
            for target in range(args.eval_every, args.steps+1, args.eval_every):
                start = time.monotonic()
                model.learn(total_timesteps=args.eval_every, reset_num_timesteps=(target==args.eval_every))
                train_seconds += time.monotonic()-start
                model.save(output/f'ppo-{target:07d}')
                stats = evaluate(dev, model, 'dev')
                stats.update(steps=model.num_timesteps, train_seconds=train_seconds)
                manifest['evaluations'].append(stats)
                rank = (stats['win_rate'], stats['mean_return'])
                if rank > best_score:
                    best_score = rank
                    model.save(output/'best-dev')
                    manifest['best_dev_steps'] = model.num_timesteps
                manifest.update(actual_steps=model.num_timesteps, train_seconds=train_seconds,
                                decisions_per_second=model.num_timesteps/train_seconds)
                atomic_json(output/'result.json', manifest)
                print('EVALUATION '+json.dumps(stats), flush=True)
            # Selection is frozen before opening the final holdout environment.
            model = PPO.load(output/'best-dev', device='cpu')
            manifest['model_sha256'] = hashlib.sha256((output/'best-dev.zip').read_bytes()).hexdigest()
            holdout = MameEnv(config, output/'holdout', difficulty, checkpoints=groups['holdout'])
            manifest['baseline_holdout'] = evaluate(holdout, model, 'baseline-holdout', baseline=True)
            manifest['selected_holdout'] = evaluate(holdout, model, 'selected-holdout')
            manifest['status'] = 'complete'
    except BaseException as error:
        manifest.update(status='invalid', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        for env in (vector, dev, holdout):
            if env is not None:
                try:
                    env.close()
                    if getattr(env, 'close_errors', None):
                        manifest['status'] = 'invalid'
                        manifest.setdefault('cleanup_errors', []).extend(env.close_errors)
                except BaseException as error:
                    manifest['status'] = 'invalid'
                    manifest.setdefault('cleanup_errors', []).append(f'{type(error).__name__}: {error}')
        manifest['wall_seconds'] = time.monotonic()-started
        atomic_json(output/'result.json', manifest)
        print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
