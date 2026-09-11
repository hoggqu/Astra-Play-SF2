"""Bounded projected-teacher probe / train-pool demonstrations, never formal play."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import signal
import time

import numpy as np
from astra_play_sf2.config import atomic_json, load_config
from astra_play_sf2.runner import sha256
from .dataset import load_dataset
from .env import MameEnv
from .projection_teacher import ProjectionTeacher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=('train', 'dev'), default='train')
    parser.add_argument('--rounds', type=int, default=33)
    parser.add_argument('--steps', type=int, default=16384)
    parser.add_argument('--lead', type=int, choices=range(13), default=2)
    parser.add_argument('--save-observations', action='store_true')
    args = parser.parse_args()
    if args.rounds <= 0 or args.steps <= 0:
        parser.error('rounds and steps must be positive')
    if args.save_observations and args.split != 'train':
        parser.error('Demonstrations may only be saved from train split')
    def interrupted(_signum, _frame):
        raise KeyboardInterrupt('Projected teacher interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    groups, difficulty = load_dataset(args.dataset)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    teacher = ProjectionTeacher()
    result = {'schema': 'astra.rl-projected-teacher.v1', 'training_only': True, 'formal_clear': False,
              'status': 'running', 'dataset_sha256': sha256(args.dataset), 'split': args.split,
              'difficulty': difficulty, 'round_budget': args.rounds, 'step_budget': args.steps,
              'lead': args.lead, 'teacher': teacher.identity, 'episodes': [], 'steps': 0}
    atomic_json(output/'result.json', result)
    by_opponent = defaultdict(list)
    for index, sample in enumerate(groups[args.split]):
        by_opponent[sample['opponent']].append(index)
    opponents = sorted(by_opponent)
    observations, actions, episode_ids, sample_ids = [], [], [], []
    mapping_counts, reasons = Counter(), Counter()
    started = time.monotonic()
    env = None
    try:
        env = MameEnv(load_config(), output/'environment', difficulty, checkpoints=groups[args.split])
        env.phase = 'projected-teacher-'+args.split
        with (output/'decisions.jsonl').open('a', encoding='utf-8') as log:
            for ordinal in range(args.rounds):
                if result['steps'] >= args.steps:
                    break
                opponent = opponents[ordinal % len(opponents)]
                pool = by_opponent[opponent]
                index = pool[(ordinal//len(opponents)) % len(pool)]
                obs, _ = env.reset(options={'checkpoint': index, 'lead': args.lead})
                teacher.reset()
                while result['steps'] < args.steps:
                    action, diagnostic = teacher.predict(env.previous)
                    if args.save_observations:
                        observations.append(obs.copy())
                        actions.append(action)
                        episode_ids.append(ordinal)
                        sample_ids.append(index)
                    mapping_counts[diagnostic['mapping']] += 1
                    reasons[str(diagnostic['reason'])] += 1
                    log.write(json.dumps({'episode': ordinal, 'checkpoint': index, 'opponent': opponent,
                                          'decision': result['steps'], 'action': action, **diagnostic})+'\n')
                    obs, _, terminated, truncated, _ = env.step(action)
                    result['steps'] += 1
                    if terminated or truncated:
                        result['episodes'].append(env.episodes[-1])
                        break
                atomic_json(output/'result.json', result)
                print(json.dumps({'episodes': len(result['episodes']), 'steps': result['steps'],
                                  'latest': result['episodes'][-1]['outcome'] if result['episodes'] else None}), flush=True)
        result['status'] = 'complete' if len(result['episodes']) == args.rounds else 'budget_exhausted'
    except BaseException as error:
        result.update(status='invalid', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        if env is not None:
            try:
                env.close()
            except BaseException as error:
                result.update(status='invalid', cleanup_error=f'{type(error).__name__}: {error}')
        if args.save_observations:
            np.savez_compressed(output/'demonstrations.npz',
                                observations=np.asarray(observations, dtype=np.float32).reshape(-1, 344),
                                actions=np.asarray(actions, dtype=np.int64),
                                episode_ids=np.asarray(episode_ids, dtype=np.int64),
                                checkpoint_indices=np.asarray(sample_ids, dtype=np.int64))
            result['demonstrations_sha256'] = sha256(output/'demonstrations.npz')
            result['demonstration_count'] = len(actions)
        result['rounds'] = dict(Counter(row['outcome'] for row in result['episodes']))
        result['per_opponent'] = {str(o): dict(Counter(row['outcome'] for row in result['episodes'] if row['opponent'] == o)) for o in opponents}
        result.update(mapping_counts=dict(mapping_counts), reason_counts=dict(reasons), wall_seconds=time.monotonic()-started)
        atomic_json(output/'result.json', result)
        print(json.dumps(result, indent=2), flush=True)
    return 0 if result['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
