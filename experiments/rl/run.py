"""Run a bounded PPO pilot: baseline, untrained, train, frozen checkpoint replay."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from astra_play_sf2.config import load_config
from .env import EVAL_LEADS, MameEnv


def evaluate(env, model, phase, baseline=False):
    env.phase, env.baseline = phase, baseline
    rows = []
    for lead in EVAL_LEADS:
        observation, _ = env.reset(options={'lead': lead})
        while True:
            action = 0 if baseline else int(model.predict(observation, deterministic=True)[0])
            observation, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                rows.append(env.episodes[-1])
                print(f'{phase}: lead={lead} {rows[-1]["outcome"]} return={rows[-1]["return"]:.3f}', flush=True)
                break
    return {'rounds': dict(Counter(row['outcome'] for row in rows)),
            'mean_return': sum(row['return'] for row in rows)/len(rows), 'episodes': len(rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Fresh directory, preferably under .local/rl-runs/')
    parser.add_argument('--steps', type=int, default=4096, help='PPO decisions; positive multiple of 256')
    parser.add_argument('--difficulty', type=int, choices=range(3, 8), default=7)
    parser.add_argument('--show-window', action='store_true', help='Show MAME; training is silent and headless by default')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.steps < 256 or args.steps % 256:
        parser.error('--steps must be a positive multiple of 256')
    torch.set_num_threads(1)
    output = args.output.resolve()
    result = {'training_only': True, 'formal_clear': False, 'seed': args.seed,
              'requested_steps': args.steps, 'status': 'running'}
    started = time.monotonic()
    env = None
    try:
        env = MameEnv(load_config(), output, args.difficulty, args.show_window)
        wrapped = Monitor(env)
        model = PPO('MlpPolicy', wrapped, seed=args.seed, device='cpu', n_steps=256,
                    batch_size=64, n_epochs=4, learning_rate=3e-4, gamma=0.99,
                    ent_coef=0.01, policy_kwargs={'net_arch': dict(pi=[64,64], vf=[64,64])}, verbose=1)
        initial = {name: tensor.detach().clone() for name, tensor in model.policy.state_dict().items()}
        result['baseline'] = evaluate(env, model, 'baseline', baseline=True)
        result['untrained'] = evaluate(env, model, 'untrained')
        env.phase, env.baseline = 'train', False
        train_start = time.monotonic()
        model.learn(total_timesteps=args.steps)
        result['train_seconds'] = time.monotonic()-train_start
        result['actual_steps'] = model.num_timesteps
        result['parameters_changed'] = any(not torch.equal(value, model.policy.state_dict()[name]) for name, value in initial.items())
        assert result['parameters_changed'], 'PPO did not update parameters'
        model.save(output/'ppo-ken')
        model = PPO.load(output/'ppo-ken', device='cpu')
        result['model_sha256'] = hashlib.sha256((output/'ppo-ken.zip').read_bytes()).hexdigest()
        result['trained'] = evaluate(env, model, 'trained')
        result['status'] = 'complete'
    except BaseException as error:
        result['status'] = 'invalid'
        result['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        if env is not None:
            result['rpc_seconds'] = env.rpc_seconds
            result['native_frames'] = env.total_frames
            result['training_rounds'] = dict(Counter(row['outcome'] for row in env.episodes if row['phase']=='train'))
            env.close()
            result['wall_seconds'] = time.monotonic()-started
            (output/'result.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
            print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
