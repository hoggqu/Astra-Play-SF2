"""Bounded optional batched-PPO prototype; benchmark/parity before adoption."""
import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import signal
import time
from unittest.mock import patch

import numpy as np
import torch
from stable_baselines3 import PPO

from astra_play_sf2.config import atomic_json, load_config
from astra_play_sf2.runner import sha256
from .batch_env import BatchEnv, BatchBridge, ATTACH
from . import env as env_module
from astra_play_sf2.transport import Bridge
from .dataset import load_dataset
from .env import MameEnv
from .vector import ManagedVec


def live_policy(model):
    """Snapshot only the current policy; value/logprob stay with frozen Torch."""
    policy = model.policy
    modules = list(policy.mlp_extractor.policy_net.children())
    layers = []
    if len(modules) % 2:
        raise ValueError('Expected Linear/Tanh layers')
    for i in range(0, len(modules), 2):
        linear, activation = modules[i:i+2]
        if not isinstance(linear, torch.nn.Linear) or not isinstance(activation, torch.nn.Tanh):
            raise ValueError('Only Linear/Tanh policy is supported')
        layers.append({'weight': linear.weight.detach().cpu().tolist(), 'bias': linear.bias.detach().cpu().tolist()})
    head = policy.action_net
    layers.append({'weight': head.weight.detach().cpu().tolist(), 'bias': head.bias.detach().cpu().tolist()})
    payload = {'schema': 'astra.rl-policy.v1', 'observations': 344, 'actions': 15,
               'history': 4, 'decision_frames': 12, 'activation': 'tanh',
               'selection': 'deterministic_argmax', 'layers': layers}
    payload['model_sha256'] = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return payload


def fill_buffer(model, chunks):
    """Preserve SB3 time/env axes; no optimization until every old-policy row arrives."""
    workers = len(chunks)
    steps = len(chunks[0]['transitions'])
    if workers != model.n_envs or steps != model.n_steps:
        raise ValueError('PPO rollout shape mismatch')
    if any(len(chunk['transitions']) != steps for chunk in chunks):
        raise ValueError('Unequal worker rollout lengths')
    obs = np.asarray([[chunks[w]['transitions'][t]['observation'] for w in range(workers)] for t in range(steps)], dtype=np.float32)
    actions = np.asarray([[chunks[w]['transitions'][t]['action'] for w in range(workers)] for t in range(steps)], dtype=np.int64)
    rewards = np.asarray([[chunks[w]['transitions'][t]['reward'] for w in range(workers)] for t in range(steps)], dtype=np.float32)
    starts = np.asarray([[chunks[w]['transitions'][t]['episode_start'] for w in range(workers)] for t in range(steps)], dtype=np.float32)
    dones = np.asarray([[chunks[w]['transitions'][t]['done'] for w in range(workers)] for t in range(steps)], dtype=bool)
    final_dones = np.asarray([chunk['episode_start'] for chunk in chunks], dtype=bool)
    if not np.array_equal(dones[:-1], starts[1:].astype(bool)) or not np.array_equal(dones[-1], final_dones):
        raise RuntimeError('Terminal flags do not match following episode starts')
    expected_logp = np.asarray([[chunks[w]['transitions'][t]['logprob'] for w in range(workers)] for t in range(steps)])
    with torch.no_grad():
        values, logp, _ = model.policy.evaluate_actions(torch.as_tensor(obs.reshape(-1, 344)), torch.as_tensor(actions.reshape(-1)))
    actual_logp = logp.reshape(steps, workers).cpu().numpy()
    error = float(np.max(np.abs(actual_logp-expected_logp)))
    if not np.isfinite(error) or error > 2e-5:
        raise RuntimeError(f'Lua/Torch old-policy logprob disagreement: {error}')
    values = values.reshape(steps, workers)
    logp = logp.reshape(steps, workers)
    model.rollout_buffer.reset()
    for t in range(steps):
        model.rollout_buffer.add(obs[t], actions[t], rewards[t], starts[t], values[t], logp[t])
    final_obs = np.asarray([chunk['observation'] for chunk in chunks], dtype=np.float32)
    final_dones = np.asarray([chunk['episode_start'] for chunk in chunks], dtype=bool)
    with torch.no_grad():
        final_values = model.policy.predict_values(torch.as_tensor(final_obs))
    model.rollout_buffer.compute_returns_and_advantage(last_values=final_values, dones=final_dones)
    return error


class ParityBridge(Bridge):
    def send(self, command, snapshot=True):
        if command == ATTACH:
            path = self.base / 'runtime/rl.lua'
            path.write_bytes((Path(__file__).parent/'corrected_reference_runtime.lua').read_bytes())
        return super().send(command, snapshot=snapshot)


def parity(config, groups, difficulty, output, count, seed):
    """Same checkpoint, lead, and forced actions; compare native trajectories exactly."""
    reference = batch = None
    try:
        with patch.object(env_module, 'Bridge', ParityBridge):
            reference = MameEnv(config, output/'reference', difficulty, checkpoints=groups['train'])
        batch = BatchEnv(config, output/'batch', difficulty, checkpoints=groups['train'])
        options = {'checkpoint': 0, 'lead': 2}
        expected, _ = reference.reset(seed=seed, options=options)
        actual, _ = batch.reset(seed=seed, options=options)
        atomic_json(output/'parity-initial-reference.json', reference.previous)
        np.testing.assert_array_equal(actual, expected)
        actions = np.random.default_rng(seed).integers(0, 15, count).tolist()
        result = batch.batch_rollout(count, actions=actions, resets=[options]*count)
        for index, (action, transition) in enumerate(zip(actions, result['transitions'])):
            np.testing.assert_array_equal(np.asarray(transition['observation'], dtype=np.float32), expected)
            expected, reward, done, truncated, _ = reference.step(action)
            if not np.isclose(reward, transition['reward'], rtol=0, atol=1e-12) or done != transition['done'] or truncated:
                raise RuntimeError(f'Native reward/done mismatch at decision {index}')
            if reference.previous != transition['state']:
                atomic_json(output/'parity-mismatch.json', {'decision': index, 'action': action,
                    'actions': actions, 'reference_state': reference.previous, 'batch_state': transition['state'],
                    'reference_observation': expected.tolist(), 'batch_before_observation': transition['observation'],
                    'reference_frames': reference.last_frames, 'batch_frames': transition['frames']})
                raise RuntimeError(f'Native state mismatch at decision {index}')
            if done:
                expected, _ = reference.reset(options=options)
        np.testing.assert_array_equal(np.asarray(result['observation'], dtype=np.float32), expected)
        return {'status': 'complete', 'decisions': count, 'native_parity': True,
                'episodes': len(result['episodes'])}
    finally:
        for env in (reference, batch):
            if env is not None:
                env.close()


def native_parity(config, groups, difficulty, output, count, seed):
    reference = batch = None
    try:
        reference = BatchEnv(config, output/'native-reference', difficulty, checkpoints=groups['train'])
        batch = BatchEnv(config, output/'batch', difficulty, checkpoints=groups['train'])
        options = {'checkpoint': 0, 'lead': 2}
        expected, _ = reference.reset(seed=seed, options=options)
        actual, _ = batch.reset(seed=seed, options=options)
        np.testing.assert_array_equal(actual, expected)
        actions = np.random.default_rng(seed).integers(0, 15, count).tolist()
        plan = dict(actions=actions, resets=[options]*count)
        old = reference.batch_rollout(count, native_reference=True, **plan)
        new = batch.batch_rollout(count, **plan)
        for index, (a,b) in enumerate(zip(old['transitions'],new['transitions'])):
            np.testing.assert_array_equal(np.asarray(a['observation'],dtype=np.float32),np.asarray(b['observation'],dtype=np.float32))
            if a['state']!=b['state'] or a['done']!=b['done'] or a['reward']!=b['reward']:
                atomic_json(output/'parity-mismatch.json', {'decision':index,'reference':a,'batch':b,'actions':actions})
                raise RuntimeError(f'Native deployment/batch mismatch at decision {index}')
        np.testing.assert_array_equal(np.asarray(old['observation'],dtype=np.float32),np.asarray(new['observation'],dtype=np.float32))
        return {'status':'complete','decisions':count,'native_parity':True,
                'reference':'native_continuous_core.lua in training-only checkpoint harness',
                'episodes':len(new['episodes'])}
    finally:
        for env in (reference,batch):
            if env is not None: env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, choices=range(1, 9), default=4)
    parser.add_argument('--steps', type=int, default=4096)
    parser.add_argument('--block', type=int, choices=(1, 16, 32, 64, 128, 256), default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--init-model', type=Path)
    parser.add_argument('--benchmark', action='store_true', help='Frozen-policy sampling only; no parameter update')
    parser.add_argument('--native-parity', action='store_true', help='No-pause deployment Core vs batch, same forced inputs')
    parser.add_argument('--parity', action='store_true', help='One old/new environment, same forced inputs; no training')
    args = parser.parse_args()
    is_parity = args.parity or args.native_parity
    if args.parity and args.native_parity:
        parser.error('Choose one reference mode')
    if args.steps <= 0 or (not is_parity and args.steps % (args.workers*256)) or (is_parity and args.steps>256):
        parser.error('Steps must be workers*256 multiples, or 1..256 for parity')
    output = args.output.resolve();output.mkdir(parents=True, exist_ok=False)
    config = load_config();groups, difficulty = load_dataset(args.dataset)
    torch.set_num_threads(1)
    result = {'schema': 'astra.rl-batch-prototype.v1', 'training_only': True, 'formal_clear': False,
              'status': 'initializing', 'workers': args.workers, 'block': args.block,
              'seed': args.seed, 'requested_steps': args.steps, 'benchmark': args.benchmark,
              'parity': args.parity, 'native_parity': args.native_parity, 'init_model_sha256': sha256(args.init_model) if args.init_model else None, 'reference': ('corrected native-time RPC; legacy runtime.lua unchanged' if args.parity else 'native-time batch sampling'), 'difficulty': difficulty, 'dataset_sha256': sha256(args.dataset),
              'sampling': 'frozen-policy categorical with worker NumPy uniforms; Torch batches old values/logprobs',
              'sources': {p.name: sha256(p) for p in Path(__file__).parent.iterdir() if p.suffix in ('.py', '.lua')},
              'iterations': []}
    atomic_json(output/'result.json', result)
    vector = None;started = time.monotonic()
    def stop(_signal, _frame):
        raise KeyboardInterrupt('SIGTERM')
    signal.signal(signal.SIGTERM, stop)
    try:
        if args.native_parity:
            result.update(native_parity(config, groups, difficulty, output, args.steps, args.seed))
        elif args.parity:
            result.update(parity(config, groups, difficulty, output, args.steps, args.seed))
        else:
            vector = ManagedVec([partial(BatchEnv, config, output/f'worker-{i:02d}', difficulty, checkpoints=groups['train']) for i in range(args.workers)])
            vector.seed(args.seed)
            model = PPO('MlpPolicy', vector, seed=args.seed, device='cpu', n_steps=256, batch_size=64, n_epochs=4,
                        learning_rate=3e-4, gamma=.99, ent_coef=.01, policy_kwargs={'net_arch': {'pi': [64,64], 'vf': [64,64]}})
            if args.init_model:
                model = PPO.load(args.init_model, env=vector, device='cpu', seed=args.seed)
                if model.n_steps != 256:
                    raise ValueError('Initial model requires n_steps=256')
            result['effective_ppo'] = {'n_steps':model.n_steps,'gamma':model.gamma,'gae_lambda':model.gae_lambda,
                'ent_coef':model.ent_coef,'learning_rate':model.learning_rate if not callable(model.learning_rate) else 'schedule',
                'n_epochs':model.n_epochs,'net_arch':model.policy.net_arch,'batch_size':model.batch_size}
            initial_parameters = {name:value.detach().clone() for name,value in model.policy.state_dict().items()} if not args.benchmark else None
            initial_updates = model._n_updates
            # Initialize SB3 schedules/logger/reset exactly once; sampling is below.
            model._setup_learn(args.steps, reset_num_timesteps=True)
            sampling = updating = 0.
            result['status'] = 'sampling'
            for target in range(args.workers*256, args.steps+1, args.workers*256):
                model.policy.set_training_mode(False)
                payload = live_policy(model)
                gathered = [{'transitions': []} for _ in range(args.workers)]
                sample_start = time.monotonic()
                for offset in range(0, 256, args.block):
                    chunks = vector.env_method('batch_rollout', args.block, payload if offset==0 else None)
                    for w, chunk in enumerate(chunks):
                        if chunk['model_sha256'] != payload['model_sha256']:
                            raise RuntimeError('Worker sampled another policy version')
                        gathered[w]['transitions'].extend(chunk['transitions'])
                        gathered[w].update(observation=chunk['observation'], episode_start=chunk['episode_start'])
                sampling += time.monotonic()-sample_start
                update_start = time.monotonic()
                error = fill_buffer(model, gathered)
                model.num_timesteps = target
                model._update_current_progress_remaining(target, args.steps)
                if not args.benchmark:
                    model.train()
                updating += time.monotonic()-update_start
                row = {'steps': target, 'sampling_seconds': sampling, 'update_seconds': updating, 'max_logprob_error': error}
                result['iterations'].append(row)
                result.update(actual_steps=target, sampling_seconds=sampling, update_seconds=updating,
                              decisions_per_sampling_second=target/sampling,
                              decisions_per_training_second=target/(sampling+updating))
                atomic_json(output/'result.json', result)
                print(json.dumps(row), flush=True)
            if not args.benchmark:
                result['parameters_changed'] = any(not torch.equal(value,model.policy.state_dict()[name]) for name,value in initial_parameters.items())
                result['optimizer_epochs_completed'] = model._n_updates-initial_updates
                if not result['parameters_changed']:
                    raise RuntimeError('PPO did not change any parameter')
                model.save(output/'ppo-batch')
                result['model_sha256'] = sha256(output/'ppo-batch.zip')
            result['status'] = 'complete'
    except BaseException as error:
        result.update(status='invalid', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        if vector is not None:
            vector.close()
            if vector.close_errors:
                result.update(status='invalid', cleanup_errors=vector.close_errors)
        result['wall_seconds'] = time.monotonic()-started
        atomic_json(output/'result.json', result)
        print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
