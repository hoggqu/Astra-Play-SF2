"""Offline policy-only behavior cloning from validated train-pool projections.

No emulator is started. Whole demonstration episodes are reserved only for
within-training-pool fitting diagnostics; they are not gameplay holdout evidence.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import signal
import time

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from astra_play_sf2.config import atomic_json
from astra_play_sf2.runner import sha256
from .projection_teacher import ProjectionTeacher


class InterfaceOnlyEnv(gym.Env):
    observation_space = gym.spaces.Box(-1, 1, (344,), dtype=np.float32)
    action_space = gym.spaces.Discrete(15)

    def reset(self, **kwargs):
        raise RuntimeError('Behavior cloning must not sample an environment')

    def step(self, action):
        raise RuntimeError('Behavior cloning must not sample an environment')


def load_demonstrations(folder, dataset):
    folder, dataset = Path(folder).resolve(), Path(dataset).resolve()
    record = json.loads((folder/'result.json').read_text(encoding='utf-8'))
    if (record.get('schema') != 'astra.rl-projected-teacher.v1' or
            record.get('status') != 'complete' or record.get('split') != 'train' or
            record.get('training_only') is not True or record.get('formal_clear') is not False):
        raise ValueError('Need completed training-only train-split projected demonstrations')
    if record.get('dataset_sha256') != sha256(dataset):
        raise ValueError('Demonstration dataset identity mismatch')
    identity = ProjectionTeacher().identity
    for key in ('fighter_sha256', 'selection_sha256', 'projection_sha256'):
        if record.get('teacher', {}).get(key) != identity[key]:
            raise ValueError('Projected teacher source identity mismatch: '+key)
    if record['teacher'].get('lossy') is not True or record['teacher'].get('decision_frames') != 12:
        raise ValueError('Unexpected teacher interface')
    path = folder/'demonstrations.npz'
    if record.get('demonstrations_sha256') != sha256(path):
        raise ValueError('Demonstration NPZ checksum mismatch')
    # BC consumes labels, never MAME states. Verify the exact original manifest
    # and split metadata without requiring cross-host checkpoint downloads.
    data = json.loads(dataset.read_text(encoding='utf-8'))
    if data.get('schema') not in ('astra.rl-openings.v1', 'astra.rl-openings.v2') or data.get('status') != 'complete':
        raise ValueError('Need completed source opening manifest')
    groups = {'train': [], 'dev': [], 'holdout': []}
    opponents = data.get('opponents', [data.get('opponent', 2)])
    seen = set()
    for sample in data['openings']:
        digest = sample.get('sha256', '')
        if (sample.get('status') != 'accepted' or sample.get('split') not in groups or
                sample.get('opponent') not in opponents or sample.get('difficulty') != data['difficulty'] or
                not isinstance(digest, str) or len(digest) != 64 or
                any(c not in '0123456789abcdef' for c in digest) or digest in seen):
            raise ValueError('Invalid source checkpoint split metadata')
        seen.add(digest)
        groups[sample['split']].append(sample)
    if not opponents or any({s['opponent'] for s in rows} != set(opponents) for rows in groups.values()):
        raise ValueError('Source manifest lacks per-opponent split coverage')
    difficulty = data['difficulty']
    if difficulty != record.get('difficulty'):
        raise ValueError('Demonstration difficulty mismatch')
    with np.load(path, allow_pickle=False) as archive:
        required = {'observations', 'actions', 'episode_ids', 'checkpoint_indices'}
        if set(archive.files) != required:
            raise ValueError('Unexpected demonstration array schema')
        arrays = {key: archive[key].copy() for key in required}
    obs, actions, episodes, checkpoints = (arrays[key] for key in ('observations', 'actions', 'episode_ids', 'checkpoint_indices'))
    n = len(actions)
    if (not n or obs.shape != (n, 344) or any(x.shape != (n,) for x in (actions, episodes, checkpoints)) or
            any(not np.issubdtype(x.dtype, np.integer) for x in (actions, episodes, checkpoints)) or
            not np.issubdtype(obs.dtype, np.floating) or not np.isfinite(obs).all() or np.any(np.abs(obs) > 1) or
            np.any(actions < 0) or np.any(actions >= 15) or np.any(checkpoints < 0) or np.any(checkpoints >= len(groups['train']))):
        raise ValueError('Invalid demonstration shapes, types, or values')
    rows = record.get('episodes', [])
    if (n != record.get('demonstration_count') or n != record.get('steps') or
            len(rows) != record.get('round_budget') or
            set(episodes.tolist()) != set(range(len(rows)))):
        raise ValueError('Incomplete demonstration episode/count evidence')
    for ordinal, row in enumerate(rows):
        selected = episodes == ordinal
        if (set(checkpoints[selected].tolist()) != {row['checkpoint']} or
                int(selected.sum()) != row['steps'] or row.get('outcome') not in ('win', 'loss', 'draw') or
                row['opponent'] != groups['train'][row['checkpoint']]['opponent'] or
                row.get('phase') != 'projected-teacher-train' or row.get('baseline') is not False):
            raise ValueError('Demonstration/native episode metadata mismatch')
    arrays['opponents'] = np.asarray([groups['train'][i]['opponent'] for i in checkpoints], dtype=np.int64)
    arrays['observations'] = obs.astype(np.float32)
    return arrays, record


def episode_partition(episodes, opponents):
    """Last whole episode per opponent held back when at least two are present."""
    diagnostics = []
    for opponent in sorted(set(opponents.tolist())):
        ids = sorted(set(episodes[opponents == opponent].tolist()))
        if len(ids) >= 2:
            diagnostics.append(ids[-1])
    # With one episode per opponent, no per-opponent reserve is possible. Reserve
    # one whole episode globally, explicitly recording the absent fit opponent.
    if not diagnostics:
        ids = sorted(set(episodes.tolist()))
        if len(ids) < 2:
            raise ValueError('At least two whole episodes needed for fitting diagnostics')
        diagnostics = [ids[-1]]
    diagnostic = np.isin(episodes, diagnostics)
    if not diagnostic.any() or diagnostic.all():
        raise ValueError('Need nonempty whole-episode fit and diagnostic sets')
    return np.flatnonzero(~diagnostic), np.flatnonzero(diagnostic)


def create_model(initial, seed):
    env = InterfaceOnlyEnv()
    model = PPO.load(initial, env=env, device='cpu', seed=seed) if initial else PPO(
        'MlpPolicy', env, seed=seed, device='cpu', n_steps=256, batch_size=64,
        n_epochs=4, learning_rate=3e-4, gamma=.99, ent_coef=.01,
        policy_kwargs={'net_arch': dict(pi=[64, 64], vf=[64, 64])})
    policy = model.policy
    if model.n_steps != 256 or policy.observation_space.shape != (344,) or policy.action_space.n != 15:
        raise ValueError('Expected compatible 344-input, 15-action PPO with n_steps 256')
    for net in (policy.mlp_extractor.policy_net, policy.mlp_extractor.value_net):
        layers = list(net.children())
        if (len(layers) != 4 or not all(isinstance(layers[i], torch.nn.Linear) for i in (0, 2)) or
                not all(isinstance(layers[i], torch.nn.Tanh) for i in (1, 3)) or
                [(layers[i].in_features, layers[i].out_features) for i in (0, 2)] != [(344, 64), (64, 64)]):
            raise ValueError('Expected two independent 64/64 Tanh MLPs')
    if any(True for _ in policy.features_extractor.parameters()):
        raise ValueError('Trainable shared feature extractor not supported for policy-only BC')
    return model


def accuracy(policy, observations, actions, indices, batch_size):
    correct, total_loss = 0, 0.
    with torch.no_grad():
        for offset in range(0, len(indices), batch_size):
            index = indices[offset:offset+batch_size]
            logits = policy.get_distribution(observations[index]).distribution.logits
            correct += int((logits.argmax(dim=1) == actions[index]).sum())
            total_loss += float(torch.nn.functional.cross_entropy(logits, actions[index], reduction='sum'))
    return {'samples': len(indices), 'accuracy': correct/len(indices), 'cross_entropy': total_loss/len(indices)}


def train(args):
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        raise ValueError('Epochs, batch size, and learning rate must be positive')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = {'schema': 'astra.rl-bc.v1', 'status': 'initializing', 'training_only': True,
              'formal_clear': False, 'method': 'lossy projected-teacher supervised policy initialization',
              'epochs_requested': args.epochs, 'batch_size': args.batch_size, 'seed': args.seed,
              'learning_rate': args.learning_rate, 'balance_opponents': args.balance_opponents,
              'evaluations': [], 'diagnostic_scope': 'whole episodes reserved within train pool; not gameplay dev/holdout',
              'value_network_trained': False, 'environment_steps': 0,
              'source_sha256': sha256(Path(__file__)),
              'checkpoint_validation': 'source manifest identity/split metadata only; no MAME states needed or consumed',
              'optimizer': 'separate Adam on policy MLP and action head only; original PPO optimizer retained'}
    atomic_json(output/'result.json', result)
    try:
        torch.set_num_threads(1)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        arrays, source = load_demonstrations(args.demonstrations, args.dataset)
        fit, diagnostic = episode_partition(arrays['episode_ids'], arrays['opponents'])
        result.update(dataset_sha256=source['dataset_sha256'], difficulty=source['difficulty'],
                      demonstration_result_sha256=sha256(Path(args.demonstrations)/'result.json'),
                      demonstrations_sha256=source['demonstrations_sha256'], teacher=source['teacher'],
                      init_model_sha256=sha256(args.init_model) if args.init_model else None,
                      fit_episode_ids=sorted(set(arrays['episode_ids'][fit].tolist())),
                      diagnostic_episode_ids=sorted(set(arrays['episode_ids'][diagnostic].tolist())),
                      fit_opponent_samples=dict(Counter(map(str, arrays['opponents'][fit]))),
                      diagnostic_opponent_samples=dict(Counter(map(str, arrays['opponents'][diagnostic]))))
        model = create_model(args.init_model, args.seed)
        policy = model.policy
        initial = {key: value.detach().clone() for key, value in policy.state_dict().items()}
        parameters = list(policy.mlp_extractor.policy_net.parameters())+list(policy.action_net.parameters())
        optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)
        observations = torch.as_tensor(arrays['observations'], dtype=torch.float32)
        actions = torch.as_tensor(arrays['actions'], dtype=torch.long)
        weights = torch.ones(len(actions), dtype=torch.float32)
        if args.balance_opponents:
            counts = Counter(arrays['opponents'][fit].tolist())
            for opponent, count in counts.items():
                weights[np.flatnonzero(arrays['opponents'] == opponent)] = len(fit)/(len(counts)*count)
        rng = np.random.default_rng(args.seed)
        def measure(epoch):
            policy.set_training_mode(False)
            row = {'epoch': epoch, 'fit': accuracy(policy, observations, actions, fit, args.batch_size),
                   'training_pool_episode_diagnostic': accuracy(policy, observations, actions, diagnostic, args.batch_size)}
            result['evaluations'].append(row)
            atomic_json(output/'result.json', result)
            print(json.dumps(row), flush=True)
        result['status'] = 'training'
        measure(0)
        for epoch in range(1, args.epochs+1):
            policy.set_training_mode(True)
            order = rng.permutation(fit)
            for offset in range(0, len(order), args.batch_size):
                index = order[offset:offset+args.batch_size]
                optimizer.zero_grad(set_to_none=True)
                logits = policy.get_distribution(observations[index]).distribution.logits
                loss = (torch.nn.functional.cross_entropy(logits, actions[index], reduction='none')*weights[index]).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, .5)
                optimizer.step()
            result['epochs_completed'] = epoch
            measure(epoch)
        unchanged = [key for key in initial if not key.startswith(('mlp_extractor.policy_net.', 'action_net.'))]
        if any(not torch.equal(initial[key], policy.state_dict()[key]) for key in unchanged):
            raise RuntimeError('Behavior cloning changed value/shared parameters')
        result['value_and_other_parameters_unchanged'] = True
        result['policy_parameters_changed'] = any(not torch.equal(value, policy.state_dict()[key]) for key, value in initial.items())
        model.save(output/'ppo-bc')
        reloaded = PPO.load(output/'ppo-bc.zip', device='cpu')
        if any(not torch.equal(value, reloaded.policy.state_dict()[key]) for key, value in policy.state_dict().items()):
            raise RuntimeError('Saved PPO policy reload mismatch')
        result.update(status='complete', model_sha256=sha256(output/'ppo-bc.zip'))
    except BaseException as error:
        result.update(status='invalid', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        result['wall_seconds'] = time.monotonic()-started
        atomic_json(output/'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demonstrations', type=Path, required=True, help='Completed projected_run output directory')
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--init-model', type=Path)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--balance-opponents', action='store_true')
    args = parser.parse_args()
    def interrupted(_signal, _frame):
        raise KeyboardInterrupt('BC interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    print(json.dumps(train(args), indent=2), flush=True)


if __name__ == '__main__':
    main()
