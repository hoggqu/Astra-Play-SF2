"""Offline function-preserving 64x64 -> 128x128 PPO expansion; never launches MAME."""
import argparse
import copy
import importlib
import json
from pathlib import Path
import sys

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO

from astra_play_sf2.config import atomic_json
from astra_play_sf2.runner import sha256

INTERFACE = 'ken_actions16_lp_mp_uppercut_v1'
ARCHITECTURE = 'linear_tanh_actor_critic_128x128_v1'


class SpacesOnly(gym.Env):
    observation_space = gym.spaces.Box(-1, 1, (344,), dtype=np.float32)
    action_space = gym.spaces.Discrete(16)


def validate_old(old):
    if (old.observation_space.shape != (344,) or old.action_space.n != 16 or
            getattr(old, 'astra_action_interface', None) != INTERFACE or
            old.policy.net_arch != {'pi': [64, 64], 'vf': [64, 64]}):
        raise ValueError('Expected identified 344-input, 16-action, 64x64 actor/critic PPO')
    if type(old.policy.optimizer) is not torch.optim.Adam:
        raise ValueError('Only standard torch Adam is supported')
    from stable_baselines3.common.torch_layers import FlattenExtractor
    if type(old.policy.features_extractor) is not FlattenExtractor:
        raise ValueError('Only flat, unnormalized observations are supported')
    for network in (old.policy.mlp_extractor.policy_net, old.policy.mlp_extractor.value_net):
        modules = list(network.children())
        if len(modules) != 4 or any(type(layer) is not (torch.nn.Linear if i % 2 == 0 else torch.nn.Tanh)
                                    for i, layer in enumerate(modules)):
            raise ValueError('Expected two standard Linear/Tanh layers')


def prefix(tensor):
    return tuple(slice(0, n) for n in tensor.shape)


def expand_model(old, seed=316):
    validate_old(old)
    kwargs = copy.deepcopy(old.policy_kwargs)
    kwargs['net_arch'] = {'pi': [128, 128], 'vf': [128, 128]}
    new = PPO('MlpPolicy', SpacesOnly(), seed=seed, device='cpu', n_steps=old.n_steps,
        batch_size=old.batch_size, n_epochs=old.n_epochs, learning_rate=old.learning_rate,
        gamma=old.gamma, gae_lambda=old.gae_lambda, clip_range=old.clip_range,
        clip_range_vf=old.clip_range_vf, normalize_advantage=old.normalize_advantage,
        ent_coef=old.ent_coef, vf_coef=old.vf_coef, max_grad_norm=old.max_grad_norm,
        target_kl=old.target_kl, policy_kwargs=kwargs)
    old_params, new_params = dict(old.policy.named_parameters()), dict(new.policy.named_parameters())
    if old_params.keys() != new_params.keys() or len(old_params) != 12:
        raise ValueError('Unexpected trainable parameter topology')
    with torch.no_grad():
        for name, parameter in new_params.items():
            source = old_params[name]
            # Old second-layer units must ignore the newly added first-layer units.
            if name.endswith('_net.2.weight'):
                parameter[:64, 64:].zero_()
            # Random new hidden features initially have no path to policy/value output.
            if name in ('action_net.weight', 'value_net.weight'):
                parameter[:, 64:].zero_()
            parameter[prefix(source)].copy_(source)
    old_names = {id(parameter): name for name, parameter in old_params.items()}
    groups = []
    for group in old.policy.optimizer.param_groups:
        replacement = {key: copy.deepcopy(value) for key, value in group.items() if key != 'params'}
        replacement['params'] = [new_params[old_names[id(parameter)]] for parameter in group['params']]
        groups.append(replacement)
    new.policy.optimizer = torch.optim.Adam(groups, **copy.deepcopy(old.policy.optimizer.defaults))
    for name, source in old_params.items():
        state = old.policy.optimizer.state.get(source, {})
        target_state = {}
        for key, value in state.items():
            if key == 'step':
                target_state[key] = value.detach().clone() if torch.is_tensor(value) else copy.deepcopy(value)
            elif key in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq') and torch.is_tensor(value) and value.shape == source.shape:
                target = torch.zeros_like(new_params[name])
                target[prefix(value)].copy_(value)
                target_state[key] = target
            else:
                raise ValueError('Unexpected Adam state field: '+name+'/'+key)
        if state:
            new.policy.optimizer.state[new_params[name]] = target_state
    for name, value in vars(old).items():
        if name.startswith('astra_'):
            setattr(new, name, copy.deepcopy(value))
    for name in ('num_timesteps', '_n_updates', '_total_timesteps', '_num_timesteps_at_start', '_current_progress_remaining'):
        setattr(new, name, copy.deepcopy(getattr(old, name)))
    new.astra_architecture = ARCHITECTURE
    new.astra_optimizer_origin = 'old_adam_moments_embedded_at_width_expansion'
    new.astra_width_migration = {'schema': 'astra.rl-width-migration.v1', 'seed': seed,
        'source_architecture': {'pi': [64, 64], 'vf': [64, 64]},
        'architecture': kwargs['net_arch'], 'optimizer_reinitialized': False,
        'new_coordinate_moments': 'zero; each tensor retains the old scalar Adam step',
        'source_steps': old.num_timesteps, 'source_optimizer_updates': old._n_updates}
    return new


def check_embedding(old, new):
    old_params, new_params = dict(old.policy.named_parameters()), dict(new.policy.named_parameters())
    assert len(old.policy.optimizer.param_groups) == len(new.policy.optimizer.param_groups)
    for a, b in zip(old.policy.optimizer.param_groups, new.policy.optimizer.param_groups):
        assert {k:v for k,v in a.items() if k != 'params'} == {k:v for k,v in b.items() if k != 'params'}
    for name, source in old_params.items():
        target = new_params[name]
        torch.testing.assert_close(source, target[prefix(source)], rtol=0, atol=0)
        a, b = old.policy.optimizer.state.get(source, {}), new.policy.optimizer.state.get(target, {})
        assert a.keys() == b.keys()
        for key, value in a.items():
            if key == 'step':
                torch.testing.assert_close(value, b[key], rtol=0, atol=0)
            else:
                torch.testing.assert_close(value, b[key][prefix(value)], rtol=0, atol=0)
                rest = b[key].clone();rest[prefix(value)] = 0
                assert torch.count_nonzero(rest).item() == 0
    for branch in ('policy_net', 'value_net'):
        assert torch.count_nonzero(new_params[f'mlp_extractor.{branch}.2.weight'][:64, 64:]).item() == 0
        assert torch.count_nonzero(new_params[f'mlp_extractor.{branch}.0.weight'][64:]).item() > 0
        assert torch.count_nonzero(new_params[f'mlp_extractor.{branch}.2.weight'][64:]).item() > 0
    for head in ('action_net.weight', 'value_net.weight'):
        assert torch.count_nonzero(new_params[head][:, 64:]).item() == 0
    return {'ok': True, 'parameters': len(old_params), 'adam_states': len(new.policy.optimizer.state),
            'old_parameter_and_moment_slices': 'bit-exact', 'new_moment_coordinates': 'zero'}


def outputs(model, observations):
    with torch.no_grad():
        x = torch.as_tensor(observations, dtype=torch.float32)
        pi, vf = model.policy.mlp_extractor(model.policy.extract_features(x))
        return model.policy.action_net(pi).cpu().numpy(), model.policy.value_net(vf).cpu().numpy()


def check_function(old, new, observations, batch_sizes=(1, 32, 256)):
    """Check expansion and batch-shape stability, including scalar deployment."""
    if not len(observations) or not batch_sizes or any(type(n) is not int or n < 1 for n in batch_sizes):
        raise ValueError('Nonempty observations and positive integer batch sizes required')
    def compare(a, av, b, bv):
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-5)
        np.testing.assert_allclose(av, bv, rtol=1e-6, atol=1e-5)
        np.testing.assert_array_equal(a.argmax(1), b.argmax(1))
        return {'max_logit_error': float(np.max(np.abs(a-b))),
                'max_value_error': float(np.max(np.abs(av-bv))), 'argmax_mismatches': 0}
    a, av = outputs(old, observations);b, bv = outputs(new, observations)
    full = compare(a, av, b, bv)
    chunks = {}
    for size in dict.fromkeys(batch_sizes):
        results = []
        for model in (old, new):
            pieces = [outputs(model, observations[i:i+size]) for i in range(0, len(observations), size)]
            results.append(tuple(np.concatenate([part[j] for part in pieces]) for j in (0, 1)))
        old_chunk, new_chunk = results
        chunks[str(size)] = {'old_vs_new': compare(*old_chunk, *new_chunk),
            'old_vs_full_batch': compare(a, av, *old_chunk),
            'new_vs_full_batch': compare(b, bv, *new_chunk),
            'chunks_per_model': (len(observations)+size-1)//size}
    comparisons = [full]+[value for group in chunks.values() for key, value in group.items() if key != 'chunks_per_model']
    return {'ok': True, 'observations': len(observations), 'rtol': 1e-6, 'atol': 1e-5,
            'max_logit_error': max(v['max_logit_error'] for v in comparisons),
            'max_value_error': max(v['max_value_error'] for v in comparisons),
            'argmax_mismatches': 0, 'full_batch_old_vs_new': full, 'batch_sizes': list(dict.fromkeys(batch_sizes)),
            'chunks': chunks}


def check_roundtrip(new, loaded):
    for name, value in new.policy.state_dict().items():
        torch.testing.assert_close(value, loaded.policy.state_dict()[name], rtol=0, atol=0)
    a, b = new.policy.optimizer.state_dict(), loaded.policy.optimizer.state_dict()
    assert a['param_groups'] == b['param_groups'] and a['state'].keys() == b['state'].keys()
    for index, values in a['state'].items():
        assert values.keys() == b['state'][index].keys()
        for key, value in values.items():
            torch.testing.assert_close(value, b['state'][index][key], rtol=0, atol=0)
    assert loaded.policy.net_arch == {'pi': [128, 128], 'vf': [128, 128]}
    assert loaded.astra_width_migration == new.astra_width_migration
    assert loaded.num_timesteps == new.num_timesteps and loaded._n_updates == new._n_updates
    return {'ok': True, 'weights_and_optimizer': 'bit-exact after SB3 save/load'}


def check_learning(copy_of_model, observations):
    """Use a disposable model only; this synthetic objective never alters the artifact."""
    model = copy_of_model
    x = torch.as_tensor(observations[:32], dtype=torch.float32)
    heads, hidden = {}, {}
    for step in range(2):
        pi, vf = model.policy.mlp_extractor(model.policy.extract_features(x))
        logits, value = model.policy.action_net(pi), model.policy.value_net(vf)
        loss = torch.nn.functional.cross_entropy(logits, torch.arange(len(x)) % 16) + (value-1).square().mean()
        model.policy.optimizer.zero_grad();loss.backward()
        parameters = dict(model.policy.named_parameters())
        if step == 0:
            heads = {name: float(parameters[name].grad[:, 64:].abs().sum())
                     for name in ('action_net.weight', 'value_net.weight')}
            model.policy.optimizer.step()
        else:
            hidden = {name: float(parameters[name].grad[64:].abs().sum()) for name in
                ('mlp_extractor.policy_net.0.weight', 'mlp_extractor.policy_net.2.weight',
                 'mlp_extractor.value_net.0.weight', 'mlp_extractor.value_net.2.weight')}
    assert all(np.isfinite(v) and v > 0 for v in [*heads.values(), *hidden.values()])
    return {'ok': True, 'disposable_tensor_update_only': True,
            'first_step_new_head_gradient_l1': heads, 'second_step_new_hidden_gradient_l1': hidden}


def migrate(model, output, observations, observation_result, code, seed=316, lua_samples=256):
    torch.set_num_threads(1)
    model, output, observations, observation_result, code = [Path(p).resolve() for p in
        (model, output, observations, observation_result, code)]
    if lua_samples < 1:
        raise ValueError('At least one Lua observation required')
    provenance = json.loads(observation_result.read_text())
    if (provenance.get('status') != 'complete' or provenance.get('split') != 'train' or
            provenance.get('training_only') is not True or provenance.get('demonstrations_sha256') != sha256(observations)):
        raise ValueError('Expected identified completed train-only demonstration observations')
    with np.load(observations, allow_pickle=False) as data:
        real = data['observations'].astype(np.float32)
    if real.ndim != 2 or real.shape[1] != 344 or not len(real) or not np.isfinite(real).all() or np.abs(real).max() > 1:
        raise ValueError('Invalid clipped 344-feature observations')
    random = np.random.default_rng(seed).uniform(-1, 1, (1024, 344)).astype(np.float32)
    boundary = np.stack([np.zeros(344), np.ones(344), -np.ones(344)]).astype(np.float32)
    source_hash = sha256(model)
    old = PPO.load(model, device='cpu');new = expand_model(old, seed)
    new.astra_width_migration['source_model_sha256'] = source_hash
    checks = {'embedding': check_embedding(old, new), 'real_observations': check_function(old, new, real),
              'random_observations': check_function(old, new, random), 'boundary_observations': check_function(old, new, boundary)}
    manifest = json.loads((code/'build.json').read_text());package = manifest['package']
    if not package.isidentifier():raise ValueError('Invalid isolated package')
    sys.path[:0] = [str(code/'src'), str(code)]
    support = importlib.import_module(package+'.support')
    if not Path(support.__file__).resolve().is_relative_to(code/package):
        raise ValueError('Another package shadows the requested verification source')
    source = support.capture_interface(model, code/package)
    output.mkdir(parents=True, exist_ok=False)
    record = {'schema': 'astra.rl-width-migration.v1', 'status': 'running', 'architecture': ARCHITECTURE,
        'action_interface': INTERFACE, 'observations': 344, 'actions': 16, 'source_model_sha256': source_hash,
        'seed': seed, 'checks': checks, 'migration': new.astra_width_migration,
        'observation_sha256': sha256(observations), 'observation_result_sha256': sha256(observation_result),
        'observation_dataset_sha256': provenance.get('dataset_sha256'), 'observation_split': 'train',
        'verification_build_sha256': source['manifest_sha256'], 'migration_source_sha256': sha256(Path(__file__)),
        'formal_clear': False, 'training_started': False}
    target = output/'ppo-width128.zip'
    try:
        temporary = output/'ppo-width128.tmp.zip';new.save(temporary);temporary.replace(target)
        loaded = PPO.load(target, device='cpu')
        checks['roundtrip'] = check_roundtrip(new, loaded)
        migrated = support.capture_interface(target, code/package)
        from lupa import LuaRuntime
        lua = LuaRuntime(unpack_returned_tuples=True)
        nn_path = code/package/'nn.lua';nn = lua.execute(nn_path.read_text())
        from .export import lua_literal
        payloads = [lua.execute('return '+lua_literal(s['payload'])) for s in (source, migrated)]
        selection = np.linspace(0, len(real)-1, min(lua_samples, len(real)), dtype=int)
        samples = np.concatenate([real[selection], random[:min(64, lua_samples)], boundary])
        torch_logits = [outputs(m, samples)[0] for m in (old, loaded)]
        errors = []
        for i, obs in enumerate(samples):
            predictions = []
            for payload, logits in zip(payloads, torch_logits):
                action, raw = nn.predict(payload, lua.table_from(obs.tolist()))
                values = np.array([raw[j] for j in range(1, 17)])
                np.testing.assert_allclose(values, logits[i], rtol=1e-6, atol=1e-5)
                assert action == int(logits[i].argmax())
                predictions.append(action);errors.append(float(np.max(np.abs(values-logits[i]))))
            assert predictions[0] == predictions[1]
        checks['lua'] = {'ok': True, 'observations': len(samples), 'max_logit_error': max(errors),
                         'argmax_mismatches': 0, 'nn_sha256': sha256(nn_path)}
        before_probe = sha256(target)
        checks['new_coordinates_learn'] = check_learning(PPO.load(target, device='cpu'), random)
        assert sha256(target) == before_probe and sha256(model) == source_hash
        assert source['manifest_sha256'] == sha256(code/'build.json')
        for name, digest in source['package_sha256'].items():assert sha256(code/package/name) == digest
        record.update(status='complete', model=target.name, model_sha256=sha256(target),
                      source_parameters=sum(p.numel() for p in old.policy.parameters()),
                      parameters=sum(p.numel() for p in loaded.policy.parameters()))
    except BaseException as error:
        record.update(status='invalid', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        atomic_json(output/'migration.json', record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'output', 'observations', 'observation-result', 'code'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--seed', type=int, default=316)
    parser.add_argument('--lua-samples', type=int, default=256)
    print(json.dumps(migrate(**vars(parser.parse_args())), indent=2))


if __name__ == '__main__':main()
