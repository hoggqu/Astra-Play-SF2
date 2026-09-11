"""Export an SB3 discrete Tanh MLP policy for local, uninterrupted Lua inference."""
import argparse
import json
import math
from pathlib import Path

from astra_play_sf2.config import atomic_json
from astra_play_sf2.runner import sha256


def lua_literal(value):
    # bool subclasses int in Python; Lua's literals are lowercase keywords.
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, dict):
        return '{' + ','.join('[' + lua_literal(k) + ']=' + lua_literal(v) for k, v in value.items()) + '}'
    if isinstance(value, (list, tuple)):
        return '{' + ','.join(lua_literal(v) for v in value) + '}'
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (int, float)) and math.isfinite(value):
        return repr(value)
    raise ValueError('Unsupported Lua export value')


def export_policy(model_path):
    import torch
    from stable_baselines3 import PPO
    model_path = Path(model_path).resolve()
    model = PPO.load(model_path, device='cpu')
    policy = model.policy
    if policy.observation_space.shape != (344,) or getattr(policy.action_space, 'n', None) != 15:
        raise ValueError('Expected 344 observations and 15 discrete actions')
    if not isinstance(policy.features_extractor, __import__('stable_baselines3').common.torch_layers.FlattenExtractor):
        raise ValueError('Only an unnormalized flat observation policy is supported')
    modules = list(policy.mlp_extractor.policy_net.children())
    if len(modules) % 2 or not modules:
        raise ValueError('Expected Linear/Tanh policy layers')
    layers = []
    for index in range(0, len(modules), 2):
        linear, activation = modules[index:index+2]
        if not isinstance(linear, torch.nn.Linear) or not isinstance(activation, torch.nn.Tanh):
            raise ValueError('Only Linear/Tanh policy layers are supported')
        layers.append({'weight': linear.weight.detach().cpu().tolist(),
                       'bias': linear.bias.detach().cpu().tolist()})
    if not isinstance(policy.action_net, torch.nn.Linear):
        raise ValueError('Expected a linear categorical action head')
    layers.append({'weight': policy.action_net.weight.detach().cpu().tolist(),
                   'bias': policy.action_net.bias.detach().cpu().tolist()})
    payload = {'schema': 'astra.rl-policy.v1', 'model_sha256': sha256(model_path),
               'observations': 344, 'actions': 15, 'history': 4, 'decision_frames': 12,
               'activation': 'tanh', 'selection': 'deterministic_argmax', 'layers': layers}
    if not all(math.isfinite(x) for layer in layers for row in layer['weight'] for x in row):
        raise ValueError('Nonfinite model weights')
    if not all(math.isfinite(x) for layer in layers for x in layer['bias']):
        raise ValueError('Nonfinite model biases')
    return payload


def write_export(model_path, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    payload = export_policy(model_path)
    atomic_json(output / 'policy.json', payload)
    (output / 'policy.lua').write_text('return ' + lua_literal(payload) + '\n', encoding='utf-8')
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    payload = write_export(args.model, args.output)
    print(json.dumps({'model_sha256': payload['model_sha256'], 'output': str(args.output)}))


if __name__ == '__main__':
    main()
