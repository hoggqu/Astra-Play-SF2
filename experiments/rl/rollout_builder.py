"""Build an isolated uniform round-chain PPO with configurable 256/1024 rollout."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from astra_play_sf2.config import atomic_json
from astra_play_sf2.runner import sha256
from .round_chain_identity import validate_chain_build

HERE = Path(__file__).resolve().parent
PACKAGE = 'astra_sf2_rl_long_rollout'


def metrics_helper(source):
    tree = ast.parse(source)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'ppo_update_metrics']
    if len(functions) != 1:
        raise ValueError('Expected one canonical read-only PPO metrics function')
    return ('import math\nfrom numbers import Integral, Real\nimport numpy as np\n\n'
            + ast.get_source_segment(source, functions[0])+'\n')


def build(output, source, default_rollout_steps=256):
    if default_rollout_steps not in (256, 1024):
        raise ValueError('Default rollout must be 256 or 1024')
    source = Path(source).resolve()
    parent_bytes = (source.parent/'build.json').read_bytes()
    parent = json.loads(parent_bytes)
    validate_chain_build(parent, source)
    if parent.get('opponent_sampling_identity') or parent.get('rollout_identity'):
        raise ValueError('Derive from a frozen uniform round-chain package')
    captured = {}
    for name, digest in parent['derived_sha256'].items():
        if Path(name).name != name:
            raise ValueError('Unsafe parent filename')
        captured[name] = (source/name).read_bytes()
        if hashlib.sha256(captured[name]).hexdigest() != digest:
            raise ValueError('Parent package differs from build: '+name)
    ancestors = {p.name: p.read_bytes() for p in source.parent.glob('*.json') if p.name != 'build.json'}
    production_root = source.parent/'src'
    production = {p.relative_to(production_root).as_posix(): p.read_bytes()
                  for p in production_root.rglob('*') if p.is_file() and p.suffix in ('.py', '.lua', '.json')}
    if 'astra_play_sf2/__init__.py' not in production:
        raise ValueError('Parent must contain frozen production src')
    metrics_source = (HERE/'batch_train.py').read_bytes()
    identity = (HERE/'rollout_identity.py').read_bytes()
    derived = {name: body.decode('utf-8') for name, body in captured.items()}
    for name in derived:
        if name.endswith('.py'):
            derived[name] = derived[name].replace(parent['package'], PACKAGE)
    patches = []
    def patch(name, old, new):
        if derived[name].count(old) != 1:
            raise ValueError('Missing/ambiguous rollout anchor in '+name+': '+old)
        derived[name] = derived[name].replace(old, new)
        patches.append({'file': name, 'old': old, 'new': new})
    name = 'batch_train.py'
    embedded_metrics = 'def ppo_update_metrics(' in derived[name]
    if embedded_metrics and metrics_helper(derived[name]) != metrics_helper(metrics_source.decode('utf-8')):
        raise ValueError('Parent contains a different PPO telemetry implementation')
    patch(name, 'from .vector import ManagedVec',
          'from .vector import ManagedVec\nfrom .rollout_identity import training_metadata'
          + ('' if embedded_metrics else '\nfrom .ppo_metrics import ppo_update_metrics'))
    patch(name, "    parser.add_argument('--block',", f"    parser.add_argument('--rollout-steps', type=int, choices=(256,1024), default={default_rollout_steps}, help='On-policy decisions per worker per PPO update')\n    parser.add_argument('--block',")
    patch(name, 'args.steps % (args.workers*256)', 'args.steps % (args.workers*args.rollout_steps)')
    patch(name, "'Steps must be workers*256 multiples, or 1..256 for parity'", "'Steps must be workers*rollout-steps multiples, or 1..256 for parity'")
    patch(name, 'checkpoint_interval(args.checkpoint_every, args.workers*256)',
          'checkpoint_interval(args.checkpoint_every, args.workers*args.rollout_steps)')
    patch(name, "'status': 'initializing', 'workers': args.workers, 'block': args.block,",
          "'status': 'initializing', 'workers': args.workers, 'block': args.block,\n              'rollout_configuration': training_metadata(Path(__file__).resolve().parent, args.rollout_steps, args.init_model),")
    patch(name, "seed=args.seed, device='cpu', n_steps=256", "seed=args.seed, device='cpu', n_steps=args.rollout_steps")
    patch(name, "PPO.load(args.init_model, env=vector, device='cpu', seed=args.seed)",
          "PPO.load(args.init_model, env=vector, device='cpu', seed=args.seed, n_steps=args.rollout_steps)")
    patch(name, "if model.n_steps != 256:\n                    raise ValueError('Initial model requires n_steps=256')",
          "if model.n_steps != args.rollout_steps or model.rollout_buffer.buffer_size != args.rollout_steps:\n                    raise ValueError('Loaded model rollout buffer differs from requested length')")
    patch(name, 'range(args.workers*256, args.steps+1, args.workers*256)',
          'range(args.workers*args.rollout_steps, args.steps+1, args.workers*args.rollout_steps)')
    patch(name, 'range(0, 256, args.block)', 'range(0, args.rollout_steps, args.block)')
    if not embedded_metrics:
        patch(name, "                if not args.benchmark:\n                    model.train()",
              "                update_metrics = None\n                if not args.benchmark:\n                    model.train()\n                    update_metrics = ppo_update_metrics(model)")
        patch(name, "                result['iterations'].append(row)",
              "                if update_metrics is not None:\n                    row['ppo'] = update_metrics\n                result['iterations'].append(row)")
    patch('support.py', 'from .chain_identity import validate_chain_build',
          'from .rollout_identity import validate_rollout_build as validate_chain_build')
    patch('native_campaign.py', "'chain_identity.py', 'native_campaign.py'",
          "'chain_identity.py', 'rollout_identity.py', 'ppo_metrics.py', 'native_campaign.py'")
    patch('native_campaign.py', "    paths.update({'astra_play_sf2/'",
          "    paths.update({name: here.parent/name for name in ('build.json', 'parent-build.json', 'rollout-parent-build.json')})\n    paths.update({'astra_play_sf2/'")
    derived['ppo_metrics.py'] = metrics_helper(metrics_source.decode('utf-8'))
    derived['rollout_identity.py'] = identity.decode('utf-8')
    for name, body in derived.items():
        if name.endswith('.py'):
            compile(body, PACKAGE+'/'+name, 'exec')
        elif name.endswith('.lua') and body.encode('utf-8') != captured[name]:
            raise RuntimeError('Long rollout must not change any native Lua source')
    if (source.parent/'build.json').read_bytes() != parent_bytes or any((source/n).read_bytes() != b for n,b in captured.items()):
        raise RuntimeError('Parent changed while building')
    if any((source.parent/n).read_bytes() != b for n,b in ancestors.items()):
        raise RuntimeError('Parent manifest/revision changed while building')
    if any((production_root/n).read_bytes() != b for n,b in production.items()):
        raise RuntimeError('Frozen production changed while building')
    if (HERE/'batch_train.py').read_bytes() != metrics_source or (HERE/'rollout_identity.py').read_bytes() != identity:
        raise RuntimeError('Read-only telemetry or identity helper changed while building')
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    package = output/PACKAGE; package.mkdir()
    for name, body in derived.items():
        (package/name).write_text(body, encoding='utf-8')
    for name, body in ancestors.items():
        (output/name).write_bytes(body)
    for name, body in production.items():
        path = output/'src'/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(body)
    (output/'rollout-parent-build.json').write_bytes(parent_bytes)
    manifest = dict(parent)
    manifest.update(package=PACKAGE, status='candidate', native_validated=False,
        rollout_identity='ppo_configurable_rollout_v1', default_rollout_steps=default_rollout_steps,
        rollout_parent_build_sha256=hashlib.sha256(parent_bytes).hexdigest(),
        rollout_source_sha256=parent['derived_sha256'], rollout_patches=patches,
        rollout_builder_sha256=sha256(Path(__file__)), telemetry_source_sha256=hashlib.sha256(metrics_source).hexdigest(),
        frozen_production_sha256={n: hashlib.sha256(b).hexdigest() for n,b in production.items()},
        derived_sha256={n: sha256(package/n) for n in derived})
    atomic_json(output/'build.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--default-rollout-steps', type=int, choices=(256,1024), default=256)
    args = parser.parse_args()
    print(json.dumps(build(args.output, args.source, args.default_rollout_steps), indent=2))


if __name__ == '__main__':
    main()
