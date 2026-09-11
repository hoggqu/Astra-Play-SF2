"""Derive a frozen round-chain package with user-supplied fixed opponent weights.

Only training reset opponent selection changes. Supply --weights JSON with
schema astra.rl-fixed-opponent-sampling.v1, probabilities keyed by all eleven
opponent IDs, formula metadata, and statistics_source metadata. No historical
weights or raw training records are built into this tool.
"""
import argparse
import hashlib
import json
from pathlib import Path
from astra_play_sf2.config import atomic_json
from astra_play_sf2.runner import sha256
from .fixed_sampling import IDENTITY, validate_config
from .round_chain_identity import validate_chain_build

HERE = Path(__file__).resolve().parent
PACKAGE = 'astra_sf2_rl_weighted_chain'


def build(output, source, weights):
    source, weights = Path(source).resolve(), Path(weights).resolve()
    parent_bytes = (source.parent/'build.json').read_bytes()
    parent = json.loads(parent_bytes)
    validate_chain_build(parent, source)
    if parent.get('opponent_sampling_identity'):
        raise ValueError('Derive from an unweighted frozen round-chain package')
    names = parent['derived_sha256']
    if any(Path(name).name != name for name in names):
        raise ValueError('Unsafe source filename')
    captured = {name: (source/name).read_bytes() for name in names}
    if any(hashlib.sha256(body).hexdigest() != names[name] for name, body in captured.items()):
        raise ValueError('Source differs from frozen chain manifest')
    ancestor_bytes = (source.parent/'parent-build.json').read_bytes()
    weights_bytes = weights.read_bytes()
    config = validate_config(json.loads(weights_bytes))
    json.dumps(config, allow_nan=False)  # Metadata must also be portable finite JSON.
    helpers = {name: (HERE/origin).read_bytes() for name, origin in
               [('fixed_sampling.py', 'fixed_sampling.py'), ('weighted_identity.py', 'weighted_sampling_identity.py')]}
    derived = {name: body.decode('utf-8') for name, body in captured.items()}
    old_package = parent['package']
    for name in derived:
        if name.endswith('.py'):
            derived[name] = derived[name].replace(old_package, PACKAGE)
    patches = []
    def patch(name, old, new):
        if derived[name].count(old) != 1:
            raise ValueError('Missing/ambiguous anchor in '+name+': '+old)
        derived[name] = derived[name].replace(old, new)
        patches.append({'file': name, 'old': old, 'new': new})
    patch('batch_env.py', '        opponents = sorted(self.checkpoint_groups)',
          '        from .fixed_sampling import probabilities\n        opponents = sorted(self.checkpoint_groups)')
    patch('batch_env.py', 'self.np_random.choice(opponents)', 'self.np_random.choice(opponents, p=probabilities(opponents))')
    patch('support.py', 'from .chain_identity import validate_chain_build',
          'from .weighted_identity import validate_weighted_build as validate_chain_build')
    patch('native_campaign.py', "'chain_identity.py', 'native_campaign.py'",
          "'chain_identity.py', 'fixed_sampling.py', 'sampling_config.py', 'weighted_identity.py', 'native_campaign.py'")
    patch('native_campaign.py', "    paths.update({'astra_play_sf2/'",
          "    paths.update({name: here.parent/name for name in ('build.json', 'parent-chain-build.json', 'parent-build.json')})\n    paths.update({'astra_play_sf2/'")
    for name in ('batch_train.py', 'native_campaign.py'):
        patch(name, 'from .dataset import load_dataset', 'from .dataset import load_dataset\nfrom .weighted_identity import sampling_metadata')
    patch('batch_train.py', "'sampling': 'frozen-policy categorical", "'opponent_sampling': sampling_metadata(Path(__file__).resolve().parent),\n              'sampling': 'frozen-policy categorical")
    patch('native_campaign.py', "'dataset_sha256': sha256(dataset)", "'opponent_sampling': sampling_metadata(Path(__file__).resolve().parent),\n              'dataset_sha256': sha256(dataset)")
    derived.update({name: body.decode('utf-8') for name, body in helpers.items()})
    derived['sampling_config.py'] = '# Frozen build-time training sampler; never used to choose gameplay actions.\nCONFIG = '+repr(config)+'\n'
    for name, body in derived.items():
        if name.endswith('.py'):
            compile(body, PACKAGE+'/'+name, 'exec')
        if name.endswith('.lua') and body.encode('utf-8') != captured[name]:
            raise RuntimeError('Unexpected gameplay/runtime change: '+name)
    if any((source/name).read_bytes() != body for name, body in captured.items()) or (source.parent/'build.json').read_bytes() != parent_bytes:
        raise RuntimeError('Chain source changed during build')
    if weights.read_bytes() != weights_bytes or (source.parent/'parent-build.json').read_bytes() != ancestor_bytes:
        raise RuntimeError('Configuration/ancestor changed during build')
    for name, origin in [('fixed_sampling.py', 'fixed_sampling.py'), ('weighted_identity.py', 'weighted_sampling_identity.py')]:
        if (HERE/origin).read_bytes() != helpers[name]:
            raise RuntimeError('Sampler helper changed during build')
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    package = output/PACKAGE; package.mkdir()
    for name, body in derived.items():
        (package/name).write_text(body, encoding='utf-8')
    (output/'parent-chain-build.json').write_bytes(parent_bytes)
    (output/'parent-build.json').write_bytes(ancestor_bytes)
    manifest = dict(parent)
    manifest.update(status='candidate', native_validated=False, package=PACKAGE,
                    opponent_sampling_identity=IDENTITY, opponent_sampling=config,
                    input_weights_sha256=hashlib.sha256(weights_bytes).hexdigest(),
                    parent_chain_build_sha256=hashlib.sha256(parent_bytes).hexdigest(),
                    weighted_source_sha256=dict(names), builder_sha256=sha256(Path(__file__)),
                    weighted_patches=patches, derived_sha256={name: sha256(package/name) for name in derived})
    atomic_json(output/'build.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.output, args.source, args.weights), indent=2))


if __name__ == '__main__':
    main()
