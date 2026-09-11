"""Audit the fixed training sampler and both preserved chain/actions16 ancestors."""
import json
from pathlib import Path
from astra_play_sf2.runner import sha256
from .chain_identity import validate_chain_build
from .fixed_sampling import IDENTITY, validate_config
from .sampling_config import CONFIG


def validate_weighted_build(manifest, package):
    package = Path(package)
    validate_chain_build(manifest, package)
    parent_path = package.parent/'parent-chain-build.json'
    if sha256(parent_path) != manifest.get('parent_chain_build_sha256'):
        raise RuntimeError('Weighted sampler chain-parent identity changed')
    parent = json.loads(parent_path.read_text())
    validate_chain_build(parent, package)
    if manifest.get('weighted_source_sha256') != parent.get('derived_sha256'):
        raise RuntimeError('Weighted source package differs from declared chain parent')
    if manifest.get('opponent_sampling_identity') != IDENTITY or manifest.get('opponent_sampling') != validate_config(CONFIG):
        raise RuntimeError('Weighted sampler configuration/identity mismatch')
    expected = manifest['derived_sha256']
    for name in ('fixed_sampling.py', 'sampling_config.py', 'weighted_identity.py'):
        if name not in expected:
            raise RuntimeError('Missing fixed sampler dependency: '+name)
    for name, digest in expected.items():
        if Path(name).name != name or sha256(package/name) != digest:
            raise RuntimeError('Weighted package changed: '+name)
    for name, digest in parent['derived_sha256'].items():
        if name.endswith('.lua') and expected.get(name) != digest:
            raise RuntimeError('Fixed sampling must preserve all game/runtime Lua: '+name)


def sampling_metadata(package):
    package = Path(package)
    manifest_path = package.parent/'build.json'
    manifest = json.loads(manifest_path.read_text())
    validate_weighted_build(manifest, package)
    return {'identity': IDENTITY, 'configuration': CONFIG,
            'configuration_file_sha256': manifest['derived_sha256']['sampling_config.py'],
            'input_weights_sha256': manifest['input_weights_sha256'],
            'build_manifest_sha256': sha256(manifest_path),
            'parent_chain_build_sha256': manifest['parent_chain_build_sha256']}
