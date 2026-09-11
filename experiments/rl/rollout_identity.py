"""Identity and metadata for an isolated long-rollout training derivative."""
import json
from pathlib import Path
import zipfile
from astra_play_sf2.runner import sha256
from .chain_identity import validate_chain_build

IDENTITY = 'ppo_configurable_rollout_v1'


def validate_rollout_build(manifest, package):
    package = Path(package)
    validate_chain_build(manifest, package)
    parent_path = package.parent/'rollout-parent-build.json'
    if (manifest.get('rollout_identity') != IDENTITY
            or sha256(parent_path) != manifest.get('rollout_parent_build_sha256')):
        raise RuntimeError('Long-rollout parent identity differs')
    parent = json.loads(parent_path.read_text(encoding='utf-8'))
    validate_chain_build(parent, package)
    if parent['derived_sha256'] != manifest.get('rollout_source_sha256'):
        raise RuntimeError('Long-rollout input lineage differs')
    for name, digest in parent['derived_sha256'].items():
        if name.endswith('.lua') and manifest['derived_sha256'].get(name) != digest:
            raise RuntimeError('Long-rollout experiment must preserve every parent Lua file')
    for name, digest in manifest['derived_sha256'].items():
        if Path(name).name != name or sha256(package/name) != digest:
            raise RuntimeError('Long-rollout package differs from build: '+name)


def training_metadata(package, rollout_steps, init_model=None):
    if rollout_steps not in (256, 1024):
        raise ValueError('Supported rollout steps are 256 and 1024')
    package = Path(package)
    manifest = json.loads((package.parent/'build.json').read_text(encoding='utf-8'))
    validate_rollout_build(manifest, package)
    initial = None
    if init_model is not None:
        with zipfile.ZipFile(init_model) as archive:
            initial = json.loads(archive.read('data'))['n_steps']
    return {'identity': IDENTITY, 'rollout_steps_per_worker': rollout_steps,
            'initial_model_rollout_steps': initial,
            'optimizer_reinitialized': False if init_model is not None else None,
            'continuation': 'PPO.load overrides n_steps before constructing the rollout buffer; policy and optimizer ZIP states are then restored',
            'build_manifest_sha256': sha256(package.parent/'build.json'),
            'parent_build_sha256': manifest['rollout_parent_build_sha256'],
            'telemetry_helper_sha256': manifest['derived_sha256']['ppo_metrics.py']}
