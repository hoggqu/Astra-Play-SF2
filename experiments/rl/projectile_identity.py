"""Fail-closed provenance guard for an independent 368-column chain candidate."""
import json
from pathlib import Path
from astra_play_sf2.runner import sha256
INTERFACE='sf2_projectiles6_owner_reciprocal_v1'
ACTION_INTERFACE='ken_actions16_lp_mp_uppercut_v1'


def validate_projectile_build(manifest,package):
    package=Path(package)
    required={'schema':'astra.rl-projectiles6-build.v1','observations':368,'actions':16,
              'action_interface':ACTION_INTERFACE,'observation_interface':INTERFACE,
              'native_decision_frames':12,'training_protocol':'native_match_round_episodes_v1'}
    if any(manifest.get(k)!=v for k,v in required.items()):raise RuntimeError('Invalid projectiles6 build identity')
    parent_path=package.parent/'parent-build.json'
    if sha256(parent_path)!=manifest.get('parent_build_sha256'):raise RuntimeError('Projectile parent manifest changed')
    parent=json.loads(parent_path.read_text())
    if parent.get('schema')!='astra.rl-round-chain-build.v1' or parent.get('observations')!=344 or parent.get('action_interface')!=ACTION_INTERFACE:
        raise RuntimeError('Expected explicit 344-column actions16 chain parent')
    if manifest['source_sha256']!=parent['derived_sha256']:raise RuntimeError('Projectile source differs from frozen chain parent')
    for name in ('actions.lua','native_continuous_core.lua','continuous_core.lua','settlement.lua'):
        if manifest['derived_sha256'].get(name)!=manifest['source_sha256'].get(name):
            raise RuntimeError('Observation candidate changed action/settlement/core: '+name)
    for name in ('projectile_features.lua','projectile_identity.py'):
        if name not in manifest['derived_sha256']:raise RuntimeError('Missing projectile interface dependency')
    for name,digest in manifest['derived_sha256'].items():
        if Path(name).name!=name or sha256(package/name)!=digest:raise RuntimeError('Projectile candidate source changed: '+name)
