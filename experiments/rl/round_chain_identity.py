"""Validate an explicitly derived chain build and its preserved actions16 parent."""
import json
from pathlib import Path
from astra_play_sf2.runner import sha256

INTERFACE='ken_actions16_lp_mp_uppercut_v1'


def validate_chain_build(manifest,package):
    package=Path(package)
    if (manifest.get('schema')!='astra.rl-round-chain-build.v1'
            or manifest.get('training_protocol')!='native_match_round_episodes_v1'
            or manifest.get('action_interface')!=INTERFACE or manifest.get('actions')!=16
            or manifest.get('observations')!=344 or manifest.get('native_decision_frames')!=12):
        raise RuntimeError('Invalid round-chain actions16 build identity')
    parent_path=package.parent/'parent-build.json'
    if sha256(parent_path)!=manifest.get('parent_build_sha256'):
        raise RuntimeError('Round-chain parent build identity changed')
    parent=json.loads(parent_path.read_text())
    if parent.get('schema')!='astra.rl-actions16-build.v1' or parent.get('action_interface')!=INTERFACE or parent.get('actions')!=16:
        raise RuntimeError('Round-chain parent is not the declared actions16 build')
    sources=manifest['source_sha256']
    if any(parent['derived_sha256'].get(name)!=digest for name,digest in sources.items()):
        raise RuntimeError('Round-chain sources differ from their parent build')
    for name in ('actions.lua','nn.lua'):
        if manifest['derived_sha256'].get(name)!=sources.get(name):
            raise RuntimeError('Round-chain changed the inherited action/observation interface')
    if manifest['derived_sha256'].get('batch_runtime.lua')!=manifest.get('runtime_sha256'):
        raise RuntimeError('Round-chain sampler identity mismatch')
    if 'chain_identity.py' not in manifest['derived_sha256']:
        raise RuntimeError('Missing round-chain identity validator')
