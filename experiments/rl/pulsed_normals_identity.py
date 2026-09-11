"""Strict derivation of pulsed normal buttons from an unchanged actions16 chain."""
import hashlib
import json
from pathlib import Path
from astra_play_sf2.runner import sha256
from .round_chain_identity import validate_chain_build

OLD = 'ken_actions16_lp_mp_uppercut_v1'
INTERFACE = 'ken_actions16_pulsed_normals_v2'
IDENTITY = 'normal_buttons_release_last_frame_v1'


def derive_sources(captured):
    derived = {}
    for name, body in captured.items():
        # Keep the old validator and old 15->16 migration as historical source.
        text = body.decode('utf-8')
        if name not in ('chain_identity.py', 'migrate.py'):
            text = text.replace(OLD, INTERFACE)
        derived[name] = text

    def patch(name, old, new):
        if derived[name].count(old) != 1:
            raise RuntimeError('Ambiguous pulse derivation anchor: '+name+' / '+old)
        derived[name] = derived[name].replace(old, new)

    patch('actions.lua', " if action<12 then return basic[action+1] end",
          " if action>=6 and action<=11 and frame==11 then return (action==8 or action==9) and 'D' or '' end\n"
          " if action<12 then return basic[action+1] end")
    patch('support.py', 'from .chain_identity import validate_chain_build',
          'from .pulsed_normals_identity import validate_pulsed_build as validate_chain_build')
    patch('support.py', "    if sha256(snapshot['manifest']) != snapshot['manifest_sha256']:",
          "    validate_chain_build(json.loads(snapshot['manifest'].read_text()), snapshot['package'])\n"
          "    if sha256(snapshot['manifest']) != snapshot['manifest_sha256']:")
    patch('batch_train.py', '    args = parser.parse_args()',
          "    args = parser.parse_args()\n    from .pulsed_normals_identity import training_metadata\n"
          "    pulse_metadata = training_metadata(Path(__file__).resolve().parent)")
    patch('batch_train.py', "'status': 'initializing', 'workers': args.workers, 'block': args.block,",
          "'status': 'initializing', 'workers': args.workers, 'block': args.block,\n"
          "              'action_semantics': pulse_metadata,")
    patch('native_campaign.py', "'chain_identity.py', 'native_campaign.py'",
          "'chain_identity.py', 'pulsed_normals_identity.py', 'native_campaign.py'")
    patch('native_campaign.py', '    return paths',
          '    from .pulsed_normals_identity import provenance_paths\n'
          '    paths.update(provenance_paths(here))\n    return paths')
    return {name: text.encode('utf-8') for name, text in derived.items()}


def validate_parent(manifest, package):
    package = Path(package)
    validate_chain_build(manifest, package)
    captured = {}
    for name, digest in manifest['derived_sha256'].items():
        if Path(name).name != name:
            raise RuntimeError('Unsafe parent filename')
        captured[name] = (package/name).read_bytes()
        if hashlib.sha256(captured[name]).hexdigest() != digest:
            raise RuntimeError('Frozen parent package changed: '+name)
    revision = manifest.get('settlement_revision')
    if revision:
        p = package.parent/revision['parent_manifest']
        if Path(revision['parent_manifest']).name != revision['parent_manifest'] or sha256(p) != revision['parent_build_sha256']:
            raise RuntimeError('Settlement revision parent changed')
        previous = json.loads(p.read_text())
        validate_chain_build(previous, package)
        expected = dict(previous['derived_sha256'])
        if expected.get('settlement.lua') != revision['old_sha256']:
            raise RuntimeError('Settlement revision old source differs')
        expected['settlement.lua'] = revision['new_sha256']
        if expected != manifest['derived_sha256']:
            raise RuntimeError('Settlement revision changed more than settlement')
    if any(manifest.get(k) for k in ('opponent_sampling_identity','rollout_identity','pulse_identity')):
        raise RuntimeError('Expected the unmodified uniform chain parent')
    return captured


def validate_pulsed_build(manifest, package):
    package = Path(package).resolve(); root = package.parent
    expected = {'schema':'astra.rl-pulsed-normals-build.v1', 'action_interface':INTERFACE,
                'pulse_identity':IDENTITY, 'actions':16, 'observations':344,
                'native_decision_frames':12, 'training_protocol':'native_match_round_episodes_v1'}
    if any(manifest.get(k) != v for k,v in expected.items()):
        raise RuntimeError('Invalid pulsed normals interface identity')
    parent_root = root/'parent-source'; parent_file = parent_root/'build.json'
    if sha256(parent_file) != manifest.get('pulse_parent_build_sha256'):
        raise RuntimeError('Pulse parent build changed')
    parent = json.loads(parent_file.read_text())
    if parent.get('package') != package.name or manifest.get('package') != package.name:
        raise RuntimeError('Pulse package namespace differs from its parent')
    for name, digest in manifest['parent_manifest_sha256'].items():
        if Path(name).name != name or sha256(parent_root/name) != digest:
            raise RuntimeError('Pulse ancestor manifest changed: '+name)
    captured = validate_parent(parent, parent_root/package.name)
    if manifest.get('pulse_source_sha256') != parent['derived_sha256']:
        raise RuntimeError('Pulse input source identity differs')
    derived = derive_sources(captured)
    if set(manifest['derived_sha256']) != set(derived)|{'pulsed_normals_identity.py'}:
        raise RuntimeError('Unexpected pulse executing dependency set')
    for name, body in derived.items():
        if (package/name).read_bytes() != body:
            raise RuntimeError('Pulse source is not the exact allowed derivation: '+name)
    for name,digest in manifest['derived_sha256'].items():
        if Path(name).name != name or sha256(package/name) != digest:
            raise RuntimeError('Pulse source hash mismatch: '+name)
    if sha256(package/'pulsed_normals_identity.py') != manifest['identity_sha256']:
        raise RuntimeError('Pulse validator identity differs')
    for name,digest in manifest['frozen_production_sha256'].items():
        p = (root/'src'/name).resolve()
        if not p.is_relative_to(root/'src') or sha256(p) != digest:
            raise RuntimeError('Pulse frozen production changed: '+name)
    if sha256(root/'launch.py') != manifest['launcher_sha256']:
        raise RuntimeError('Pulse launcher changed')
    return parent


def provenance_paths(package):
    package = Path(package).resolve();root = package.parent
    manifest = json.loads((root/'build.json').read_text())
    validate_pulsed_build(manifest, package)
    paths = {'pulse/build.json':root/'build.json','pulse/launch.py':root/'launch.py'}
    for name in manifest['parent_manifest_sha256']:
        paths['pulse/parent-source/'+name] = root/'parent-source'/name
    for name in manifest['pulse_source_sha256']:
        paths['pulse/parent-source/'+package.name+'/'+name] = root/'parent-source'/package.name/name
    for name in manifest['frozen_production_sha256']:
        paths['pulse/src/'+name] = root/'src'/name
    return paths


def training_metadata(package):
    package=Path(package);m=json.loads((package.parent/'build.json').read_text())
    validate_pulsed_build(m,package)
    return {'identity':IDENTITY,'action_interface':INTERFACE,'parent_interface':OLD,
            'changed_actions':[6,7,8,9,10,11],'release_frame_zero_based':11,
            'crouching_actions_keep_down':[8,9], 'decision_frames':12,
            'build_manifest_sha256':sha256(package.parent/'build.json'),
            'parent_build_sha256':m['pulse_parent_build_sha256'],
            'combat_policy':'shared PPO only; no V4 fallback'}
