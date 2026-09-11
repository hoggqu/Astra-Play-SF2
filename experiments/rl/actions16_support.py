"""Pure interface guards and action-covering diagnostics for the isolated candidate."""
import numpy as np
import json
from pathlib import Path
from unittest.mock import patch

from astra_play_sf2.config import atomic_json
from astra_play_sf2.runner import sha256, seal_run

INTERFACE = 'ken_actions16_lp_mp_uppercut_v1'


def validate_model(model):
    if (getattr(model, 'astra_action_interface', None) != INTERFACE
            or model.observation_space.shape != (344,)
            or getattr(model.action_space, 'n', None) != 16):
        raise ValueError('Expected an explicitly identified 344-observation actions16 model')
    return model


def validate_model_file(path):
    from stable_baselines3 import PPO
    return validate_model(PPO.load(path, device='cpu'))


def parity_actions(count, seed):
    if not 16 <= count <= 256:
        raise ValueError('Actions16 parity needs 16..256 decisions to cover every action')
    rng = np.random.default_rng(seed)
    return rng.permutation(16).tolist() + rng.integers(0, 16, count-16).tolist()


def optimizer_provenance(model, loaded):
    validate_model(model)
    return {'loaded_from_init_model': bool(loaded),
            'origin': getattr(model, 'astra_optimizer_origin', 'fresh_actions16_training'),
            'initial_updates': model._n_updates,
            'initial_state_entries': len(model.policy.optimizer.state),
            'migration': getattr(model, 'astra_migration', None)}


def capture_interface(model_path, package):
    """Validate the generated snapshot and model before starting an emulator."""
    from .export import export_policy, lua_literal
    package = Path(package).resolve()
    manifest_path = package.parent/'build.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if (manifest.get('schema') != 'astra.rl-actions16-build.v1'
            or manifest.get('action_interface') != INTERFACE or manifest.get('actions') != 16):
        raise RuntimeError('Invalid actions16 build identity')
    expected = manifest['derived_sha256']
    for name in ('support.py','actions.lua','nn.lua','native_continuous.py','export.py'):
        if name not in expected: raise RuntimeError('Missing generated dependency: '+name)
    for name, digest in expected.items():
        if Path(name).name != name or sha256(package/name) != digest:
            raise RuntimeError('Generated package source differs from build: '+name)
    model_path = Path(model_path).resolve()
    payload = export_policy(model_path)
    for key, value in {'schema':'astra.rl-policy.actions16.v1','action_interface':INTERFACE,
                       'actions':16,'observations':344,'decision_frames':12,
                       'selection':'deterministic_argmax'}.items():
        if payload.get(key) != value: raise RuntimeError('Wrong exported interface: '+key)
    return {'package':package,'manifest':manifest_path,'manifest_sha256':sha256(manifest_path),
            'package_sha256':dict(expected),'model':model_path,'model_sha256':payload['model_sha256'],
            'payload_text':'return '+lua_literal(payload)+'\n','payload':payload,'staged':None}


def check_staged_interface(snapshot, run, hashes):
    """Bind exact exported model bytes and derived Lua files to the staged runtime."""
    runtime = Path(run)/'training/runtime'
    if (runtime/'rl_policy.lua').read_bytes() != snapshot['payload_text'].encode('utf-8'):
        raise RuntimeError('Staged model payload differs from validated export')
    for target, source in (('rl_actions.lua','actions.lua'),('rl_nn.lua','nn.lua'),
                           ('rl_continuous_core.lua','native_continuous_core.lua'),
                           ('rl_settlement.lua','settlement.lua')):
        if hashes.get(target) != snapshot['package_sha256'][source]:
            raise RuntimeError('Staged interface source mismatch: '+target)
    for name, digest in hashes.items():
        if Path(name).name != name or sha256(runtime/name) != digest:
            raise RuntimeError('Staged runtime hash mismatch: '+name)
    if snapshot['staged'] is not None and hashes != snapshot['staged']:
        raise RuntimeError('Staged runtime identity changed')
    snapshot['staged'] = dict(hashes)


def audit_interface(snapshot, run, result):
    """Return success only after checking immutable source/model and every match."""
    if sha256(snapshot['manifest']) != snapshot['manifest_sha256']:
        raise RuntimeError('Build manifest changed during verification')
    for name, digest in snapshot['package_sha256'].items():
        if sha256(snapshot['package']/name) != digest:
            raise RuntimeError('Package source changed during verification: '+name)
    if sha256(snapshot['model']) != snapshot['model_sha256']:
        raise RuntimeError('Model changed during verification')
    if not snapshot['staged'] or result.get('runtime_sha256') != snapshot['staged']:
        raise RuntimeError('Missing or mismatched staged runtime record')
    check_staged_interface(snapshot, run, result['runtime_sha256'])
    if (result.get('schema') != 'astra.rl-continuous.actions16.v1'
            or result.get('action_interface') != INTERFACE or result.get('actions') != 16
            or result.get('model_sha256') != snapshot['model_sha256']):
        raise RuntimeError('Run interface/model identity mismatch')
    matches = 0
    for attempt in result['attempts']:
        for relative in attempt['matches']:
            path = (Path(run)/relative).resolve()
            if not path.is_relative_to(Path(run).resolve()):
                raise RuntimeError('Match path escaped this run')
            summary = json.loads(path.read_text(encoding='utf-8'))['summary']
            if (summary.get('schema') != 'mame.rl-continuous-play.actions16.v1'
                    or summary.get('action_interface') != INTERFACE
                    or summary.get('model_sha256') != snapshot['model_sha256']
                    or summary.get('policy_kind') != 'ppo'):
                raise RuntimeError('Match interface/model identity mismatch')
            matches += 1
    if not matches: raise RuntimeError('No completed match identity evidence')
    return {'ok':True,'action_interface':INTERFACE,'actions':16,'observations':344,
            'decision_frames':12,'selection':'deterministic_argmax',
            'model_sha256':snapshot['model_sha256'],'build_manifest_sha256':snapshot['manifest_sha256'],
            'checked_package_files':len(snapshot['package_sha256']),
            'checked_staged_files':len(snapshot['staged']),'checked_matches':matches,
            'package_sha256':snapshot['package_sha256'],'staged_sha256':snapshot['staged']}


def interface_evaluate(evaluator, namespace, args, kwargs):
    """Wrap only the generated native evaluator; failures remain invalid, sealed evidence."""
    model = kwargs.get('model', args[1] if len(args)>1 else None)
    output = Path(kwargs.get('output', args[2] if len(args)>2 else '')).resolve()
    if output.exists(): raise FileExistsError('Never reuse verification output: '+str(output))
    result = {'schema':'astra.rl-continuous.actions16.v1','status':'invalid',
              'action_interface':INTERFACE,'actions':16,'attempts':[],
              'frozen_v4_certification':False,'training_during_run':False}
    result['action_interface_audit'] = {'ok':False,'reason':'Run did not complete'}
    try:
        snapshot = capture_interface(model, namespace['HERE'])
        original_stage = namespace['stage_native_policy']
        def checked_stage(run, level, payload):
            if payload != snapshot['payload']: raise RuntimeError('Model export changed before staging')
            hashes = original_stage(run, level, payload)
            check_staged_interface(snapshot, run, hashes)
            return hashes
        with patch.dict(namespace, {'stage_native_policy':checked_stage}):
            result = evaluator(*args, **kwargs)
        result['action_interface_audit'] = {'ok':False,'reason':'Run did not complete'}
        if result['status'] == 'complete':
            result['action_interface_audit'] = audit_interface(snapshot, output, result)
    except (Exception, KeyboardInterrupt) as error:
        result.update(status='invalid',error=f'Action interface audit: {type(error).__name__}: {error}')
        result['action_interface_audit'] = {'ok':False,'reason':str(error)}
    output.mkdir(parents=True,exist_ok=True)
    atomic_json(output/'result.json', result)
    seal_run(output)
    return result
