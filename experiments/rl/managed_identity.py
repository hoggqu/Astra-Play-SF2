"""Standalone training lifecycle/device revision; immutable timing parent."""
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from .perception128_identity import ARCHITECTURE, INTERFACE, OBSERVATION_INTERFACE, digest, validate_architecture

PACKAGE = 'astra_sf2_rl_managed'
SCHEMA = 'astra.rl-managed-perception128.v10'
PROTOCOL = 'native-pip-ko-late-ko-time-and-confirmed-draw-v12'
TIMING_PROTOCOL = 'restore-held-input-before-rpc-resume-v1'
STOP_PROTOCOL = 'ppo-update-stop-file-v1'
SAMPLING_PROTOCOL = 'adaptive_reset_probability_multipliers_v2'
INPUT_NAMES = ('managed_builder.py', 'managed_identity.py', 'managed_runtime.py', 'adaptive_sampling.py', 'weighted_sampling.py', 'rollout_control.py')


def validate_parent(root):
    root = Path(root).resolve()
    manifest = json.loads((root/'build.json').read_text())
    package = root/manifest['package']
    name = '_managed_parent_'+digest(str(root).encode())[:16]
    spec = importlib.util.spec_from_file_location(name, package/'__init__.py', submodule_search_locations=[str(package)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    importlib.import_module(name+'.perception128_draw_identity').validate_build(manifest, package)
    return manifest, {n:(package/n).read_bytes() for n in manifest['derived_sha256']}


def replace_once(body, old, new):
    if body.count(old) != 1:
        raise RuntimeError('Unexpected managed trainer anchor: '+old[:100].decode())
    return body.replace(old, new)


def derive(captured, inputs):
    out = dict(captured)
    out['managed_identity.py'] = inputs['managed_identity.py']
    out['managed_runtime.py'] = inputs['managed_runtime.py']
    out['adaptive_sampling.py'] = inputs['adaptive_sampling.py']
    out['weighted_sampling.py'] = inputs['weighted_sampling.py']
    out['rollout_control.py'] = inputs['rollout_control.py']
    for name in ('initialize.py', 'support.py', 'batch_train.py'):
        out[name] = out[name].replace(b'from .perception128_draw_identity import', b'from .managed_identity import')
    body = out['batch_train.py']
    changes = [
        (b"type=int, choices=range(1, 17), default=8", b"type=positive_workers, default=8"),
        (b'from .batch_env import BatchEnv, BatchBridge, ATTACH', b'from .batch_env import BatchEnv, BatchBridge, ATTACH\nfrom .managed_runtime import stop_request, select_device, positive_workers, PROTOCOL as STOP_PROTOCOL'),
        (b"    parser.add_argument('--dataset', type=Path, required=True)", b"    parser.add_argument('--device', choices=('cpu','cuda','mps','auto'), default=os.environ.get('ASTRA_RL_DEVICE','cpu'))\n    parser.add_argument('--dataset', type=Path, required=True)"),
        (b'    is_parity = args.parity or args.native_parity', b'    device = select_device(args.device)\n    is_parity = args.parity or args.native_parity'),
        (b"    atomic_json(output/'result.json', result)\n    vector = None;started = time.monotonic()", b"    result.update(device_requested=args.device, device_resolved=device, cooperative_stop_protocol=STOP_PROTOCOL)\n    atomic_json(output/'result.json', result)\n    vector = None;started = time.monotonic()"),
        (b"    def stop(_signal, _frame):\n        raise KeyboardInterrupt('SIGTERM')\n    signal.signal(signal.SIGTERM, stop)", b"    signal_stop = {}\n    def stop(_signal, _frame):\n        signal_stop['reason'] = 'user_cancelled'\n    signal.signal(signal.SIGTERM, stop)\n    signal.signal(signal.SIGINT, stop)"),
        (b"seed=args.seed, device='cpu', n_steps=256", b"seed=args.seed, device=device, n_steps=256"),
        (b"PPO.load(args.init_model, env=vector, device='cpu', seed=args.seed)", b"PPO.load(args.init_model, env=vector, device=device, seed=args.seed)"),
        (b'torch.as_tensor(obs.reshape(-1, 4516)), torch.as_tensor(actions.reshape(-1))', b'torch.as_tensor(obs.reshape(-1, 4516), device=model.device), torch.as_tensor(actions.reshape(-1), device=model.device)'),
        (b'    actual_logp = logp.reshape(steps, workers).cpu().numpy()', b'    values = values.detach().cpu()\n    logp = logp.detach().cpu()\n    actual_logp = logp.reshape(steps, workers).numpy()'),
        (b'model.policy.predict_values(torch.as_tensor(final_obs))', b'model.policy.predict_values(torch.as_tensor(final_obs, device=model.device))'),
        (b"                print(json.dumps(row), flush=True)\n            if not args.benchmark:", b"                print(json.dumps(row), flush=True)\n                reason = signal_stop.get('reason') or stop_request()\n                if reason:\n                    result.update(budget_stop=True, stop_reason=reason)\n                    break\n            if not args.benchmark:"),
    ]
    for old, new in changes:
        body = replace_once(body, old, new)
    sampling_changes = [
        (b'from .dataset import load_dataset', b'from .dataset import load_dataset\nfrom .adaptive_sampling import OpponentSampler'),
        (b"    parser.add_argument('--benchmark',", b"    parser.add_argument('--opponent-sampling', choices=('adaptive','uniform'), default=os.environ.get('ASTRA_RL_OPPONENT_SAMPLING','adaptive'))\n    parser.add_argument('--benchmark',"),
        (b"            model._setup_learn(args.steps, reset_num_timesteps=True)", b"            sampler = OpponentSampler(sorted({r['opponent'] for r in groups['train']}), args.opponent_sampling, getattr(model, 'astra_opponent_sampler', None))\n            vector.env_method('set_opponent_probabilities', sampler.probabilities)\n            result['opponent_sampling_initial'] = sampler.snapshot()\n            model._setup_learn(args.steps, reset_num_timesteps=True)"),
        (b"gathered = [{'transitions': []} for _ in range(args.workers)]", b"gathered = [{'transitions': [], 'episodes': []} for _ in range(args.workers)]"),
        (b"gathered[w]['transitions'].extend(chunk['transitions'])", b"gathered[w]['transitions'].extend(chunk['transitions'])\n                        gathered[w]['episodes'].extend(chunk['episodes'])"),
        (b"                    if target % effective_checkpoint_interval == 0:", b"                    sampler.observe(gathered)\n                    model.astra_opponent_sampler = sampler.state()\n                    vector.env_method('set_opponent_probabilities', sampler.probabilities)\n                    result['opponent_sampling'] = sampler.snapshot()\n                    if target % effective_checkpoint_interval == 0:"),
        (b"                row['chain_metrics'] =", b"                row['opponent_sampling'] = sampler.snapshot()\n                row['chain_metrics'] ="),
    ]
    for old, new in sampling_changes:
        body = replace_once(body, old, new)
    from_anchor = b'from .adaptive_sampling import OpponentSampler'
    body = replace_once(body, from_anchor, from_anchor+b'\nfrom .rollout_control import configure, collect, validate_sizes, PROTOCOL as ROLLOUT_PROTOCOL')
    begin=body.index(b'def fill_buffer(model, chunks):')
    end=body.index(b'class ParityBridge',begin)
    body=body[:begin]+b'from .rollout_control import fill_buffer\n\n\n'+body[end:]
    body=replace_once(body,b"    parser.add_argument('--steps', type=int, default=4096)",b"    parser.add_argument('--steps', type=int, default=4096)\n    parser.add_argument('--rollout-steps', type=int, default=4096)\n    parser.add_argument('--minibatch-size', type=int, default=64)")
    body=replace_once(body,b'    device = select_device(args.device)',b'    validate_sizes(args.rollout_steps, args.minibatch_size, args.workers)\n    device = select_device(args.device)')
    body=body.replace(b'args.workers*256',b'args.rollout_steps')
    body=body.replace(b'Steps must be workers*256 multiples',b'Steps must be rollout-steps multiples')
    body=replace_once(body,b'n_steps=256, batch_size=64, n_epochs=4',b'n_steps=2, batch_size=2, n_epochs=4')
    body=replace_once(body,b"PPO.load(args.init_model, env=vector, device=device, seed=args.seed)",b"PPO.load(args.init_model, env=vector, device=device, seed=args.seed, custom_objects={'n_steps':2,'batch_size':2})")
    body=replace_once(body,b"                if model.n_steps != 256:\n                    raise ValueError('Initial model requires n_steps=256')",b'            configure(model, args.rollout_steps, args.minibatch_size)')
    start=body.index(b"                gathered = [{'transitions': [], 'episodes': []}")
    end=body.index(b'                sampling +=',start)
    body=body[:start]+b"                sample_start = time.monotonic()\n                gathered = collect(vector, args.rollout_steps, args.block, payload, target//args.rollout_steps-1)\n"+body[end:]
    body=replace_once(body,b"    result.update(device_requested=args.device",b"    result.update(rollout_steps=args.rollout_steps, minibatch_size=args.minibatch_size, rollout_protocol=ROLLOUT_PROTOCOL)\n    result.update(device_requested=args.device")
    body=replace_once(body,b"                row['chain_metrics'] =",b"                row['worker_decisions'] = [len(c['transitions']) for c in gathered]\n                row['chain_metrics'] =")
    body=replace_once(body,b'from .managed_runtime import stop_request,',b'from .managed_runtime import positive_learning_rate, configure_learning_rate, stop_request,')
    body=replace_once(body,b"    parser.add_argument('--device',",b"    parser.add_argument('--learning-rate', type=positive_learning_rate, default=os.environ.get('ASTRA_RL_LEARNING_RATE') or None)\n    parser.add_argument('--device',")
    body=replace_once(body,b'            configure(model, args.rollout_steps, args.minibatch_size)',b"            configure(model, args.rollout_steps, args.minibatch_size)\n            result['learning_rate_override'] = configure_learning_rate(model, args.learning_rate)")
    body=replace_once(body,b'from .adaptive_sampling import OpponentSampler',b'from .weighted_sampling import OpponentSampler')
    out['batch_train.py'] = body
    env = captured['batch_env.py']
    env = replace_once(env, b'    def reset_choice(self):', b"    def set_opponent_probabilities(self, values):\n        from .adaptive_sampling import probability_map\n        self.opponent_probabilities = probability_map(sorted(self.checkpoint_groups), values)\n\n    def reset_choice(self):")
    env = replace_once(env, b'opponent = int(self.np_random.choice(opponents))', b"weights = getattr(self, 'opponent_probabilities', None)\n        opponent = int(self.np_random.choice(opponents, p=[weights[str(o)] for o in opponents] if weights else None))")
    out['batch_env.py'] = env
    return out


def validate_build(manifest, package):
    package = Path(package).resolve(); root = package.parent
    expected = {'schema':SCHEMA, 'package':PACKAGE, 'architecture':ARCHITECTURE,
                'action_interface':INTERFACE, 'observation_interface':OBSERVATION_INTERFACE,
                'observations':4516, 'actions':85, 'decision_frames':12,
                'settlement_protocol':PROTOCOL, 'input_timing_protocol':TIMING_PROTOCOL,
                'cooperative_stop_protocol':STOP_PROTOCOL, 'devices_supported':['cpu','cuda','mps','auto'],
                'opponent_sampling_protocol':SAMPLING_PROTOCOL,
                'rollout_protocol':'exact_global_rollout_per_worker_gae_v1'}
    if any(manifest.get(k) != v for k,v in expected.items()):
        raise RuntimeError('Wrong managed training identity')
    for name, value in manifest['frozen_files_sha256'].items():
        path = (root/name).resolve()
        if not path.is_relative_to(root) or digest(path.read_bytes()) != value:
            raise RuntimeError('Frozen managed source changed: '+name)
    _, captured = validate_parent(root/'managed-parent')
    if manifest['parent_build_sha256'] != digest((root/'managed-parent/build.json').read_bytes()):
        raise RuntimeError('Managed parent changed')
    inputs = {n:(root/'managed-inputs'/n).read_bytes() for n in INPUT_NAMES}
    derived = derive(captured, inputs)
    if set(derived) != set(manifest['derived_sha256']):
        raise RuntimeError('Managed file set changed')
    if {p.name for p in package.iterdir() if p.suffix in ('.py','.lua','.json')} != set(derived):
        raise RuntimeError('Unexpected managed executing file')
    for name, body in derived.items():
        if (package/name).read_bytes() != body or digest(body) != manifest['derived_sha256'][name]:
            raise RuntimeError('Managed derivation changed: '+name)
    return manifest


def training_metadata(package):
    package = Path(package)
    manifest = json.loads((package.parent/'build.json').read_text())
    validate_build(manifest, package)
    return {'perception_build_sha256':digest((package.parent/'build.json').read_bytes()),
            'parent_build_sha256':manifest['parent_build_sha256'], 'settlement_protocol':PROTOCOL,
            'input_timing_protocol':TIMING_PROTOCOL, 'architecture':ARCHITECTURE,
            'net_arch':manifest['net_arch'], 'inference':manifest['inference'], 'worker_limit':None,
            'action_interface':INTERFACE, 'observation_interface':OBSERVATION_INTERFACE,
            'actions':85, 'observations':4516, 'decision_frames':12,
            'cooperative_stop_protocol':STOP_PROTOCOL, 'opponent_sampling_protocol':SAMPLING_PROTOCOL,
            'rollout_protocol':'exact_global_rollout_per_worker_gae_v1',
            'combat_policy':'Shared PPO128; unchanged actions, observations, reward and input timing; explicit CPU/CUDA and stop at completed update'}
