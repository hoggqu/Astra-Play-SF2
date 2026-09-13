"""Exact, architecture-only derivation of structured perception PPO 128x128."""
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys

PACKAGE = 'astra_sf2_rl_perception128'
SCHEMA = 'astra.rl-screen-perception-width128.v1'
ARCHITECTURE = 'mlp_tanh_pi128x128_vf128x128_v1'
INTERFACE = 'ken_actions85_full_v1'
OBSERVATION_INTERFACE = 'sf2_screen_perception_v2'
INPUT_NAMES = ('perception128_builder.py', 'perception128_identity.py')


def digest(body):
    return hashlib.sha256(body).hexdigest()


def patch(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError('Width128 source anchor changed: ' + old[:100])
    return text.replace(old, new)


def validate_parent(root):
    root = Path(root).resolve()
    manifest = json.loads((root/'build.json').read_text())
    package = root/manifest['package']
    name = '_width_parent_' + digest(str(root).encode())[:16]
    spec = importlib.util.spec_from_file_location(name, package/'__init__.py', submodule_search_locations=[str(package)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    importlib.import_module(name+'.perception_identity').validate_build(manifest, package)
    return manifest, {n:(package/n).read_bytes() for n in manifest['derived_sha256']}


def validate_architecture(model):
    """Check actual policy AND critic layers, not just a user-editable label."""
    import torch
    policy = model.policy
    if getattr(model, 'astra_architecture', None) != ARCHITECTURE:
        raise ValueError('Expected explicitly identified perception128 architecture')
    for network in (policy.mlp_extractor.policy_net, policy.mlp_extractor.value_net):
        layers = list(network.children())
        if len(layers) != 4:
            raise ValueError('Expected two Linear/Tanh hidden layers')
        for i, dims in ((0, (4516,128)), (2, (128,128))):
            linear = layers[i]
            if (not isinstance(linear, torch.nn.Linear)
                    or (linear.in_features, linear.out_features) != dims
                    or not isinstance(layers[i+1], torch.nn.Tanh)):
                raise ValueError('Wrong perception128 hidden architecture')
    for head, size in ((policy.action_net, 85), (policy.value_net, 1)):
        if not isinstance(head, torch.nn.Linear) or (head.in_features, head.out_features) != (128,size):
            raise ValueError('Wrong perception128 output architecture')
    return model


def derive(captured, inputs):
    out = dict(captured)
    out['perception128_identity.py'] = inputs['perception128_identity.py']
    for name in ('initialize.py', 'support.py', 'batch_train.py'):
        out[name] = out[name].replace(b'from .perception_identity import', b'from .perception128_identity import')
    s = out['initialize.py'].decode()
    s = patch(s, "'pi':[64,64], 'vf':[64,64]", "'pi':[128,128], 'vf':[128,128]")
    s = patch(s, "    model.astra_action_interface = INTERFACE", "    model.astra_architecture = '"+ARCHITECTURE+"'\n    model.astra_action_interface = INTERFACE")
    s = s.replace('fresh_screen_perception_v2', 'fresh_screen_perception128_v1').replace('astra.rl-screen-perception-initialization.v2', 'astra.rl-screen-perception-width128-initialization.v1')
    out['initialize.py'] = s.encode()
    s = out['batch_train.py'].decode()
    s = patch(s, "choices=range(1, 9), default=4", "choices=range(1, 17), default=8")
    s = patch(s, "'pi': [64,64], 'vf': [64,64]", "'pi': [128,128], 'vf': [128,128]")
    s = patch(s, "            model.astra_action_interface = 'ken_actions85_full_v1'", "            model.astra_architecture = '"+ARCHITECTURE+"'\n            model.astra_action_interface = 'ken_actions85_full_v1'")
    out['batch_train.py'] = s.encode()
    s = out['support.py'].decode()
    s = patch(s, '    return model\n', '    from .perception128_identity import validate_architecture\n    return validate_architecture(model)\n')
    out['support.py'] = s.encode()
    for name in ('batch_train.py', 'export.py'):
        s = out[name].decode()
        s = patch(s, "payload = {'schema':", "payload = {'architecture': '"+ARCHITECTURE+"', 'schema':")
        out[name] = s.encode()
    s = out['nn.lua'].decode()
    s = patch(s, ' local x=obs\n', " assert(model.architecture=='"+ARCHITECTURE+"','Wrong perception128 architecture')\n assert(#model.layers==3 and #model.layers[1].bias==128 and #model.layers[2].bias==128 and #model.layers[3].bias==85,'Wrong perception128 width')\n local x=obs\n")
    s = patch(s, ' local x=obs\n', ' local nonzero={}\n for j,v in ipairs(obs) do if v~=0 then nonzero[#nonzero+1]=j end end\n local x=obs\n')
    s = patch(s, '   for j,w in ipairs(row) do value=value+w*x[j] end',
              '   if index==1 then\n    for _,j in ipairs(nonzero) do value=value+row[j]*x[j] end\n'
              '   else\n    for j,w in ipairs(row) do value=value+w*x[j] end\n   end')
    out['nn.lua'] = s.encode()
    return out


def validate_build(manifest, package):
    package = Path(package).resolve(); root = package.parent
    expected = {'schema':SCHEMA, 'package':PACKAGE, 'architecture':ARCHITECTURE,
                'action_interface':INTERFACE, 'observation_interface':OBSERVATION_INTERFACE,
                'observations':4516, 'actions':85, 'decision_frames':12,
                'frame_features':1129, 'history':4, 'net_arch':{'pi':[128,128], 'vf':[128,128]}}
    expected.update(worker_limit=16, worker_default=8, inference='ordered_nonzero_first_layer_v1')
    if any(manifest.get(k) != v for k,v in expected.items()):
        raise RuntimeError('Wrong perception128 build identity')
    for name, value in manifest['frozen_files_sha256'].items():
        path = (root/name).resolve()
        if not path.is_relative_to(root) or digest(path.read_bytes()) != value:
            raise RuntimeError('Frozen perception128 source changed: '+name)
    parent, captured = validate_parent(root/'width-parent')
    if manifest['parent_build_sha256'] != digest((root/'width-parent/build.json').read_bytes()):
        raise RuntimeError('Perception parent identity changed')
    inputs = {n:(root/'width-inputs'/n).read_bytes() for n in INPUT_NAMES}
    derived = derive(captured, inputs)
    if set(derived) != set(manifest['derived_sha256']):
        raise RuntimeError('Perception128 file set changed')
    if {p.name for p in package.iterdir() if p.suffix in ('.py','.lua','.json')} != set(derived):
        raise RuntimeError('Unexpected perception128 executing file')
    for name, body in derived.items():
        if (package/name).read_bytes() != body or digest(body) != manifest['derived_sha256'][name]:
            raise RuntimeError('Perception128 derivation changed: '+name)
    return manifest


def training_metadata(package):
    package = Path(package)
    manifest = json.loads((package.parent/'build.json').read_text())
    validate_build(manifest, package)
    return {'perception_build_sha256':digest((package.parent/'build.json').read_bytes()),
            'architecture':ARCHITECTURE, 'net_arch':manifest['net_arch'],
            'inference':manifest['inference'], 'worker_limit':manifest['worker_limit'],
            'parent_build_sha256':manifest['parent_build_sha256'],
            'action_interface':INTERFACE, 'observation_interface':OBSERVATION_INTERFACE,
            'actions':85, 'observations':4516, 'decision_frames':12,
            'combat_policy':'Shared PPO128; unchanged screen perception/actions/reward; no masks'}
