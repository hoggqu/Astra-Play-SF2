"""Result-only native KO revision, bound to an immutable perception128 parent."""
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from .perception128_identity import (ARCHITECTURE, INTERFACE, OBSERVATION_INTERFACE,
                                     digest, validate_architecture)

PACKAGE='astra_sf2_rl_perception128_ko'
SCHEMA='astra.rl-screen-perception128-settlement.v6'
PROTOCOL='native-pip-ko-time-and-confirmed-draw-v6'
INPUT_NAMES=('perception128_settlement_builder.py','perception128_settlement_identity.py','settlement_v6.lua')


def validate_parent(root):
    root=Path(root).resolve();manifest=json.loads((root/'build.json').read_text())
    package=root/manifest['package'];name='_ko_parent_'+digest(str(root).encode())[:16]
    spec=importlib.util.spec_from_file_location(name,package/'__init__.py',submodule_search_locations=[str(package)])
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    importlib.import_module(name+'.perception128_identity').validate_build(manifest,package)
    return manifest,{n:(package/n).read_bytes() for n in manifest['derived_sha256']}


def derive(captured,inputs):
    out=dict(captured)
    out['settlement.lua']=inputs['settlement_v6.lua']
    out['perception128_settlement_identity.py']=inputs['perception128_settlement_identity.py']
    for name in ('initialize.py','support.py','batch_train.py'):
        out[name]=out[name].replace(b'from .perception128_identity import',b'from .perception128_settlement_identity import')
    return out


def validate_build(manifest,package):
    package=Path(package).resolve();root=package.parent
    expected={'schema':SCHEMA,'package':PACKAGE,'settlement_protocol':PROTOCOL,
              'architecture':ARCHITECTURE,'action_interface':INTERFACE,
              'observation_interface':OBSERVATION_INTERFACE,'observations':4516,'actions':85,
              'decision_frames':12,'history':4,'frame_features':1129,
              'net_arch':{'pi':[128,128],'vf':[128,128]},'worker_limit':16,'worker_default':8,
              'inference':'ordered_nonzero_first_layer_v1'}
    if any(manifest.get(k)!=v for k,v in expected.items()):raise RuntimeError('Wrong KO revision identity')
    for name,value in manifest['frozen_files_sha256'].items():
        path=(root/name).resolve()
        if not path.is_relative_to(root) or digest(path.read_bytes())!=value:raise RuntimeError('Frozen KO revision changed: '+name)
    _,captured=validate_parent(root/'settlement-parent')
    if manifest['parent_build_sha256']!=digest((root/'settlement-parent/build.json').read_bytes()):raise RuntimeError('KO parent identity changed')
    inputs={n:(root/'settlement-inputs'/n).read_bytes() for n in INPUT_NAMES}
    derived=derive(captured,inputs)
    if set(derived)!=set(manifest['derived_sha256']):raise RuntimeError('KO revision file set changed')
    if {p.name for p in package.iterdir() if p.suffix in ('.py','.lua','.json')}!=set(derived):raise RuntimeError('Unexpected KO executing file')
    for name,body in derived.items():
        if (package/name).read_bytes()!=body or digest(body)!=manifest['derived_sha256'][name]:raise RuntimeError('KO derivation changed: '+name)
    return manifest


def training_metadata(package):
    package=Path(package);manifest=json.loads((package.parent/'build.json').read_text())
    validate_build(manifest,package)
    return {'perception_build_sha256':digest((package.parent/'build.json').read_bytes()),
            'parent_build_sha256':manifest['parent_build_sha256'],'settlement_protocol':PROTOCOL,
            'architecture':ARCHITECTURE,'net_arch':manifest['net_arch'],
            'inference':manifest['inference'],'worker_limit':16,
            'action_interface':INTERFACE,'observation_interface':OBSERVATION_INTERFACE,
            'actions':85,'observations':4516,'decision_frames':12,
            'combat_policy':'Shared PPO128; unchanged input/actions/reward; native KO result revision only'}
