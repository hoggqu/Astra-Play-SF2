"""Batch-boundary input-latch revision, bound to an immutable late-KO parent."""
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from .perception128_identity import (ARCHITECTURE, INTERFACE, OBSERVATION_INTERFACE,
                                     digest, validate_architecture)

PACKAGE='astra_sf2_rl_perception128_timing'
SCHEMA='astra.rl-screen-perception128-input-timing.v1'
PROTOCOL='native-pip-ko-late-ko-time-and-confirmed-draw-v7'
TIMING_PROTOCOL='restore-held-input-before-rpc-resume-v1'
INPUT_NAMES=('perception128_timing_builder.py','perception128_timing_identity.py')


def validate_parent(root):
    root=Path(root).resolve();manifest=json.loads((root/'build.json').read_text())
    package=root/manifest['package'];name='_timing_parent_'+digest(str(root).encode())[:16]
    spec=importlib.util.spec_from_file_location(name,package/'__init__.py',submodule_search_locations=[str(package)])
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    importlib.import_module(name+'.perception128_late_ko_identity').validate_build(manifest,package)
    return manifest,{n:(package/n).read_bytes() for n in manifest['derived_sha256']}


def patch_runtime(body):
    """Keep the extra same-time input poll on resume identical to native play.

    MAME 0.288's video frame hook precedes the MACHINE_NOTIFY_FRAME input
    update. Unpausing in that hook causes an input poll at unchanged emulated
    time. Restore the OLD held override for this poll; the new macro remains
    deferred until the next advancing frame hook. Never edit a frozen parent.
    """
    replacements=(
        (b"local function input(text)\n release();for k in (text or ''):gmatch",
         b"local held_input,resume_input='',''\nlocal function input(text)\n text=text or '';held_input=text\n release();for k in text:gmatch"),
        (b"assert(not active_action,'Cannot pause halfway through an action')\n release();emu.pause()",
         b"assert(not active_action,'Cannot pause halfway through an action')\n resume_input=held_input;release();emu.pause()"),
        (b"metrics.loads=metrics.loads+1;release();m:load(path)",
         b"metrics.loads=metrics.loads+1;input('');m:load(path)"),
        (b"    start_action()\n    pending.deferred_after=m.time:as_double()",
         b"    input(resume_input)\n    start_action()\n    pending.deferred_after=m.time:as_double()"),
    )
    for old,new in replacements:
        if body.count(old)!=1:raise RuntimeError('Unexpected batch input timing source anchor: '+old.decode())
        body=body.replace(old,new)
    return body


def derive(captured,inputs):
    out=dict(captured)
    out['batch_runtime.lua']=patch_runtime(out['batch_runtime.lua'])
    out['perception128_timing_identity.py']=inputs['perception128_timing_identity.py']
    for name in ('initialize.py','support.py','batch_train.py'):
        out[name]=out[name].replace(b'from .perception128_late_ko_identity import',b'from .perception128_timing_identity import')
    return out


def validate_build(manifest,package):
    package=Path(package).resolve();root=package.parent
    expected={'schema':SCHEMA,'package':PACKAGE,'settlement_protocol':PROTOCOL,'input_timing_protocol':TIMING_PROTOCOL,
              'architecture':ARCHITECTURE,'action_interface':INTERFACE,
              'observation_interface':OBSERVATION_INTERFACE,'observations':4516,'actions':85,
              'decision_frames':12,'history':4,'frame_features':1129,
              'net_arch':{'pi':[128,128],'vf':[128,128]},'worker_limit':16,'worker_default':8,
              'inference':'ordered_nonzero_first_layer_v1'}
    if any(manifest.get(k)!=v for k,v in expected.items()):raise RuntimeError('Wrong input timing revision identity')
    for name,value in manifest['frozen_files_sha256'].items():
        path=(root/name).resolve()
        if not path.is_relative_to(root) or digest(path.read_bytes())!=value:raise RuntimeError('Frozen input timing revision changed: '+name)
    _,captured=validate_parent(root/'timing-parent')
    if manifest['parent_build_sha256']!=digest((root/'timing-parent/build.json').read_bytes()):raise RuntimeError('input timing parent identity changed')
    inputs={n:(root/'timing-inputs'/n).read_bytes() for n in INPUT_NAMES}
    derived=derive(captured,inputs)
    if set(derived)!=set(manifest['derived_sha256']):raise RuntimeError('input timing revision file set changed')
    if {p.name for p in package.iterdir() if p.suffix in ('.py','.lua','.json')}!=set(derived):raise RuntimeError('Unexpected input timing executing file')
    for name,body in derived.items():
        if (package/name).read_bytes()!=body or digest(body)!=manifest['derived_sha256'][name]:raise RuntimeError('input timing derivation changed: '+name)
    return manifest


def training_metadata(package):
    package=Path(package);manifest=json.loads((package.parent/'build.json').read_text())
    validate_build(manifest,package)
    return {'perception_build_sha256':digest((package.parent/'build.json').read_bytes()),
            'parent_build_sha256':manifest['parent_build_sha256'],'settlement_protocol':PROTOCOL,'input_timing_protocol':TIMING_PROTOCOL,
            'architecture':ARCHITECTURE,'net_arch':manifest['net_arch'],
            'inference':manifest['inference'],'worker_limit':16,
            'action_interface':INTERFACE,'observation_interface':OBSERVATION_INTERFACE,
            'actions':85,'observations':4516,'decision_frames':12,
            'combat_policy':'Shared PPO128; unchanged actions/reward; batch pause resumes prior input latch before next native decision frame'}
