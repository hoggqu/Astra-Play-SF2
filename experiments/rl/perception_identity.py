"""Strict source derivation for the separate structured screen observation."""
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys

PACKAGE='astra_sf2_rl_perception'
INTERFACE='ken_actions85_full_v1'
OBSERVATION_INTERFACE='sf2_screen_perception_v2'
FRAME_FEATURES=1129
OBSERVATIONS=4516
SCHEMA='astra.rl-screen-perception-code.v2'
INPUT_NAMES=('perception_builder.py','perception_identity.py','perception_features.py',
             'perception_features.lua','perception_execution.py','perception_execution.lua',
             'fighter_perception.py','fighter_perception.lua','fighter_animation_map.lua','fighter_render.lua',
             'visible_projectiles.py','visible_projectiles.lua','visible_sprite_buffer.lua')
RUNTIME_FILES=('perception_features','perception_execution','fighter_perception',
               'fighter_animation_map','fighter_render','visible_projectiles','visible_sprite_buffer')


def digest(body): return hashlib.sha256(body).hexdigest()


def patch(text,old,new):
    if text.count(old)!=1: raise RuntimeError('Perception anchor changed: '+old[:100])
    return text.replace(old,new)


def validate_parent(root):
    root=Path(root).resolve();manifest=json.loads((root/'build.json').read_text())
    package=root/manifest['package']
    name='_perception_parent_'+digest(str(root).encode())[:16]
    spec=importlib.util.spec_from_file_location(name,package/'__init__.py',submodule_search_locations=[str(package)])
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    identity=importlib.import_module(name+'.full_interface_identity')
    identity.validate_build(manifest,package)
    return manifest,{n:(package/n).read_bytes() for n in manifest['derived_sha256']}


def derive(captured,inputs):
    out=dict(captured)
    names=('env.py','nn.lua','export.py','initialize.py','batch_train.py','support.py',
           'continuous.py','native_continuous.py')
    for name in names:
        s=out[name].decode().replace('sf2_state86_visible_feedback114_history4_v1',OBSERVATION_INTERFACE)
        s=s.replace('full85-feedback.v1','screen-perception.v2').replace('800',str(OBSERVATIONS))
        out[name]=s.encode()
    out['initialize.py']=out['initialize.py'].replace(b'astra.rl-full85-initialization.v1',b'astra.rl-screen-perception-initialization.v2').replace(b'fresh_full85_feedback',b'fresh_screen_perception_v2')
    for name in INPUT_NAMES:
        if name!='perception_builder.py': out[name]=inputs[name]
    # The old identities remain unchanged as parent evidence, not validators of
    # this new package or authorities for silently accepting an old model.
    for name in ('support.py','initialize.py','batch_train.py'):
        out[name]=out[name].replace(b'from .full_interface_identity import',b'from .perception_identity import')
    s=out['env.py'].decode()
    start=s.index('def features(state):');end=s.index('\n\ndef reward(',start)
    s=s[:start]+'def features(state):\n    from .perception_features import features as encode\n    return encode(state)\n'+s[end:]
    s=patch(s,'shape=(4*200,)','shape=(4*'+str(FRAME_FEATURES)+',)')
    out['env.py']=s.encode()
    s=out['nn.lua'].decode()
    start=s.index('function N.features(s)');end=s.index('local function tanh(',start)
    s=s[:start]+"local Perception=assert(loadfile('training/runtime/rl_perception_features.lua'))()\nfunction N.features(s) return Perception.features(s) end\n"+s[end:]
    s=patch(s,"local Feedback=assert(loadfile('training/runtime/rl_visible_feedback.lua'))()\n",'')
    out['nn.lua']=s.encode()

    s=out['native_continuous_core.lua'].decode()
    s=patch(s,"local Feedback=assert(loadfile('training/runtime/rl_visible_feedback.lua'))()",
            "local Feedback=assert(loadfile('training/runtime/rl_perception_execution.lua'))()\n"
            "local Fighters=assert(loadfile('training/runtime/rl_fighter_perception.lua'))()\n"
            "local Projectiles=assert(loadfile('training/runtime/rl_visible_projectiles.lua'))()\n"
            "local function rendered_projectiles(s)\n"
            " local raw=Projectiles.read(mem,s)\n"
            " assert(raw and raw.known==true,'unsupported-render-reader: current CPS1 sprite buffer/atlas unavailable; requires MAME0.288 sf2 World910522')\n"
            " return raw\nend\n"
            "local function finish_perception(s)\n"
            " s.visible_sprite_buffer=nil;s.p1.visible_fighter=nil;s.p2.visible_fighter=nil\nend")
    s=patch(s,' local monitor=Feedback.new(Actions);monitor:reset(s)',
            ' local fighters=Fighters.new();local projectiles=Projectiles.new()\n'
            ' Fighters.read(mem,s);fighters:reset(s);projectiles:reset(s,rendered_projectiles(s));finish_perception(s)\n'
            ' local monitor=Feedback.new(Actions);monitor:reset(s)')
    s=patch(s,'self.rl_feedback=monitor;return self',
            'self.rl_feedback=monitor;self.rl_fighters=fighters;self.rl_projectiles=projectiles;return self')
    s=patch(s,' self.rl_feedback:tick(s)',
            ' Fighters.read(mem,s);self.rl_fighters:tick(s)\n'
            ' self.rl_projectiles:tick(s,rendered_projectiles(s));finish_perception(s)\n self.rl_feedback:tick(s)')
    s=patch(s,' self.rl_feedback:reset(s)',
            ' Fighters.read(mem,s);self.rl_fighters:reset(s)\n'
            ' self.rl_projectiles:reset(s,rendered_projectiles(s));finish_perception(s)\n self.rl_feedback:reset(s)')
    out['native_continuous_core.lua']=s.encode()

    s=out['batch_env.py'].decode()
    anchor="            for name in ('visible_feedback', 'visible_animation_map'):"
    extra=repr(RUNTIME_FILES)
    s=patch(s,anchor,"            for name in ('visible_feedback', 'visible_animation_map') + "+extra+':')
    anchor="'rl_visible_animation_map.lua': sha256(self.run / 'training/runtime/rl_visible_animation_map.lua'),"
    s=patch(s,anchor,anchor+'\n                '+',\n                '.join(
        repr('rl_'+n+'.lua')+": sha256(self.run / 'training/runtime/rl_"+n+".lua')" for n in RUNTIME_FILES)+',')
    out['batch_env.py']=s.encode()
    s=out['continuous.py'].decode()
    anchor="('visible_animation_map.lua', 'rl_visible_animation_map.lua')]:"
    s=patch(s,anchor,"('visible_animation_map.lua', 'rl_visible_animation_map.lua'),\n                         "+
            ',\n                         '.join(repr((n+'.lua','rl_'+n+'.lua')) for n in RUNTIME_FILES)+']:')
    # Previous feedback is column29; append immutable current screen snapshots.
    anchor="    path.write_text('-- EXPERIMENTAL RL ADAPTER; NOT FROZEN V4 POLICY CERTIFICATION.\\n' + text, encoding='utf-8')"
    extra="""    text = replace_once(text, "'effective_difficulty','visible_feedback'}", "'effective_difficulty','visible_feedback','fighter_perception','visible_projectiles'}")
    text = replace_once(text, 'self.row[29]=copy(assert(s.visible_feedback))', 'self.row[29]=copy(assert(s.visible_feedback));self.row[30]=copy(assert(s.fighter_perception));self.row[31]=copy(assert(s.visible_projectiles))')
    text = replace_once(text, 'visible_feedback=copy(assert(s.visible_feedback))}', 'visible_feedback=copy(assert(s.visible_feedback)),fighter_perception=copy(assert(s.fighter_perception)),visible_projectiles=copy(assert(s.visible_projectiles))}')
"""
    tail="'training/runtime/rl_visible_animation_map.lua'}) do"
    more="'training/runtime/rl_visible_animation_map.lua',"+','.join(repr('training/runtime/rl_'+n+'.lua') for n in RUNTIME_FILES)+"}) do"
    extra+='    text = replace_once(text, '+repr(tail)+', '+repr(more)+')\n'
    s=patch(s,anchor,extra+anchor);out['continuous.py']=s.encode()

    s=out['support.py'].decode()
    anchor="('rl_visible_animation_map.lua','visible_animation_map.lua')):"
    s=patch(s,anchor,"('rl_visible_animation_map.lua','visible_animation_map.lua'),\n                           "+
            ',\n                           '.join(repr(('rl_'+n+'.lua',n+'.lua')) for n in RUNTIME_FILES)+'):')
    out['support.py']=s.encode()
    s=out['native_continuous.py'].decode()
    s=patch(s,'from .visible_feedback import features as feedback_features','from .perception_execution import features as feedback_features')
    anchor="    frame_index=columns.index('frame');time_index=columns.index('emulated_seconds')"
    s=patch(s,anchor,"    from .fighter_perception import features as fighter_features\n"
            "    from .visible_projectiles import features as projectile_features\n"
            "    for name,encoder in (('fighter_perception',fighter_features),('visible_projectiles',projectile_features)):\n"
            "        index=columns.index(name)\n"
            "        for row in rows: encoder({name:row[index]})\n"
            "        for decision in trace['decisions']: encoder({name:decision[name]})\n"+anchor)
    out['native_continuous.py']=s.encode()
    return out


def validate_build(manifest,package):
    package=Path(package).resolve();root=package.parent
    expected={'schema':SCHEMA,'package':package.name,'action_interface':'ken_actions85_full_v1',
              'observation_interface':OBSERVATION_INTERFACE,'observations':OBSERVATIONS,
              'frame_features':FRAME_FEATURES,'actions':85,'decision_frames':12,'history':4}
    if any(manifest.get(k)!=v for k,v in expected.items()):raise RuntimeError('Wrong perception build identity')
    for name,value in manifest['frozen_files_sha256'].items():
        path=(root/name).resolve()
        if not path.is_relative_to(root) or digest(path.read_bytes())!=value:
            raise RuntimeError('Frozen perception source changed: '+name)
    _,captured=validate_parent(root/'perception-parent')
    inputs={n:(root/'perception-inputs'/n).read_bytes() for n in INPUT_NAMES}
    expected=derive(captured,inputs)
    if set(expected)!=set(manifest['derived_sha256']):raise RuntimeError('Perception file set changed')
    if {p.name for p in package.iterdir() if p.suffix in ('.py','.lua','.json')}!=set(expected):
        raise RuntimeError('Unexpected perception executing file')
    for name,body in expected.items():
        if (package/name).read_bytes()!=body or digest(body)!=manifest['derived_sha256'][name]:
            raise RuntimeError('Perception derivation changed: '+name)
    return manifest


def training_metadata(package):
    package=Path(package);manifest=json.loads((package.parent/'build.json').read_text())
    validate_build(manifest,package)
    return {'perception_build_sha256':digest((package.parent/'build.json').read_bytes()),
            'action_interface':'ken_actions85_full_v1','observation_interface':OBSERVATION_INTERFACE,
            'actions':85,'observations':OBSERVATIONS,'decision_frames':12,
            'combat_policy':'PPO only; screen-visible structured input; unchanged action/reward; no masks'}
