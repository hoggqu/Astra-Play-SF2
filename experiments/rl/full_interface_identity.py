"""Exact, isolated full-action/visible-feedback derivation from a frozen parent."""
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys

INTERFACE = 'ken_actions85_full_v1'
OBSERVATION_INTERFACE = 'sf2_state86_visible_feedback114_history4_v1'
SCHEMA = 'astra.rl-full85-feedback-code.v1'
INPUT_NAMES = ('full_actions.py', 'full_actions.lua', 'visible_feedback.py',
               'visible_feedback.lua', 'visible_animation_map.lua',
               'full_interface_identity.py', 'full_interface_initialize.py',
               'full_interface_dataset.py', 'full_interface_bootstrap.py')


def digest(body):
    return hashlib.sha256(body).hexdigest()


def patch(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError('Full-interface derivation anchor changed: ' + old[:100])
    return text.replace(old, new)


def parent_validate(root):
    """The parent retains its original validators and complete ancestor tree."""
    root = Path(root)
    manifest = json.loads((root/'build.json').read_text())
    package = root/manifest['package']
    name = '_full85_parent_' + digest(str(root.resolve()).encode())[:16]
    spec = importlib.util.spec_from_file_location(name, package/'__init__.py',
                                                submodule_search_locations=[str(package)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if manifest.get('schema') == 'astra.rl-specialist-code.v1':
        validator = importlib.import_module(name+'.specialist_identity')
        validator.validate_specialist_build(manifest, package)
    elif manifest.get('schema') == 'astra.rl-pulsed-normals-build.v1':
        validator = importlib.import_module(name+'.pulsed_normals_identity')
        validator.validate_pulsed_build(manifest, package)
    else:
        raise RuntimeError('Expected a frozen pulsed or specialist-pulsed source')
    return manifest, {n: (package/n).read_bytes() for n in manifest['derived_sha256']}


def derive(captured, inputs):
    """Only explicit dimension/identity and feedback hooks change execution."""
    out = dict(captured)
    # Schemas and shapes change everywhere they are consumed. Integer 16 is
    # deliberately not replaced globally: block sizes and native timing remain.
    for name, body in out.items():
        if name.endswith(('.py', '.lua')) and name not in (
                'specialist_identity.py', 'pulsed_normals_identity.py', 'chain_identity.py'):
            text = body.decode().replace('ken_actions16_pulsed_normals_v2', INTERFACE)
            text = text.replace('.actions16.v1', '.full85-feedback.v1').replace('344', '800')
            for old, new in (('actions==16', 'actions==85'), ('count==16', 'count==85'),
                             ('#x==16', '#x==85'), ("'actions':16", "'actions':85"),
                             ("'actions': 16", "'actions': 85"), ('!= 16', '!= 85'),
                             ('minlength=16', 'minlength=85')):
                text = text.replace(old, new)
            out[name] = text.encode()
    out['actions.lua'] = inputs['full_actions.lua']
    out['full_actions.py'] = inputs['full_actions.py']
    for name in ('visible_feedback.py', 'visible_feedback.lua', 'visible_animation_map.lua'):
        out[name] = inputs[name]
    out['full_interface_identity.py'] = inputs['full_interface_identity.py']
    out['initialize.py'] = inputs['full_interface_initialize.py']
    out['specialist_dataset.py'] = captured['dataset.py']
    out['dataset.py'] = inputs['full_interface_dataset.py']

    s = out['env.py'].decode()
    start = s.index('ACTION_NAMES = '); stop = s.index('TRAIN_LEADS = ', start)
    s = s[:start] + 'from .full_actions import ACTION_NAMES\nfrom .visible_feedback import features as feedback_features\n' + s[stop:]
    s = patch(s, '    return np.clip(np.asarray(values, dtype=np.float32), -1, 1)',
              '    values.extend(feedback_features(state))\n    assert len(values) == 200\n    return np.clip(np.asarray(values, dtype=np.float32), -1, 1)')
    s = patch(s, 'shape=(4*86,)', 'shape=(4*200,)')
    out['env.py'] = s.encode()
    # The old synchronous/V4 runtime is not an alternative controller here.
    out['runtime.lua'] = b"error('Full85 requires native BatchEnv; legacy scripted runtime is disabled')\n"
    out['corrected_reference_runtime.lua'] = b"error('Use the full85 native Core reference gate')\n"

    s = out['nn.lua'].decode()
    s = patch(s, 'local N={}', "local N={}\nlocal Feedback=assert(loadfile('training/runtime/rl_visible_feedback.lua'))()")
    s = patch(s, 'assert(#out==86);return out',
              'assert(#out==86);for _,v in ipairs(Feedback.features(s)) do append(out,v) end;assert(#out==200);return out')
    s = patch(s, "assert(#obs==800 and model.activation", "assert(model.observation_interface=='"+OBSERVATION_INTERFACE+"')\n assert(#obs==800 and model.activation")
    s = patch(s, "return seq,'rl_action_'..action", "return seq,'rl_action_'..action,action")
    out['nn.lua'] = s.encode()

    s = out['native_continuous_core.lua'].decode()
    anchor = "local Actions=assert(loadfile('training/runtime/rl_actions.lua'))()"
    s = patch(s, anchor, anchor + '''
local Feedback=assert(loadfile('training/runtime/rl_visible_feedback.lua'))()
local original_new,original_tick,original_begin=Core.new,Core.tick,Core.begin_round
function Core.new(options,s)
 local monitor=Feedback.new(Actions);monitor:reset(s)
 local self=original_new(options,s);self.rl_feedback=monitor;return self
end
function Core:tick(s)
 self.rl_feedback:tick(s)
 return original_tick(self,s)
end
function Core:begin_round(s,effects)
 self.rl_feedback:reset(s)
 return original_begin(self,s,effects)
end
function Core:rl_request(action,s) self.rl_feedback:request(action,s) end
function Core:rl_pick(s,reset)
 local seq,reason,action=self.choose(s.p1,s.p2,self.mode,s,reset)
 if action~=nil then self:rl_request(action,s) end
 return seq
end
''')
    s = patch(s, 'self.rl_sequence=self.choose(s.p1,s.p2,self.mode,s,true)', 'self.rl_sequence=self:rl_pick(s,true)')
    s = patch(s, 'self.rl_sequence=self.choose(s.p1,s.p2,self.mode,s,false)', 'self.rl_sequence=self:rl_pick(s,false)')
    out['native_continuous_core.lua'] = s.encode()
    # Non-native entry is unavailable; native staging replaces this file before
    # its code is loaded. This prevents a superficially compatible wrong clock.
    out['continuous_core.lua'] = b"error('Full85 requires native_continuous')\n"
    s = out['batch_runtime.lua'].decode()
    s = patch(s, 'active_action={action=action,logprob=logprob,before=state,observation=obs,episode_start=episode_start}',
              'active_action={action=action,logprob=logprob,before=state,observation=obs,episode_start=episode_start}\n core:rl_request(action,state)')
    out['batch_runtime.lua'] = s.encode()

    s = out['batch_env.py'].decode()
    anchor = "            (runtime / 'rl_settlement.lua').write_bytes((HERE / 'settlement.lua').read_bytes())"
    s = patch(s, anchor, anchor + "\n            for name in ('visible_feedback', 'visible_animation_map'):\n                (runtime / ('rl_'+name+'.lua')).write_bytes((HERE / (name+'.lua')).read_bytes())")
    anchor = "'rl_settlement.lua': sha256(self.run / 'training/runtime/rl_settlement.lua'),"
    s = patch(s, anchor, anchor + "\n                'rl_visible_feedback.lua': sha256(self.run / 'training/runtime/rl_visible_feedback.lua'),\n                'rl_visible_animation_map.lua': sha256(self.run / 'training/runtime/rl_visible_animation_map.lua'),")
    out['batch_env.py'] = s.encode()

    s = out['continuous.py'].decode()
    s = patch(s, "('continuous_core.lua', 'rl_continuous_core.lua')]:",
              "('continuous_core.lua', 'rl_continuous_core.lua'),\n                         ('visible_feedback.lua', 'rl_visible_feedback.lua'),\n                         ('visible_animation_map.lua', 'rl_visible_animation_map.lua')]:")
    s = patch(s, "action_interface=Model.action_interface,policy_kind='ppo'",
              "action_interface=Model.action_interface,observation_interface=Model.observation_interface,observations=800,policy_kind='ppo'")
    # These replacements operate on the frozen staged play adapter, not assets.
    anchor = "    path.write_text('-- EXPERIMENTAL RL ADAPTER; NOT FROZEN V4 POLICY CERTIFICATION.\\n' + text, encoding='utf-8')"
    additions = '''    text = replace_once(text, 'local seq,reason=policy(a,b,selected,s,reset)', 'local seq,reason,action=policy(a,b,selected,s,reset)')
    text = replace_once(text, "trace_record(r,'decision',r.core.frame,r.core.round,a,b,selected,seq,reason)", "trace_record(r,'decision',r.core.frame,r.core.round,a,b,selected,seq,reason,s)")
    text = replace_once(text, 'return seq,reason\\n end', 'return seq,reason,action\\n end')
    text = replace_once(text, 'function Recorder:decision(frame,round,a,b,mode,seq,reason)', 'function Recorder:decision(frame,round,a,b,mode,seq,reason,s)')
    text = replace_once(text, 'cpu=copy(b)}', 'cpu=copy(b),visible_feedback=copy(assert(s.visible_feedback))}')
    text = replace_once(text, "'difficulty_mirror','effective_difficulty'}", "'difficulty_mirror','effective_difficulty','visible_feedback'}")
    text = replace_once(text, 'function Recorder:after(effects)', 'function Recorder:after(effects,s)')
    text = replace_once(text, "assert(self.row,'Missing pre-tick observation')", "assert(self.row,'Missing pre-tick observation');self.row[29]=copy(assert(s.visible_feedback))")
    text = replace_once(text, "trace_record(r,'after',effects)", "trace_record(r,'after',effects,r.latest)")
    text = replace_once(text, "'training/runtime/rl_continuous_core.lua'}) do", "'training/runtime/rl_continuous_core.lua','training/runtime/rl_visible_feedback.lua','training/runtime/rl_visible_animation_map.lua'}) do")
'''
    s = patch(s, anchor, additions + anchor)
    s = patch(s, "        score = summary['score']", "        if summary.get('observation_interface') != '"+OBSERVATION_INTERFACE+"' or summary.get('observations') != 800:\n            raise RuntimeError('Wrong full85 observation identity')\n        score = summary['score']")
    out['continuous.py'] = s.encode()

    s = out['native_continuous.py'].decode()
    anchor = "    frame_index=columns.index('frame');time_index=columns.index('emulated_seconds')"
    s = patch(s, anchor, "    from .visible_feedback import features as feedback_features\n"
              "    feedback_index = columns.index('visible_feedback')\n"
              "    for row in rows: feedback_features({'visible_feedback':row[feedback_index]})\n"
              "    for decision in trace['decisions']:\n"
              "        feedback_features({'visible_feedback':decision['visible_feedback']})\n" + anchor)
    out['native_continuous.py'] = s.encode()

    s = out['support.py'].decode()
    old_import = ('from .specialist_identity import validate_specialist_build as validate_chain_build'
                  if 'specialist_identity.py' in captured else
                  'from .pulsed_normals_identity import validate_pulsed_build as validate_chain_build')
    s = patch(s, old_import,
              'from .full_interface_identity import validate_build as validate_chain_build, OBSERVATION_INTERFACE')
    s = patch(s, "            or model.observation_space.shape", "            or getattr(model, 'astra_observation_interface', None) != OBSERVATION_INTERFACE\n            or model.observation_space.shape")
    s = s.replace('if not 16 <= count', 'if not 85 <= count').replace('permutation(16)', 'permutation(85)').replace('integers(0, 16, count-16)', 'integers(0, 85, count-85)')
    s = patch(s, "'selection':'deterministic_argmax'}.items()", "'selection':'deterministic_argmax','observation_interface':OBSERVATION_INTERFACE}.items()")
    s = patch(s, "('rl_settlement.lua','settlement.lua')):", "('rl_settlement.lua','settlement.lua'),\n                           ('rl_visible_feedback.lua','visible_feedback.lua'),\n                           ('rl_visible_animation_map.lua','visible_animation_map.lua')):")
    s = patch(s, "            or result.get('model_sha256')", "            or result.get('observation_interface') != OBSERVATION_INTERFACE\n            or result.get('model_sha256')")
    s = patch(s, "                    or summary.get('model_sha256')", "                    or summary.get('observation_interface') != OBSERVATION_INTERFACE\n                    or summary.get('model_sha256')")
    s = patch(s, "return {'ok':True,'action_interface':INTERFACE,'actions':85,'observations':800,",
              "return {'ok':True,'action_interface':INTERFACE,'actions':85,'observations':800,'observation_interface':OBSERVATION_INTERFACE,")
    out['support.py'] = s.encode()

    for name in ('export.py', 'batch_train.py'):
        s = out[name].decode()
        s = patch(s, '    policy = model.policy',
                  "    from .support import validate_model\n    validate_model(model)\n    policy = model.policy")
        s = s.replace("'observations': 800", "'observation_interface': '"+OBSERVATION_INTERFACE+"', 'observations': 800")
        if name == 'batch_train.py':
            old_import = ('from .specialist_identity import training_metadata' if 'specialist_identity.py' in captured
                          else 'from .pulsed_normals_identity import training_metadata')
            s = patch(s, old_import, 'from .full_interface_identity import training_metadata')
            if 'specialist_identity.py' in captured:
                s = patch(s, "    if not args.init_model: parser.error('Specialist training requires complete baseline PPO ZIP')\n", '')
            else:
                s = patch(s, '    config = load_config();groups, difficulty = load_dataset(args.dataset)',
                          '    config = load_config();groups, difficulty = load_dataset(args.dataset)\n    from .dataset import metadata\n    specialist_metadata = metadata(args.dataset)')
                s = patch(s, "              'action_semantics': pulse_metadata,", "              'action_semantics': pulse_metadata,\n              'specialist_training': specialist_metadata,")
            s = patch(s, "            model.astra_action_interface = '"+INTERFACE+"'", "            model.astra_action_interface = '"+INTERFACE+"'\n            model.astra_observation_interface = '"+OBSERVATION_INTERFACE+"'")
            s = patch(s, "            result['optimizer_initialization']", "            from .support import validate_model\n            validate_model(model)\n            result['optimizer_initialization']")
            s = s.replace('args.steps < 16', 'args.steps < 85').replace('16..256 steps', '85..256 steps')
        out[name] = s.encode()
    # Result identities (in addition to policy payload identities).
    for name in ('continuous.py', 'batch_train.py', 'support.py'):
        s = out[name].decode()
        if name == 'continuous.py':
            s = patch(s, "'actions': 85, 'frozen_v4_certification'",
                      "'actions': 85, 'observations':800, 'observation_interface':'"+OBSERVATION_INTERFACE+"', 'frozen_v4_certification'")
        elif name == 'batch_train.py':
            s = patch(s, "'actions': 85,\n              'status'",
                      "'actions': 85, 'observations':800, 'observation_interface':'"+OBSERVATION_INTERFACE+"',\n              'status'")
        s = s.replace("'actions':85,'attempts'", "'actions':85,'observations':800,'observation_interface':'"+OBSERVATION_INTERFACE+"','attempts'")
        out[name] = s.encode()
    # Legacy training/campaign selectors do not understand this new schema.
    for name in ('train.py', 'native_campaign.py', 'campaign.py', 'migrate.py'):
        if name in out:
            out[name] = b"raise RuntimeError('Use this versioned package batch_train or native_continuous entry')\n"
    # Remove stale human-facing dimensional labels while retaining legacy
    # ancestor validators, whose wording and behavior must stay unchanged.
    for name in ('support.py','export.py','batch_train.py','env.py','nn.lua','continuous.py'):
        out[name] = out[name].replace(b'actions16', b'full85').replace(b'Actions16', b'Full85')
        out[name] = out[name].replace(b'16 discrete actions',b'85 discrete actions')
        out[name] = out[name].replace(b'4 x 86 features and 16 actions',b'4 x 200 features and 85 actions')
    return out


def validate_build(manifest, package):
    package = Path(package).resolve(); root = package.parent
    if (manifest.get('schema') != SCHEMA or manifest.get('package') != package.name
            or manifest.get('action_interface') != INTERFACE
            or manifest.get('observation_interface') != OBSERVATION_INTERFACE
            or manifest.get('actions') != 85 or manifest.get('observations') != 800
            or manifest.get('decision_frames') != 12):
        raise RuntimeError('Wrong full85-feedback build')
    for name, expected in manifest['frozen_files_sha256'].items():
        path = (root/name).resolve()
        if not path.is_relative_to(root) or digest(path.read_bytes()) != expected:
            raise RuntimeError('Frozen full85 input changed: '+name)
    _, captured = parent_validate(root/'parent-source')
    inputs = {name: (root/'integration-inputs'/name).read_bytes() for name in INPUT_NAMES}
    expected = derive(captured, inputs)
    if set(expected) != set(manifest['derived_sha256']):
        raise RuntimeError('Full85 executing file set changed')
    actual = {p.name for p in package.iterdir() if p.suffix in ('.py', '.lua', '.json')}
    if actual != set(expected): raise RuntimeError('Unexpected full85 executing dependency')
    for name, body in expected.items():
        if (package/name).read_bytes() != body or digest(body) != manifest['derived_sha256'][name]:
            raise RuntimeError('Full85 source changed: '+name)
    return manifest


def training_metadata(package):
    package = Path(package)
    manifest = json.loads((package.parent/'build.json').read_text())
    validate_build(manifest, package)
    return {'full_interface_build_sha256': digest((package.parent/'build.json').read_bytes()),
            'action_interface': INTERFACE, 'observation_interface': OBSERVATION_INTERFACE,
            'actions':85, 'observations':800, 'decision_frames':12,
            'combat_policy':'PPO only; no V4 fallback; no legality mask or reward change'}
