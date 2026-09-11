"""Identity and final sampling audit for an isolated fixed-seed actions16 evaluator."""
import json
from pathlib import Path
from astra_play_sf2.config import atomic_json
from astra_play_sf2.runner import sha256, seal_run

INTERFACE='ken_actions16_lp_mp_uppercut_v1'


def validate_build(manifest,package):
    parent=Path(package).parent/'parent-build.json'
    if (manifest.get('schema')!='astra.rl-stochastic16-build.v1'
            or manifest.get('action_interface')!=INTERFACE or manifest.get('actions')!=16
            or manifest.get('observations')!=344 or manifest.get('native_decision_frames')!=12
            or manifest.get('selection')!='categorical_softmax'
            or manifest.get('policy_prng')!='park_miller_48271_v1'
            or not 1<=manifest.get('policy_seed',0)<2147483647):
        raise RuntimeError('Invalid stochastic actions16 build identity')
    if sha256(parent)!=manifest.get('parent_build_sha256'):
        raise RuntimeError('Stochastic parent manifest changed')
    original=json.loads(parent.read_text())
    if (original.get('schema')!='astra.rl-actions16-build.v1' or original.get('action_interface')!=INTERFACE
            or original.get('actions')!=16 or original.get('observations')!=344):
        raise RuntimeError('Expected original actions16 parent')
    if manifest['source_sha256']!=original['derived_sha256']:
        raise RuntimeError('Stochastic sources differ from parent')
    if manifest['derived_sha256']['actions.lua']!=original['derived_sha256']['actions.lua']:
        raise RuntimeError('Stochastic candidate changed action macros')
    for name in ('sampling.py','stochastic_nn.lua','stochastic_identity.py'):
        if name not in manifest['derived_sha256']:raise RuntimeError('Missing sampling dependency: '+name)


def stage_sampling(run,payload,source,package):
    from .continuous import replace_once
    runtime=Path(run)/'training/runtime'
    (runtime/'rl_stochastic_nn.lua').write_bytes((Path(package)/'stochastic_nn.lua').read_bytes())
    source=replace_once(source,"local NN=assert(loadfile('training/runtime/rl_nn.lua'))()",
        "local BaseNN=assert(loadfile('training/runtime/rl_nn.lua'))()\n"
        "local NN=assert(loadfile('training/runtime/rl_stochastic_nn.lua'))()(BaseNN)")
    source=replace_once(source,"policy_kind='ppo',model_sha256=Model.model_sha256,",
        "policy_kind='ppo',model_sha256=Model.model_sha256,selection=Model.selection,policy_seed=Model.policy_seed,policy_prng=Model.policy_prng,")
    source=replace_once(source,"'training/runtime/rl_nn.lua','training/runtime/rl_actions.lua'",
        "'training/runtime/rl_nn.lua','training/runtime/rl_stochastic_nn.lua','training/runtime/rl_actions.lua'")
    atomic_json(Path(run)/'sampling-protocol.json',{k:payload[k] for k in ('selection','policy_seed','policy_prng','model_sha256')})
    return source


def finish_sampling(result,args,kwargs,package):
    from .support import capture_interface, audit_interface
    from .sampling import audit_sampling
    model=kwargs.get('model',args[1] if len(args)>1 else None)
    output=Path(kwargs.get('output',args[2] if len(args)>2 else '')).resolve()
    result['selection']='categorical_softmax'
    result['evaluation_kind']='fixed-weight categorical policy; separate from deterministic argmax'
    result['sampling_audit']={'ok':False,'reason':'Run did not complete'}
    if result['status']=='complete':
        try:
            snapshot=capture_interface(model,package)
            if snapshot['manifest_sha256']!=result['action_interface_audit']['build_manifest_sha256']:
                raise RuntimeError('Build changed before sampling audit')
            payload=snapshot['payload']
            result.update(policy_seed=payload['policy_seed'],policy_prng=payload['policy_prng'])
            result['stochastic_source_sha256']={name:snapshot['package_sha256'][name]
                for name in ('sampling.py','stochastic_nn.lua','stochastic_identity.py','nn.lua')}
            protocol=json.loads((output/'sampling-protocol.json').read_text())
            if protocol!={k:payload[k] for k in ('selection','policy_seed','policy_prng','model_sha256')}:
                raise RuntimeError('Sampling protocol changed')
            result['sampling_audit']=audit_sampling(output,result,payload)
            if result['sampling_audit']['decisions']!=result['native_timing_audit']['decisions']:
                raise RuntimeError('Sampling/native decision coverage differs')
            snapshot['staged']=dict(result['runtime_sha256'])
            result['action_interface_audit']=audit_interface(snapshot,output,result)
        except (Exception,KeyboardInterrupt) as error:
            result.update(status='invalid',error=f'Sampling audit: {type(error).__name__}: {error}')
            result['sampling_audit']={'ok':False,'reason':str(error)}
            result['action_interface_audit']={'ok':False,'reason':'Final sampling/source audit failed: '+str(error)}
    atomic_json(output/'result.json',result);seal_run(output)
    return result
