"""Immutable eleven-branch PPO bundles; building/exporting never starts MAME."""
import hashlib
import io
import json
from pathlib import Path
import zipfile

OPS=(0,1,2,3,5,6,7,8,9,10,11)
INTERFACE='ken_actions16_pulsed_normals_v2'
SCHEMA='astra.rl-opponent-bundle.v1'
POLICY_SCHEMA='astra.rl-opponent-bundle-policy.v1'
def encoded(value):return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def digest(body):return hashlib.sha256(body).hexdigest()
def route_digest(route,branches):
    return digest(encoded({'route':route,'branch_models':{k:b['model_sha256'] for k,b in branches.items()}}))

def actor_from_zip(body):
    from stable_baselines3 import PPO
    import torch
    model=PPO.load(io.BytesIO(body),device='cpu')
    if (getattr(model,'astra_action_interface',None)!=INTERFACE or model.observation_space.shape!=(344,) or model.action_space.n!=16):
        raise ValueError('Only explicit pulsed16/344 PPO branches are allowed')
    policy=model.policy
    from stable_baselines3.common.torch_layers import FlattenExtractor
    if not isinstance(policy.features_extractor,FlattenExtractor):raise ValueError('Expected flat observation policy')
    modules=list(policy.mlp_extractor.policy_net.children());layers=[]
    if not modules or len(modules)%2:raise ValueError('Expected Linear/Tanh actor')
    for i in range(0,len(modules),2):
        if not isinstance(modules[i],torch.nn.Linear) or not isinstance(modules[i+1],torch.nn.Tanh):raise ValueError('Expected Linear/Tanh actor')
        layers.append({'weight':modules[i].weight.detach().cpu().tolist(),'bias':modules[i].bias.detach().cpu().tolist()})
    if not isinstance(policy.action_net,torch.nn.Linear):raise ValueError('Expected linear action head')
    layers.append({'weight':policy.action_net.weight.detach().cpu().tolist(),'bias':policy.action_net.bias.detach().cpu().tolist()})
    actor={'schema':'astra.rl-policy.actions16.v1','action_interface':INTERFACE,'observations':344,'actions':16,
           'history':4,'decision_frames':12,'activation':'tanh','selection':'deterministic_argmax','model_sha256':digest(body),'layers':layers}
    encoded(actor) # Reject nonfinite exported values.
    return actor

def build_bundle(path, models, routing, selection):
    """Each logical branch owns a complete PPO ZIP, even when initial bytes match."""
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    if set(routing)!=set(OPS):raise ValueError('Exactly eleven opponent routes required')
    bodies={name:Path(p).read_bytes() for name,p in models.items()}
    actors={name:actor_from_zip(body) for name,body in bodies.items()}
    files={};route={};branches={}
    for op in OPS:
        name=routing[op]
        if name not in bodies:raise ValueError('Unknown route source')
        branch=f'op{op:02d}';route[str(op)]=branch
        model_name=f'branches/{branch}/model.zip';actor_name=f'branches/{branch}/actor.json'
        files[model_name]=bodies[name];files[actor_name]=encoded(actors[name])
        branches[branch]={'opponent':op,'source_candidate':name,'model_file':model_name,'model_sha256':digest(bodies[name]),
                          'actor_file':actor_name,'actor_sha256':digest(files[actor_name])}
    manifest={'schema':SCHEMA,'action_interface':INTERFACE,'actions':16,'observations':344,'decision_frames':12,
              'policy_kind':'ppo_opponent_bundle','architecture':'11 independently owned actor/critic PPO branches, fixed opponent router',
              'route':route,'route_sha256':route_digest(route,branches),'branches':branches,'selection_provenance':selection,
              'formal_win_rate_claim':False,'fallback':False,'training_during_play':False}
    files['manifest.json']=encoded(manifest);path.parent.mkdir(parents=True,exist_ok=True)
    # Stable ZIP metadata makes the complete model identity reproducible.
    with path.open('xb') as stream:
        with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for name,body in sorted(files.items()):
                info=zipfile.ZipInfo(name,(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,body)
    return {'model_sha256':digest(path.read_bytes()),'route_sha256':manifest['route_sha256'],'manifest':manifest}

def export_bundle(path):
    body=Path(path).read_bytes()
    with zipfile.ZipFile(io.BytesIO(body)) as z:
        names=z.namelist()
        if len(names)!=len(set(names)):raise ValueError('Duplicate bundle member')
        manifest=json.loads(z.read('manifest.json'))
        expected={'schema':SCHEMA,'action_interface':INTERFACE,'actions':16,'observations':344,'decision_frames':12,
                  'policy_kind':'ppo_opponent_bundle','fallback':False,'training_during_play':False}
        if any(manifest.get(k)!=v for k,v in expected.items()):raise ValueError('Wrong bundle identity')
        route=manifest['route'];branches=manifest['branches']
        if set(route)!=set(map(str,OPS)) or len(set(route.values()))!=11 or set(branches)!=set(route.values()):raise ValueError('Incomplete or aliased opponent branch route')
        if route_digest(route,branches)!=manifest['route_sha256']:raise ValueError('Route hash differs')
        allowed={'manifest.json'};exported={};cache={}
        for op in OPS:
            branch=route[str(op)];meta=branches[branch]
            if branch!=f'op{op:02d}' or meta['opponent']!=op:raise ValueError('Opponent route differs')
            if meta['model_file']!=f'branches/{branch}/model.zip' or meta['actor_file']!=f'branches/{branch}/actor.json':raise ValueError('Invalid bundle member path')
            allowed.update((meta['model_file'],meta['actor_file']))
            weights=z.read(meta['model_file']);saved=z.read(meta['actor_file'])
            if digest(weights)!=meta['model_sha256'] or digest(saved)!=meta['actor_sha256']:raise ValueError('Branch checksum differs')
            if meta['model_sha256'] not in cache:cache[meta['model_sha256']]=actor_from_zip(weights)
            actor=cache[meta['model_sha256']]
            if json.loads(saved)!=actor:raise ValueError('Stored actor is not exported from the bundled PPO')
            exported[branch]={'opponent':op,'source_model_sha256':meta['model_sha256'],'actor_sha256':meta['actor_sha256'],'actor':actor}
        if set(names)!=allowed:raise ValueError('Unexpected or missing bundle member')
    return {'schema':POLICY_SCHEMA,'action_interface':INTERFACE,'actions':16,'observations':344,'history':4,
            'decision_frames':12,'activation':'tanh','selection':'deterministic_argmax','policy_kind':'ppo_opponent_bundle',
            'model_sha256':digest(body),'route':route,'route_sha256':manifest['route_sha256'],'branches':exported}
