"""Zero-column observation expansion; preserve the 16-action policy and Adam state."""
import argparse
import copy
import json
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import FlattenExtractor
from astra_play_sf2.config import atomic_json
from astra_play_sf2.runner import sha256

INTERFACE='sf2_projectiles6_owner_reciprocal_v1'
ACTION_INTERFACE='ken_actions16_lp_mp_uppercut_v1'
FIRST={'mlp_extractor.policy_net.0.weight','mlp_extractor.value_net.0.weight'}
COLUMNS=np.array([92*h+j for h in range(4) for j in range(86)])
class SpacesOnly(gym.Env):
    observation_space=gym.spaces.Box(-1,1,(368,),dtype=np.float32)
    action_space=gym.spaces.Discrete(16)


def expand(value):
    if tuple(value.shape)!=(64,344): raise ValueError('Unexpected first-layer/moment shape')
    out=value.new_zeros((64,368));out[:,COLUMNS]=value
    return out


def expand_observations(old,extra=None):
    old=np.asarray(old,dtype=np.float32)
    if old.shape[-1]!=344: raise ValueError('Expected four 86-column histories')
    out=np.zeros(old.shape[:-1]+(368,),dtype=np.float32);out[...,COLUMNS]=old
    if extra is not None:out.reshape(*old.shape[:-1],4,92)[...,86:]=np.asarray(extra,dtype=np.float32)
    return out


def migrate_model(old,seed=42):
    if (old.observation_space.shape!=(344,) or old.action_space.n!=16
            or getattr(old,'astra_action_interface',None)!=ACTION_INTERFACE):
        raise ValueError('Expected identified 344-observation actions16 PPO')
    if old.policy.net_arch!={'pi':[64,64],'vf':[64,64]} or not isinstance(old.policy.features_extractor,FlattenExtractor):
        raise ValueError('Only flat 64/64 actor and critic are supported')
    for network in (old.policy.mlp_extractor.policy_net,old.policy.mlp_extractor.value_net):
        layers=list(network.children())
        if len(layers)!=4 or any(type(layer)!=(torch.nn.Linear if i%2==0 else torch.nn.Tanh) for i,layer in enumerate(layers)):
            raise ValueError('Expected standard Linear/Tanh')
    new=PPO('MlpPolicy',SpacesOnly(),seed=seed,device='cpu',n_steps=old.n_steps,batch_size=old.batch_size,
            n_epochs=old.n_epochs,learning_rate=old.learning_rate,gamma=old.gamma,gae_lambda=old.gae_lambda,
            clip_range=old.clip_range,clip_range_vf=old.clip_range_vf,normalize_advantage=old.normalize_advantage,
            ent_coef=old.ent_coef,vf_coef=old.vf_coef,max_grad_norm=old.max_grad_norm,target_kl=old.target_kl,
            policy_kwargs=copy.deepcopy(old.policy_kwargs))
    weights={name:expand(value) if name in FIRST else value.detach().clone() for name,value in old.policy.state_dict().items()}
    new.policy.load_state_dict(weights,strict=True)
    old_names={id(p):n for n,p in old.policy.named_parameters()};new_params=dict(new.policy.named_parameters())
    # Bind optimizer entries by parameter name, never by incidental serialized ID.
    groups=[]
    for group in old.policy.optimizer.param_groups:
        g={k:copy.deepcopy(v) for k,v in group.items() if k!='params'}
        g['params']=[new_params[old_names[id(p)]] for p in group['params']];groups.append(g)
    if type(old.policy.optimizer)!=torch.optim.Adam:raise ValueError('Only Adam state migration is supported')
    new.policy.optimizer=torch.optim.Adam(groups)
    for param,state in old.policy.optimizer.state.items():
        name=old_names[id(param)];new_state={}
        for key,value in state.items():
            if isinstance(value,torch.Tensor):
                if value.ndim==0:new_state[key]=value.clone()
                elif value.shape==param.shape:new_state[key]=expand(value) if name in FIRST else value.clone()
                else:raise ValueError('Unrecognized optimizer tensor: '+name+'/'+key)
            else:new_state[key]=copy.deepcopy(value)
        new.policy.optimizer.state[new_params[name]]=new_state
    new.astra_action_interface=ACTION_INTERFACE;new.astra_observation_interface=INTERFACE
    new.astra_optimizer_origin='projectiles6_preserved_adam_zero_columns'
    new.astra_migration={'optimizer_reinitialized':False,'source_steps':old.num_timesteps,'source_updates':old._n_updates,
                         'columns':'86*h+j -> 92*h+j; six new columns zero','seed':seed}
    new.num_timesteps=old.num_timesteps;new._n_updates=old._n_updates
    new._current_progress_remaining=old._current_progress_remaining
    return new


def compare(old,new,observations,extra=None,atol=2e-5):
    x=np.asarray(observations,dtype=np.float32);y=expand_observations(x,extra)
    if x.ndim!=2 or not len(x) or not np.isfinite(y).all():raise ValueError('Finite nonempty observations required')
    maxima={'logits':0.,'value':0.};count=0;minimum_margin=float('inf')
    # Include batch-one dispatch, which can accumulate Linear products differently.
    for batch in (1,32,256):
        for start in range(0,len(x),batch):
            with torch.no_grad():
                a=torch.from_numpy(x[start:start+batch]);b=torch.from_numpy(y[start:start+batch])
                la=old.policy.action_net(old.policy.mlp_extractor.policy_net(a))
                lb=new.policy.action_net(new.policy.mlp_extractor.policy_net(b))
                va=old.policy.predict_values(a);vb=new.policy.predict_values(b)
            for key,l,r in [('logits',la,lb),('value',va,vb)]:
                delta=float((l-r).abs().max());maxima[key]=max(maxima[key],delta)
                if not torch.isfinite(r).all() or delta>atol:raise RuntimeError(f'Migration {key} difference {delta} > {atol}')
            if not torch.equal(la.argmax(1),lb.argmax(1)):raise RuntimeError('Migration changed argmax (including a near-tie); reject candidate')
            best=torch.topk(la,2,dim=1).values
            minimum_margin=min(minimum_margin,float((best[:,0]-best[:,1]).min()))
            count+=len(a)
    return {'ok':True,'samples':len(x),'evaluations':count,'batch_sizes':[1,32,256],'max_abs_difference':maxima,
            'minimum_old_argmax_margin':minimum_margin,'atol':atol,'argmax_equal':True,'bitwise_equivalence_claimed':False}


def migrate(model,output,seed=42,observations=None):
    torch.set_num_threads(1);model=Path(model).resolve();old=PPO.load(model,device='cpu');new=migrate_model(old,seed)
    rng=np.random.default_rng(seed);x=rng.uniform(-1,1,(512,344)).astype(np.float32)
    x=np.concatenate([x,np.zeros((1,344),np.float32),np.ones((1,344),np.float32),-np.ones((1,344),np.float32)])
    sources=[]
    if observations:
        path=Path(observations);actual=np.load(path,allow_pickle=False);x=np.concatenate([actual,x]);sources=[{'sha256':sha256(path),'rows':len(actual)}]
    check=compare(old,new,x,rng.uniform(-1,1,(len(x),4,6)))
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False);target=output/'ppo-projectiles6.zip';new.save(target)
    reloaded=PPO.load(target,device='cpu');reload_check=compare(old,reloaded,x,rng.uniform(-1,1,(len(x),4,6)))
    report={'schema':'astra.rl-projectiles6-migration.v1','status':'offline_pass','native_validated':False,
            'action_interface':ACTION_INTERFACE,'observation_interface':INTERFACE,'observations':368,
            'source_model_sha256':sha256(model),'model_sha256':sha256(target),'model':target.name,
            'source_steps':old.num_timesteps,'source_updates':old._n_updates,'optimizer_reinitialized':False,
            'new_steps':new.num_timesteps,'new_updates':new._n_updates,
            'effective_ppo':{k:getattr(new,k) for k in ('n_steps','batch_size','n_epochs','gamma','gae_lambda','ent_coef','vf_coef','max_grad_norm','normalize_advantage')},
            'migration_source_sha256':sha256(Path(__file__)),'optimizer_entries':len(new.policy.optimizer.state),'comparison':check,'reload_comparison':reload_check,
            'observation_sources':sources,'seed':seed}
    atomic_json(output/'migration.json',report);return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--observations',type=Path)
    p.add_argument('--seed',type=int,default=42);print(json.dumps(migrate(**vars(p.parse_args())),indent=2))
if __name__=='__main__':main()
