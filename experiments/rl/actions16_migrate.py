"""Split the old LP uppercut probability into LP/MP heads; reset optimizer explicitly."""
import argparse
import json
import math
from pathlib import Path
import uuid

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO

from astra_play_sf2.config import atomic_json
from astra_play_sf2.runner import sha256

INTERFACE='ken_actions16_lp_mp_uppercut_v1'


class SpacesOnly(gym.Env):
    observation_space=gym.spaces.Box(-1,1,(344,),dtype=np.float32)
    action_space=gym.spaces.Discrete(16)


def migrate_model(old,seed=42):
    if old.observation_space.shape!=(344,) or old.action_space.n!=15:
        raise ValueError('Expected original 344-observation/15-action PPO')
    if old.policy.net_arch!={'pi':[64,64],'vf':[64,64]}:
        raise ValueError('This migration preserves only the declared 64/64 actor and critic')
    for network in (old.policy.mlp_extractor.policy_net,old.policy.mlp_extractor.value_net):
        layers=list(network.children())
        if len(layers)!=4 or any(not isinstance(layer,torch.nn.Linear if i%2==0 else torch.nn.Tanh)
                                for i,layer in enumerate(layers)):
            raise ValueError('Expected the original Linear/Tanh actor and critic')
    new=PPO('MlpPolicy',SpacesOnly(),seed=seed,device='cpu',n_steps=old.n_steps,
            batch_size=old.batch_size,n_epochs=old.n_epochs,learning_rate=old.learning_rate,
            gamma=old.gamma,gae_lambda=old.gae_lambda,clip_range=old.clip_range,
            clip_range_vf=old.clip_range_vf,normalize_advantage=old.normalize_advantage,
            ent_coef=old.ent_coef,vf_coef=old.vf_coef,max_grad_norm=old.max_grad_norm,
            target_kl=old.target_kl,policy_kwargs=old.policy_kwargs)
    weights={name:value.detach().clone() for name,value in old.policy.state_dict().items()}
    old_bias=weights['action_net.bias'];old_weight=weights['action_net.weight']
    weights['action_net.weight']=torch.cat((old_weight,old_weight[13:14]),dim=0)
    bias=torch.cat((old_bias,old_bias[13:14]),dim=0);bias[13]-=math.log(2);bias[15]-=math.log(2)
    weights['action_net.bias']=bias
    new.policy.load_state_dict(weights,strict=True)
    new.astra_action_interface=INTERFACE
    new.astra_optimizer_origin='reinitialized_at_actions16_migration'
    new.astra_migration={'optimizer_reinitialized':True,'source_steps':old.num_timesteps,
                         'source_optimizer_updates':old._n_updates,'seed':seed}
    new.num_timesteps=0;new._n_updates=0
    if new.policy.optimizer.state_dict()['state']:
        raise RuntimeError('Migrated optimizer was expected to be fresh')
    return new


def migrate(model,output,seed=42):
    torch.set_num_threads(1)
    model=Path(model).resolve();old=PPO.load(model,device='cpu');new=migrate_model(old,seed)
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    target=output/'ppo-actions16.zip';temporary=output/(uuid.uuid4().hex+'.tmp.zip')
    try:new.save(temporary);temporary.replace(target)
    except BaseException:temporary.unlink(missing_ok=True);raise
    report={'schema':'astra.rl-actions16-migration.v1','status':'complete','action_interface':INTERFACE,
            'source_model_sha256':sha256(model),'model_sha256':sha256(target),'model':target.name,
            'source_steps':old.num_timesteps,'source_optimizer_updates':old._n_updates,
            'new_steps':0,'new_optimizer_updates':0,'optimizer_reinitialized':True,
            'preserved':'actor trunk, critic and value head; non-DP action logits',
            'head_split':'actions13 and15 copy old13 weights, both biases subtract log(2)',
            'caveat':'categorical DP-family mass preserved; deterministic argmax can change after splitting',
            'seed':seed}
    atomic_json(output/'migration.json',report);return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=42);args=parser.parse_args()
    print(json.dumps(migrate(**vars(args)),indent=2))


if __name__=='__main__':main()
