"""Relabel action semantics explicitly; retain every policy/Adam ZIP member byte."""
import argparse
import json
from pathlib import Path
import zipfile
import torch
from stable_baselines3 import PPO
from astra_play_sf2.config import atomic_json
from astra_play_sf2.runner import sha256
from .pulsed_normals_identity import OLD, INTERFACE, IDENTITY


def identical(left,right):
    if isinstance(left,torch.Tensor):
        if not isinstance(right,torch.Tensor) or left.dtype!=right.dtype or not torch.equal(left,right):
            raise RuntimeError('Changed tensor parameter or optimizer state')
    elif isinstance(left,dict):
        if left.keys()!=right.keys():raise RuntimeError('Changed state dictionary keys')
        for k in left:identical(left[k],right[k])
    elif isinstance(left,(tuple,list)):
        if type(left)!=type(right) or len(left)!=len(right):raise RuntimeError('Changed state sequence')
        for a,b in zip(left,right):identical(a,b)
    elif left!=right:raise RuntimeError('Changed scalar optimizer state')


def migrate(model,output):
    torch.set_num_threads(1);model=Path(model).resolve();source_hash=sha256(model)
    old=PPO.load(model,device='cpu')
    if getattr(old,'astra_action_interface',None)!=OLD or old.observation_space.shape!=(344,) or old.action_space.n!=16:
        raise ValueError('Expected explicit original 344-observation actions16 v1 model')
    with zipfile.ZipFile(model) as archive:
        entries=[(item,archive.read(item.filename)) for item in archive.infolist()]
    if len({x.filename for x,_ in entries})!=len(entries):raise ValueError('Duplicate ZIP member')
    data=json.loads(dict((i.filename,b) for i,b in entries)['data'])
    if data.get('astra_action_interface')!=OLD or 'astra_pulsed_normals' in data:
        raise ValueError('Model metadata already changed or unrecognized')
    data['astra_action_interface']=INTERFACE
    data['astra_pulsed_normals']={'identity':IDENTITY,'source_interface':OLD,'target_interface':INTERFACE,
        'source_model_sha256':source_hash,'optimizer_reinitialized':False,
        'weights_changed':False,'semantics_changed':True,'source_steps':old.num_timesteps,'source_updates':old._n_updates}
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False);target=output/'ppo-pulsed.zip'
    with zipfile.ZipFile(target,'w') as archive:
        for info,body in entries:
            archive.writestr(info,json.dumps(data,indent=2).encode() if info.filename=='data' else body)
    with zipfile.ZipFile(target) as archive:
        for info,body in entries:
            if info.filename!='data' and archive.read(info.filename)!=body:
                raise RuntimeError('Non-metadata ZIP entry changed')
    loaded=PPO.load(target,device='cpu')
    identical(old.policy.state_dict(),loaded.policy.state_dict())
    identical(old.policy.optimizer.state_dict(),loaded.policy.optimizer.state_dict())
    if loaded.astra_action_interface!=INTERFACE or loaded.num_timesteps!=old.num_timesteps or loaded._n_updates!=old._n_updates:
        raise RuntimeError('Reload changed semantic identity or training counters')
    if sha256(model)!=source_hash:raise RuntimeError('Original model changed during migration')
    report={'schema':'astra.rl-pulsed-normals-migration.v1','status':'offline_pass','native_validated':False,
        'action_interface':INTERFACE,'source_interface':OLD,'model':target.name,'source_model_sha256':source_hash,
        'model_sha256':sha256(target),'weights_and_adam':'bit-exact tensors and unchanged ZIP member bytes',
        'changed_zip_members':['data'],'changed_metadata_keys':['astra_action_interface','astra_pulsed_normals'],
        'optimizer_reinitialized':False,'optimizer_entries':len(loaded.policy.optimizer.state),
        'source_steps':old.num_timesteps,'source_updates':old._n_updates,'migration_source_sha256':sha256(Path(__file__))}
    atomic_json(output/'migration.json',report);return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    print(json.dumps(migrate(**vars(p.parse_args())),indent=2))
if __name__=='__main__':main()
