"""Offline-only real-state migration and Torch/Lua checks; never starts an emulator."""
import argparse
import importlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
from stable_baselines3 import PPO
from lupa.lua54 import LuaRuntime
from astra_play_sf2.config import atomic_json
from astra_play_sf2.runner import sha256
from .env import features
from .export import lua_literal
from .projectile_migrate import migrate, expand_observations

# Same absolute bound as the native PPO sampler's old-policy logprob gate.
ATOL=2e-5
MLP='''function(layers,x)
 for index,layer in ipairs(layers) do
  local out={}
  for i,row in ipairs(layer.weight) do
   local value=layer.bias[i]
   for j,w in ipairs(row) do value=value+w*x[j] end
   if index<#layers then local e=math.exp(-2*math.abs(value));local y=(1-e)/(1+e);value=value<0 and -y or y end
   out[i]=value
  end
  x=out
 end
 return x
end'''

def layers(policy,critic=False):
    trunk=policy.mlp_extractor.value_net if critic else policy.mlp_extractor.policy_net
    head=policy.value_net if critic else policy.action_net
    return [{'weight':m.weight.detach().tolist(),'bias':m.bias.detach().tolist()} for m in [*list(trunk.children())[::2],head]]


def actual_observations(paths):
    out=[];sources=[]
    for path in paths:
        path=Path(path);record=json.loads(path.read_text());before=len(out)
        if record.get('training_only') is not True:raise ValueError('Only explicit training diagnostics are accepted')
        history=[]
        for row in record['decisions']:
            f=features(row['state'])
            if row['reset_history']:history=[f]*4
            else:history=history[1:]+[f]
            if len(history)!=4:raise ValueError('Missing diagnostic history reset')
            out.append(np.concatenate(history))
        sources.append({'sha256':sha256(path),'samples':len(out)-before})
    if not out:raise ValueError('Actual diagnostic observations required')
    return np.asarray(out,np.float32),sources


def lua_check(old,new,package,x):
    rng=np.random.default_rng(101);extra=rng.uniform(-1,1,(len(x),4,6)).astype(np.float32)
    y=expand_observations(x,extra);lua=LuaRuntime(unpack_returned_tuples=True)
    nn=lua.execute((package/'nn.lua').read_text());evaluate=lua.eval(MLP)
    actor_old=lua.table_from(layers(old.policy),recursive=True);actor_new=lua.table_from(layers(new.policy),recursive=True)
    critic_old=lua.table_from(layers(old.policy,True),recursive=True);critic_new=lua.table_from(layers(new.policy,True),recursive=True)
    maxima={'old_logits':0.,'new_logits':0.,'old_value':0.,'new_value':0.,'logprob':0.}
    old_outputs=[];new_outputs=[];old_values=[];new_values=[]
    for a,b in zip(x,y):
        ax=lua.table_from(a.tolist());bx=lua.table_from(b.tolist())
        old_outputs.append(list(evaluate(actor_old,ax).values()));new_outputs.append(list(evaluate(actor_new,bx).values()))
        old_values.append(list(evaluate(critic_old,ax).values()));new_values.append(list(evaluate(critic_new,bx).values()))
    for a,b in ((old_outputs,new_outputs),(old_values,new_values)):
        if not np.array_equal(a,b):raise RuntimeError('Lua zero-column migration changed exact arithmetic result')
    with torch.no_grad():
        old_t=old.policy.action_net(old.policy.mlp_extractor.policy_net(torch.from_numpy(x))).numpy()
        new_t=new.policy.action_net(new.policy.mlp_extractor.policy_net(torch.from_numpy(y))).numpy()
        old_v=old.policy.predict_values(torch.from_numpy(x)).numpy();new_v=new.policy.predict_values(torch.from_numpy(y)).numpy()
    for name,a,b in [('old_logits',old_outputs,old_t),('new_logits',new_outputs,new_t),('old_value',old_values,old_v),('new_value',new_values,new_v)]:
        maxima[name]=float(np.max(np.abs(np.asarray(a)-b)))
        if maxima[name]>ATOL or not np.isfinite(b).all():raise RuntimeError(f'Torch/Lua {name} mismatch: {maxima[name]}')
    if not np.array_equal(np.argmax(old_outputs,1),old_t.argmax(1)) or not np.array_equal(np.argmax(new_outputs,1),new_t.argmax(1)):
        raise RuntimeError('Torch/Lua argmax changed')
    # Exercise the actual candidate NN.predict identity and forward path too.
    payload={'schema':'astra.rl-policy.projectiles6.v1','observations':368,'actions':16,'action_interface':'ken_actions16_lp_mp_uppercut_v1',
             'observation_interface':'sf2_projectiles6_owner_reciprocal_v1','activation':'tanh','selection':'deterministic_argmax','layers':layers(new.policy)}
    model=lua.table_from(payload,recursive=True)
    for i,obs in enumerate(y):
        action,out=nn.predict(model,lua.table_from(obs.tolist()))
        if action!=int(new_t[i].argmax()) or list(out.values())!=new_outputs[i]:raise RuntimeError('Actual candidate NN.predict differs')
    z=np.asarray(new_outputs);z=z-z.max(1,keepdims=True);lua_logp=z-np.log(np.exp(z).sum(1,keepdims=True))
    torch_logp=torch.log_softmax(torch.from_numpy(new_t),dim=1).numpy()
    maxima['logprob']=float(np.abs(lua_logp-torch_logp).max())
    if maxima['logprob']>ATOL:raise RuntimeError('Native sampler logprob tolerance exceeded')
    return {'ok':True,'samples':len(x),'lua_old_new_exact':True,'argmax_equal':True,'atol':ATOL,'max_abs_difference':maxima}


def probe_check(paths,package):
    lua=LuaRuntime(unpack_returned_tuples=True);module=lua.execute((package/'projectile_features.lua').read_text())
    rows=0;positive=0;bytype={};sources=[]
    for path in paths:
        path=Path(path);data=json.loads(path.read_text())
        if data.get('training_only') is not True:raise ValueError('Only training probes are accepted')
        count=0
        for row in data['trace']:
            s=row['state'];objects=[dict(o,owner=o['owner_words'][0]) for o in s['projectiles']]
            links=[words[0] for words in s['projectile_pointer_words']]
            out,diag=module.encode(lua.table_from(s,recursive=True),lua.table_from(objects,recursive=True),lua.table_from(links))
            values=list(out.values());expected=[]
            for owner in (0,1):
                selected=[o for o in objects if o['status']==257 and o['hp']==256 and o['type'] in(0,1,3,4)
                          and o['owner']==(0x83c6,0x86c6)[owner] and links[owner]==o['base']%65536]
                selected.sort(key=lambda o:(abs(o['x']-s['p1']['x']),abs(o['y']-s['p1']['y']),o['slot']))
                if selected:
                    o=selected[0];expected.extend([1,np.clip((o['x']-s['p1']['x'])/512,-1,1),np.clip((o['y']-s['p1']['y'])/256,-1,1)])
                    bytype[str(o['type'])]=bytype.get(str(o['type']),0)+1;positive+=1
                else:expected.extend([0,0,0])
            if values!=expected:raise RuntimeError('Shared Lua classifier differs from independent raw-field oracle')
            count+=1;rows+=1
        sources.append({'sha256':sha256(path),'frames':count})
    return {'ok':True,'frames':rows,'positive_owner_frames':positive,'types':bytype,'sources':sources,
            'scope':'read-only replay of existing train probe telemetry; no MAME and no collision-semantics certification'}


def accept(model,package,output,diagnostics,probes=()):
    torch.set_num_threads(1);package=Path(package).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    report={'schema':'astra.rl-projectiles6-offline-acceptance.v1','status':'invalid','native_validated':False}
    try:
        x,sources=actual_observations(diagnostics);np.save(output/'observations.npy',x)
        report['migration']=migrate(model,output/'migration',observations=output/'observations.npy')
        old=PPO.load(model,device='cpu');new=PPO.load(output/'migration/ppo-projectiles6.zip',device='cpu')
        actual=x[np.linspace(0,len(x)-1,min(128,len(x)),dtype=int)]
        stress=np.random.default_rng(5).uniform(-1,1,(64,344)).astype(np.float32)
        sample=np.concatenate([actual,stress,np.zeros((1,344),np.float32),np.ones((1,344),np.float32),-np.ones((1,344),np.float32)])
        report['lua']=lua_check(old,new,package,sample);report['raw_probes']=probe_check(probes,package)
        report['actual_observation_sources']=sources;report['build_sha256']=sha256(package.parent/'build.json')
        sys.path.insert(0,str(package.parent))
        try:
            support=importlib.import_module(package.name+'.support');native=importlib.import_module(package.name+'.native_continuous')
            snapshot=support.capture_interface(output/'migration/ppo-projectiles6.zip',package)
            staged=output/'staged';staged.mkdir();hashes=native.stage_native_policy(staged,3,snapshot['payload'])
            support.check_staged_interface(snapshot,staged,hashes)
            lua=LuaRuntime()
            for path in (staged/'training/runtime').glob('*.lua'):lua.execute('assert(load(...))',path.read_text())
            report['staging']={'ok':True,'runtime_sha256':hashes}
        finally:sys.path.remove(str(package.parent))
        report['status']='offline_pass'
    except Exception as error:
        report['error']=f'{type(error).__name__}: {error}';atomic_json(output/'acceptance.json',report);raise
    atomic_json(output/'acceptance.json',report);return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','package','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--diagnostics',type=Path,nargs='+',required=True);p.add_argument('--probes',type=Path,nargs='*',default=[])
    print(json.dumps(accept(**vars(p.parse_args())),indent=2))
if __name__=='__main__':main()
