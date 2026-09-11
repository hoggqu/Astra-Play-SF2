"""Offline fixed routing, complete bundle integrity and native staging tests."""
import copy,importlib,io,json,os
from unittest.mock import patch
from .export import lua_literal
from pathlib import Path
import shutil,sys,tempfile,unittest,zipfile
import numpy as np
import torch
from stable_baselines3 import PPO
from lupa.lua54 import LuaRuntime
from astra_play_sf2.runner import sha256
from .actions16_builder import build as action_build,PACKAGE as AP
from .round_chain_builder import build as chain_build,PACKAGE
from .pulsed_normals_builder import build as pulse_build
from .opponent_bundle_builder import build
from .opponent_bundle import build_bundle,export_bundle,OPS,encoded,digest
from .opponent_bundle_identity import audit_branches
from .width_migrate import SpacesOnly

class BundleTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  torch.set_num_threads(1);cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name)
  action_build(cls.root/'a');chain_build(cls.root/'c',cls.root/'a'/AP)
  shutil.copytree(Path(__file__).resolve().parents[2]/'src',cls.root/'c/src',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
  pulse_build(cls.root/'p',cls.root/'c'/PACKAGE);cls.parent=cls.root/'p'/PACKAGE
  cls.m=build(cls.root/'code',cls.parent);cls.package=cls.root/'code'/PACKAGE
  cls.models={};cls.policies={}
  for i in range(2):
   model=PPO('MlpPolicy',SpacesOnly(),n_steps=256,batch_size=64,seed=i,policy_kwargs={'net_arch':{'pi':[64,64],'vf':[64,64]}})
   model.astra_action_interface='ken_actions16_pulsed_normals_v2'
   with torch.no_grad():model.policy.action_net.bias[i]+=2
   p=cls.root/f'm{i}.zip';model.save(p);cls.models[str(i)]=p;cls.policies[str(i)]=model
  cls.route={op:str(op%2) for op in OPS};cls.bundle=cls.root/'mixed.zip'
  build_bundle(cls.bundle,cls.models,cls.route,{'kind':'test_only'})
  cls.payload=export_bundle(cls.bundle)
  build_bundle(cls.root/'baseline.zip',cls.models,{op:'0' for op in OPS},{'kind':'test_only'})
  sys.path.insert(0,str(cls.root/'code'))
  cls.native=importlib.import_module(PACKAGE+'.native_continuous');cls.support=importlib.import_module(PACKAGE+'.support')
  cls.identity=importlib.import_module(PACKAGE+'.opponent_bundle_identity')
  cls.lua=LuaRuntime(unpack_returned_tuples=True);base=cls.lua.execute((cls.package/'bundle_base_nn.lua').read_text());cls.lua.globals().Base=base
  cls.lua.execute("loadfile=function(p) assert(p=='training/runtime/rl_bundle_base_nn.lua');return function() return Base end end")
  cls.nn=cls.lua.execute((cls.package/'nn.lua').read_text())
 @classmethod
 def tearDownClass(cls):
  sys.path.remove(str(cls.root/'code'))
  for name in list(sys.modules):
   if name==PACKAGE or name.startswith(PACKAGE+'.'):del sys.modules[name]
  cls.temp.cleanup()
 def obs(self,op,seed):
  a=np.random.default_rng(seed).uniform(-1,1,(4,86)).astype(np.float32)
  a[:,10:22]=0;a[:,10+op]=1;return a.reshape(-1)
 def test_all_opponent_torch_lua_and_baseline_equivalence(self):
  for payload,route in [(self.payload,self.route),(export_bundle(self.root/'baseline.zip'),{o:'0' for o in OPS})]:
   lua_model=self.lua.execute('return '+lua_literal(payload))
   for op in OPS:
    obs=self.obs(op,op);model=self.policies[route[op]]
    with torch.no_grad():logits=model.policy.action_net(model.policy.mlp_extractor.forward_actor(torch.from_numpy(obs[None,:])))[0].numpy()
    action,actual,bid,h=self.nn.predict(lua_model,self.lua.table_from(obs.tolist()))
    self.assertEqual(action,int(logits.argmax()));np.testing.assert_allclose([actual[i] for i in range(1,17)],logits,atol=2e-6)
    self.assertEqual(bid,f'op{op:02d}');self.assertEqual(h,sha256(self.models[route[op]]))
 def test_unknown_ambiguous_opponents_and_wrong_macro_table_rejected(self):
  model=self.lua.execute('return '+lua_literal(self.payload))
  for op in [4]:
   with self.assertRaises(Exception):self.nn.predict(model,self.lua.table_from(self.obs(op,1).tolist()))
  bad=self.obs(0,2);bad[3*86+11]=1
  with self.assertRaises(Exception):self.nn.predict(model,self.lua.table_from(bad.tolist()))
  with self.assertRaises(Exception):self.nn.new(model,self.lua.table_from({'count':16,'frames':11,'interface':'ken_actions16_pulsed_normals_v2'}))
 def test_bundle_route_weights_and_actor_tamper_rejected(self):
  original={}
  with zipfile.ZipFile(self.bundle) as z:original={n:z.read(n) for n in z.namelist()}
  for kind in ['route','weights','actor']:
   files=dict(original);m=json.loads(files['manifest.json'])
   if kind=='route':m['route']['0']='op01';files['manifest.json']=encoded(m)
   elif kind=='weights':files['branches/op00/model.zip']+=b'tamper'
   else:
    n='branches/op00/actor.json';a=json.loads(files[n]);a['layers'][-1]['bias'][0]+=1;files[n]=encoded(a);m['branches']['op00']['actor_sha256']=digest(files[n]);files['manifest.json']=encoded(m)
   p=self.root/f'tamper-{kind}.zip'
   with zipfile.ZipFile(p,'w') as z:
    for n,b in files.items():z.writestr(n,b)
   with self.assertRaises(ValueError):export_bundle(p)
 def test_source_identity_and_native_staging_keep_audits(self):
  self.identity.validate_bundle_build(self.m,self.package)
  for name in ['actions.lua','native_continuous_core.lua','settlement.lua']:
   self.assertEqual((self.package/name).read_bytes(),(self.parent/name).read_bytes())
  run=self.root/'staged';run.mkdir()
  snap=self.support.capture_interface(self.bundle,self.package)
  hashes=self.native.stage_native_policy(run,3,self.payload);self.support.check_staged_interface(snap,run,hashes)
  play=(run/'training/runtime/play.lua').read_text()
  self.assertIn("policy_kind='ppo_opponent_bundle'",play);self.assertIn('branch_source_model_sha256',play)
  self.assertIn('native_period_checks',play);self.assertIn('rl_bundle_base_nn.lua',play)
  self.assertIn('pauses_during_match',play)
  b=(self.package/'bundle_base_nn.lua').read_bytes()
  try:
   (self.package/'bundle_base_nn.lua').write_bytes(b+b'\n')
   with self.assertRaises(RuntimeError):self.identity.validate_bundle_build(self.m,self.package)
  finally:(self.package/'bundle_base_nn.lua').write_bytes(b)
 def test_failed_branch_audit_marks_wrapper_invalid(self):
  run=self.root/'failed-wrapper'
  def evaluator(**kw):return {'status':'complete','attempts':[]}
  ns={'HERE':self.package,'stage_native_policy':lambda *a:None}
  with patch.object(self.support,'audit_interface',return_value={'ok':True}),patch.object(self.identity,'audit_branches',side_effect=RuntimeError('wrong branch')):
   r=self.support.interface_evaluate(evaluator,ns,(),{'model':self.bundle,'output':run})
  self.assertEqual(r['status'],'invalid');self.assertFalse(r['bundle_audit']['ok'])
  self.assertFalse(r['action_interface_audit']['ok']);self.assertIn('wrong branch',r['error'])
 def test_match_and_each_decision_branch_identity_audited(self):
  p=self.payload;op=0;bid=p['route']['0'];h=p['branches'][bid]['source_model_sha256'];run=self.root/'fake-match';run.mkdir()
  match={'summary':{'opponent':op,'branch_id':bid,'branch_source_model_sha256':h,'bundle_route_sha256':p['route_sha256'],'model_sha256':p['model_sha256'],'policy_kind':'ppo_opponent_bundle'},'trace':{'decisions':[{'cpu':{'char':op},'reason':f'rl_bundle_{bid}_{h}_action_0'}]}}
  result={'schema':'astra.rl-continuous.opponent-bundle.v1','policy_kind':'ppo_opponent_bundle','bundle_route_sha256':p['route_sha256'],'model_sha256':p['model_sha256'],'branch_model_sha256':{k:b['source_model_sha256'] for k,b in p['branches'].items()},'attempts':[{'matches':['m.json']}]}
  path=run/'m.json';path.write_text(json.dumps(match));self.assertTrue(audit_branches(p,run,result)['ok'])
  for key in ['branch_id','branch_source_model_sha256']:
   bad=copy.deepcopy(match);bad['summary'][key]='wrong';path.write_text(json.dumps(bad))
   with self.assertRaises(RuntimeError):audit_branches(p,run,result)
  match['trace']['decisions'][0]['reason']='rl_action_0';path.write_text(json.dumps(match))
  with self.assertRaises(RuntimeError):audit_branches(p,run,result)

if __name__=='__main__':unittest.main()
