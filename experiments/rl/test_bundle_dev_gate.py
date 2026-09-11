import copy,unittest
from pathlib import Path
from unittest.mock import patch
from experiments.rl import bundle_dev_gate as g
class BundleDevGateTests(unittest.TestCase):
 def pair(self):
  old={'opening':{},'state':{},'terminal':{},'rounds':[],'observed_frames':1,'initial_loads':1,'in_play_pauses':0,'trace':[{'frame':1,'state':{},'consumed_input':'L LP'}],'decisions':[{'frame':0,'action':6,'state':{},'round':1,'reset_history':True}]}
  actual=copy.deepcopy(old);actual['trace'][0]['state']['native_ports']=[65517,255];actual['decisions'][0].update(branch_id='op00',branch_source_model_sha256='a'*64)
  return actual,old
 def test_runtime_routes_and_reads_ports(self):
  text=g.runtime((Path(g.__file__).parent/'chain_reference_runtime.lua').read_text())
  self.assertIn('branch_source_model_sha256=hash',text);self.assertIn("ports[':IN1']:read()",text);self.assertNotIn('mem:write',text);self.assertNotIn("reason:match('rl_action_",text)
 def test_old_parity_new_ports_explicit_boundary(self):
  a,b=self.pair()
  with patch.object(g,'audit',return_value={'ok':True}):r=g.compare(a,b,0,'op00','a'*64)
  self.assertTrue(r['ok']);self.assertFalse(r['old_actual_ports_compared']);self.assertEqual(r['actual_port_frames_checked'],1)
 def test_native_state_input_and_decision_changes_rejected(self):
  for kind in ('state','input','action','branch','port','frames'):
   a,b=self.pair()
   if kind=='state':a['trace'][0]['state']['hp']=12
   elif kind=='input':a['trace'][0]['consumed_input']='R LP'
   elif kind=='action':a['decisions'][0]['action']=7
   elif kind=='branch':a['decisions'][0]['branch_id']='op01'
   elif kind=='port':a['trace'][0]['state']['native_ports']=[65535,255]
   else:a['trace'].append(a['trace'][0])
   with self.subTest(kind=kind),patch.object(g,'audit',return_value={'ok':True}),self.assertRaises(RuntimeError):g.compare(a,b,0,'op00','a'*64)
 def test_kick_release_masks(self):
  g.check_ports({'frame':1,'state':{'native_ports':[65531,253]},'consumed_input':'D MK'})
  g.check_ports({'frame':2,'state':{'native_ports':[65531,255]},'consumed_input':'D'})
  with self.assertRaises(RuntimeError):g.check_ports({'frame':1,'state':{'native_ports':[65535,255]},'consumed_input':'C'})
if __name__=='__main__':unittest.main()
