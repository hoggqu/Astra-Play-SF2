"""Predeclared route ranking and evidence-separation tests (no MAME)."""
import copy
import unittest
from .opponent_bundle_select import ORDER,OPS,dev_selection,formal_selection,wilson_lower

class SelectionTests(unittest.TestCase):
 def dev(self):
  d={'status':'complete','split':'dev','holdout_opened':False,'selection':'deterministic_argmax','matches_requested':22,'dataset_sha256':'dataset','cases':[]}
  for op in OPS:
   for lead in (0,12):d['cases'].append({'status':'complete','opponent':op,'checkpoint_sha256':str(op),'lead':lead,'audit':{'ok':True,'outcome':'win','round_outcomes':['win','win']}})
  return {n:copy.deepcopy(d) for n in ORDER}
 def test_dev_aligned_tie_and_no_formal_use(self):
  d=self.dev();route,p=dev_selection(d)
  self.assertEqual(set(route.values()),{'f26'});self.assertFalse(p['formal_data_used']);self.assertFalse(p['independent_win_rate'])
  d['f26']['cases'][0]['audit']['outcome']='loss'
  with self.assertRaises(ValueError):dev_selection(d)
  d['f26']['cases'][0]['audit']['round_outcomes']=['loss','loss'];route,p=dev_selection(d);self.assertEqual(route[0],'local-c2')
  d['remote85']['cases'][0]['lead']=99
  with self.assertRaises(ValueError):dev_selection(d)
 def test_wilson_draw_zero_and_predeclared_tie(self):
  self.assertEqual(wilson_lower(0,0,0),-1);self.assertAlmostEqual(wilson_lower(3,0,2),wilson_lower(3,2,0))
  self.assertGreater(wilson_lower(20,2,0),wilson_lower(2,0,0))
  d={n:{'status':'complete','attempts':20,'invalid':0,'rounds':{str(op):{'win':1,'loss':0,'draw':0} for op in OPS}} for n in ORDER}
  del d['f26']['rounds']['8'];route,p=formal_selection(d)
  self.assertEqual(route[0],'f26');self.assertEqual(route[8],'local-c2')
  self.assertTrue(p['formal_data_used']);self.assertFalse(p['confidence_guarantee']);self.assertFalse(p['independent_win_rate'])
  d['f26']['attempts']=19
  with self.assertRaises(ValueError):formal_selection(d)
 def test_negative_or_fractional_counts_rejected(self):
  for w,l,d in [(-1,0,0),(1,2.5,0),(True,0,0)]:
   with self.assertRaises(ValueError):wilson_lower(w,l,d)
if __name__=='__main__':unittest.main()
