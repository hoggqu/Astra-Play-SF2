from copy import deepcopy
import unittest
import json
from pathlib import Path
import tempfile
from lupa.lua54 import LuaRuntime
from astra_play_sf2.runner import sha256
from .deterministic_train_eval import HERE,audit,runtime,select_states,ALL_OPPONENTS

class DeterministicTrainEvalTests(unittest.TestCase):
    def test_dev_selection_does_not_open_train_or_holdout(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);rows=[]
            for opponent in ALL_OPPONENTS:
                path=root/f'dev-{opponent}.sta';path.write_bytes(str(opponent).encode())
                rows.append({'opponent':opponent,'split':'dev','status':'accepted','difficulty':3,'path':path.name,'sha256':sha256(path)})
                for split in ('train','holdout'):
                    rows.append({'opponent':opponent,'split':split,'path':'must-not-open.sta'})
            manifest=root/'manifest.json';manifest.write_text(json.dumps({'status':'complete','difficulty':3,'openings':rows}))
            chosen=select_states(manifest,'dev',ALL_OPPONENTS,1)
            self.assertEqual([x['opponent'] for x in chosen],list(ALL_OPPONENTS))
            with self.assertRaises(ValueError):select_states(manifest,'holdout',ALL_OPPONENTS,1)
            with self.assertRaises(ValueError):select_states(manifest,'dev',ALL_OPPONENTS,2)
            (root/'dev-0.sta').write_bytes(b'changed')
            with self.assertRaises(ValueError):select_states(manifest,'dev',ALL_OPPONENTS,1)
    def evidence(self):
        state={'emulated_seconds':0.,'native_frame_period':.016768,'effective_difficulty':3,
               'p1':{'char':4,'wins':0},'p2':{'char':3,'wins':0}}
        rows=[]
        for n in range(1,6):
            s=deepcopy(state);s['emulated_seconds']=n*.016768;s['p1']['wins']=2
            events=[{'kind':'round_stop','round':1}] if n==1 else [{'kind':'round_start','round':2}] if n==3 else [{'kind':'round_stop','round':2}] if n==4 else []
            rows.append({'frame':n,'phase':'complete' if n==5 else 'settling','state':s,'events':events})
        return {'opening':state,'trace':rows,'initial_loads':1,'in_play_pauses':0,'terminal':{'valid':True,'result':'ken_win','score':[2,0]},
                'rounds':[{'outcome':'win'},{'outcome':'win'}],
                'decisions':[{'frame':0,'round':1,'reset_history':True},{'frame':3,'round':2,'reset_history':True}]}
    def test_generated_runtime_compiles_and_requires_complete_match(self):
        text=runtime((HERE/'chain_reference_runtime.lua').read_text());LuaRuntime().execute('assert(load(...))',text)
        self.assertIn("if core.phase=='complete' then",text)
        self.assertNotIn("or frames>=pending.frames",text)
        self.assertIn('pending.native_loads=pending.native_loads+1',text)
        self.assertNotIn('mem:write',text)
    def test_mature_whole_match_and_natural_second_round(self):
        result=audit(self.evidence(),3);self.assertEqual(result['outcome'],'win');self.assertEqual(result['round_outcomes'],['win','win'])
    def test_truncation_missing_action_extra_load_or_wrong_pips_invalid(self):
        for mutation in ('partial','decision','load','pips'):
            with self.subTest(mutation=mutation):
                r=self.evidence()
                if mutation=='partial':r['trace'][-1]['phase']='fighting'
                elif mutation=='decision':r['decisions'].pop()
                elif mutation=='load':r['initial_loads']=2
                else:r['terminal']['score']=[1,0]
                with self.assertRaises(RuntimeError):audit(r,3)
