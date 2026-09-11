"""Pure stochastic inference and staged identity tests; no emulator required."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from lupa.lua54 import LuaRuntime

from .export import lua_literal
from .stochastic_continuous import next_random, stochastic_payload, stage_policy, audit_sampling

HERE=Path(__file__).resolve().parent


class StochasticTests(unittest.TestCase):
    def setUp(self):
        self.lua=LuaRuntime(unpack_returned_tuples=True)
        self.base=self.lua.execute((HERE/'nn.lua').read_text())
        self.sampling=self.lua.execute((HERE/'stochastic_nn.lua').read_text())(self.base)
        self.actions=self.lua.execute((HERE/'actions.lua').read_text())
        self.payload={'schema':'astra.rl-policy.v1','model_sha256':'test','observations':344,'actions':15,
                      'history':4,'decision_frames':12,'activation':'tanh','selection':'deterministic_argmax',
                      'layers':[{'weight':[[0.]*344 for _ in range(15)],'bias':[i*.1 for i in range(15)]}]}
        self.state={'p1':{'hp':144,'x':100,'y':40,'a':0,'char':4},
                    'p2':{'hp':144,'x':200,'y':40,'a':0,'char':2},'timer':99}

    def test_seed_stream_matches_python_and_does_not_use_global_rng(self):
        self.lua.execute('math.random=function() error("must not use shared RNG") end;math.randomseed=math.random')
        for seed in (1,42,2147483646):
            state=seed
            for _ in range(100):
                expected,u=next_random(state);actual,v=self.sampling.next_random(state)
                self.assertEqual(expected,actual);self.assertEqual(u,v);state=actual
        for seed in (0,-1,2147483647):
            with self.assertRaises(ValueError):next_random(seed)

    def test_softmax_actions_and_logprobs_match_torch(self):
        rng=np.random.default_rng(8);state=42
        for _ in range(100):
            logits=rng.normal(0,4,15);state,u=next_random(state)
            probs=torch.softmax(torch.tensor(logits),dim=0).numpy()
            expected=int(np.searchsorted(np.cumsum(probs),u,side='right'))
            action,logp=self.sampling.sample(self.lua.table_from(logits.tolist()),u)
            self.assertEqual(action,expected)
            self.assertAlmostEqual(logp,float(torch.log_softmax(torch.tensor(logits),dim=0)[action]),places=12)

    def test_round_and_match_history_resets_do_not_reset_sampling(self):
        payload=stochastic_payload(self.payload,42)
        policy=self.sampling.new(self.lua.execute('return '+lua_literal(payload)),self.actions)
        reasons=[];state=42
        for reset in (True,False,True,True):
            seq,reason=policy.choose(policy,self.lua.table_from(self.state,recursive=True),reset)
            self.assertEqual(len(seq),12);reasons.append(reason)
            state,_=next_random(state)
            self.assertEqual(policy.rng_state,state)
        self.assertEqual(policy.draws,4)
        self.assertIn('policy_draw=4;',reasons[-1])

    def test_mode_identity_must_be_explicit_and_original_export_unchanged(self):
        altered=stochastic_payload(self.payload,42)
        self.assertEqual(self.payload['selection'],'deterministic_argmax')
        self.assertEqual(altered['selection'],'categorical_softmax')
        with self.assertRaises(Exception):
            self.sampling.new(self.lua.execute('return '+lua_literal(self.payload)),self.actions)

    def test_staging_records_new_source_and_explicit_selection(self):
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/'run';payload=stochastic_payload(self.payload,42)
            sources=stage_policy(output,3,payload)
            self.assertIn('rl_stochastic_nn.lua',sources)
            source=(output/'training/runtime/play.lua').read_text()
            self.assertIn('selection=Model.selection',source)
            self.assertIn("'training/runtime/rl_nn.lua','training/runtime/rl_stochastic_nn.lua'",source)
            self.assertIn('categorical_softmax',(output/'training/runtime/rl_policy.lua').read_text())

    def test_sampling_audit_covers_sequence_across_attempts_and_rejects_reset(self):
        payload=stochastic_payload(self.payload,42)
        policy=self.sampling.new(self.lua.execute('return '+lua_literal(payload)),self.actions)
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder);result={'attempts':[]}
            for n in (1,2):
                _,reason=policy.choose(policy,self.lua.table_from(self.state,recursive=True),True)
                match={'summary':{k:payload[k] for k in ('selection','policy_seed','policy_prng')},
                       'trace':{'columns':['frame','timer'],'rows':[],
                                'decisions':[{'frame':0,'round':1,'reason':reason,'ken':self.state['p1'],'cpu':self.state['p2']}]},
                       'events':[{'kind':'match_start','state':self.state}]}
                name=f'match{n}.json';(output/name).write_text(json.dumps(match))
                result['attempts'].append({'matches':[name]})
            self.assertEqual(audit_sampling(output,result,payload)['decisions'],2)
            match['trace']['decisions'][0]['reason']=match['trace']['decisions'][0]['reason'].replace('policy_draw=2','policy_draw=1')
            (output/'match2.json').write_text(json.dumps(match))
            with self.assertRaisesRegex(RuntimeError,'stream reset'):
                audit_sampling(output,result,payload)


if __name__=='__main__':unittest.main()
