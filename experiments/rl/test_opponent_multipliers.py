import json
from pathlib import Path
import unittest
from unittest.mock import patch
from .adaptive_sampling import OpponentSampler as Base
from .weighted_sampling import OpponentSampler, parse_multipliers
from .test_adaptive_sampling import OPS, chunk
from .sampling_comparison import arm_arguments
from .lr_comparison import training_args


class WeightedSamplingTests(unittest.TestCase):
    def test_control_exact_and_multiplier_does_not_change_adaptive_state(self):
        base=Base(OPS)
        a=OpponentSampler(OPS,state=base.state(),multipliers={})
        b=OpponentSampler(OPS,state=base.state(),multipliers={'9':2})
        for _ in range(20):
            self.assertEqual(a.probabilities,base.probabilities)
            p=base.probabilities['9']
            self.assertAlmostEqual(b.probabilities['9'],2*p/(1+p))
            self.assertAlmostEqual(sum(b.probabilities.values()),1)
            for k,v in base.probabilities.items():
                if k!='9':self.assertAlmostEqual(b.probabilities[k],v/(1+p))
                self.assertGreater(b.probabilities[k],0)
            data=[chunk(o,['loss' if o==9 else 'win']*8) for o in OPS]
            for obj in (base,a,b):obj.observe(data)
            self.assertEqual(b.state()['base'],base.state())
            self.assertEqual(a.state()['base'],base.state())

    def test_resume_no_compounding_explicit_override_and_environment(self):
        a=OpponentSampler(OPS,multipliers={'9':2});a.observe([chunk(9,['loss']*8)])
        for _ in range(4):
            with patch.dict('os.environ',{'ASTRA_RL_OPPONENT_MULTIPLIERS':''}):
                b=OpponentSampler(OPS,state=json.loads(json.dumps(a.state())))
            self.assertEqual(a.state(),b.state());self.assertEqual(a.probabilities,b.probabilities);a=b
        with patch.dict('os.environ',{'ASTRA_RL_OPPONENT_MULTIPLIERS':'{}'}):
            control=OpponentSampler(OPS,state=a.state())
        self.assertEqual(control.probabilities,a.base.probabilities)
        with patch.dict('os.environ',{'ASTRA_RL_OPPONENT_MULTIPLIERS':'{"9":2}'}):
            self.assertEqual(OpponentSampler(OPS,state=control.state()).probabilities,a.probabilities)
        snap=a.snapshot()['per_opponent']['9']
        self.assertEqual(snap['next_probability'],a.probabilities['9'])
        self.assertEqual(snap['base_probability'],a.base.probabilities['9'])

    def test_invalid_and_absent_opponents_rejected(self):
        for value in ('[]','{"4":2}','{"9":0}','{"9":true}','{"9":NaN}','{"9":101}'):
            with self.assertRaises(ValueError):parse_multipliers(value)
        with self.assertRaises(ValueError):OpponentSampler([0],multipliers={'9':2})

    def test_single_arm_and_only_multiplier_changes_training_configuration(self):
        settings=dict(model='model.zip',model_sha256='x',dataset='manifest.json',dataset_sha256='y',device='cpu',config='config.json')
        a=arm_arguments(dict(settings,arm='A'),Path('/tmp/example'))
        b=arm_arguments(dict(settings,arm='B'),Path('/tmp/example'))
        self.assertEqual((a.rounds,a.steps_per_round,a.final_attempts),(2,409600,40))
        aa=vars(training_args(a,Path('/tmp/output'),Path('/tmp/model.zip')))
        bb=vars(training_args(b,Path('/tmp/output'),Path('/tmp/model.zip')))
        self.assertEqual(json.loads(aa.pop('opponent_multipliers')), {})
        self.assertEqual(json.loads(bb.pop('opponent_multipliers')), {'9':2})
        self.assertEqual(aa,bb)
        for invalid in (dict(settings,arm=['A','B']),dict(settings,arm='A',order=['A','B'])):
            with self.assertRaises(ValueError):arm_arguments(invalid,Path('/tmp/example'))
