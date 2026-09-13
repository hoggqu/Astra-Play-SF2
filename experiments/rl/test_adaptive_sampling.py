import copy
import json
import unittest
from .adaptive_sampling import OpponentSampler, probability_map, WINDOW

OPS = [0,1,2,3,5,6,7,8,9,10,11]


def chunk(op, outcomes, length=10):
    return dict(transitions=[{'state':{'p2':{'char':op}}} for _ in range(len(outcomes)*length)],
                episodes=[dict(opponent=op,outcome=o,steps=length) for o in outcomes])


class AdaptiveSamplingTests(unittest.TestCase):
    def test_weak_opponents_gain_probability_but_all_keep_coverage(self):
        s=OpponentSampler(OPS)
        for _ in range(25):
            s.observe([chunk(o,['loss' if o in (0,8,11) else 'win']*8) for o in OPS])
        self.assertAlmostEqual(sum(s.probabilities.values()),1.)
        for o in OPS:
            self.assertGreaterEqual(s.probabilities[str(o)],.5/11)
            self.assertLessEqual(s.probabilities[str(o)],.25)
        self.assertGreater(s.probabilities['11'],s.probabilities['1'])
        self.assertGreater(s.probabilities['11'],1/11)

    def test_single_hard_opponent_cannot_take_all_sampling(self):
        s=OpponentSampler(OPS)
        for _ in range(100):
            s.observe([chunk(o,['loss' if o==11 else 'win']*8) for o in OPS])
        self.assertAlmostEqual(s.probabilities['11'],.25,places=10)
        self.assertAlmostEqual(sum(s.probabilities.values()),1.)
        for o in OPS[:-1]:self.assertGreater(s.probabilities[str(o)],.5/11)

    def test_uniform_control_and_small_pools(self):
        s=OpponentSampler(OPS,'uniform');s.observe([chunk(11,['loss']*128)])
        self.assertTrue(all(p==1/11 for p in s.probabilities.values()))
        for ops in ([0],[0,8],[0,8,11]):
            s=OpponentSampler(ops);s.observe([chunk(0,['loss'])])
            self.assertAlmostEqual(sum(s.probabilities.values()),1.)

    def test_win_history_ages_out_and_improvement_reduces_extra_practice(self):
        s=OpponentSampler(OPS)
        for _ in range(20):s.observe([chunk(o,['loss' if o==11 else 'win']*8) for o in OPS])
        before=s.probabilities['11']
        for _ in range(20):s.observe([chunk(o,['win' if o==11 else 'loss']*8) for o in OPS])
        self.assertLess(s.probabilities['11'],before)
        self.assertTrue(all(len(v)==WINDOW for v in s.history.values()))

    def test_checkpoint_resume_preserves_window_and_probabilities_not_stage_counts(self):
        a=OpponentSampler(OPS);a.observe([chunk(0,['win','loss']),chunk(11,['loss'])])
        b=OpponentSampler(OPS,state=json.loads(json.dumps(a.state())))
        self.assertTrue(b.restored);self.assertEqual(a.state(),b.state())
        self.assertEqual(b.snapshot()['decisions'],0)
        future=[chunk(8,['win','draw']),chunk(11,['loss'])]
        a.observe(future);b.observe(future);self.assertEqual(a.state(),b.state())

    def test_exposure_counts_short_fights_and_unfinished_round_decisions(self):
        s=OpponentSampler(OPS)
        partial=dict(transitions=[{'state':{'p2':{'char':0}}}]*3,episodes=[])
        s.observe([chunk(8,['loss'],length=5),chunk(1,['win'],length=20),partial])
        snap=s.snapshot();self.assertEqual(snap['decisions'],28)
        self.assertEqual(snap['per_opponent']['8']['mean_round_decisions'],5)
        self.assertEqual(snap['per_opponent']['0']['rounds'],0)
        self.assertAlmostEqual(snap['per_opponent']['8']['decision_share'],5/28)

    def test_invalid_probabilities_outcomes_or_checkpoint_fail_closed(self):
        s=OpponentSampler(OPS)
        for value in (0,float('nan'),2):
            p=dict(s.probabilities);p['0']=value
            with self.assertRaises(ValueError):probability_map(OPS,p)
        bad=copy.deepcopy(s.state());bad['opponents']=[0]
        with self.assertRaises(ValueError):OpponentSampler(OPS,state=bad)
        with self.assertRaises(ValueError):s.observe([chunk(0,['invalid'])])


if __name__=='__main__':unittest.main()
