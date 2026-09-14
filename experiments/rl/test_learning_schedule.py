import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from .learning_schedule import configure, rate_at, prepare_update, complete_update
from .schedule_comparison import arm_arguments
from .lr_comparison import training_args
from .autotrain import schedule_environment

PLAN=dict(kind='cosine',start=1e-4,end=3e-5,total_steps=64)


class ScheduleTests(unittest.TestCase):
    def test_endpoints_monotone_floor_and_invalid(self):
        self.assertAlmostEqual(rate_at(PLAN,0),1e-4)
        self.assertAlmostEqual(rate_at(PLAN,32),6.5e-5)
        self.assertEqual(rate_at(PLAN,64),3e-5)
        self.assertEqual(rate_at(PLAN,128),3e-5)
        values=[rate_at(PLAN,i) for i in range(65)]
        self.assertEqual(values,sorted(values,reverse=True))
        for plan in (dict(PLAN,end=2e-4),dict(PLAN,total_steps=True),dict(PLAN,start=float('nan'))):
            with self.assertRaises(ValueError):rate_at(plan,0)

    def test_actual_ppo_update_save_reload_and_resume_equivalence(self):
        import numpy as np
        import torch
        from stable_baselines3 import PPO
        from stable_baselines3.common.logger import configure as logger
        torch.set_num_threads(1)
        model=PPO('MlpPolicy','CartPole-v1',n_steps=16,batch_size=16,n_epochs=1,learning_rate=1e-4,seed=42)
        model.set_logger(logger(folder=None,format_strings=[]))
        configure(model,PLAN)
        def update(m):
            # Exercise actual PPO.train(), including its own schedule overwrite.
            m.rollout_buffer.reset()
            for _ in range(16):
                m.rollout_buffer.add(np.zeros((1,4),dtype=np.float32),np.array([0]),np.array([1.]),np.array([False]),torch.tensor([0.]),torch.tensor([-.69]))
            m.rollout_buffer.compute_returns_and_advantage(torch.tensor([0.]),np.array([False]))
            m._current_progress_remaining=0.  # Per-cycle progress must be irrelevant.
            value=prepare_update(m,16)
            np.random.seed(123)
            m.train()
            self.assertEqual(m.policy.optimizer.param_groups[0]['lr'],value)
            complete_update(m,16)
            return value
        first=update(model)
        optimizer=copy.deepcopy(model.policy.optimizer.state_dict())
        with tempfile.TemporaryDirectory() as tmp:
            file=Path(tmp)/'model.zip';model.save(file)
            resumed=PPO.load(file);resumed.set_logger(logger(folder=None,format_strings=[]))
            with patch.dict(os.environ,{'ASTRA_RL_LR_SCHEDULE':''}):
                configure(resumed)
            self.assertEqual(resumed.astra_learning_schedule['completed_steps'],16)
            self.assertEqual(len(resumed.policy.optimizer.state),len(optimizer['state']))
            for _ in range(3):
                self.assertEqual(update(model),update(resumed))
            self.assertEqual(resumed.astra_learning_schedule['completed_steps'],64)
            for a,b in zip(model.policy.parameters(),resumed.policy.parameters()):
                self.assertTrue(torch.equal(a,b))
            # Reapplying same plan across subprocess cycles cannot reset progress.
            configure(resumed,PLAN,learning_rate_override=1e-4)
            self.assertEqual(resumed.astra_learning_schedule['completed_steps'],64)
            self.assertLess(prepare_update(resumed,16),first)
            configure(resumed,{})
            self.assertIsNone(resumed.astra_learning_schedule)

    def test_ab_only_schedule_differs_and_plan_spans_cycles(self):
        settings=dict(model='model.zip',model_sha256='x',dataset='manifest.json',dataset_sha256='y',device='mps',config='config.json')
        a=training_args(arm_arguments(dict(settings,arm='A'),'/tmp/example'),Path('/tmp/out'),Path('/tmp/model.zip'))
        b=training_args(arm_arguments(dict(settings,arm='B'),'/tmp/example'),Path('/tmp/out'),Path('/tmp/model.zip'))
        import json
        plan=json.loads(schedule_environment(b)['ASTRA_RL_LR_SCHEDULE'])
        self.assertEqual(plan['total_steps'],819200)
        aa=vars(a).copy();bb=vars(b).copy()
        for key in ('lr_schedule','lr_end'):aa.pop(key);bb.pop(key)
        self.assertEqual(aa,bb)
        with self.assertRaises(ValueError):arm_arguments(dict(settings,arm='A',order=['A','B']),'/tmp/example')


if __name__=='__main__':unittest.main()
