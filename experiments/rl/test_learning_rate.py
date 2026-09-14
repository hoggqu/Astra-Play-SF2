from copy import deepcopy
from pathlib import Path
import argparse
import tempfile
import unittest

from .managed_runtime import positive_learning_rate, configure_learning_rate
from .versioned_campaign import evaluate_result
from .lr_comparison import parser, training_args


class LearningRateTests(unittest.TestCase):
    def test_override_preserves_loaded_adam_and_survives_schedule_and_save(self):
        import gymnasium as gym
        import torch
        from stable_baselines3 import PPO
        torch.set_num_threads(1)
        env=gym.make('CartPole-v1')
        self.addCleanup(env.close)
        model=PPO('MlpPolicy',env,n_steps=8,batch_size=4,n_epochs=1,learning_rate=3e-4,
                  policy_kwargs={'net_arch':[8,8]},seed=42,device='cpu')
        model.learn(16)
        with tempfile.TemporaryDirectory() as folder:
            original=Path(folder)/'initial.zip';model.save(original)
            model=PPO.load(original,env=env,device='cpu')
            before=deepcopy(model.policy.optimizer.state_dict());updates=model._n_updates
            weights=deepcopy(model.policy.state_dict())
            info=configure_learning_rate(model,1e-4)
            self.assertEqual(info['previous'],3e-4)
            self.assertEqual(model._n_updates,updates)
            for key,value in weights.items():torch.testing.assert_close(value,model.policy.state_dict()[key],rtol=0,atol=0)
            for key,state in before['state'].items():
                for name,value in state.items():torch.testing.assert_close(value,model.policy.optimizer.state_dict()['state'][key][name],rtol=0,atol=0)
            for progress in (1.0,0.5,0.0):self.assertEqual(model.lr_schedule(progress),1e-4)
            from stable_baselines3.common.logger import configure
            model.set_logger(configure(folder, []))
            model._update_learning_rate(model.policy.optimizer)
            self.assertTrue(all(g['lr']==1e-4 for g in model.policy.optimizer.param_groups))
            modified=Path(folder)/'modified.zip';model.save(modified)
            loaded=PPO.load(modified,env=env,device='cpu')
            self.assertEqual(loaded.learning_rate,1e-4)
            self.assertEqual(loaded._n_updates,updates)
            self.assertEqual(configure_learning_rate(loaded,None)['effective'],1e-4)
            loaded.learn(8,reset_num_timesteps=False)
            self.assertGreater(loaded._n_updates,updates)
            self.assertTrue(all(g['lr']==1e-4 for g in loaded.policy.optimizer.param_groups))

    def test_invalid_rates_and_full_evaluation_contract(self):
        for value in ('nan','inf','-1','0','bad'):
            with self.assertRaises(argparse.ArgumentTypeError):positive_learning_rate(value)
        identity={'action_interface':'a','observation_interface':'o'}
        row=dict(outcome='rl_gameplay_clear',audit={'ok':True},match_wins=11,matches=['m']*11)
        value=dict(status='complete',difficulty=3,model_sha256='x',**identity,native_timing=True,
                   native_timing_audit={'ok':True},action_interface_audit={'ok':True},attempts=[row,row])
        self.assertTrue(evaluate_result(value,0,'x',2,identity,all_attempts=True))
        with self.assertRaises(RuntimeError):evaluate_result(value,0,'x',2,identity)
        with self.assertRaises(RuntimeError):evaluate_result(value,0,'x',3,identity,all_attempts=True)
        value['attempts']=[dict(row,audit={'ok':False})]
        with self.assertRaises(RuntimeError):evaluate_result(value,0,'x',1,identity,all_attempts=True)

    def test_comparison_arms_only_change_lr_and_always_run_full_evaluations(self):
        def arm(lr):
            args=parser().parse_args(['--checkpoint','model.zip','--checkpoint-sha256','x',
                    '--dataset','manifest.json','--dataset-sha256','y','--learning-rate',lr])
            self.assertEqual((args.rounds,args.workers,args.final_attempts),(10,12,20))
            return training_args(args,Path('/tmp/output'),Path('/tmp/model.zip'))
        high,low=arm('3e-4'),arm('1e-4')
        self.assertEqual(high.learning_rate,3e-4);self.assertEqual(low.learning_rate,1e-4)
        self.assertTrue(high.all_attempts);self.assertFalse(high.stop_on_clear)
        a,b=vars(high).copy(),vars(low).copy();a.pop('learning_rate');b.pop('learning_rate')
        self.assertEqual(a,b)
        self.assertEqual((high.rollout_steps,high.minibatch_size,high.steps_per_round),(16384,256,409600))


if __name__=='__main__':unittest.main()
