import unittest
from types import SimpleNamespace
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer
from .rollout_control import quotas, validate_sizes, configure, fill_buffer, collect


class Policy:
    def evaluate_actions(self,obs,actions):
        return obs[:,0:1],torch.zeros(len(obs)),None
    def predict_values(self,obs):
        return obs[:,0:1]


def model(workers,total=8,mini=4):
    m=SimpleNamespace(n_envs=workers,observation_space=spaces.Box(-100,100,(2,),dtype=np.float32),
        action_space=spaces.Discrete(2),device='cpu',gamma=1.,gae_lambda=1.,policy=Policy())
    configure(m,total,mini)
    return m


def chunk(dones,starts,tail):
    rows=[dict(observation=[0.,0.],action=0,reward=1.,logprob=0.,done=d,episode_start=s)
          for d,s in zip(dones,starts)]
    return dict(transitions=rows,observation=[tail,0.],episode_start=dones[-1])


class RolloutTests(unittest.TestCase):
    def test_exact_quotas_rotate_remainder_for_arbitrary_workers(self):
        for workers in (1,3,6,11,20,40):
            totals=np.zeros(workers,dtype=int)
            for i in range(workers):
                q=quotas(4096,workers,i)
                self.assertEqual(sum(q),4096);self.assertLessEqual(max(q)-min(q),1)
                totals+=q
            self.assertTrue(np.all(totals==4096))
        for args in ((4096,3,20),(32,64,2),(8,4,9),(8,1,2)):
            with self.assertRaises(ValueError):validate_sizes(*args)

    def test_unequal_trajectories_bootstrap_separately_and_stop_at_terminal(self):
        m=model(3)
        chunks=[chunk([False]*3,[True,False,False],10.),
                chunk([False,True,False],[True,False,True],10.),
                chunk([False,True],[True,False],100.)]
        self.assertEqual(fill_buffer(m,chunks),0.)
        np.testing.assert_array_equal(m.rollout_buffer.returns[:,0],[13,12,11,2,1,11,2,1])
        batches=list(m.rollout_buffer.get(m.batch_size))
        self.assertEqual([len(b.actions) for b in batches],[4,4])
        chunks[0]['transitions'][1]['episode_start']=True
        with self.assertRaisesRegex(RuntimeError,'Terminal'):fill_buffer(m,chunks)

    def test_equal_quotas_match_original_sb3_vector_gae(self):
        m=model(2);m.gamma=.99;m.gae_lambda=.95
        chunks=[chunk([False,True,False,False],[True,False,True,False],5.),
                chunk([False,False,False,True],[True,False,False,False],99.)]
        old=RolloutBuffer(4,m.observation_space,m.action_space,device='cpu',n_envs=2,gamma=m.gamma,gae_lambda=m.gae_lambda)
        for i in range(4):
            old.add(np.zeros((2,2)),np.zeros(2),np.ones(2),
                    np.asarray([c['transitions'][i]['episode_start'] for c in chunks]),torch.zeros(2),torch.zeros(2))
        old.compute_returns_and_advantage(torch.tensor([[5.],[99.]]),np.array([False,True]))
        fill_buffer(m,chunks)
        np.testing.assert_array_equal(m.rollout_buffer.advantages[:,0],old.advantages.T.reshape(-1))
        np.testing.assert_array_equal(m.rollout_buffer.returns[:,0],old.returns.T.reshape(-1))

    def test_collector_uses_exact_budget_without_padding_or_extra_actions(self):
        class Remote:
            def __init__(self):self.requests=[]
            def send(self,request):self.requests.append(request);self.request=request
            def recv(self):
                _,(_,args,_)=self.request;n,payload=args
                return dict(model_sha256='abc',transitions=[{}]*n,episodes=[],
                            observation=[0,0],episode_start=False,chain_metrics={})
        vec=SimpleNamespace(num_envs=3,remotes=[Remote() for _ in range(3)],waiting=False)
        got=collect(vec,8,2,{'model_sha256':'abc'},0)
        self.assertEqual([len(c['transitions']) for c in got],[3,3,2])
        for remote in vec.remotes:
            self.assertIsNotNone(remote.requests[0][1][1][1])
            for r in remote.requests[1:]:self.assertIsNone(r[1][1][1])


if __name__=='__main__':unittest.main()
