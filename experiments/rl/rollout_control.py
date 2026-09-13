"""Exact global PPO rollout budgets with independent worker and minibatch sizes."""
PROTOCOL = 'exact_global_rollout_per_worker_gae_v1'


def validate_sizes(total, minibatch, workers):
    if any(type(v) is not int or v < 1 for v in (total, minibatch, workers)):
        raise ValueError('Rollout, minibatch and workers must be positive integers')
    if minibatch < 2 or total < minibatch or total % minibatch:
        raise ValueError('rollout-steps must be a multiple of minibatch-size, which must be at least 2')
    if workers > total:
        raise ValueError('workers cannot exceed rollout-steps')


def quotas(total, workers, iteration=0):
    if total < workers or workers < 1:
        raise ValueError('Every worker must receive at least one decision')
    base, extra = divmod(total, workers)
    return [base + int((i-iteration) % workers < extra) for i in range(workers)]


def configure(model, total, minibatch):
    from stable_baselines3.common.buffers import RolloutBuffer
    validate_sizes(total, minibatch, model.n_envs)
    model.n_steps = (total + model.n_envs - 1) // model.n_envs
    model.batch_size = minibatch
    model.astra_rollout_steps = total
    model.astra_rollout_protocol = PROTOCOL
    # Flatten only after computing GAE separately on each real environment.
    model.rollout_buffer = RolloutBuffer(total, model.observation_space, model.action_space,
        device=model.device, gamma=model.gamma, gae_lambda=model.gae_lambda, n_envs=1)


def collect(vector, total, block, payload, iteration):
    counts = quotas(total, vector.num_envs, iteration)
    gathered = [dict(transitions=[], episodes=[]) for _ in counts]
    while any(len(c['transitions']) < n for c,n in zip(gathered,counts)):
        active = []
        for i,(chunk,needed) in enumerate(zip(gathered,counts)):
            count = min(block, needed-len(chunk['transitions']))
            if count <= 0:
                continue
            args = (count, payload if not chunk['transitions'] else None)
            vector.remotes[i].send(('env_method',('batch_rollout',args,{})))
            active.append((i,count))
        vector.waiting = True
        for i,count in active:
            chunk = vector.remotes[i].recv()
            if chunk['model_sha256'] != payload['model_sha256'] or len(chunk['transitions']) != count:
                raise RuntimeError('Incomplete rollout or mixed policy versions')
            gathered[i]['transitions'].extend(chunk['transitions'])
            gathered[i]['episodes'].extend(chunk['episodes'])
            for key in ('observation','episode_start','chain_metrics'):
                gathered[i][key] = chunk[key]
        vector.waiting = False
    return gathered


def fill_buffer(model, chunks):
    import numpy as np
    import torch
    from stable_baselines3.common.buffers import RolloutBuffer
    counts = [len(c['transitions']) for c in chunks]
    if len(chunks) != model.n_envs or min(counts) < 1 or sum(counts) != model.astra_rollout_steps:
        raise ValueError('Exact rollout budget/worker mismatch')
    rows = [r for c in chunks for r in c['transitions']]
    observations = np.asarray([r['observation'] for r in rows],dtype=np.float32)
    actions = np.asarray([r['action'] for r in rows],dtype=np.int64)
    expected = np.asarray([r['logprob'] for r in rows])
    with torch.no_grad():
        values,logp,_ = model.policy.evaluate_actions(torch.as_tensor(observations,device=model.device),
                                                    torch.as_tensor(actions,device=model.device))
        tails = model.policy.predict_values(torch.as_tensor(
            np.asarray([c['observation'] for c in chunks],dtype=np.float32),device=model.device))
    values,logp,tails = values.cpu(),logp.cpu(),tails.cpu()
    error = float(np.max(np.abs(logp.numpy().reshape(-1)-expected)))
    if not np.isfinite(error) or error > 2e-5:
        raise RuntimeError(f'Lua/Torch old-policy logprob disagreement: {error}')
    buffers,offset = [],0
    for i,(chunk,count) in enumerate(zip(chunks,counts)):
        transitions=chunk['transitions']
        dones=[r['done'] for r in transitions]
        starts=[r['episode_start'] for r in transitions]
        if dones[:-1] != starts[1:] or dones[-1] != chunk['episode_start']:
            raise RuntimeError('Terminal flags do not match following episode starts')
        b=RolloutBuffer(count,model.observation_space,model.action_space,device=model.device,
                       gamma=model.gamma,gae_lambda=model.gae_lambda,n_envs=1)
        for j,r in enumerate(transitions):
            k=offset+j
            b.add(observations[k:k+1],actions[k:k+1],np.asarray([r['reward']]),
                  np.asarray([r['episode_start']]),values[k:k+1],logp[k:k+1])
        b.compute_returns_and_advantage(last_values=tails[i:i+1],dones=np.asarray([dones[-1]]))
        buffers.append(b);offset+=count
    target=model.rollout_buffer
    target.reset()
    for name in ('observations','actions','rewards','episode_starts','values','log_probs','advantages','returns'):
        setattr(target,name,np.concatenate([getattr(b,name) for b in buffers],axis=0))
    target.pos=sum(counts);target.full=True
    return error
