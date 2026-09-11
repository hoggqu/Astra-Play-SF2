# Configurable rollout candidate

This isolated training derivative changes the number of on-policy decisions
collected per worker before a PPO update. The CLI defaults to 256 for
compatibility; the proposed experiment explicitly requests
`--rollout-steps 1024`. Eight workers then collect 8,192 decisions per rollout,
so 409,600 total decisions contain 50 rollouts instead of 200. The minibatch
size remains 64 and each rollout still receives four epochs: both lengths use
1,638,400 training-sample visits and 25,600 Adam minibatch steps per stage.
This is not a claim of fourfold wall-clock acceleration.

The experiment keeps the uniform opponent sampler, observations 344, actions
16, network architecture, reward, gamma 0.99, GAE lambda 0.95, learning rate
0.0003 and entropy coefficient 0.01. Native RPCs remain blocks of 64 decisions.
The longer buffer spans 16 RPC blocks without changing the policy midway;
history and natural R2/R3 transitions remain owned by the unchanged Lua runtime.
`PPO.load(..., n_steps=...)` constructs the requested rollout buffer before
restoring saved policy and optimizer tensors. Adam moments, steps and parameter
groups are preserved. Episode starts/dones still cut GAE at each mature small
round, including a terminal at the last buffer entry; refill observations do
not bootstrap that terminal. Network observation history remains four decision-time observations
(12 native frames between decisions); increasing rollout length does not add model memory.

## Motivation and limits

A read-only reconstruction of two completed 409,600-decision stages found mean
update coverage of 8.155/11 opponents under uniform sampling and 7.40/11 under
fixed weighted sampling. Grouping each four adjacent historical updates raises
coverage to 10.58 and 9.84, respectively. In the uniform sequence, Guile, Ryu and
Zangief absence falls from 14.5%, 15%, 27.5% to 2% each; in the weighted sequence
the corresponding figures change from 28.5%, 18%, 54% to 2%, 2%, 16%.
The historical four updates used different policies, so grouping them estimates
exposure, not actual behavior under a newly trained 1,024-step policy. Absence
is not proof of forgetting or a software bug.

The observed old training stages took about 15.3–15.5 minutes and their full-20
verification took 7.8–9.1 minutes. The new pipeline already overlaps verification
and the next training stage. Long rollouts therefore aim to improve the data
mix at similar decision cost; their throughput and clear rate still need real
measurement. Additional budget alone or less frequent verification is not part
of this single-parameter candidate.

## Build and proposed gate

Build from an already frozen uniform round-chain package with its copied
production `src`. The builder copies all Lua files byte-for-byte, including the
parent's settlement revision, and preserves its manifests and provenance.
For the current candidate, use the validated v5 settlement parent. A strict
rollout-only comparison requires both 256 and 1,024 arms to use that same v5
runtime; an older v4 run is historical context, not a clean single-variable
control. Any later input-interface repair requires a new parent and new build.

```sh
python -m experiments.rl.rollout_builder \
  --source FROZEN_V5/astra_sf2_rl_round_chain \
  --output NEW_ROLLOUT_CODE
```

Run the following from `NEW_ROLLOUT_CODE`, with its root and `src` on
`PYTHONPATH`, using the RL Python environment and an explicit local MAME config:

```sh
python -m astra_sf2_rl_long_rollout.batch_train \
  --dataset DATASET/manifest.json --output NEW_SMOKE \
  --init-model ORIGINAL_MODEL.zip --workers 8 --block 64 \
  --rollout-steps 1024 --steps 8192 --seed 130
```

This is a proposed **real** smoke gate, not something the builder runs.
Require complete status, 8,192 actual decisions, changed parameters, inherited
optimizer state, `effective_ppo.n_steps=1024`, matching rollout configuration,
and a finite or explicitly unavailable PPO telemetry record. Preserve the
smoke output, then start any formal 409,600-decision training from
`ORIGINAL_MODEL.zip`, not the smoke weights. A 2,048-decision job with eight
workers cannot fill this buffer and is rejected before starting an emulator.

After formal training, use the separate frozen native verifier for **all 20**
natural coins with the resulting immutable ZIP. The goal remains at least ten
clears in that complete group, with all native/model/action-interface audits.
No prefix screen, training rounds, or results from another model enter those
20 attempts. Keep the explicit `--rollout-steps 1024` on every training launch;
existing serial/pipeline schedulers do not automatically inject this option.
A future isolated driver may supply it, or a clearly recorded build may set
`--default-rollout-steps 1024`; the general CLI default remains 256.

The current PPO logger is copied as a read-only helper, recording entropy,
approximate KL, clip fraction, value loss, explained variance, policy loss,
learning rate and update count. Missing/nonfinite fields remain explicit nulls.
The builder and tests do not start MAME, and offline buffer tests are not a
claim that the new end-to-end training candidate has passed its real smoke.
