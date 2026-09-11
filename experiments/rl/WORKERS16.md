# Sixteen-worker pulsed-input candidate

This isolated candidate raises the allowed worker count to 16. It derives from
an explicitly identified frozen pulsed-normal-button package and does not modify
that parent. All Lua files, including the action waveform, settlement and native
cadence, remain byte-identical. Only the batch/native campaign CLI limits and
their exact source-derivation validator change. The parent pulse manifest and
original validator are retained and checked during training and native audit.

Each worker still collects 256 decisions before an update. Sixteen workers
therefore provide 4,096 decisions per global rollout. A 1,638,400-decision stage
contains 400 updates, each with the unchanged batch size 64 and four epochs.
This changes the number of environments and the aggregate on-policy buffer;
it is not a clean single-variable comparison against the former two separate
8-worker learners, especially when the input interface also changes. No
throughput or clear-rate gain has yet been established by these offline tests.
PPO.load constructs the new 16-environment buffer while restoring the original
policy, critic and Adam state. Environment/RNG trajectories restart per stage;
this is not exact interrupted-trajectory recovery.

```sh
python -m experiments.rl.workers16_builder \
  --source FROZEN_PULSE/astra_sf2_rl_round_chain \
  --output NEW_TRAIN_CODE --driver-output NEW_DRIVER_CODE
```

The optional driver copies the existing reliability campaign and one-generation
pipeline into an independent tree with production `src`. It accepts up to 16
workers and requires the pulsed action-interface identity. All twenty natural
coins, matching model hash, native timing and action-interface audits are still
mandatory. It pins every Python/Lua/JSON provenance dependency recursively;
old held-input results cannot satisfy its full-20 checker. The old schedulers
and source snapshots remain unchanged. Defaults stay at their previous worker
counts; launches must explicitly request 16.

Before any long training, run one real smoke using the RL Python environment,
explicit MAME config, and the frozen tree's `src` and root on `PYTHONPATH`:

```sh
python -m astra_sf2_rl_round_chain.batch_train \
  --dataset DATASET/manifest.json --output NEW_SMOKE \
  --init-model PULSED_MODEL.zip --workers 16 --block 64 \
  --steps 4096 --seed 400
```

Require complete status, actual steps 4096, changed parameters, 16 workers,
`effective_ppo.n_steps=256`, and inherited optimizer state. A 2,048-step budget
cannot fill this rollout and is rejected before MAME starts. Preserve the smoke
output and start formal training from the original migrated pulsed model,
not the smoke weights. Formal validation remains one immutable complete-20
group with at least ten clears; never pool models or replace invalid attempts.

Proposed remote resource envelope, subject to the owner's launch gate:

- One learner with 16 worker MAME processes; at most one formal evaluator.
- The existing shared `astra-rl-normal.slice`: 20 logical CPU quota, 24 GiB RAM.
- A single owned service capped at the same 20 CPU / 24 GiB, Nice 10, all
  Torch/BLAS/OpenMP thread variables set to one, silent/headless.
- Existing owned process-group cleanup and raw cancellation/checkpoint retention.

No command in the builder launches MAME. Six offline tests cover unchanged
runtime bytes, source/provenance tampering, limits and complete-update budgets,
16-environment buffer creation with exact weight/Adam preservation, and the
unchanged strict full-20 audit requirements. Real smoke and native pulsed-input
gates remain prerequisites for deployment.
