# Single-generation reliability pipeline

This optional experiment overlaps verification of frozen model N with training
of N+1. It does not modify the serial reliability campaign, active snapshots,
the PPO learner, game inputs, or native verification protocol. It currently
supports POSIX systems (Linux/macOS); Windows keeps the serial campaign because
this scheduler requires audited ownership and cleanup of a whole process group.

The goal remains **at least 10 clears in one complete group of 20 natural coins
with one fixed model**, using Normal difficulty 3. A partial group, pooled
successes from different models, or an exit code of zero alone cannot meet it.
The existing strict full-20 checker and native/actions16 audits are reused.

At most one trainer and one verifier run simultaneously. The trainer reads the
same immutable ZIP being evaluated and writes a new output directory. Training
seed, budget and input interface are declared before any results arrive; partial
verification outcomes never change training. If the trainer finishes first, its
completed ZIP waits for the current full-20 audit. N+2 cannot start until N's
verification has completed and N+1 is ready. `--cycles 5` permits at most five
candidate evaluations and four new training stages; there is no sixth-model
prefetch after the final candidate.

When a complete group reaches the goal, its evidence is preserved and any owned
unfinished prefetch is stopped. The child training result remains unchanged
(normally `invalid` with SIGTERM); the parent separately records
`owner_cancelled_after_verified_goal`, all completed-update progress, and the
last checksum-verified recoverable checkpoint. Cancellation is neither a game
loss nor a completed training stage. A genuine error stops the pipeline without
replacement coins or automatic recovery. A completion racing cancellation is
validated normally, so an already-failed child is not relabelled as a benign
cancellation.

Use separate immutable code roots for the orchestrator, trainer and verifier;
each execution root must include its frozen `src/astra_play_sf2`. The verifier
must support `--all-attempts`. Run from the frozen orchestrator root using a
Python environment with the existing RL dependencies, and set `PYTHONPATH` to
that root and its `src`. The scheduler supplies isolated paths to its children:

```sh
python -m experiments.rl.reliability_pipeline \
  --dataset DATASET/manifest.json --output NEW_OUTPUT \
  --init-model COMPLETED_MODEL.zip \
  --initial-training-result COMPLETED_TRAIN/result.json \
  --training-code TRAIN_CODE --training-package astra_sf2_rl_round_chain \
  --verification-code VERIFY_CODE --verification-package astra_sf2_rl_round_chain \
  --workers 8 --cycles 5 --steps-per-cycle 409600 --seed 129
```

The initial candidate is evaluated while its successor trains; do not use an
active or partially written checkpoint. Weighted training uses its own frozen
package instead of the uniform package, preserving configuration metadata.
PPO ZIP parameters and optimizer continue; environment/RNG startup is new at
each training stage, not an exact trajectory resume. All source and input hashes
are checked before and after each stage. Models awaiting evaluation retain
their previously captured hash. No existing output is appended or overwritten.

Process count limits do not enforce CPU or memory limits. On the existing Linux
runner, place each arm in a 10-CPU/12-GiB service under the common
20-CPU/24-GiB slice, with Nice 10. The scheduler forces numerical-library threads
to one. Eight training MAME processes plus one verifier fit the declared
per-arm process budget; measure contention before claiming a speedup. Stage
timestamps and wall time are retained. On cancellation, only the newly created
child process group is eligible for termination, including live descendants
whose parent has already exited; unrelated controllers are never signalled.

Exit codes: 0 = audited full-20 goal and owned cleanup succeeded; 1 = complete
candidate budget without meeting the goal; 2 = setup, control, audit or cleanup
failure. If verification reached the goal but a simultaneous training/cleanup
failure occurred, the pipeline remains invalid and keeps `verified_goal_cycle`
and the complete original verification evidence for independent review.
