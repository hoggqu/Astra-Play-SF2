# Local capacity A/B protocol

## Execution status

The first paired experiment was interrupted and cannot support a completed
capacity comparison. A finished its 20 attempts with 0 clears. B stopped after
113,664 completed decisions on an unrecognized native TIME-loss settlement;
the last saved model contains 102,400 decisions, and the remaining 11,264
completed decisions are not recoverable from that checkpoint. No B full-20
evaluation started. The invalid run and its raw trace remain unchanged.

A new settlement adapter has independently reproduced and recognized the
native loss using the previous frame's HP, the awarded pip and stable locked
HP. Any continuation from the saved B checkpoint uses a new frozen source,
output directory and restarted environment RNG. It is a separate reliability
experiment, not completion of the original matched-seed A/B trajectory.

## Original predeclared design

The local capacity comparison starts from one completed 64×64 actor/critic
checkpoint. Arm A continues that model. Arm B starts from its function-preserving
128×128 expansion, including the embedded old Adam state described in
[width migration](WIDTH_MIGRATION.md). It keeps observations 344, actions 16,
native action timing, reward, frozen training and verification code, local
44-opening dataset, four workers, block size 64, and seed 316 unchanged.

B first performs a separate 2,048-decision real PPO smoke test. It must complete,
change parameters, report 128×128 policy/value networks, and inherit the old
optimizer state. The smoke model is retained for diagnostics and **never becomes
the formal B training parent**. Failure stops the flow. On success, B restarts
from the original migrated ZIP for its full 409,600 decisions.

**Only A's corresponding 409,600-decision stage and B's first 409,600-decision
stage, followed by each fixed model's complete 20 natural-coin attempts, form
the paired capacity comparison.** Do not compare B with an earlier A baseline,
merge different models' clears, or describe later continued training as part of
this single paired test. Matching initial functions and random seeds does not
mean that the learned policies or game trajectories must remain equal.

After B's initial training, a bounded three-candidate reliability pipeline
evaluates that fixed model for all 20 attempts while pretraining the next
candidate for 409,600 decisions with seed 317. The final possible successor
uses seed 318. These later candidates are separate reliability trials. Each
candidate must independently achieve at least 10 clears in its own complete
20-attempt group; invalid execution stops the flow, and cancelled prefetch
training is preserved separately from gameplay failures.

All B processes use new output directories and frozen orchestration code,
single-thread numerical libraries, Nice 10, silent/headless MAME, and separate
controllers. The existing A controller is neither modified nor stopped.
Actual elapsed time and throughput must be measured: additional capacity
increases policy computation, and concurrent work can change wall-clock speed.
No capacity conclusion should be drawn before both paired groups finish and
their native, model and action-interface audits pass.
