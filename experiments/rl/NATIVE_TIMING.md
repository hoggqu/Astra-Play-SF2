# Native-time training and deployment contract

The native-time interface targets **twelve advancing
emulated frames per fighting decision**, four observations of history, and the
same action macros. Zero-time machine-frame notifications do not consume an
action frame. Action frame zero is applied in `frame_done`; frames one through
eleven use machine-frame notifications. A macro's facing direction is captured
when the decision is made and remains fixed through that macro.

The native deployment adapter runs every complete opponent match without a
pause, model replacement, or Python action RPC. It reuses the unchanged native
result state machine and natural-coin route orchestration. It has its own staged
Core and source identities; legacy `continuous.py`, `train.py`, `env.py`,
`runtime.lua`, and frozen V4 sources remain unchanged.

```sh
python -m experiments.rl.native_continuous \
  --model MODEL.zip --output .local/rl-runs/FRESH_NATIVE_EVALUATION \
  --difficulty 3 --attempts 3 --speed fast
```

The result schema and exit codes are those of [continuous RL evaluation](CONTINUOUS.md),
with `native_timing: true` and a separate `native_timing_audit`. For a completed
run, that audit requires each match's native timing flag, complete frame rows,
strictly increasing native times spaced by the independently read
`screen.frame_period`, and complete twelve-frame decision coverage from every
round opening through its stop (no missing first, intermediate, or last
decisions). The screen period is checked unchanged on every native frame. Cross-round settlement/opening
intervals are excluded from the decision-spacing check. Native-Core source hashes
are computed after staging, and result/evidence checksums are resealed after this
audit. A failed timing audit invalidates the run.

## Why this is a new interface

During a strict transport comparison, the old paused RPC controller counted
machine-frame callbacks without checking whether emulated time advanced. In the
observed example, two nominal twelve-frame steps counted 24 callbacks while
emulated time advanced only 22 native frames. The no-pause sampler did not have
the same repeated boundary callback behavior. This issue concerns the old RL
experiment's sampling interface; it does not modify frozen V4 results or code.

An isolated `corrected_reference_runtime.lua` removes duplicate-time callbacks
from a copy of the old RPC controller. That correction aligned the clocks, but
strict comparisons still found divergent native trajectories once fighting
began. We did not ignore animation, positions, opponent actions, or other state
fields to manufacture a pass, and we did not add artificial combat pauses to the
new deployment interface. Failed diagnostic runs remain invalid evidence.

Instead, the new acceptance test compares the batch action executor against
`native_continuous_core.lua` in a training-only checkpoint harness. Both paths
run without within-batch pauses and independently drive the same predetermined
macro actions. A 256-decision native test, including a mature round result and
automatic checkpoint reset, matched complete native states (including emulated
time), float32 observations, rewards, and terminal flags exactly. This is finite-sample evidence for that tested trajectory, not a general proof
of sampling/deployment equivalence or reproduction of the earlier paused RPC
trajectory. The later batch-boundary investigation below found a separate
input-latch defect that this sample did not expose.

Earlier models can initialize new native-time training, but their previous
win-rate evidence does not certify this new interface. New training and fresh
continuous-game validation are required.

The pinned [MAME 0.288 Lua binding](https://github.com/mamedev/mame/blob/mame0288/src/frontend/mame/luaengine.cpp#L1817) exposes `screen.frame_period` directly as seconds; the audit does not infer the expected period from its own trace.

## Batch-resume input latch correction

A later full-match replay exposed a second timing issue despite correct native
frame counts. At a 33-decision batch boundary, the paused sampler installed the
next action while emulated time was unchanged. MAME then latched the new input
one native frame earlier than the uninterrupted deployment Core. In the real
port trace, IN1 changed from neutral `65535` to right `65534` at time `33.619840`;
the uninterrupted reference changed at `33.636608`. The first visible game-state
difference appeared only 32 frames after that boundary. Removing the pause-time
input release did not fix it.

This agrees with pinned MAME 0.288: `video.cpp` calls the Lua frame-done hook
before the machine-frame notification, while `ioport.cpp` skips its frame update
when paused. Ignoring duplicate-time Lua notifications alone does not control
when MAME latches a button. `held_input` logs cannot replace actual `IN1`/`IN2`
port reads when checking this mechanism.

The corrected `round_chain_runtime.lua` leaves the existing neutral
boundary input in place on RPC resume, and applies action zero only in the first
**advancing** frame-done callback. The same gate also applies to the initial explicit paused `reset` followed by
`rollout`; live `reset_rollout` startup already defers action zero to the next
frame-done callback. A 600-native-frame gate compared both full states and actual
ports around the boundary: both paths stayed neutral at `33.619840` and latched
right at `33.636608`, with no state divergence. The older 256/640-decision proofs
remain retained historical samples; they do not certify arbitrary batch
boundaries. Full-match and fresh-training acceptance must use the corrected
source identity and explicitly report their tested scope.

The final candidate also passed a three-path startup probe: explicit paused
`reset` + `rollout`, live `reset_rollout`, and the uninterrupted native reference
matched all 600 native states, with actual IN1/IN2 samples matching on all first
460 frames. The corrected full-match replay matched 4,889 native frames and 316
decisions through R1→R2 (`win/win`), including the 33-decision first batch and
subsequent 64-decision boundaries. Its reference loaded once initially and never
paused during the match. Observations, two mature round rewards and GAE masks
also passed (maximum serialized observation difference below 5e-15). This is
bounded training-harness evidence, not a no-load formal clear and not a proof
covering every possible input sequence.


The correction is now the canonical round-chain runtime, and the v4 settlement
adapter is the canonical `settlement.lua`. Existing frozen packages remain
unchanged and retain their own identities. The older R1-only `batch_runtime.lua`
has not received this resume correction; do not extend the corrected chain's
parity claim to that older sampler. A two-worker, 2,048-decision real PPO smoke
updated parameters successfully using the corrected chain. Its parent model also
completed a fresh natural-coin evaluation: two match wins followed by a valid
Honda loss, with native and 16-action/chain identity audits passing. Neither
check established a gameplay clear.
