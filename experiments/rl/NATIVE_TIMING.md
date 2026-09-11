# Native-time training and deployment contract

The optional `batch_train` / `native_continuous` pair uses **twelve advancing
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
time), float32 observations, rewards, and terminal flags exactly. This establishes
that tested sampling/deployment interface, not a claim that the earlier paused
RPC policy trajectory is reproduced.

Earlier models can initialize new native-time training, but their previous
win-rate evidence does not certify this new interface. New training and fresh
continuous-game validation are required.

The pinned [MAME 0.288 Lua binding](https://github.com/mamedev/mame/blob/mame0288/src/frontend/mame/luaengine.cpp#L1817) exposes `screen.frame_period` directly as seconds; the audit does not infer the expected period from its own trace.
