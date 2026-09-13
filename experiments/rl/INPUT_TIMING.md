# Batch-boundary input timing

The `astra_sf2_rl_perception128_timing` candidate fixes a training transport
boundary that could alter a match despite identical opening state and actions.
Its policy, 85 action waveforms, 12-frame decisions, observations, native Core,
reward, optimizer and v7 settlement code remain unchanged. The new identity is
`astra.rl-screen-perception128-input-timing.v1`; the input protocol is
`restore-held-input-before-rpc-resume-v1`.

## Cause and correction

In MAME 0.288, `video_manager::frame_update` invokes the Lua `frame_done` hook
before `MACHINE_NOTIFY_FRAME`. The input manager updates its cached digital
ports in that notifier when the machine is unpaused. Resuming from a paused
`frame_done` hook can therefore poll inputs again at the **same emulated time**.
The training observer correctly ignores duplicate-time callbacks, but this does
not prevent the input manager's own poll.

The old trainer released controls when returning a completed batch. On resume,
the extra poll could replace the previous macro's still-latched final input
with neutral. Merely applying the next macro immediately instead is also wrong:
it can make that input arrive earlier than it does in uninterrupted play.

The correction records the input held before pausing, releases controls during
the pause, and restores that previous input immediately before resuming. The
existing `deferred_after` guard still waits for advancing emulated time before
applying the next macro. A checkpoint reset explicitly clears the recorded
input. Thus a pause between decisions preserves the uninterrupted input latch;
it does not insert an additional decision or change an action waveform.

This applies to training RPC boundaries only. Continuous gameplay still runs
without in-match pauses. Old frozen packages and the records they produced are
preserved; this correction does not retroactively certify their timing.

Relevant pinned primary implementation: MAME 0.288
[`video.cpp`](https://github.com/mamedev/mame/blob/mame0288/src/emu/video.cpp),
[`ioport.cpp`](https://github.com/mamedev/mame/blob/mame0288/src/emu/ioport.cpp), and
[`luaengine.cpp`](https://github.com/mamedev/mame/blob/mame0288/src/frontend/mame/luaengine.cpp).

## Reproduce the source package

From a clean checkout with the RL Python dependencies installed:

```sh
python -m experiments.rl.perception128_timing_builder --output .local/rl-code/my-timing
python -m unittest experiments.rl.test_perception128_timing -v
```

The bootstrap reconstructs its parent using public source. Alternatively,
`--source PATH` accepts a validated frozen v7 late-KO package root. The builder
creates a new directory, captures the complete parent and recipe, checks every
source hash, and marks the package as a candidate. It never modifies its parent.
The generated package is portable with its `src`, launcher, manifest and captured
parents; no ROM, checkpoint, learned model or private historical data is needed
to build it. Running an emulator evaluation still needs the normal compatible
MAME/ROM setup and a suitable model/dataset.

## Validation boundary

Six portable tests execute the captured Lua input, pause, reset and resume
functions with a mocked input latch, check all 85 macro boundary waveforms,
retain the deferred-time guard, reject mutated executing source, and validate
the package from an isolated Python import path. These tests check callback
semantics; they do not simulate the game CPU.

A bounded native diagnosis compared five 540-frame trajectories from the same
opening and forced actions. Uninterrupted native play, the unsplit batch, and
the corrected split batch agreed frame by frame. The original split batch and
an attempted guard removal diverged at frame 460, immediately after a batch
boundary at frame 456. Full-frame port snapshots alone were identical even on
the failing paths: the harmful poll occurs between these snapshots. Testing
must therefore compare game trajectories as well as sampled ports.

This diagnosis supports the correction but is not a gameplay result. Each final
frozen package must pass its separate native action/observation timing gates
before it is used for a new training or natural-coin evaluation campaign.
