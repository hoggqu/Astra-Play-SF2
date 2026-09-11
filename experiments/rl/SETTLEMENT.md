# Native result adapter

`settlement.lua` is an experimental result recognizer shared by native batch
training and native continuous evaluation. Packaged `play_core.lua` remains
byte-identical to the frozen version. The adapter does not change inputs,
policy decisions, game RAM or recorded snapshots.

Two deterministic native replays exposed results that the frozen recognizer
could not classify. Both original runs remain invalid; repaired replays use
new source identities and new output directories.

The first adapter version handled these two specific cases; its implementation
is retained in Git history. The current default uses version 3 below.

- **Ken / Chun-Li, zero-health time draw:** both fighters were alive at zero HP
  when time expired. Ken entered the time-loss pose before a late spinning kick
  changed his live HP to -1. Displayed and time-latched HP stayed zero, native
  pips stayed unchanged, and the game subsequently opened the next round at the
  same score. The existing Ken / Chun-Li late-KO draw rule required a positive
  time latch. The adapter admits zero only when stop-state live/display HP are
  zero on both sides and Ken is observed alive in action 18 before the KO.
  The original mature draw pose, unchanged-pip and 360-frame gates still apply.
- **Ken / Guile, time loss followed by late KO:** the native game awarded Guile
  one pip with latched/displayed HP 17 versus Ken's 6 before a flash kick reduced
  Ken's live HP to -1. Ken's final grounded KO animation retained action 8;
  the frozen recognizer accepted actions 0 and 12 for this late-KO case.
  The adapter requires the observed pre-KO time-result award, exact unchanged
  latched/displayed HP, Guile's grounded winner pose, Ken's exact final KO
  animation/action, the unique one-pip increase and 360-frame maturity. It
  records a loss and retains the original native fields.

Unknown outcomes still invalidate the run. Batch failures now write the full
settlement trajectory to `training/rl-batch-unresolved-settlement.json`; this
never creates a terminal transition or reward. Valid terminal rows are persisted
before checkpoint reset as before. The adapter and staged copy are covered by
runtime source hashes.

Offline tests cover both accepted cases, missing chronology, mismatched fields
and premature results. They are distinct from native emulator replays. See
`test_settlement.py`; no private raw trajectories are included in the project.

## Version 2: native time-result protocol

Version 2 introduced the protocol below, retained in the current default.
Both batch and native continuous entry points stage the adapter in the
`rl_settlement.lua` runtime role. Never
replace the adapter beneath an active controller. A further Dhalsim time loss retained
Ken's action 4 in the final KO sprite, demonstrating that enumerating each
retained action would repeatedly reject genuine native results.

The new fallback is independent of opponent identity and final animation:

1. Time must have expired. Observe exactly one legal native pip increase from
   this round's initial score.
2. At that observation, each time-latched HP must be an integer in 0..144 and
   equal that fighter's displayed **and live** HP. The winner's latched HP must
   be strictly greater. This records the game awarding its actual time result
   before any later live-HP changes.
3. Keep that pip and both displayed/latched HP values unchanged for at least
   360 advancing native frames. A change invalidates the evidence instead of
   silently starting another candidate. Callers already filter duplicate-time
   callbacks.
4. Both fighters must be grounded at acceptance. Preserve the native opening,
   stop, award and final snapshots; record the stable duration and protocol ID.

This fallback never guesses a draw from absent pips. The separately confirmed
Chun-Li zero-time draw extension and original Core result/failure rules remain.
It does not override an original Core result or failure, including arrival of
an unresolved next round. Ordinary KO results continue through the original
Core; time-HP evidence does not apply to a non-time result.

Offline replay of recorded trajectories resolves Chun-Li as a draw,
Guile as a loss after 360 stable award frames, and Dhalsim as a loss after 376
(the latter waits longer for grounding). A fourth trace, where Ken won the
Dhalsim time result, is accepted without another code change. All mature before
their next native round. Native validation subsequently passed 256-decision
batch/deployment parity, 102400 and 61440 real PPO decisions, and one continuous
attempt per resulting model. Those continuous attempts were valid losses;
they do not demonstrate a clear. `test_settlement_v2.py` includes missing evidence,
health bounds, score/HP changes, airborne states and early next-round cases.


## Version 3: confirm equal-time draws from the natural next round

A later Dhalsim timeout left both time-latched and displayed HP at zero, then
produced a late KO without changing either pip. Its final poses were outside
the old draw whitelist. Version 3 retains version 2's win/loss protocol and
adds a draw confirmation independent of character-specific animation lists:

1. Observe equal, integer time-latched HP in 0..144, matching both fighters'
   live and displayed HP after time expires; native pips must remain unchanged.
2. Retain a grounded previous-round snapshot at least 360 native frames after
   the stop. Changes to pips or displayed/latched HP disqualify the candidate.
   Actor initialization through zero animation pointers is permitted only
   after this mature evidence exists.
3. Require the game's actual full-health, timer-99 next-round opening with the
   same actors and score. Until then, there is no draw result. Unknown evidence
   still invalidates; an already invalid Core is never revived.

The terminal row separates the previous round's raw `settled` snapshot from
its raw next-round `confirmation`. Batch terminal damage reward uses only
`settled`, preventing next-round health refill from creating spurious reward.
Continuous play keeps the natural next round and uses its normal readiness and
input cadence. The adapter never edits snapshots or game RAM.

Validation included the original failing seed's complete 102400-decision PPO
replay, a same-start-phase 256-decision batch/native-Core replay with 746
identical settlement frames, and a separate native continuation probe. In the
last probe, the draw confirmed at frame 3797, round 2 began at 3886, and the
next twelve frames consumed the intended light-punch input. There was one
initial training checkpoint load and no subsequent load or pause through frame
3898. These are training diagnostics, not a formal clear.

`test_settlement_v3.py` covers rejected evidence and separated terminal reward.
The portable `draw_parity`, `draw_block_probe`, and `draw_continuation` diagnostic
modules require captured private action plans and explicitly staged diagnostic
runtimes; they are not normal verification entry points. A paused checkpoint
reset and a live in-batch reset have different first-input phases: reproducing
the captured draw required the live reset phase. Ordinary 64-decision batch
boundaries were ruled out as the cause. Input-start normalization remains a
separate experiment; this settlement fix does not change action timing.
