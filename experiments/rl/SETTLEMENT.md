# Native result adapter

`settlement.lua` is an experimental result recognizer shared by native batch
training and native continuous evaluation. Packaged `play_core.lua` remains
byte-identical to the frozen version. The adapter does not change inputs,
policy decisions, game RAM or recorded snapshots.

Two deterministic native replays exposed results that the frozen recognizer
could not classify. Both original runs remain invalid; repaired replays use
new source identities and new output directories.

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
