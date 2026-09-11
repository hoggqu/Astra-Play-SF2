# Native TIME assignment: adjacent-frame evidence

The separate `settlement_v5.lua` adapter covers a native TIME-loss transition
observed during experimental PPO training. On the frame that awards the winner
a pip, SF2 can latch both timeout HP values and replace the loser's live HP with
`-1`. Requiring both live HP values to still equal the latches on that same
frame misses this legitimate result.

V5 additionally accepts the immediately preceding observed frame as chronology
evidence: timer zero, matching fighters, unchanged old score, valid live HP equal
to displayed HP, and those HP values equal to the newly awarded timeout latches.
The new frame must have exactly one winner pip, strictly greater winner HP,
winner live HP equal to its latch and loser live HP exactly `-1`. Existing
stable-score/latch checks, 360-frame maturity and grounded settlement remain.
Unknown or inconsistent evidence still fails closed. The adapter writes no RAM
and does not select combat actions.

Adjacent adapter ticks are not by themselves proof of native-frame continuity.
Callers must still deduplicate callbacks and audit native timing. The existing
settlement source and active frozen packages are unchanged; select V5 only in
a new source snapshot and rebuild its derived-file manifest.

## Validation

The original invalid run remains invalid. Its 866-frame raw settlement trace
replays to an unresolved result under V4 and a loss under V5. An independent
training-only MAME replay used the original opening and 254 recorded actions.
All 391 compared settlement snapshots matched exactly: previous HP `1:42` at
frame 3073, native CPU award and Ken live HP `-1` at 3074, stable loss `0:1` at
3434. The probe continued naturally into round two and through its first
12-frame macro, ending after 4050 frames. It used one initial training load and
zero in-play pauses; it is not a formal attempt or a win-rate sample.

The separate V5 suites cover both winner directions, missing or inconsistent
previous evidence, changed pips/latches, wrong sentinels, premature openings,
airborne settlement, old TIME/draw/double-KO paths and new-round cleanup.
All 38 tests passed:

```sh
python -m unittest experiments.rl.test_settlement_v5 \
  experiments.rl.test_settlement_v5_regression
```

The original V4 tests remain unchanged. Their old assertion that same-frame
loser `-1` must always remain unresolved is deliberately replaced in the V5
regression suite by missing-previous-frame and previously-KO negative cases.
