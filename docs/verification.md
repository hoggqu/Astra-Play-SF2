# Evidence and review

## Automatic gameplay evidence

A `gameplay_clear` means the runner observed eleven native whole-match wins in one Ken game under its enforced run constraints. A whole match contains its rounds; one won round is not a won match. Gameplay uses ordinary P1 inputs and reads current game state in Lua. It does not use screenshot recognition to choose moves and does not inspect future CPU decisions.

The Python runner arranges the session, records outcomes, and advances through natural transitions. Its fixed policy handles combat without an AI service. No state loads, state saves, continues, resets between attempts, or pauses inside a whole match are part of a valid gameplay attempt. Initial boot of a new isolated session is distinct from resetting a failed attempt inside that session.

Native health, timer, round-win markers, and settled state are evidence together. Zero HP or a stopped timer alone is not a win. Damage can occur between the timer stopping and native score locking, or after a result has already locked; an audit must not replace the native result with a later live-health comparison.

## Difficulty is part of the evidence

From 0.1.1, native DIP is configured before boot. A valid run records the boot
configuration and checks both DIP and game-decoded difficulty at startup, before
a match and every Core tick. Each requested difficulty boots a separate session;
attempts within that level use the same process and natural coin insertion.
The multi-level root audit verifies all child sessions and their shared runtime.

Legacy `astra.run.v1` files do not contain internal difficulty evidence. Their
results remain readable as provisional history, but the new audit will not
certify them or accept a new visual approval as a substitute for missing
measurements. See [the difficulty correction](difficulty-fix.md).

## Optional visual review

The automatic result and visual review answer different questions:

| Record | What it establishes | What it does not establish |
|---|---|---|
| Automatic gameplay clear | Eleven native match wins and recorded gameplay constraints. | That a person or AI inspected the selection or ending images. |
| Named offline review | A reviewer explicitly approves or rejects the saved visual evidence. | That the code itself performs vision, or that failed gameplay becomes valid. |
| Historical validation | Results observed in the specified original environment. | Guaranteed success or port validation on a different environment. |

The runner saves a selection screenshot, a Bison-result screenshot, and three ending screenshots for a completed attempt's review. Open the actual images associated with that attempt, not screenshots from another run. Confirm Ken was selected, the terminal Bison result supports a clear, and the ending sequence supports Ken's completion. Check the accompanying audit for invalid or incomplete gameplay evidence.

Only then record your decision:

```sh
astra-sf2 review RUN_DIR --attempt ATTEMPT_ID --reviewer "Reviewer name" --decision approve
```

If you cannot support approval, use `reject`. The CLI records the declared decision; it does not know whether you really looked. Do not name a person or agent that did not perform the review. Visual review is offline and does not pause combat or change the policy.

Report automatic clears and review status separately. Do not describe an unreviewed automated clear as “visually verified.” Likewise, a recorded approval is not permission to ignore missing gameplay evidence.

## Denominators and stopping rules

Keep every started attempt, including losses, interruptions, and invalid or pending records. Distinguish a bounded campaign's cumulative clear rate from a final consecutive streak. Five consecutive wins after earlier losses is not a 100% campaign win rate.

For opponent-level reporting, retain individual round wins, losses, and draws. A 2:1 match contributes two won rounds and one lost round. Do not silently discard a losing match's already played rounds or treat uncertain results as victories. Whole-game clears, whole-match wins, and round wins are three different counts.

Starting another session does not erase the previous session's failures or import its streak. Repeating a checkpoint or a closely related opening in training is also not an independent random trial; this CLI's gameplay verification does not load training checkpoints.

## Preserve the evidence

Retain the complete run directory, including the policy identity, run settings, outcome records, screenshots, and reviews that the runner produced. Use `audit RUN_DIR` and `report RUN_DIR` instead of manually rewriting results. Do not edit evidence to turn an interrupted attempt into a clear.

An audit can check recorded consistency and attestations; it cannot reconstruct an image a reviewer never inspected or prove that a particular input change would have won an unplayed counterfactual. Unit tests, simulator tests, and native emulator runs must be labeled separately.
