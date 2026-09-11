# Fixed-weight categorical evaluation of the 16-action policy

This is a separate evaluation candidate: the same PPO model ZIP and the same 16
action macros, selecting from the network's softmax probabilities instead of
argmax. It performs no training, opponent-specific action selection, or game RNG
edits. Its results must stay separate from deterministic-policy statistics.

Build from a completed original actions16 snapshot, not a round-chain snapshot.
The seed is explicitly fixed when building; use a fresh build for another seed.
Neither the parent snapshot nor the model is modified.

```console
python -m experiments.rl.stochastic16_builder --source ACTIONS16_CODE/astra_sf2_rl16 --output .local/stochastic16-code-001 --policy-seed 42
python .local/stochastic16-code-001/launch.py native_continuous --model SAME_MODEL.zip --output .local/stochastic16-play-001 --difficulty 3 --attempts 3 --speed fast
```

Use the project's RL Python environment and configured MAME/ROM. The generated
launcher supplies its isolated namespace to Python. Native playback defaults to
silent/headless. Attempts share one emulator session and use natural coins, with
no continue, state load, reset, or in-match pause. `native_continuous` and `export`
are the only supported CLI operations in this candidate; training entries reject
execution. No weight migration is needed.

The payload and per-match summaries explicitly declare `categorical_softmax`,
the policy seed, and `park_miller_48271_v1`. The derived NN accepts the real
categorical label; it does not disguise the payload as deterministic to pass a
guard. The original reviewed sampling math and audit functions are copied by
the builder into an isolated helper module, with their source hashes recorded.
Lua selects from softmax once every 12 native frames. The independent policy
PRNG advances continuously across history resets, rounds, opponents and coins.

A complete result requires three audits:

- `native_timing_audit`: all fighting decisions and native-frame coverage.
- `action_interface_audit`: parent identity, package/model hashes before and
  after, exact exported payload, staged Lua files, and every match identity.
- `sampling_audit`: reconstruct every network output, random draw, chosen action
  and log probability, including continuity across natural coin attempts.

All new executing Python/Lua dependencies are included in the generated build
manifest and interface audit. Staged stochastic Lua is also in the controller's
source checks. The final audit rechecks sources after sampling reconstruction;
any mismatch marks the run invalid and reseals the evidence. Native and sampling
decision totals must agree. Exit codes remain 0 for a clear, 1 for valid losses,
2 for an invalid run. A valid loss is still a failed clearance attempt.

Offline tests do not start MAME:

```console
python -m unittest experiments.rl.test_stochastic16 -v
```

They cover 16-logit Lua/Torch softmax agreement, seed continuity without global
RNG calls, truthful mode guards, preserved parent/macros/weights, all three audits
on synthetic traces, and rejection of stream resets or dependency tampering.
These tests alone make no gameplay or clearance claim.

## Native evaluation result (2026-09-12)

Using one frozen 16-action R1-trained model, policy seed 42, and Normal difficulty
3, the separate 20-attempt natural-coin test achieved **0/20 clears**. All 20
attempts ended in valid losses: Honda 7, Guile 7, Ryu 5, Chun-Li 1. The deepest
attempt won six opponent matches; none reached the boss section. Across the run,
Ken won 58 of 78 opponent matches and 127 of 177 individual rounds.

Native timing, action-interface and sampling audits all passed: 409,017 native
frames and 25,927 decisions were checked. LP uppercut was selected 4,071 times
(15.70%) and MP uppercut 477 times (1.84%). Model and executing source hashes,
including the frozen production dependencies, stayed unchanged.

A preceding one-attempt smoke test also passed all three audits and lost to
Honda; it is excluded from the 20-attempt denominator. Both sessions started with
the same policy seed, so their first attempts are not independent samples. One
earlier configuration failure occurred before MAME started and was retained
separately; it is not a gameplay attempt.

This sample does not support adopting categorical deployment as an improvement
for this model. It does not establish that every model or sampling seed would
perform worse than argmax. The experiment stopped at its fixed budget, without
policy changes or further training.
