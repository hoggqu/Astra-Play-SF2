# Continuous learned-policy evaluation

This experiment runs an SB3 PPO policy through a new Ken game. It is separate
from the frozen V4 CLI and uses `astra.rl-continuous.v1` result records.
No learned-policy result is frozen-policy certification.

```sh
python -m experiments.rl.continuous \
  --model .local/rl-runs/TRAIN_RUN/best-dev.zip \
  --output .local/rl-runs/FRESH_EVALUATION \
  --difficulty 3 --attempts 3 --speed fast
```

The output directory must not exist. The command owns one isolated MAME process,
uses the configured MAME 0.288/compatible ROM, and defaults to silent/headless.
`--show-window` is optional. It does not require a GPU, Agent service, or action
RPC to Python during combat. The optional training Python dependencies are
needed to load/export the SB3 model before MAME starts.

Each invocation boots once with the native DIP already configured. It validates
the decoded difficulty word, inserts a coin, selects Ken, and plays the native
route. Failed attempts end naturally before another ordinary coin. No continue,
reset, state load/save, or within-match pause is allowed. The run stops after its
first clear or the attempt cap; preceding losses remain in `result.json`.

## What is learned and what is reused

Only the exported network chooses fighting actions. There is no V4 fallback,
timeout defense, opening lead override, or hit/crossup macro cancellation.
The unchanged production entry and route orchestration handles character
selection, bonus stages, and transitions. The unchanged native settlement core
recognizes completed rounds and whole matches. Those administrative operations
are not learned. All modifications are to a copied experimental adapter under
the fresh run's `training/runtime/`; production/frozen source is unchanged.

The adapter exports only the policy MLP, with Linear/Tanh hidden layers and a
15-logit categorical head. The value network is not needed for inference. It
selects the largest logit deterministically, matching SB3 deterministic
prediction. The exported weights are captured before boot; model/runtime SHA256
identities are recorded, source changes invalidate an active match, and the
bridge rejects policy changes during fighting.

Observations and macros match the training interface:

- The same 86 clipped current-state features, stacked as four observations.
- At the first match opening, history contains four copies of the observed
  opening. The first action's frame-zero input is issued before unpausing.
- A new network decision every 12 emulated frames; intermediate macro inputs
  use the facing direction captured at decision time, even after crossing sides.
- History is reinitialized at each subsequent native round's opening. These
  rounds use the unchanged Core opening readiness gate; they are not checkpoint
  resets and need not have the same RNG/animation timing as a training reset.
- Neutral input during mature settlement and inter-round transitions.

## Result and exit status

`result.json` contains every attempt and references its native lifecycle record,
per-match result, round list, all-frame traces, and optional screenshot evidence.
A learned-policy clear has outcome `rl_gameplay_clear`: eleven mature native
whole-match wins in one attempt. It does not assert that ending images were
visually reviewed. A lost round is not itself a lost attempt; the match must be
lost natively. Invalid control or unresolved settlement is an invalid execution,
not a scored defeat. All preceding artifacts are retained.

Exit codes: **0** at least one clear, **1** valid attempt cap without a clear,
**2** setup/control/evidence failure. Use this experiment's `result.json`, not
`astra-sf2 audit`, which applies the frozen V4 contract and identity.

Offline checks:

```sh
python -m unittest experiments.rl.test_continuous -v
```

They compare Lua logits with Torch, feature ordering/history, twelve-frame input
cadence across hits and side switches, native round history resets, source
staging, and rejection of incomplete clears/lifecycle violations. They do not
replace native MAME evaluation.


## Fixed-sample natural-coin reliability runs

`--all-attempts` disables the historical stop-on-first-clear behavior in both
`continuous` and `native_continuous`, including generated 16-action/chain
packages. For example, run the frozen package's native evaluator with
`--attempts 20 --all-attempts` to retain exactly twenty valid attempts. A clear
finishes through the existing ending sequence; the next attempt uses the same
process and waits for native coin readiness before inserting a new coin. There
is no reset, load, continue, model replacement, or in-match pause.

An invalid execution stops the run immediately and remains in the evidence; it
is not replaced to fill the sample. Results record `stop_on_first_clear: false`.
A 10-of-20 target requires a completed run, twenty audited attempts, and at least
ten `rl_gameplay_clear` outcomes from the same model. Ten early clears never
shorten the twenty-attempt run. Individual native and action-interface audits
still apply. CLI exit 0 retains its existing meaning of at least one clear;
callers must check the full sample and their requested success threshold rather
than treating exit 0 alone as 10-of-20 success. Without the flag, the original
stop-on-first-clear default is preserved.
