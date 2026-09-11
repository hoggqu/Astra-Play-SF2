# Independent LP/MP uppercut action experiment

This candidate adds one network-selected action to the original 15-action PPO
interface. It does not change the frozen V4 policy or edit the original RL
execution files. It is an experiment, with no clearance claim from offline tests.

The identity is `ken_actions16_lp_mp_uppercut_v1`. Original actions 0–14 retain
their exact inputs. New action 15 copies action 13's fixed-facing uppercut:
two frames forward, two down, two down-forward plus **MP**, six neutral. Action
13 still uses LP. The network chooses between them; there is no opponent rule.
The observation remains 344 numbers, actor and critic remain 64/64 Tanh MLPs,
reward and native 12-frame cadence are unchanged.

Migration preserves the actor trunk, critic, value head, and all non-uppercut
logits. Actions 13 and 15 share the old action-13 weights; each bias is reduced by
`log(2)`. Therefore each gets half the former categorical probability and their
combined probability stays the same. Other action probabilities stay the same,
up to floating-point precision. **Deterministic argmax can change** because the
individual uppercut logits decrease. This is a new action interface, not an
identical deterministic policy.

The migration deliberately starts a fresh optimizer and resets training counters.
It retains PPO hyperparameters but is not an exact optimizer continuation.
`migration.json` records original and derived model hashes, counters, and this
reset. Later training checkpoints preserve the new optimizer normally.

## Build and run

Use the project's installed Python environment with optional RL dependencies.
Generate the snapshot only after the desired shared native settlement protocol
is stable. Every output below must be a fresh directory. These commands work on
macOS, Linux and Windows; substitute actual dataset/model paths.

```console
python -m experiments.rl.actions16_builder --output .local/actions16-code-001
python .local/actions16-code-001/launch.py migrate --model OLD15.zip --output .local/actions16-model-001 --seed 42
python .local/actions16-code-001/launch.py batch_train --dataset DATASET.json --output .local/actions16-parity-001 --init-model .local/actions16-model-001/ppo-actions16.zip --native-parity --workers 1 --steps 256 --seed 42
python .local/actions16-code-001/launch.py batch_train --dataset DATASET.json --output .local/actions16-smoke-001 --init-model .local/actions16-model-001/ppo-actions16.zip --workers 2 --steps 512 --block 64 --seed 42
python .local/actions16-code-001/launch.py batch_train --dataset DATASET.json --output .local/actions16-train-001 --init-model .local/actions16-model-001/ppo-actions16.zip --workers 8 --steps 102400 --block 64 --seed 42
python .local/actions16-code-001/launch.py native_continuous --model .local/actions16-train-001/ppo-batch.zip --output .local/actions16-verify-001 --difficulty 3 --attempts 3 --speed fast
```

First require parity `status=complete`, `native_parity=true`, 256 decisions and
positive counts for every action in `action_counts`. Its first 16 decisions cover
all actions, including MP uppercut. This diagnostic uses forced actions, not
neural decisions; `--init-model` validates identity only. It owns two emulator
processes (reference and batch), regardless of `--workers 1`.

The 512-decision smoke test performs a real PPO update, not `--benchmark`.
Require `status=complete`, `actual_steps=512`, `parameters_changed=true`,
`optimizer_epochs_completed=4`, and the saved final model/hash. Its result's
`optimizer_initialization` records the migrated optimizer's origin and initial
zero updates/state entries. Training, export, campaign and verification entries
reject a 15-action model before starting MAME; only migration accepts the old
interface. Parity requires at least 16 decisions.

For an authorized bounded campaign, replace the final 102400-decision training
and verification commands with:

```console
python .local/actions16-code-001/launch.py native_campaign --dataset DATASET.json --output .local/actions16-campaign-001 --init-model .local/actions16-model-001/ppo-actions16.zip --workers 8 --cycles 5 --steps-per-cycle 102400 --verification-attempts 1 --seed 42
```

The campaign starts from the migrated model, not the smoke-test output, keeping
the smoke update outside the formal comparison budget. Do not resume the same
output directory or silently add diagnostic learning to one comparison arm.

The generated `astra_sf2_rl16` namespace contains an isolated copy of necessary
RL sources. `launch.py` also exposes `export`; it propagates this package path to
campaign subprocesses. It still requires the installed `astra_play_sf2` package,
its original assets, and a working configured MAME/ROM. The copied generic
`campaign.py` supplies lifecycle helpers only; use `native_campaign`, not that
older campaign or the old 15-action training entries.

`build.json` records original and generated source hashes and every checked patch
anchor. Unknown source changes cause a build error rather than approximate
patching. Runtime payloads/results have distinct actions16 schemas and explicit
interface identity. Export, training and Lua reject a mismatched or unlabelled
model. Campaigns freeze actual generated execution sources plus production
dependencies for the run. Original 15-action sources remain available unchanged.

The generated native verifier also writes `action_interface_audit`. Before MAME
starts it checks package files against the build manifest, the model identity,
the exported schema/dimensions/cadence, exact staged payload bytes, and the staged
actions/NN/core/settlement files. After the run it rechecks package, model and
staged hashes, plus every match's model/interface identity. Only successful
checks yield `ok=true`; failures mark the run invalid and seal the evidence.
The audit reports checked file/match counts and source hashes for the report CLI.

Before substantial training, perform the existing native parity and short
benchmark/update checks in this generated namespace; offline neural parity does
not certify MAME timing or the native settlement protocol. Verification retains
the native audit, natural coins, loss records and no pause/load/continue rule.
Keep action-interface results separate from 15-action statistics.

## Offline checks

```console
python -m unittest experiments.rl.test_actions16 -v
```

Tests cover original macros in both directions, the MP-only addition, probability
mass preservation, unchanged critic/trunk, fresh optimizer, persisted identity,
16-output Lua/Torch parity, rejecting incompatible models, native staging and the
portable launcher. Tests do not start an emulator or alter shared sources.
They also check complete parity action coverage, preflight rejection before
emulator startup, optimizer provenance and byte-identical copying of the current
default settlement, batch runtime and native core. Rebuild a fresh snapshot after
the default native protocol is promoted; existing snapshots remain unchanged.
