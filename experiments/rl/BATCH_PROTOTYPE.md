# Optional batched PPO sampler prototype

This module is independent of the active `train.py` pipeline. Do not adopt it
merely because its offline tests pass: first run native parity and a paired
throughput benchmark. It changes transport and policy execution placement, not
the reward, observation, action macros, or PPO update objective.

```sh
# Fresh directories; each command owns only its own MAME process(es).
python -m experiments.rl.batch_train --dataset DATASET/manifest.json \
  --output .local/rl-runs/batch-parity-001 --native-parity --steps 256

python -m experiments.rl.batch_train --dataset DATASET/manifest.json \
  --output .local/rl-runs/batch-benchmark-001 --benchmark \
  --workers 4 --steps 4096 --block 64 --init-model MODEL.zip

# Bounded prototype training, only after parity/throughput checks:
python -m experiments.rl.batch_train --dataset DATASET/manifest.json \
  --output .local/rl-runs/batch-train-001 \
  --workers 4 --steps 4096 --block 64 --init-model MODEL.zip
```

`--native-parity` compares the new deployment Core driving path and batch
execution with the same checkpoint, lead, and predetermined actions. The
reference uses `native_continuous_core.lua` in a training-only checkpoint harness,
with fighting inputs issued by its independently implemented `drive` method. It requires equal float32
observations, exact observed native state, and equal terminal flags; rewards
allow only 1e-12 JSON floating-point serialization error. A short parity run may
not reach a terminal reset; run up to 256 decisions and inspect `episodes` before
claiming native reset parity. The original and batch environments both remain
training-only: state loads and synchronous boundary pauses are permitted.

The older `--parity` diagnostic compares against an isolated native-time
corrected copy of the paused RPC interface. It is retained for investigation,
not the acceptance gate: removing its duplicate-time callbacks aligned native
time but still produced different trajectories. See [native timing](NATIVE_TIMING.md).

Batch size can be 1, 16, 32, 64, 128, or 256. PPO still updates only after 256
transitions per worker. One weight snapshot is sent to each worker at the start
of that rollout; later blocks reuse it. Within a block, Lua samples actions
from policy softmax probabilities using uniform numbers supplied by the worker,
executes each fixed twelve-frame macro, waits for mature native terminal
results, and restores the next scheduled checkpoint after terminal episodes.
Uniform-opponent/within-opponent checkpoint sampling and training lead choices
are preserved. RNG streams are reproducible per worker but are not the old
Torch categorical RNG stream, so a matching seed does not imply an identical
stochastic training trajectory.

Lua returns each transition's old observation, action, reward, episode-start and
terminal flags, selected-action log probability, and native state. Torch then
computes old values and log probabilities in a batch **before any update**. A
Lua/Torch log-probability discrepancy above 2e-5 invalidates the run. SB3's
unchanged rollout buffer computes GAE/returns and its unchanged PPO `train()`
performs the updates. Final-state bootstrapping respects terminal reset masks.
The value network does not need a Lua implementation.

This prototype does not include development model selection or final held-out
validation; its final `ppo-batch.zip` is a last-iteration model, not a selected
best model. Use existing round and continuous evaluators to assess it. Its
`astra.rl-batch-prototype.v1` result cannot establish a gameplay clear.

`sampling_seconds` excludes model serialization before the sampling loop and
PPO updates; `update_seconds` includes batched Torch value/log-probability
calculation, GAE, and optimization (or only validation/GAE in benchmark mode).
`wall_seconds` additionally includes boot, preflight, staging, and cleanup.
Use end-to-end wall time as well as sampling throughput when comparing methods.
Benchmarking a frozen learned policy is not directly interchangeable with the
old random-action benchmark: different actions cause different episode lengths
and terminal/reset overhead.

Completed native round records are appended to
`worker-NN/training/rl-batch-episodes.jsonl` before every automatic reset. They
remain available if a later error interrupts the enclosing batch. Completed
Python batch results also populate the usual worker `episodes.jsonl`; do not
add those two copies together. Failed or interrupted runs remain invalid and
are never replayed into the same output directory.

Offline tests:

```sh
python -m unittest experiments.rl.test_batch -v
```

## Bounded native evidence (2026-09-11)

The deployment-Core/batch harness passed **256 decisions including one mature
round ending and automatic reset**, with identical complete native states,
float32 observations, rewards, and terminal flags. This is training-interface
parity, not a game clear. Earlier failed RPC comparisons remain invalid records.

Under a two-CPU/two-GiB service cap within the shared training slice, a two-worker
frozen-policy benchmark completed 2,048 decisions in **16.744 sampling seconds
(122.31 decisions/s)**, with 36.00 seconds total wall time. A separate real PPO
smoke completed 512 decisions, ran four optimization epochs, and confirmed that
network parameters changed. Its sampling took 3.954 seconds and its batched
value/logprob/GAE/optimization work took 0.189 seconds. Maximum Lua/Torch selected
log-probability error was **4.90e-7**, below the 2e-5 rejection threshold.
These measurements do not establish the eventual eight-worker speedup or policy
strength, and the frozen-policy benchmark did not update model parameters.
