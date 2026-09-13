# CUDA training on an NVIDIA GPU

The managed training interface supports `--device cpu`, `--device cuda`, and
`--device auto`. CPU remains the default. CUDA moves the PyTorch policy/value
calculations and PPO gradient updates to the GPU; MAME game execution, structured
perception and the Lua policy used during sampling remain on the CPU. A GPU does
not make the whole training loop GPU-based.

Requesting CUDA when unavailable fails with an actionable error. `auto` selects
CUDA when available and CPU otherwise; the requested and resolved device are
recorded. Explicit device selection is preferable for reproducible experiments.
The same checkpoint includes the full optimizer state when continued on another
device. The action interface, observations, rewards, terminal rules and sampling
protocol remain unchanged. CPU and CUDA training need not produce bit-identical
updates or identical future game trajectories.

## Independent environment

Install CUDA-enabled PyTorch in a separate project-local environment while an
existing CPU environment is in use. For example, on a compatible Linux/WSL2
machine with Python 3.12:

```sh
python3 -m venv .local/rl-cuda-venv
.local/rl-cuda-venv/bin/python -m pip install \
  torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130
.local/rl-cuda-venv/bin/python -m pip install stable-baselines3==2.7.1
.local/rl-cuda-venv/bin/python -c \
  "import torch; print(torch.__version__, torch.cuda.is_available())"
```

These are the versions used for the reported comparison. Other platforms or
drivers may need a different supported CUDA wheel; use the
[official PyTorch installation instructions](https://pytorch.org/get-started/locally/).
Use this environment's Python to run the managed trainer, adding `--device cuda`
to the existing training command. No separate GPU inference server is required.

For WSL2, the Windows NVIDIA driver supplies CUDA support. Do not install a Linux
display driver inside WSL. If `nvidia-smi` is absent from PATH, try
`/usr/lib/wsl/lib/nvidia-smi`; this is a documented WSL configuration.
[NVIDIA's WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).

## Measured benefit and its limits

A bounded RTX4090 comparison used the same 4516-input, 128×128 actor/critic,
85-action model, 4096 samples, minibatch64 and four PPO epochs. On the same CUDA
PyTorch build, GPU gradient updates were approximately **2.2× faster** than CPU
updates. The comparison used recorded training observations and fixed synthetic
advantages; the resulting parameter updates were discarded. It demonstrates
compute throughput, not improved playing strength.

Policy serialization back to CPU/Lua remained essentially unchanged. In the
measured full CPU training window, roughly one quarter of wall time belonged to
the update section. Applying the measured pure-gradient savings would imply
roughly **8% total improvement**, not 2.2× end-to-end speed. Batch value/logprob
calculations may provide additional savings, while transfers and setup can reduce
them. This projection must be checked with real end-to-end runs; it is not a
promised training speedup.

Small MLP policies can be slower on GPU in other configurations, which is also
noted in the [SB3 PPO documentation](https://stable-baselines3.readthedocs.io/en/v2.7.1/modules/ppo.html).
Do not change batch size, epochs or learning parameters just to make the GPU busy
and then attribute the resulting learning differences solely to the device.

## Numerical and runtime checks

The managed CUDA path keeps TF32 disabled and retains the existing Lua/Torch
old-policy log-probability tolerance. Observations and actions are placed on the
model's device; the complete old value/logprob arrays are copied back once before
filling the CPU rollout buffer. This avoids many small device transfers without
changing GAE or terminal masks. Lua payloads and final exports remain ordinary
CPU values; formal gameplay still uses the same frozen Lua neural inference.
The device/buffer behavior follows the
[SB3 on-policy implementation](https://stable-baselines3.readthedocs.io/en/v2.7.1/_modules/stable_baselines3/common/on_policy_algorithm.html).

The compute comparison passed the existing tolerance on the tested observations
and actions. A fresh bounded native run is also required before using a new
runtime/device configuration for long training. This checks real sampling,
old-policy log probabilities, updates, complete checkpoint/Adam saving, and
emulator cleanup. It is a training validation, not a formal clear or win-rate
certificate.

A separate real native CUDA test has now completed **4096 steps with four
workers**. All four updates passed Lua/Torch old-policy checks (maximum error
below 1.8×10⁻⁶), parameters changed, the complete checkpoint reloaded with all
12 Adam states and the expected update counts, and every owned emulator exited.
Frozen source and staged-runtime hashes also matched. This validates the CUDA
sampling/update/save/cleanup chain; it is not a long-run throughput measurement
or a gameplay performance certificate.

Resource limits must cover the combined jobs sharing a parent cgroup, including
validation jobs. A per-job memory limit is not additional capacity beyond that
shared limit. Before adding a GPU test alongside CPU training, account for its
CPU workers and host RAM as well as its GPU memory; testing sequentially avoids
that contention. A worker EOF is only a symptom: inspect its first underlying
exception and memory events before calling it an out-of-memory failure.
