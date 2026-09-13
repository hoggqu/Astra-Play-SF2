# Watch the current PPO model play

Run this independent viewer from the source checkout using the Python environment
that has the optional RL dependencies installed. It opens a silent MAME window
and plays one natural coin at normal speed. It does not start training, choose a
new model after play begins, or add results to a training campaign.

```sh
python -m experiments.rl.watch --model path/to/ppo-batch.zip
```

To watch the latest **published completed checkpoint** of a campaign:

```sh
python -m experiments.rl.watch --campaign .local/rl-runs/my-campaign --speed 2x
```

`--campaign` accepts the campaign output directory or its `result.json`. The
Top-level autonomous-training outputs with `run/result.json` are also accepted. The
record must contain `latest_model` and `latest_model_sha256`. The viewer checks
the hash and copies that ZIP before launch; an active trainer may publish a newer
checkpoint without changing the model being watched. It never guesses a model
from modification times or opens a checkpoint still being written. If the
campaign has not yet published a checkpoint, use a known completed `--model`
ZIP or wait for publication.

The viewer uses the current **128×128 PPO**, 85-action, 4516-observation
perception interface. A model must be a compatible full PPO ZIP, including its
policy, critic and optimizer. It cannot load the frozen rule-based policy or an
older 16-action network. Supply checkpoints you trust, as with other PyTorch
model loading.

## Setup and options

Configure MAME 0.288 and the compatible World 910522 ROM using the normal
[installation instructions](../../docs/installation.md). Windows and Linux need
a graphical desktop session for the visible window. On macOS it is a normal
native window. The runtime starts with `-noreadconfig -window -sound none` and
never enables always-on-top or repeatedly raises the window. When launched
through SSH on WSL2, an existing WSLg `/tmp/.X11-unix/X0` socket is detected if
`DISPLAY` is missing; the viewer then uses `DISPLAY=:0` and SDL X11. Explicit
display settings are preserved. This opens the window on the remote Windows
machine's WSLg desktop, not on the SSH client's desktop.

Without `--code`, the viewer builds an isolated current managed revision (including v8 draw settlement) from public
sources inside its own output directory. To reuse an existing validated timing build or managed training build:

```sh
python -m experiments.rl.watch --model path/to/ppo-batch.zip \
  --code path/to/timing-build --config path/to/config.json \
  --difficulty 7 --speed 2x
```

| Option | Behavior |
| --- | --- |
| `--model ZIP` | Explicit completed checkpoint; mutually exclusive with campaign |
| `--campaign PATH` | Read a published latest checkpoint and its hash |
| `--code PATH` | Reuse a validated perception128 input-timing build |
| `--config FILE` | Use this configuration without changing the user's default |
| `--difficulty 3..7` | Normal through Hardest; default 3, set before boot |
| `--speed normal/2x/4x/fast` | Default normal; fast removes throttling |
| `--attempts N` | Default one natural coin; additional coins follow native game end |
| `--output PATH` | New viewing directory; default `.local/rl-watch/<unique-id>` |

The process uses the native continuous-play executor: no state loads, Continue,
combat pauses, model replacements or speed changes inside a match. It retains
its lifecycle/timing checks so a broken run is visible instead of being presented
as normal gameplay. Those checks do not schedule an automatic verification
campaign. A loss finishes the viewing session normally. If multiple attempts are
requested, every requested coin is played even after a clear.

Results, the fixed model copy and source identity belong only to the viewing
directory. `watch.json` marks `purpose=visual_play_only` and
`training_statistics=false`; `play/` contains the isolated native runtime and
its evidence. No campaign result or training statistics file is edited.

Exit 0 means the requested viewing session finished, whether Ken won or lost.
Exit 2 means setup, runtime failure or interruption. Ctrl+C stops the viewer;
the interruption is retained, and the viewer does not retry the coin. Closing
the MAME window also ends the session and may be recorded as an interrupted run.

Do not point `--output` at an existing training or watching directory. The viewer
requires a new directory and owns only its own MAME process. Watching consumes
one additional emulator process; choose a free local machine while larger
training jobs run.
