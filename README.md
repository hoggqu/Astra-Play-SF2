# Astra-Play-SF2

[中文说明](README.zh-CN.md) · [Runtime architecture](docs/architecture.md)

Play **Street Fighter II: The World Warrior (World 910522)** as Ken with a fixed Lua policy and a Python command-line runner. Gameplay runs locally in MAME; it needs no AI model, API key, or online service.

The runner uses ordinary player-one controls and current game state. Each verification starts an isolated session, then uses natural game completion and coin insertion between attempts. It does not load states, continue a defeated game, reset between attempts, or pause inside a match to change its policy.

**Required:** Python 3.10+ and **MAME 0.288**, installed separately, plus your own compatible `sf2` ROM set. MAME, ROMs, save states, screenshots, and historical run logs are not included in the package. See [installation](docs/installation.md) for official download links.

## Quick start

From a local checkout, install into a virtual environment:

```sh
# macOS / Linux
bash scripts/bootstrap.sh
source .venv/bin/activate
astra-sf2 configure --mame /absolute/path/to/mame --rom-dir /absolute/path/to/roms
astra-sf2 doctor
astra-sf2 verify --difficulty 3
```

```powershell
# Windows PowerShell
.\scripts\bootstrap.ps1
.\.venv\Scripts\astra-sf2.exe configure --mame "C:\MAME\mame.exe" --rom-dir "C:\MAME\roms"
.\.venv\Scripts\astra-sf2.exe doctor
.\.venv\Scripts\astra-sf2.exe verify --difficulty 3
```

The bootstrap scripts create `.venv`, install this checkout with pip, and run diagnostics. They do not install Python or MAME, obtain ROMs, elevate privileges, or start a game. You can instead create a virtual environment yourself and run `python -m pip install .`.

`verify` defaults to one attempt at normal speed. To run a bounded experiment:

```sh
astra-sf2 verify --difficulty 7 --attempts 5 --speed fast
astra-sf2 verify --difficulty all --attempts 20 --consecutive 5 --speed fast
```

`--attempts` is the cap **per difficulty**. The second command tries levels 3 through 7 and stops each level after five consecutive gameplay clears or its cap. Losses remain in the report and break the streak; a cap does not guarantee success. See [CLI usage](docs/usage.md).

配置文件默认保存在 `~/.astra-play-sf2/config.json`。首次运行先执行 `configure` 和 `doctor`；不需要 Agent 在每局手动操作。默认正常速度，只有显式传入 `--speed fast` 才快进。

## What counts as a clear?

An automatic `gameplay_clear` records eleven native match wins under the run's gameplay constraints. It is **not** a claim that a person or AI has visually checked the character selection or ending.

The runner saves selection, Bison-result, and three ending screenshots for optional offline review. Inspect those images and the audit before recording a review:

```sh
astra-sf2 audit RUN_DIR
astra-sf2 review RUN_DIR --attempt ATTEMPT_ID --reviewer "Your name" --decision approve
astra-sf2 report RUN_DIR
```

Use the run directory and attempt ID printed by the CLI. `approve` records the named reviewer's decision; it does not make the program inspect images, turn a failed game into a clear, or erase previous attempts. Read [evidence and review](docs/verification.md) before publishing results.

## Validation and history

The original V4 policy completed five consecutive clears at each difficulty from Normal to Hardest on its original macOS setup. Across **55 attempts**, it cleared **45** and lost **10**: an observed 81.8% clearance rate, not a guarantee. The standalone runner is a separate port; historical results do not certify this release on another OS or emulator build. Details and the frozen source identity are in [validation history](docs/validation.md).

The Python tooling is designed for macOS, Linux, and Windows. CI defines unit tests and wheel builds on all three systems with Python 3.10 and 3.12. CI does not run copyrighted game data or certify emulator gameplay. Actual platform smoke-test evidence must be reported separately.

## Working with an agent

Agents read [AGENTS.md](AGENTS.md) and use the same CLI. One controller owns a session; other agents may inspect completed files. No agent needs to intervene during gameplay. Any visual review must come from actually inspecting that attempt's saved evidence, not from assuming an automatic result is visually verified.

For development:

```sh
python -m pip install .
python -m unittest discover -s tests -v
python -m pip wheel --no-deps --wheel-dir dist .
```

No hosted service or GitHub publication is needed to run locally.
