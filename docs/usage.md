# CLI usage

The entry point is `astra-sf2`. Run `astra-sf2 --help` or add `--help` after a command for its current syntax. All examples assume the installed executable is on `PATH`; otherwise use `.venv/bin/astra-sf2` or `.venv\Scripts\astra-sf2.exe` directly.

## Commands

| Command | Purpose |
|---|---|
| `configure --mame PATH --rom-dir PATH [--data-dir PATH]` | Store local executable, ROM, and data paths. |
| `doctor` | Check the local setup and report what needs attention. |
| `verify --difficulty 3..7\|all [--attempts N] [--speed normal\|2x\|4x\|fast] [--consecutive N]` | Run a new isolated, bounded gameplay session. |
| `audit RUN_DIR` | Inspect the recorded gameplay evidence and constraints. |
| `review RUN_DIR --attempt ID --reviewer NAME --decision approve\|reject` | Record an offline review decision for one attempt. |
| `report RUN_DIR` | Report results for the preserved session. |

`3..7` means one integer: `3`, `4`, `5`, `6`, or `7`; do not type the literal range. Level 3 is Normal and level 7 is Hardest. Use `all` for all five levels. A bare `verify` defaults to difficulty `3`, one attempt, and `normal` speed.

| Speed | Target |
|---|---|
| `normal` | Original arcade speed (default) |
| `2x` | Twice original speed |
| `4x` | Four times original speed |
| `fast` | Unthrottled; as fast as the host can run |

Fixed rates are targets: a slow host may not sustain them. Entry waits and
gameplay use the chosen setting. Every match checks throttling, target rate and
the native base speed factor on every frame; changing them mid-match invalidates
the run. Version 0.1.4 adds the fixed rates without modifying the frozen policy.

```sh
astra-sf2 verify --difficulty 7 --speed 2x
astra-sf2 verify --difficulty 7 --speed 4x
```

## Single run and bounded verification

```sh
astra-sf2 verify --difficulty 3
astra-sf2 verify --difficulty 7 --attempts 5 --speed fast
astra-sf2 verify --difficulty all --attempts 20 --consecutive 5 --speed fast
```

A fresh `verify` creates a new run directory and one preconfigured MAME session per requested difficulty. `--attempts N` is a maximum per requested difficulty. `--consecutive N` ends a level early after that many consecutive automatic gameplay clears; every failure is retained and breaks the streak. When the cap is reached without the requested streak, the record remains unsuccessful rather than manufacturing more attempts. Audit failures and incomplete runs are not clears.

Without `--consecutive`, the runner uses the requested bounded attempt count. The speed is chosen before play. A failed game is allowed to finish naturally before a new coin; no continue, soft reset, hard reset, or state restoration substitutes for that transition. Each level boots a separate preconfigured process. All attempts within that level share it; a loss does not restart the emulator.

From 0.1.2, the runner waits for native new-game readiness instead of always
idling for 9000 frames. It inserts a coin when the attract task is active and
the preceding game has exited, then waits for the title screen to accept Start.
Each wait has a 9000-frame timeout; timeout invalidates the run without retrying
coin or Start. The CLI prints actual neutral frames waited. These waits use the
selected speed and never pause or alter an active policy-controlled match.
See [native entry readiness](entry-readiness.md) for the state checks and evidence.

Do not start another runner, attach a second input sender, reload Lua, alter policy files, or pause a live match. If you stop the process or something fails, retain the run and audit it; do not splice successful fragments into a complete attempt. A later command starts a new session rather than repairing the old record into a win.

A single-level run uses `astra.run.v2`. For `--difficulty all`, the root report
combines the five child directories `sessions/l3` through `sessions/l7`.
`audit`, `report` and `review` accept the root run directory and the usual attempt
IDs. Startup prints requested difficulty, raw DIP bits, and game-internal level;
for difficulty 7 the expected values are `7`, `0`, and `7`.

## Exit status

For `verify`, exit `0` means all requested attempts cleared, or the requested consecutive-clear target was met. Exit `1` means valid gameplay losses or an unmet target. Exit `2` indicates a setup, controller, or audit error. Preserve the run and inspect the report instead of treating every nonzero exit as an installation failure.

## Read and review results

Use the exact `RUN_DIR` and attempt identifiers from your command's output (for example, `l3-001`):

```sh
astra-sf2 audit RUN_DIR
astra-sf2 report RUN_DIR
astra-sf2 review RUN_DIR --attempt ATTEMPT_ID --reviewer "Alice" --decision approve
astra-sf2 report RUN_DIR
```

Before approval, actually inspect that attempt's saved selection screenshot, Bison result, and three ending screenshots. Read [verification rules](verification.md) for the distinction between native gameplay evidence and visual review. If the images contradict the record or do not support approval, record `reject` instead.

The same commands serve human users and agents. No AI is called to choose moves, wait between rounds, or determine the native gameplay result. An agent that reviews images is an optional offline reviewer and must be named as such.

## Troubleshooting

- **Executable not found:** activate the virtual environment or use its complete executable path. Run `configure` with the actual MAME executable path.
- **Version or driver mismatch:** install the supported MAME 0.288 and verify the intended `sf2` ROM set. Do not bypass the check to reuse memory addresses with a different game.
- **ROM audit failure:** correct your local ROM files using MAME's diagnostic output. A filename alone does not establish compatibility.
- **Session already owned:** inspect the existing runner. Do not remove a live ownership lock or start a second input sender to make progress.
- **Incomplete evidence or interrupted run:** preserve the directory and use `audit`. Start a fresh session only after the existing owner has stopped; never report partial progress as a clear.
- **Difficulty mismatch or legacy evidence:** upgrade to 0.1.1 and start a new run. Do not override a mismatch. Old v1 runs remain readable but cannot certify internal difficulty; see [the correction](difficulty-fix.md).
- **Review pending:** automatic gameplay can finish without an offline reviewer. Report it as such, or inspect the evidence and explicitly record a review; do not fabricate an approval.

This port's runs are independent of completed historical V4 IDs, bootstrap instances, and local compatibility paths. Do not use legacy training controllers as substitutes for this CLI.
