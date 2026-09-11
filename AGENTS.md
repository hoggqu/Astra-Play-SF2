# Astra-Play-SF2: Agent handoff

This checkout is a standalone Python CLI plus a frozen MAME Lua policy. An AI
model is not part of its runtime. Use the same commands as a human operator.

## Start here

1. Read [README](README.md), [installation](docs/installation.md), and
   [verification rules](docs/verification.md).
2. Install with `scripts/bootstrap.sh` (macOS/Linux) or `scripts/bootstrap.ps1`
   (Windows). Python 3.10+ and MAME **0.288** are required.
3. Find the user's MAME executable and compatible `sf2` ROM directory. If missing,
   ask for those paths; do not download or distribute game ROMs.
4. Run `astra-sf2 configure --mame PATH --rom-dir PATH` and `astra-sf2 doctor`.
5. Run `astra-sf2 verify --difficulty 3`. Default: one attempt, normal speed.
   `--speed fast` enables fast-forward when requested. `--difficulty all
   --attempts 5` runs five attempts at each level, Normal (3) to Hardest (7).
6. Keep the printed run directory. Run `astra-sf2 audit RUN_DIR` and
   `astra-sf2 report RUN_DIR`. Report every attempt, loss and invalid execution.

## Verification contract

- One CLI owns one MAME process and inbox. Never attach another sender. The data
  directory has `verify.lock`; inspect its PID and process before removing a
  stale lock. Never remove a live lock or kill another controller.
- Let the CLI finish. It inserts a new coin after the native game ends. No
  continue, state load/save, or reset between attempts. Each whole opponent
  match runs continuously without pause, speed change or policy replacement.
- No policy edits, training, reset-based opponent selection, failure deletion,
  or state recovery during verification. A timed-out command is never replayed.
  An interruption invalidates the run; preserve it and start a new directory.
- `gameplay_clear` means eleven mature native match wins, not an inspected Ken
  ending. For visual review, actually inspect the selection, Bison result and
  all three ending images, then run `review RUN_DIR --attempt l3-001
  --reviewer NAME --decision approve|reject`. Never fabricate a review.
- `--consecutive 5 --attempts 20` is a bounded streak test per difficulty. Retain
  preceding losses in the denominator; a streak is not 100% overall clearance.
  Exit 0: requested success criterion met; 1: valid loss or unmet streak;
  2: setup/control/audit failure.
- The Lua policy reads current-state RAM and issues ordinary P1 button inputs.
  It never writes game RAM. All-frame traces and lifecycle audits are retained.

## Development

- Code: `src/astra_play_sf2`; packaged Lua/JSON: `assets`; tests: `tests`.
  Run `python -m unittest discover -s tests -v` and build/install the wheel.
  CI defines Windows/Linux/macOS checks without an emulator or ROM.
- `fighter.lua`, `play_core.lua`, and `selection.json` are byte-identical V4
  inputs. Identity is in `assets/provenance.json`. Do not silently change them
  or apply historical win-rate claims to a different policy/runtime.
- Use pathlib, subprocess argument lists, UTF-8 files and relative Lua paths.
  Outputs belong in the configured data directory. Never commit ROMs, states,
  screenshots, private logs or absolute user paths.
- The original workspace may contain ignored `MAME/` and `skills/`. They are
  historical material, not portable dependencies. Before explicitly requested
  legacy training, read local `skills/mame-sf2-ken/SKILL.md` and
  `MAME/training/handoff.txt`. Synchronize the local and personal copies if
  modifying that legacy skill. Standalone verification needs neither.
