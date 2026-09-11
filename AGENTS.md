# Astra-Play-SF2: Agent handoff

This checkout is a standalone Python CLI plus a frozen MAME Lua policy. An AI
model is not part of its runtime. Use the same commands as a human operator.

## Start here

1. Read [README](README.md), [installation](docs/installation.md), and
   [verification rules](docs/verification.md).
   Read the [0.1.1 difficulty correction](docs/difficulty-fix.md): historical DIP-only
   cross-difficulty results are pending revalidation, not current certification.
2. Install with `scripts/bootstrap.sh` (macOS/Linux) or `scripts/bootstrap.ps1`
   (Windows). Python 3.10+ and MAME **0.288** are required.
3. Find an existing MAME 0.288 executable and compatible `sf2` ROM directory.
   If either is missing, you are authorized to download and install the missing
   dependency **inside this checkout** using the procedure below. Do not stop
   merely to ask permission for a project-local download or installation.
4. Run `astra-sf2 configure --mame PATH --rom-dir PATH` and `astra-sf2 doctor`.
5. Run `astra-sf2 verify --difficulty 3`. Default: one attempt, normal speed.
   `--speed fast` enables fast-forward when requested. `--difficulty all
   --attempts 5` runs five attempts at each level, Normal (3) to Hardest (7).
6. Keep the printed run directory. Run `astra-sf2 audit RUN_DIR` and
   `astra-sf2 report RUN_DIR`. Report every attempt, loss and invalid execution.

## Missing dependencies: automatic project-local setup

Use `.local/` under the repository root; it is ignored by Git. Keep downloads in
`.local/downloads/`, MAME in `.local/mame/0.288/`, compatible ROMs in
`.local/roms/`, and verification data in `.local/data/`. Reuse a working existing
installation when available. Do not overwrite the original `MAME/` training
archive or replace dependencies used by an active session.

- **Search autonomously:** you may freely search for suitable MAME and compatible
  ROM downloads; the recommended links are starting points, not an allowlist.
  Find, download, install and check missing files yourself without requiring the
  player to supply a URL or approve each source first. Prefer official or
  reputable sources. Ask the player for help only after searching and trying
  suitable alternatives without finding a usable download or installation path;
  explain what you tried and the specific remaining obstacle.
- **MAME:** detect the OS and CPU architecture, then download, extract or build
  **0.288**, starting with the recommended links below. Install the executable and required
  runtime files together inside the project. On macOS/Linux, build the tagged
  source if a matching binary is unavailable. Follow the platform build
  instructions for prerequisites; a system-required privilege/login prompt may
  need the user, but ordinary local setup does not require another confirmation.
- **ROM:** independently search for and download a compatible `sf2` / World
  910522 set. You do not need a player-provided download URL to begin. The absence
  of a recommended SF2 link in this document is a reason to search, not to stop
  and ask the player. Preserve the archive as
  `.local/roms/sf2.zip`; do not change another SFII edition's filename to make it
  appear compatible. If a candidate fails the ROM audit, look for another
  compatible set. Only if that search finds no usable source, finish the
  emulator/CLI setup and ask for the compatible ROM or its download location.
  Do not substitute the unrelated free games below.
- **Verify and continue:** confirm `mame -version`, run `configure` with the
  actual local executable/ROM/data paths, then `doctor`. Proceed with the
  requested verification only after its version and ROM audit pass. A filename
  alone is not a compatibility check. Record source URLs, downloaded file hashes
  and versions locally; do not add downloads or ROMs to commits or releases.

### Recommended downloads (checked 2026-09-11)

| Dependency/platform | Recommended source |
|---|---|
| MAME 0.288 release and asset list | [Official pinned release](https://github.com/mamedev/mame/releases/tag/mame0288) |
| Windows x64 | [mame0288b_x64.exe](https://github.com/mamedev/mame/releases/download/mame0288/mame0288b_x64.exe) — extract this self-extracting archive into the local MAME directory |
| Windows ARM64 | [mame0288b_arm64.exe](https://github.com/mamedev/mame/releases/download/mame0288/mame0288b_arm64.exe) |
| Official binary checksums | [SHA256SUMS](https://github.com/mamedev/mame/releases/download/mame0288/SHA256SUMS) |
| macOS/Linux and source builds | [Source at tag mame0288](https://github.com/mamedev/mame/tree/mame0288) and [platform build instructions](https://docs.mamedev.org/initialsetup/compilingmame.html) — use this tag, even if the documentation describes a newer version |
| Python | [Official downloads](https://www.python.org/downloads/) — Python 3.10+ |
| Authorized free MAME games, for reference only | [MAME's free ROM downloads](https://www.mamedev.org/roms/) — **does not include this project's SF2 set** |

Project-local commands and the ROM compatibility requirement are detailed in
[installation](docs/installation.md#agent-managed-project-local-installation).

## Verification contract

- One CLI owns at most one MAME process and inbox at a time. Each difficulty
  boots a separate preconfigured session; attempts within that level share the
  same process and use natural coin insertion. Never attach another sender. The data
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
- Configure native DIP before process launch. Check both the port and the game's
  decoded difficulty word at `0xFF82C6` before play and throughout each match.
  Never bypass a mismatch or patch RAM to make it pass. Legacy v1 evidence lacks
  this check and cannot certify a difficulty; preserve it as provisional history.

## Development

- Training knowledge is included as portable documentation: read
  [training method](docs/training-method.md), [training history](docs/training-history.md)
  and [matchup lessons](docs/matchup-lessons.md) when asked to understand or improve
  the policy. These Chinese-language guides replace the need for the private
  historical Skill when learning the method; they contain no raw run data.
  The release provides verification, not the old checkpoint collectors or batch
  trainer. Do not invent a `train` command or assume private scripts are present.
  New training needs an isolated experiment and explicit new policy identity;
  reading these guides alone is not an instruction to start playing or training.

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
