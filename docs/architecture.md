# Runtime and policy handoff

Python handles setup, bounded attempts, natural coin insertion, evidence and
reports. Lua executes every decision during a whole opponent match. Neither
layer calls an AI service. A reviewer can inspect screenshots after the run.

## Execution flow

```text
configure → doctor → unique run directory → isolated MAME process
  → preconfigured native DIP at boot → Lua checks DIP and decoded RAM difficulty
  → wait until native attract task is ready (9000-frame upper bound)
  → coin/start/Ken selection → validate live R1 → play_match
  → mature match settlement → next opponent / bonus stage
  → defeat: retain loss and let continue countdown expire
  → 11 wins: save Ken ending evidence
  → next ordinary coin, in the same MAME process
  → process exit → evidence hashes → audit → report → optional visual review
```

The file inbox is private to one run directory. The Python sender logs a command
before publishing it, waits for acceptance and (for a bounded navigation action)
a new paused, idle screenshot observation. MAME rejects arbitrary bridge
commands during an active match. A timeout is not a retry instruction.

Physical input providers are disabled for the verification process. Whole-match
input consists only of the ten fighting directions/buttons; Coin and Start are
restricted to orchestration. The game RAM is read, not patched. Closing or
interrupting an owned process invalidates incomplete work.

## Source map

| Source | Responsibility |
|---|---|
| `src/astra_play_sf2/cli.py` | Human/Agent entry point and exit codes |
| `config.py` | Per-user paths; MAME version and ROM preflight |
| `runner.py` | Process ownership, difficulty/attempt budgets, transitions and sealing |
| `transport.py` | Single-writer inbox, command log, no-replay waits |
| `opening.py` | Strict live R1 or narrowly recognized uninitialized-intro predicates |
| `assets/bootstrap.lua` | Noninteractive module loading and capability checks |
| `assets/difficulty.lua` | Read-only DIP, game mirror, decoded difficulty and AI diagnostic fields |
| `assets/session.lua` | Native reset/load/save counters and fixed-difficulty session checks |
| `assets/control.lua`, `bridge.lua`, `observe.lua` | Ordinary input jobs, guarded inbox and current-state snapshots |
| `assets/fighter.lua` | Frozen V4 policy and current fighter reads |
| `assets/play_core.lua` | Frozen pure whole-match/round state machine |
| `assets/play.lua` | MAME adapter, all-frame trace, input and lifecycle enforcement |
| `assets/selection.json`, `provenance.json` | Chosen mode per opponent and source hashes |
| `evidence.py` | Raw-log integrity/settlement checks, separate review records and reports |

## Current-state observation

This address map is for **MAME 0.288, `sf2`, World 910522 only**. For player index
`i` (Ken 0, CPU 1), the fighter structure starts at `0xff83c6 + i * 0x300`.

| Offset/address | Read |
|---|---|
| `+6`, `+10` | Signed x and y position |
| `+42` | Signed internal HP |
| `+3`, `+0x1a` | Action byte and animation pointer |
| `+0x291` | Character ID; Ken is 4 |
| `+0x290` | Native rounds won |
| `+0x1bc`, `+0x164` | Displayed HP and timeout HP |
| `0xff8ace` | BCD round timer |
| I/O port `:DSWB & 7` | Difficulty bits: `7 - CLI difficulty` |
| `0xff808b & 7` | Inverted DIP B mirror, equals CLI difficulty |
| `0xff82c6` (16-bit) | Game-decoded difficulty, equals CLI difficulty |

Lua uses `manager.machine.devices[':maincpu'].spaces['program']` for reads.
PNG screenshots are paired with observation JSON for review; they are not the
source of position/HP or automatic round-winner detection.

Do not infer a winner from HP alone. The Core observes native pip changes,
waits at least 360 frames after a round stop, checks mature poses and handles
narrowly specified timeout/draw cases. The adapter records every Core tick and
decision. An unexpected scene/route or unfinished telemetry fails closed.

## Frozen strategy

The three frozen policy inputs retain their original bytes. V4 selects different
modes for each opponent (see `selection.json`); every difficulty uses that same
selection. Timeout guards apply to Honda, Chun-Li, Bison and Vega. Bison alone
has a two-frame opening lead. This port adapts file paths and lifecycle setup,
not the selected moves.

Develop new strategies in a separately versioned experiment. Do not edit a
verification run, replace its copied runtime or splice together successful
matches. Existing reports should remain attributable to their original inputs.

Native DIP is written to the isolated `cfg/sf2.cfg` before process launch. The
initial XML is preserved as `boot-config.xml`, because MAME can rewrite CFG on
exit. The bootstrap validates both native DIP and decoded RAM difficulty;
matching the port alone does not prove the game consumed a changed setting.

Each difficulty has its own process and working directory. A single-level run
uses `astra.run.v2`; a multi-level `astra.batch.v1` contains sequential child
sessions at `sessions/l3` through `sessions/l7` and a combined report. Attempts
within a level retain the same process and natural coin sequence. Policy/runtime
hashes must agree across children except the per-level `settings.lua`.

There is no in-session difficulty change. The adapter passively checks internal
difficulty every Core tick alongside DIP, speed and lifecycle checks. See the
[difficulty correction](difficulty-fix.md) for the diagnosis and legacy evidence limits.

## Status file publication (0.1.3)

During a match, Lua appends progress to `PREFIX-progress.jsonl`. A progress
reader must tolerate an incomplete last line. Python polls only
`PREFIX-status.json`, which remains absent until the match has finished and its
full log and screenshot have been written. Lua closes a temporary JSON file and
renames it to that fresh terminal name exactly once. Later status queries never
replace the published file.

This avoids deleting a status file while Python has it open on Windows. The
previous replacement fallback could raise `Permission denied` during a match.
Both files are included in the local evidence seal; the existing terminal/raw
summary comparison still applies. See [Windows status I/O](windows-status-io.md).
