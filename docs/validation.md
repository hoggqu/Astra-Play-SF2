# Validation history and portability

## Fixed speed targets — 0.1.4, 2026-09-11

The CLI accepts `normal`, `2x`, `4x`, and `fast`. Fixed targets use MAME's
native throttle multiplier. Every match frame checks throttling, multiplier,
and base speed factor; the offline audit checks the recorded speed evidence.
Reports include the selected speed. Frozen V4 fighter/core/selection are unchanged.

Native diagnostics on macOS 26.5.2 arm64, Python 3.12.14, MAME 0.288,
`sf2` World 910522, difficulty 7:

| Mode | 1200 native frames (20.102 game seconds) | Complete opponent match |
|---|---:|---|
| `2x` | 10.104 wall seconds | Ken beat Dhalsim 2–0; all 5783 speed checks passed |
| `4x` | 5.084 wall seconds | Ken beat Dhalsim 2–0; all 5783 speed checks passed |

Each diagnostic used a fresh CLI-owned process and natural coin entry, with
no state loading or policy changes. Match evidence passed the offline audit.
These are two speed functionality tests, not full-game clears or win-rate
measurements. Actual speed remains limited by the host's performance.

Local tests: 76 discovered, 75 passed, one Windows-only reproduction skipped.
The built 0.1.4 wheel installed and staged its runtime outside the checkout,
including the new speed module and unchanged frozen policy hashes. The Windows
real-Lua CI job now also exercises all four modes and rejects speed tampering.

## Windows status publication — 0.1.3, 2026-09-11

The [status I/O change](windows-status-io.md) adds real Lua 5.4 file tests and
a mandatory Windows job in addition to the existing six-platform/version matrix.
The initial Windows run `34578854723` reproduced the old held-reader deletion
failure and passed terminal publication, no-overwrite and partial-write checks.
Its concurrent test used a raw Python read and failed on a transient read-side
sharing denial during rename. The revised test uses the actual CLI reader's
existing bounded polling, while still strictly checking completed file bytes.
The failed test run is retained; it is not counted as a passing run.

Local macOS 26.5.2 arm64 / Python 3.12.14 checks: 69 tests discovered, 68 passed,
one Windows-only reproduction explicitly skipped; built/installed wheel version
and runtime staging passed outside the checkout. The test-only Lua runtime was
Lupa 2.8, explicitly using its Lua 5.4 module.

Native MAME 0.288, World 910522 smoke `20260911T081919Z-f8aab2e1`: Hardest,
one fast attempt, first opponent Blanka, **1/1 audited gameplay clear**, round
W/L/D **22/0/1**. The run used the passive local preview captured in runtime
hashes. Frozen V4 fighter/Core/selection were unchanged. All five images were
inspected; the ending capture missed the wedding, so visual approval was
rejected even though the eleven native match wins and file audit passed.
This macOS gameplay run is separate from Windows file-I/O validation; no Windows
MAME full-game test or new win-rate guarantee is claimed.

## Native entry readiness — 0.1.2, 2026-09-11

The [entry readiness change](entry-readiness.md) replaces fixed 9000-frame idle
blocks with neutral native-state waits. Frozen V4 fighter/Core/selection hashes
are unchanged. Tests used macOS 26.5.2 arm64, Python 3.12.14, MAME 0.288 and
`sf2` World 910522.

- A normal-speed startup diagnostic reached coin readiness in **580 frames,
  about 9.9 seconds**, plus the existing five-second autoboot stage. Start
  readiness took 33 native frames after coin input.
- A separate passive diagnostic rejected coin readiness during an active game
  with a two-frame budget. It then let the game lose naturally and finish its
  Continue countdown before accepting a new coin, followed by a verified new
  Ken R1. These intentional idle losses are setup diagnostics, not policy trials.
- Source-runner smoke test `20260911T080337Z-da4d9e5a`: Normal, two attempts,
  fast speed, **2/2 gameplay clears**, **44/2/0 round W/L/D**, both visually
  inspected and approved. First/second coin waits were 580/5726 frames; Start
  waits were 33/48 frames. The same MAME process handled both attempts without
  reset, load or Continue. The second wait allowed the native ending to finish.
  Internal difficulty, entry readiness and continuous-play audit all passed.
- The smoke runtime included the same optional passive local preview module
  recorded in its hashes. No native Windows/Linux run or new five-level
  win-rate certification is claimed here.
- **65 local unit tests passed.** The wheel was built and installed in an
  isolated environment, then CLI help, runtime staging and evidence review were
  exercised outside the checkout without `PYTHONPATH`.

Failed diagnostics were retained, including the early version that checked
title mode but missed its fade wait. The final check requires fade completion
before sending Start. No diagnostic failure was reclassified as a policy win.

## Difficulty validity correction — 0.1.1

The [difficulty investigation](difficulty-fix.md) reproduced a startup bug:
0.1.0 could report requested Hardest with a correct DIP readback while SF2 still
used cached Normal. The port and original training histories below checked DIP,
not the internal decoded value. Preserve these historical observations, but treat
their cross-difficulty certification as **pending revalidation**. Do not assume
all legacy runs used Normal; the old workspace had different boot settings.

0.1.1 configures each level before boot and requires internal RAM verification.
New validation results are recorded below; old successes are not
retrospectively upgraded.

### 0.1.1 local validation — 2026-09-11

Environment: macOS 26.5.2 arm64, Python 3.12.14, MAME 0.288, `sf2` World 910522.
The frozen fighter, Core and selection hashes remain unchanged.

All five levels passed separate boot and initialized first-round checks:

| Requested level | DIP low bits | Mirror / decoded level | First-round rank / index |
|---|---:|---|---|
| 3 | 4 | 3 / 3 | 56 / 7 |
| 4 | 3 | 4 / 4 | 72 / 9 |
| 5 | 2 | 5 / 5 | 88 / 11 |
| 6 | 1 | 6 / 6 | 104 / 13 |
| 7 | 0 | 7 / 7 | 112 / 14 |

A source-runner batch used levels `[3, 7]`, two attempts per level, fast speed.
Each level booted once; its second attempt used natural coin insertion.
An optional local read-only screenshot module was included in the runtime
hashes for the live preview. It did not change the policy or send inputs.

| Difficulty | Audited gameplay clears | Round W/L/D | Visual approvals |
|---|---:|---:|---:|
| 3 Normal | 2/2 | 44/3/0 | 2/2 |
| 7 Hardest | 2/2 | 44/1/0 | 1/2 |

Batch `20260911T073801Z-39359e0b` passed the parent and both child audits,
including internal difficulty on every match frame. All five images from each
attempt were inspected. The first Hardest attempt's ending capture started
early: it showed the final result, reunion and portrait, but missed the wedding.
Its visual review was rejected; the eleven audited match wins remain a
`gameplay_clear`. This capture limitation is separate from difficulty validity.

The built 0.1.1 wheel was installed in an isolated environment with spaces and
Chinese characters in its path, and imported outside the checkout without
`PYTHONPATH`. A deliberate diagnostic supplied Normal boot configuration while
requesting Hardest: native Lua rejected it before any attempt, reporting
`requested=7 DIP=4 mirror=3 internal=3` and exit 2. The invalid run
`20260911T074715Z-42b890ce` was retained.

The release has **59 passing local unit tests**, covering boot-before-launch,
cross-level session identity, internal mismatch rejection, sealed initialization
failures and legacy evidence rejection in addition to the existing checks.
These are finite smoke tests, not a five-level win-rate certification. Levels
4–6 had startup checks only; native Windows/Linux gameplay was not rerun here.

## Original V4 result (historical labels; difficulty pending revalidation)

The original project used the same frozen V4 strategy across native difficulty 3 (Normal) through 7 (Hardest), with Ken in MAME 0.288's `sf2` World 910522. Each labelled level finished with five consecutive natural-coin clears under the original DIP-only checks. Root reviewed the relevant selection, terminal result, and ending evidence in that original macOS environment.

| Difficulty | Cumulative clears / attempts | Final consecutive clears | Round W/L/D |
|---|---:|---:|---:|
| 3 Normal | 8/10 | 5 | 205/19/1 |
| 4 | 9/12 | 5 | 247/28/1 |
| 5 | 9/10 | 5 | 206/12/0 |
| 6 | 8/10 | 5 | 199/18/1 |
| 7 Hardest | 11/13 | 5 | 257/19/0 |

Totals: **55 attempts, 45 clears, 10 losses, no invalid attempts**; **565 opponent matches** (555 wins, 10 losses); **1114 round wins, 96 losses, 3 draws**. The cumulative observed clearance rate was **81.8%**, not 100%. Earlier failures were retained. The five final streaks did not carry over wins from retired V1, V2, or V3 strategies.

The original final-input snapshot had 36 pinned sources and SHA-256:

```text
e8061e601bc28ca17805521f584f0705c21e0045228ecc47daf87b19683493a5
```

The original fighter, core, and selection identities were:

```text
fighter.lua     80e3a83c3db48c14825ec39aea8248440fed9432cd5d476f7d4b9d23db4c6a99
play_core.lua   28153fcf787baba6fe3c7636739d4ffc608302ddba58ac1fc4ec08e9285909e7
selection.json  cd2d6c8462f27e8569afdafde94491ed33b1ea988b64aef05b313251ec6042e4
```

These identify historical inputs, not an assertion that every file in this standalone package has those same bytes. V4 selected the Dhalsim far-start guard; it did not select the experimental Blanka fast-MP mode. Registered experimental modes are not automatically the chosen strategy.

The original local audit was stored at `MAME/training/coin22/staged/v4-final-report/report.md` and `report.json` in the development archive. Those logs, images, ROMs, states, and emulator binaries are not distributed with this standalone package. The original machine retained `Play_Games` as a compatibility symlink after the project directory was renamed; a fresh installation does not need that path or symlink.

## Standalone port (original 0.1.0 checks)

### Local release smoke tests — 2026-09-11

Environment: macOS 26.5.2, Apple Silicon arm64, Python 3.12.14, MAME 0.288,
`sf2` World 910522. The tests below used an isolated CLI-owned process and
unchanged V4 fighter/core/selection files. The strategy was not trained or tuned.

| Test | Result | Native round W/L/D | Evidence |
|---|---|---:|---|
| Source CLI, difficulty 3, two natural-coin attempts, fast | 2/2 clears; 22 opponent matches won | 44/3/0 | Audit passed; all five images per attempt inspected and approved |
| Installed wheel, difficulty 7, initial startup | Invalid before any attempt started | — | DIP readback was checked before a native frame latched the update; failure preserved |
| Corrected installed wheel, difficulty 7, one attempt, fast | 1/1 clear; 11 opponent matches won | 22/2/0 | Audit passed; all five images inspected and approved |

The Normal run ID was `20260911T050815Z-2ce8cc17` (about 234 seconds for two
attempts). The corrected Hardest run ID was `20260911T051426Z-d15bd35a`
(about 123 seconds). The failed startup ID was `20260911T051318Z-09d0f677`.
Raw evidence is retained locally, excluded from source and wheel distributions.

The startup correction adds two neutral native frames **outside an attempt**
after setting DIP, followed by observation readback; it introduces no reset or
policy change. The Normal source run preceded this correction. The corrected
Hardest smoke test installed a wheel into a separate virtual environment and
ran outside the source working directory, with Chinese characters and spaces
in both the installation and data paths. It did not depend on `PYTHONPATH`.

The release has **47 passing standard-library tests** on local Python 3.12.
They cover evidence corruption/maturity/lifecycle checks, explicit reviews,
per-difficulty streak and failure accounting, malformed configuration,
space/Unicode paths, packaged source identities, ownership locks, no-replay
timeouts and incorrect-DIP rejection before any attempt starts.

These are packaging and execution smoke tests, not a fresh statistical claim
of 100% win rate. Normal-speed gameplay and Windows/Linux native MAME execution
were not rerun for this release. The CI matrix remains configured, not executed
on a remote host in this work session.

The CLI, installation flow, isolated sessions, and evidence interface are a separate software port. Historical V4 results are useful policy provenance, **not proof that this release has passed native gameplay on every platform**.

The CI workflow defines a macOS/Linux/Windows matrix on Python 3.10 and 3.12, running standard-library `unittest` tests and building a wheel. It contains no ROMs and does not run MAME gameplay. A configured workflow is not evidence that a remote job has already executed successfully.

Native smoke-test reports should state the OS and architecture, Python and exact MAME build, game revision, command and budget, actual result, and whether an offline visual review occurred. Record failures and limitations along with successes. Testing a local macOS port does not establish Windows or Linux gameplay compatibility.

There is no claim of a guaranteed win rate. A policy may encounter unfamiliar openings, lose rounds, or fail an attempt even on the original validated configuration. Keep a fixed strategy during an attempt and use separate future development sessions for changes; do not reinterpret a frozen run after changing the code.
