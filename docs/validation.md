# Validation history and portability

## Original V4 result

The original project used the same frozen V4 strategy across native difficulty 3 (Normal) through 7 (Hardest), with Ken in MAME 0.288's `sf2` World 910522. Each level finished with five consecutive certified natural-coin clears. Root reviewed the relevant selection, terminal result, and ending evidence in that original macOS environment.

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

## Standalone port

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
