# Pulsed-input baseline: complete 20 natural coins

The first full-20 candidate with pulsed normal buttons completed **2 clears,
18 losses and 0 invalid attempts**. It did not meet the 10-clear goal.
Clears occurred on attempts 1 and 3; every attempt finished under one unchanged
model, with no continue, load, reset or pause inside an opponent match.

Model `f26bb8be52583209f9d29f9f816b73bf7704f3eafa790824b10ca1dbfda724eb`
retains the exact policy/critic and Adam bytes of local baseline `7750e164…`.
Its interface is `ken_actions16_pulsed_normals_v2`, with the validated V5
settlement adapter. It contains no PPO updates from the disposable smoke.
The corresponding old-interface model had completed 0/20. These are two
separate development evaluations; this small comparison does not establish a
reliable magnitude of improvement.

## Audited round results

Round win rate includes draws in the denominator. These counts belong only to
this frozen model's complete 20 attempts; they are not training statistics.

| Opponent | Round wins | Losses | Draws | Round win rate | Attempt-ending losses |
|---|---:|---:|---:|---:|---:|
| Ryu | 28 | 19 | 0 | 59.6% | 6 |
| Honda | 29 | 6 | 0 | 82.9% | 1 |
| Blanka | 26 | 3 | 0 | 89.7% | 0 |
| Guile | 25 | 7 | 1 | 75.8% | 2 |
| Chun-Li | 28 | 3 | 0 | 90.3% | 0 |
| Zangief | 28 | 3 | 0 | 90.3% | 0 |
| Dhalsim | 28 | 8 | 0 | 77.8% | 2 |
| Balrog | 16 | 4 | 0 | 80.0% | 1 |
| Vega | 13 | 5 | 0 | 72.2% | 2 |
| Sagat | 9 | 7 | 0 | 56.2% | 2 |
| Bison | 6 | 6 | 0 | 50.0% | 2 |

Later opponents have fewer observations because attempts stop on defeat.
Do not interpret their smaller failure totals as proof that they are easier.

Independent review recomputed 131 matches, 746,671 native frames and 47,731
decisions. All 1,562 sealed files, the final native/interface checks, both
11-opponent clears, source ancestry and metadata-only model migration passed.
The final result remains 2/20, regardless of these successful integrity checks.

The original full result and per-attempt logs remain in the local run archive;
raw historical data and model weights are not included in Git. Further PPO
training uses new outputs and each new model receives its own complete-20
evaluation under the [reliability protocol](RELIABILITY.md).
