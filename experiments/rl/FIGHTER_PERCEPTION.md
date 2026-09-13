# Current rendered fighter perception

This experimental interface describes both fighters from the current rendered
picture and its observed history. It does not replace the frozen V4 or full85 v1
interfaces, and it does not establish a new trained policy's win rate.

## What the observer sees

`visible_sprite_buffer.lua` reads MAME 0.288's **currently latched CPS1 object
buffer** (`0/m_buffered_obj`). `fighter_render.lua` compares its visible tiles
with static SF2 World 910522 sprite compositions. The actor's current RAM
animation pointer is not used to select the picture. This matters because RAM
can advance before the associated picture appears; a fixed frame delay is not
assumed.

The reference library contains metadata and composition addresses for all twelve
characters. The ROM remains an external dependency. Tile layouts come from the
version's renderer (composition/layout handlers at `0x7F190` and `0x7F60A`), with
current visible tile positions and flips supplied by the latched buffer. A match
requires every on-screen tile of a candidate composition and at least three
matching tiles. Any visible piece can anchor a match, including when part of a
fighter is clipped at the edge. Palette differences are ignored by the fighter
matcher so palette flashes do not invent different moves; it does not supply
color-specific status information.

The observer reports:

- Current pose identity, coarse status, and specific move/variant when that
  picture distinguishes them. Status vocabulary covers idle, walking, crouching,
  jumping, attack, special, throw, guard, hit, falling, down, getup, dizzy,
  victory, defeat and unknown. A vocabulary entry is not a promise that every
  corresponding picture has a unique label.
- Observed time in the current status, action and pose, plus observed pose
  changes. These are elapsed observations, never remaining recovery or an
  animation's future frame count.
- Current screen position as the visible tile bounding box's horizontal center
  and bottom, in a 384×224 viewport with top-left origin. These are image
  coordinates, not a collision center or hidden world position. Their changes
  include changes in the shape of an attack pose.
- Facing when the rendered orientation is unambiguous, and the latest recognized
  attack with its age. The latter retains brief attacks occurring between policy
  decisions. All histories update once per native frame and reset each round.

The classifier merges **all semantic labels** for visually aliased poses. For
example, indistinguishable strength variants cannot reveal different move IDs,
strength or elapsed histories through different animation addresses. Runtime
matches with conflicting labels also merge to unknown. Ambiguous positions are
unknown rather than assigned using hidden actor coordinates. Dhalsim's shared
Yoga Fire/Flame casting picture is `yoga_cast`; it does not expose which hidden
variant the engine selected. Other verified casting families include Hadouken,
Sonic Boom and Tiger Shot.

A recognized pose can still have an unknown semantic label. The model retains
that distinct visible pose so it can learn its consequences. A recognized normal
attack's `started` signal is a distinctive visible gesture, which may appear
later than the engine accepted the input. It is not a claim of exact internal
startup timing. No action is masked and no reward is added by this observer.

## Explicit limits

There is no fallback to raw actor action-state, animation pointer, coordinates,
facing or round-win counter. Character identity selects the corresponding visual
reference library; the game displays that identity. Hidden stun accumulation,
AI intent, input acceptance flags and future animation transitions are excluded.
The current implementation leaves grounded status and win-icon counts unknown:
neither has a verified decoder here. Visible dizzy poses are recognized when
unambiguous; a separate bird/star overlay detector during an otherwise ambiguous
knockdown is not implemented. Unknown does not mean false.

Sprite composition matching describes the rendered object list, rather than
performing screenshot OCR or a neural image encoding. This is a game-specific
perception adapter. It does not inspect collision boxes. Heavy clipping,
unmapped compositions or ambiguous simultaneous pictures can return unknown.

## API and feature contract

`fighter_perception.lua` exports `interface='sf2_visible_fighters_v1'`,
`feature_count=872` and `actor_feature_count=436`.

1. `F.read(memory, state)` resolves both currently rendered pictures. It requires
   staged `rl_fighter_animation_map.lua`, `rl_fighter_render.lua` and
   `rl_visible_sprite_buffer.lua` beside the runtime.
2. `observer = F.new()`; `observer:reset(state)` starts a new round or post-load
   history. `observer:tick(state)` advances exactly once per native frame after
   reading the picture and before recording the decision observation.
3. Both attach a fresh `state.fighter_perception` snapshot. `F.features(state)`
   only encodes it; it never advances history. `fighter_perception.py` mirrors
   that pure encoder.
4. After the fighter and projectile observers have consumed a frame, integration
   may remove transient `state.visible_sprite_buffer` and
   `state.p1/p2.visible_fighter`. Histories retain semantic snapshots only.

Each actor's 436 scalars contain 16 status categories, 65 move categories,
17 variant categories, 257 pose categories, four elapsed/progression scalars,
two image-motion scalars, grounded and recognized scalars, three facing
categories, four win-count categories, 65 latest-attack categories and its age.
IDs are character-local visual labels; zero is unknown. Elapsed counts cap at
120 for encoding; unavailable ages are -1. Grounded is -1/0/1 for
unknown/false/true. The optional `visible_wins` input accepts only 0/1/2 from a
separately verified visible decoder; absent is -1. The present runtime supplies
unknown. Out-of-contract enums, dimensions or non-finite values fail validation.

## Validation evidence

The reference lookup covers all twelve characters in an existing inventory of
190 matches: 708,022 native game frames, or 1,416,044 actor-frame observations.
Every recorded current animation has a reference pose. **This measures reference
coverage, not screen recognition or complete semantic understanding.** The raw
animation inventory is used only for this offline audit.

An independent replay of 7,200 recorded current object buffers against Ryu,
Guile, Dhalsim and Sagat recognized 14,350 of 14,400 actor pictures (99.65%). Forty
unknown observations occurred when both fighters disappeared during a match
transition; ten were an unresolved late Guile animation. These observations are
retained. Screen samples of casting poses were also visually inspected. Replay
of both actors took approximately 2.8 seconds on the development machine; this
is an observer replay measurement, not end-to-end training throughput.

`test_fighter_perception.py` and `test_fighter_render.py` check alias invariance,
immutable history, Python/Lua parity, short-attack retention, clipped matches,
missing tiles, ambiguous duplicate pictures and unavailable-buffer behavior.
Run with the optional RL environment:

```sh
python -m unittest experiments.rl.test_fighter_perception experiments.rl.test_fighter_render -v
```

Raw buffers, screenshots, local ROM inspection and replay outputs stay under
ignored `.local/`; they are not distributed. The static structure was cross-read
with [the SF2 lineage engine reference](https://github.com/ROMArchaeology/sf2-lineage/blob/d88f5624835ea8d35346f7a95ba583f2ebc2cd8c/engine/ENGINE.md)
and checked against the compatible local World 910522 program. Later native
integration gates must pin the actual module hashes and report unknown coverage
separately from policy performance.
