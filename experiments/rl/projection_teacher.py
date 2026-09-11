"""Lossy, training-only projection of frozen V4 into the existing 15 actions.

This calls only the pure choose() selector via Lua. It neither operates MAME nor
executes V4 sequences. The resulting action is executed by the ordinary learner
interface; projected-teacher performance must be measured separately from V4.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import astra_play_sf2
from lupa.lua54 import LuaRuntime

from .env import ACTION_NAMES


def project_sequence(sequence, forward):
    """Map the actual sequence, never keywords in a guard's descriptive reason."""
    back = 'L' if forward == 'R' else 'R'
    keys = [set(step[1].split()) for step in sequence]
    first = keys[0]
    punches = {'LP', 'MP', 'HP'}
    # Recognize full directional prefixes before projecting individual buttons.
    if len(keys) >= 3:
        if keys[0] == {forward} and keys[1] == {'D'} and {'D', forward} <= keys[2] and keys[2] & punches:
            return 13, 'uppercut_strength_timing_projected'
        if keys[0] == {'D'} and keys[1] == {'D', forward} and forward in keys[2] and keys[2] & punches:
            return 12, 'fireball_strength_timing_projected'
    if 'U' in first:
        if back in first:
            return 5, 'jump_back_timing_projected'
        if any('HK' in step for step in keys[1:]):
            return 14, 'jump_kick_timing_projected'
        return 4, 'jump_forward_or_neutral_projected'
    if 'HK' in first:
        return (9, 'sweep_timing_projected') if 'D' in first else (11, 'heavy_kick_timing_projected')
    if 'MK' in first or 'LK' in first:
        return 10, 'kick_height_strength_projected'
    if first & punches:
        if 'HP' in first:
            return 7, 'fierce_or_throw_projected'  # No forward+HP throw action exists.
        return (8, 'crouch_jab_strength_projected') if 'D' in first else (6, 'jab_strength_projected')
    if 'D' in first:
        return 3, 'crouch_guard_timing_projected'
    if back in first:
        return 2, 'back_timing_projected'
    if forward in first:
        return 1, 'forward_timing_projected'
    if not first:
        return 0, 'neutral_timing_projected'
    raise ValueError(f'Unmapped teacher sequence: {sequence}')


class ProjectionTeacher:
    def __init__(self):
        assets = Path(astra_play_sf2.__file__).parent/'assets'
        source = (assets/'fighter.lua').read_text(encoding='utf-8')
        start, end = 'local function c19_extension(', '\nif bot_subscription then'
        if source.count(start) != 1 or source.count(end) != 1:
            raise ValueError('Frozen selector extraction boundary changed')
        # Only function definitions; no emulator callbacks or RAM access installed.
        selector = source[source.index(start):source.index(end)]
        self.lua = LuaRuntime(unpack_returned_tuples=True)
        self.lua.execute(selector)
        self.choose = self.lua.globals().choose
        self.modes = json.loads((assets/'selection.json').read_text(encoding='utf-8'))
        self.identity = {'fighter_sha256': hashlib.sha256(source.encode()).hexdigest(),
                         'selection_sha256': hashlib.sha256((assets/'selection.json').read_bytes()).hexdigest(),
                         'projection_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                         'actions': ACTION_NAMES, 'decision_frames': 12,
                         'training_only': True, 'lossy': True,
                         'history': '12-frame sampled velocity/attack-age/rebound estimates; not native per-frame V4 history',
                         'privileged_teacher_features': ['anim'],
                         'learner_features_unchanged': True}
        self.reset()

    def reset(self):
        self.previous = None
        self.frame = 0
        self.attack_frame = None
        self.last_vy = 0.
        self.rebound = False

    def predict(self, state, elapsed_frames=12):
        if type(elapsed_frames) is not int or elapsed_frames <= 0:
            raise ValueError('elapsed_frames must be positive integer')
        a, b = deepcopy(state['p1']), deepcopy(state['p2'])
        if str(b['char']) not in self.modes:
            raise ValueError('Unsupported opponent')
        for p in (a, b):
            if 'anim' not in p:
                raise ValueError('Projection teacher requires observed animation identity')
        if self.previous is not None:
            self.frame += elapsed_frames
            if b['a'] in (10, 12) and b['a'] != self.previous['a']:
                self.attack_frame = self.frame
            dy = b['y']-self.previous['y']
            if b['y'] <= 40:
                self.rebound = False
            elif dy > 0 and self.last_vy < 0 and self.previous['y'] > 50:
                self.rebound = True
            if dy != 0:
                self.last_vy = dy/elapsed_frames
        b['vy'] = 0 if b['y'] <= 40 else self.last_vy
        b['rebound'] = self.rebound
        b['attack_age'] = self.frame-self.attack_frame if self.attack_frame is not None else 999
        self.previous = deepcopy(b)
        result = self.choose(self.lua.table_from(a), self.lua.table_from(b), self.modes[str(b['char'])])
        seq, reason = result if isinstance(result, tuple) else (result, None)
        sequence = [(int(seq[i][1]), str(seq[i][2] or '')) for i in range(1, len(seq)+1)]
        action, mapping = project_sequence(sequence, 'R' if a['x'] < b['x'] else 'L')
        return action, {'reason': reason, 'mapping': mapping, 'sequence': sequence,
                        'estimated_vy': b['vy'], 'estimated_attack_age': b['attack_age'],
                        'estimated_rebound': b['rebound']}
