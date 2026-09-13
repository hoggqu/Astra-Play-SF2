"""Versioned Ken World Warrior input vocabulary; no game-state action masks.

IDs 0..15 retain the frozen pulsed-normal interface. IDs 16..78 expose every
relative stick direction with no button or any one of the six attack buttons.
IDs 79..84 complete the nine strength-specific special-move input macros.
These are requests, not promises that the game will start the requested move.
"""

INTERFACE = 'ken_actions85_full_v1'
FRAMES = 12
LEGACY_INTERFACE = 'ken_actions16_pulsed_normals_v2'
LEGACY_COUNT = 16
DIRECTIONS = ('N', 'F', 'B', 'D', 'DF', 'DB', 'U', 'UF', 'UB')
BUTTONS = ('', 'LP', 'MP', 'HP', 'LK', 'MK', 'HK')
_STRENGTH = {'': 'none', 'LP': 'light', 'LK': 'light', 'MP': 'medium',
             'MK': 'medium', 'HP': 'heavy', 'HK': 'heavy'}


def _spec(action, name, direction='N', button='', target='contextual', kind='raw'):
    return {'id': action, 'name': name, 'kind': kind, 'direction': direction,
            'button': button, 'target': target, 'strength': _STRENGTH[button]}


_legacy = (
    ('neutral', 'N', ''), ('forward', 'F', ''), ('back', 'B', ''),
    ('crouch_guard', 'DB', ''), ('jump_forward', 'UF', ''), ('jump_back', 'UB', ''),
    ('jab', 'N', 'LP'), ('fierce', 'N', 'HP'), ('crouch_jab', 'D', 'LP'),
    ('sweep', 'D', 'HK'), ('medium_kick', 'N', 'MK'), ('heavy_kick', 'N', 'HK'),
)
_specs = [_spec(i, *row) for i, row in enumerate(_legacy)]
_specs.extend((
    _spec(12, 'fireball', 'F', 'LP', 'hadouken', 'macro'),
    _spec(13, 'uppercut', 'F', 'LP', 'shoryuken', 'macro'),
    _spec(14, 'jump_heavy_kick', 'UF', 'HK', 'jump_attack', 'macro'),
    _spec(15, 'medium_uppercut', 'F', 'MP', 'shoryuken', 'macro'),
))
for _direction in DIRECTIONS:
    for _button in BUTTONS:
        _specs.append(_spec(len(_specs), f'raw_{_direction.lower()}_{_button.lower() or "none"}',
                            _direction, _button))
for _target, _button in (('hadouken', 'MP'), ('hadouken', 'HP'), ('shoryuken', 'HP'),
                         ('tatsumaki', 'LK'), ('tatsumaki', 'MK'), ('tatsumaki', 'HK')):
    _specs.append(_spec(len(_specs), f'{_target}_{_button.lower()}',
                        'B' if _target == 'tatsumaki' else 'F', _button, _target, 'macro'))
SPECS = tuple(_specs)
ACTION_NAMES = tuple(row['name'] for row in SPECS)
COUNT = len(SPECS)
assert COUNT == 85


def descriptor(action):
    if isinstance(action, bool) or not isinstance(action, int) or not 0 <= action < COUNT:
        raise ValueError('Invalid action')
    return dict(SPECS[action])


def raw_id(direction, button=''):
    """Look up an unconditional stick + single-button request."""
    return LEGACY_COUNT + DIRECTIONS.index(direction) * len(BUTTONS) + BUTTONS.index(button)


def _direction_keys(direction, forward):
    back = 'L' if forward == 'R' else 'R'
    return {'N': '', 'F': forward, 'B': back, 'D': 'D', 'DF': 'D ' + forward,
            'DB': 'D ' + back, 'U': 'U', 'UF': 'U ' + forward, 'UB': 'U ' + back}[direction]


def keys(action, frame, state=None, forward='R'):
    """Independent Python reference for the Lua waveform (state is unused).

    Freeze forward at decision time. Do not turn around, cancel, wait, retry or
    suppress inputs in response to game state during the twelve-frame macro.
    """
    spec = descriptor(action)
    if isinstance(frame, bool) or not isinstance(frame, int) or not 0 <= frame < FRAMES:
        raise ValueError('Invalid action frame')
    if forward not in ('L', 'R'):
        raise ValueError('Invalid forward direction')
    direction = _direction_keys(spec['direction'], forward)
    button = spec['button']
    if spec['kind'] == 'raw':
        return ' '.join(x for x in (direction, button if frame < 11 else '') if x)
    target = spec['target']
    if target in ('hadouken', 'tatsumaki'):
        if frame < 3:
            return 'D'
        if frame < 6:
            return 'D ' + direction
        return direction + ' ' + button if frame < 8 else ''
    if target == 'shoryuken':
        if frame < 2:
            return forward
        if frame < 4:
            return 'D'
        return 'D ' + forward + ' ' + button if frame < 6 else ''
    if target == 'jump_attack':
        return 'U ' + forward if frame < 3 else (button if frame < 9 else '')
    raise ValueError('Unsupported action target')
