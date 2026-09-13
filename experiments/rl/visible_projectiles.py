"""Pure encoding of current rendered projectile observations (no game RAM)."""
import math
INTERFACE = 'sf2_visible_projectiles_v1'
FEATURE_COUNT = 121
KINDS = ('unknown', 'hadouken', 'yoga_fire', 'yoga_flame', 'sonic_boom', 'tiger_shot')
OWNERS = ('unknown', 'p1', 'p2')

def features(state):
    data = state['visible_projectiles']
    if data['interface'] != INTERFACE or len(data['slots']) != 8:
        raise ValueError('Wrong visible projectile interface')
    out = []
    for slot in data['slots']:
        for key in ('present', 'motion_known'):
            if type(slot[key]) is not bool:
                raise ValueError('Invalid visible projectile flag')
        if slot['owner'] not in OWNERS or slot['kind'] not in KINDS:
            raise ValueError('Invalid visible projectile label')
        out.append(float(slot['present']))
        for key, scale in (('x', 384), ('y', 224), ('vx', 16), ('vy', 16)):
            value = slot[key]
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError('Invalid visible projectile coordinate')
            out.append(max(-1.0, min(1.0, value / scale)))
        out.append(float(slot['motion_known']))
        out.extend(float(slot['owner'] == value) for value in OWNERS)
        out.extend(float(slot['kind'] == value) for value in KINDS)
    if type(data['scan_known']) is not bool:
        raise ValueError('Invalid sprite scan flag')
    out.append(float(data['scan_known']))
    assert len(out) == FEATURE_COUNT
    return out
