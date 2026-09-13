"""Pure encoding of executor-produced visible feedback; no inferred action labels."""
INTERFACE = 'ken_visible_feedback_v1'
FEATURE_COUNT = 114
RESULTS = ('none', 'pending', 'started', 'other', 'unconfirmed', 'unknown')
FAMILIES = ('unknown', 'normal_punch', 'normal_kick', 'hadouken', 'shoryuken', 'tatsumaki', 'throw', 'jump', 'locomotion')
STRENGTHS = ('unknown', 'light', 'medium', 'heavy')
DIZZY_STATES = ('unknown', 'absent', 'present')


def features(state):
    v = state['visible_feedback']
    if v.get('interface') != INTERFACE:
        raise ValueError('Wrong visible feedback interface')
    if type(v['request_action']) is not int or not -1 <= v['request_action'] < 85:
        raise ValueError('Invalid request action')
    for field, values in (('result', RESULTS), ('actual_family', FAMILIES), ('actual_strength', STRENGTHS), ('p1_dizzy', DIZZY_STATES), ('p2_dizzy', DIZZY_STATES)):
        if v[field] not in values:
            raise ValueError('Invalid visible feedback label')
    if any(type(v[field]) is not int or v[field] < -1 for field in ('request_age', 'start_age')):
        raise ValueError('Invalid feedback age')
    if type(v['started_since_request']) is not bool:
        raise ValueError('Invalid start flag')
    out = [float(v['request_action'] == action) for action in range(-1, 85)]
    for field, values in (('result', RESULTS), ('actual_family', FAMILIES), ('actual_strength', STRENGTHS)):
        out.extend(float(v[field] == value) for value in values)
    out.extend(-1.0 if v[field] < 0 else min(v[field], 120) / 120 for field in ('request_age', 'start_age'))
    for field in ('p1_dizzy', 'p2_dizzy'):
        out.extend(float(v[field] == value) for value in DIZZY_STATES)
    out.append(float(v['started_since_request']))
    assert len(out) == FEATURE_COUNT
    return out
