"""Encoding of current visible fighter pose observations; no game-memory reads."""
INTERFACE = 'sf2_visible_fighters_v1'
FEATURE_COUNT = 872
ACTOR_FEATURE_COUNT = 436
STATUSES = ('unknown','idle','walk','crouch','jump','attack','special','throw','guard','hit','falling','down','getup','dizzy','victory','defeat')
FACES = ('unknown','left','right')


def features(state):
    import math
    data=state['fighter_perception']
    if data.get('interface')!=INTERFACE:raise ValueError('Wrong fighter perception interface')
    out=[]
    def hot(value,values):
        if value not in values:raise ValueError('Unknown fighter enum')
        out.extend(float(value==v) for v in values)
    def integer(v,lo,hi):
        if type(v)is not int or not lo<=v<=hi:raise ValueError('Invalid fighter integer')
        return v
    for key in ('p1','p2'):
        p=data[key];hot(p['status'],STATUSES)
        for field,maximum in (('move_id',64),('variant',16),('pose_id',256)):
            hot(integer(p[field],0,maximum),range(maximum+1))
        for field in ('status_age','action_age','pose_age','pose_changes'):
            age=integer(p[field],-1,2147483647);out.append(-1.0 if age<0 else min(age,120)/120)
        for field in ('dx','dy'):
            value=p[field]
            if not isinstance(value,(int,float)) or not math.isfinite(value):raise ValueError('Invalid observed movement')
            out.append(max(-1,min(1,value/16)))
        if p['grounded']!='unknown' and type(p['grounded'])is not bool:raise ValueError('Invalid grounded state')
        out.append(-1.0 if p['grounded']=='unknown' else float(p['grounded']))
        if type(p['recognized'])is not bool:raise ValueError('Invalid visible flag')
        out.append(float(p['recognized']))
        hot(p['facing'],FACES);hot(p['wins'],(-1,0,1,2))
        hot(integer(p['last_attack_move'],0,64),range(65))
        age=integer(p['last_attack_age'],-1,2147483647);out.append(-1.0 if age<0 else min(age,120)/120)
    assert len(out)==FEATURE_COUNT
    return out
