"""Structured screen observation; never expose hidden action-state one-hots."""
import math
import numpy as np
from .perception_execution import features as execution_features
from .fighter_perception import features as fighter_features
from .visible_projectiles import features as projectile_features

INTERFACE = 'sf2_screen_perception_v2'
FRAME_FEATURES = 1129
OBSERVATIONS = FRAME_FEATURES*4


def health(player):
    value = player.get('displayed_hp')
    # -1 is the game's exhausted displayed-bar sentinel at a mature KO.
    return max(0,value)/144 if type(value) in (int,float) and math.isfinite(value) and -1<=value<=144 else -1.


def features(state):
    a,b = state['p1'],state['p2']
    actors=state['fighter_perception']; va,vb=actors['p1'],actors['p2']
    def position(p):
        if p.get('position_known') is not True:return None
        x,y=p['visible_x'],p['visible_y']
        if any(type(v) not in (int,float) or not math.isfinite(v) for v in (x,y)):
            raise ValueError('Invalid rendered actor position')
        return x,y
    def grounded(p):
        value=p['grounded']
        if value=='unknown':return -1.
        if type(value)is not bool:raise ValueError('Invalid visible grounded state')
        return float(value)
    pa,pb=position(va),position(vb)
    out = [health(a),health(b),state['timer']/99,(pb[0]-pa[0])/384 if pa and pb else -1.,
           pa[0]/384 if pa else -1.,pb[0]/384 if pb else -1.,
           pa[1]/224 if pa else -1.,pb[1]/224 if pb else -1.,grounded(va),grounded(vb)]
    out.extend(float(b['char']==i) for i in range(12))
    out.extend(execution_features(state))
    out.extend(fighter_features(state))
    out.extend(projectile_features(state))
    assert len(out)==FRAME_FEATURES
    array = np.asarray(out,dtype=np.float32)
    if not np.isfinite(array).all(): raise ValueError('Nonfinite screen perception')
    return np.clip(array,-1,1)
