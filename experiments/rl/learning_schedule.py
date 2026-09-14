"""Decision-budget cosine decay, persisted across trainer processes and resumes."""
import copy
import json
import math
import os
from .managed_runtime import configure_learning_rate

PROTOCOL = 'global-completed-decisions-cosine-v1'


def validate_plan(plan):
    if not isinstance(plan, dict) or set(plan) != {'kind', 'start', 'end', 'total_steps'}:
        raise ValueError('Invalid learning-rate schedule plan')
    if plan['kind'] != 'cosine':
        raise ValueError('Only cosine plans carry progress')
    if type(plan['total_steps']) is not int or plan['total_steps'] <= 0:
        raise ValueError('Schedule total_steps must be a positive integer')
    for key in ('start', 'end'):
        value = plan[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError('Schedule rates must be positive and finite')
    if plan['end'] > plan['start']:
        raise ValueError('Cosine decay end must not exceed start')
    return dict(plan)


def rate_at(plan, completed):
    validate_plan(plan)
    if type(completed) is not int or completed < 0:
        raise ValueError('Invalid completed decision count')
    progress = min(completed / plan['total_steps'], 1.)
    return plan['end'] + (plan['start']-plan['end']) * .5 * (1+math.cos(math.pi*progress))


def configure(model, requested=None, learning_rate_override=None):
    """Explicit {} selects constant; absent plan inherits unless scalar LR overrides."""
    if requested is None:
        raw = os.environ.get('ASTRA_RL_LR_SCHEDULE', '')
        requested = json.loads(raw) if raw else None
    saved = getattr(model, 'astra_learning_schedule', None)
    if requested == {} or (requested is None and learning_rate_override is not None):
        model.astra_learning_schedule = None
        return None
    if requested is not None:
        plan = validate_plan(requested)
        if saved is None or saved.get('plan') != plan:
            saved = dict(protocol=PROTOCOL, plan=plan, completed_steps=0)
    if saved is None:
        return None
    if saved.get('protocol') != PROTOCOL:
        raise ValueError('Unknown learning-rate schedule checkpoint protocol')
    value = rate_at(saved['plan'], saved['completed_steps'])
    model.astra_learning_schedule = copy.deepcopy(saved)
    configure_learning_rate(model, value)
    return copy.deepcopy(saved)


def prepare_update(model, decisions):
    state = getattr(model, 'astra_learning_schedule', None)
    if type(decisions) is not int or decisions <= 0:
        raise ValueError('Update decisions must be positive')
    if state is not None:
        # Set the SB3 schedule itself, not just optimizer groups: PPO.train() reapplies it.
        configure_learning_rate(model, rate_at(state['plan'], state['completed_steps']+decisions))
    return float(model.lr_schedule(model._current_progress_remaining))


def complete_update(model, decisions):
    state = getattr(model, 'astra_learning_schedule', None)
    if state is not None:
        state['completed_steps'] += decisions
    return copy.deepcopy(state)
