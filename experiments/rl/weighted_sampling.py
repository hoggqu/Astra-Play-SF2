"""Multiply reset-selection probabilities without changing adaptive history or PPO loss."""
import json
import math
import os
from .adaptive_sampling import OpponentSampler as BaseSampler, probability_map

PROTOCOL = 'adaptive_reset_probability_multipliers_v2'


def parse_multipliers(value):
    if value is None or value == '':
        return None
    data = json.loads(value) if isinstance(value, str) else value
    if not isinstance(data, dict):
        raise ValueError('Opponent multipliers must be a JSON object')
    for key, number in data.items():
        if (key not in {str(i) for i in range(12) if i != 4}
                or type(number) not in (int, float) or not math.isfinite(number) or not 0 < number <= 100):
            raise ValueError('Opponent multiplier requires a valid character ID and finite factor in (0,100]')
    return dict(data)


class OpponentSampler:
    def __init__(self, opponents, mode='adaptive', state=None, multipliers=None):
        saved = state if state and state.get('protocol') == PROTOCOL else None
        self.base = BaseSampler(opponents, mode, saved['base'] if saved else state)
        configured = parse_multipliers(multipliers if multipliers is not None
                                       else os.environ.get('ASTRA_RL_OPPONENT_MULTIPLIERS'))
        self.multipliers = (configured if configured is not None
                            else parse_multipliers(saved['multipliers']) if saved else {})
        if not set(self.multipliers) <= set(self.base.probabilities):
            raise ValueError('Weighted opponent is absent from the training pool')

    @property
    def probabilities(self):
        base = self.base.probabilities
        if all(v == 1 for v in self.multipliers.values()):
            return dict(base)  # Exact baseline, including its floating-point values.
        weights = {k: v * self.multipliers.get(k, 1) for k, v in base.items()}
        total = sum(weights.values())
        return probability_map(self.base.opponents, {k: v / total for k, v in weights.items()})

    def observe(self, chunks):
        self.base.observe(chunks)

    def state(self):
        # Persist UNWEIGHTED adaptive probabilities. Never multiply an already
        # weighted checkpoint again at the next cycle or resume.
        return dict(protocol=PROTOCOL, base=self.base.state(), multipliers=dict(self.multipliers))

    def snapshot(self):
        result = self.base.snapshot()
        result.update(protocol=PROTOCOL, base_protocol=result['protocol'],
                      multipliers=dict(self.multipliers),
                      multiplier_scope='after adaptive bounds/smoothing; reset probabilities renormalized; no loss reweighting')
        selected = self.probabilities
        for key, row in result['per_opponent'].items():
            row['base_probability'] = row['next_probability']
            row['multiplier'] = self.multipliers.get(key, 1)
            row['next_probability'] = selected[key]
        return result
