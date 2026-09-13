"""Training-only opponent selection from recent on-policy round outcomes.

No replay, loss reweighting, action overrides or changes to formal evaluation.
State is plain JSON-compatible data so a full PPO checkpoint can carry it.
"""
from collections import deque
import math

PROTOCOL = 'recent_round_win_opponent_sampling_v1'
WINDOW = 128
PRIOR_ROUNDS = 20
UNIFORM_MIX = 0.5
MAX_PROBABILITY = 0.25
ADJUSTMENT = 0.25


def probability_map(opponents, values):
    if set(values) != {str(o) for o in opponents}:
        raise ValueError('Sampler probabilities must cover the exact training opponent pool')
    numbers = [values[str(o)] for o in opponents]
    if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in numbers):
        raise ValueError('Sampler probabilities must be positive finite numbers')
    if not math.isclose(sum(numbers), 1., rel_tol=0, abs_tol=1e-10):
        raise ValueError('Sampler probabilities must sum to one')
    return dict(zip(map(str, opponents), numbers))


class OpponentSampler:
    def __init__(self, opponents, mode='adaptive', state=None):
        self.opponents = sorted(opponents)
        if (not self.opponents or len(set(self.opponents)) != len(self.opponents)
                or any(type(o) is not int or o not in range(12) or o == 4 for o in self.opponents)):
            raise ValueError('Invalid training opponent pool')
        if mode not in ('adaptive', 'uniform'):
            raise ValueError('Opponent sampling must be adaptive or uniform')
        self.mode = mode
        self.history = {str(o): deque(maxlen=WINDOW) for o in self.opponents}
        self.probabilities = {str(o): 1 / len(self.opponents) for o in self.opponents}
        self.restored = state is not None
        if state is not None:
            if (state.get('protocol') != PROTOCOL or state.get('opponents') != self.opponents
                    or state.get('settings') != self.settings()):
                raise ValueError('Incompatible checkpoint sampler state')
            self.probabilities = probability_map(self.opponents, state['probabilities'])
            if set(state['history']) != set(self.history):
                raise ValueError('Incomplete checkpoint sampler history')
            for key, rows in state['history'].items():
                if (not isinstance(rows, list) or len(rows) > WINDOW
                        or any(not isinstance(r, list) or len(r) != 2 or r[0] not in ('win','loss','draw')
                               or type(r[1]) is not int or r[1] < 1 for r in rows)):
                    raise ValueError('Invalid checkpoint sampler history')
                self.history[key].extend(rows)
            low, high = UNIFORM_MIX / len(self.opponents), max(MAX_PROBABILITY, 1 / len(self.opponents))
            if any(not low-1e-10 <= p <= high+1e-10 for p in self.probabilities.values()):
                raise ValueError('Checkpoint sampler probability outside bounds')
        if mode == 'uniform':
            self.probabilities = {str(o): 1 / len(self.opponents) for o in self.opponents}
        self.stage = {str(o): dict(decisions=0, rounds=0, wins=0, losses=0, draws=0, round_decisions=0)
                      for o in self.opponents}

    @staticmethod
    def settings():
        return dict(window=WINDOW, prior_rounds=PRIOR_ROUNDS, uniform_mix=UNIFORM_MIX,
                    max_probability=MAX_PROBABILITY, adjustment=ADJUSTMENT, hardness_power=2,
                    metric='round_win_rate; draws count as non-wins')

    def win_rate(self, key):
        rows = self.history[key]
        return (sum(r[0] == 'win' for r in rows) + PRIOR_ROUNDS / 2) / (len(rows) + PRIOR_ROUNDS)

    def target(self):
        count = len(self.opponents)
        base = UNIFORM_MIX / count
        capacity = max(MAX_PROBABILITY, 1 / count) - base
        weights = {k: (1 - self.win_rate(k)) ** 2 for k in self.history}
        extra, remaining = {}, 1 - UNIFORM_MIX
        while weights:
            total = sum(weights.values())
            saturated = [k for k, v in weights.items() if remaining * v / total > capacity]
            if not saturated:
                extra.update({k: remaining * v / total for k, v in weights.items()})
                break
            for k in saturated:
                extra[k] = capacity
                remaining -= capacity
                del weights[k]
        return probability_map(self.opponents, {k: base + extra[k] for k in self.history})

    def observe(self, chunks):
        # Stable worker order makes aggregation reproducible despite worker timing.
        for chunk in chunks:
            for transition in chunk['transitions']:
                key = str(transition['state']['p2']['char'])
                if key not in self.stage:
                    raise ValueError('Rollout opponent outside training pool')
                self.stage[key]['decisions'] += 1
            for row in chunk['episodes']:
                key = str(row['opponent'])
                outcome, steps = row['outcome'], row['steps']
                if (key not in self.history or outcome not in ('win','loss','draw')
                        or type(steps) is not int or steps < 1 or row.get('incomplete')):
                    raise ValueError('Invalid completed training round for sampler')
                self.history[key].append([outcome, steps])
                totals = self.stage[key]
                totals['rounds'] += 1
                totals[{'win':'wins','loss':'losses','draw':'draws'}[outcome]] += 1
                totals['round_decisions'] += steps
        if self.mode == 'adaptive':
            target = self.target()
            self.probabilities = probability_map(self.opponents, {
                k: (1 - ADJUSTMENT) * self.probabilities[k] + ADJUSTMENT * target[k]
                for k in self.probabilities})

    def state(self):
        return dict(protocol=PROTOCOL, opponents=self.opponents, settings=self.settings(),
                    probabilities=dict(self.probabilities), history={k:list(v) for k,v in self.history.items()})

    def snapshot(self):
        total = sum(v['decisions'] for v in self.stage.values())
        return dict(protocol=PROTOCOL, mode=self.mode, restored=self.restored, settings=self.settings(),
                    decisions=total, probability_scope='next whole-match reset; not current-match reassignment',
                    per_opponent={k:dict(**v, next_probability=self.probabilities[k],
                        decision_share=v['decisions']/total if total else None,
                        recent_rounds=len(self.history[k]), smoothed_win_rate=self.win_rate(k),
                        mean_round_decisions=v['round_decisions']/v['rounds'] if v['rounds'] else None)
                        for k,v in self.stage.items()})
