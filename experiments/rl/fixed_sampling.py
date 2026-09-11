"""Pure validation and lookup for frozen, training-only opponent probabilities."""
import math

OPPONENTS = (0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11)
IDENTITY = 'fixed_opponent_probabilities_v1'


def validate_config(config):
    if config.get('schema') != 'astra.rl-fixed-opponent-sampling.v1':
        raise ValueError('Unsupported opponent sampling configuration')
    values = config.get('probabilities', {})
    if set(values) != {str(opponent) for opponent in OPPONENTS}:
        raise ValueError('Specify all eleven opponents, with no extras')
    probabilities = [values[str(opponent)] for opponent in OPPONENTS]
    if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in probabilities):
        raise ValueError('Every opponent must have a positive finite probability')
    if not math.isclose(sum(probabilities), 1., rel_tol=0, abs_tol=1e-10):
        raise ValueError('Opponent probabilities must sum to one; no silent normalization')
    if not isinstance(config.get('formula'), dict) or not isinstance(config.get('statistics_source'), dict):
        raise ValueError('Record formula and statistics_source metadata (user-supplied weights are allowed)')
    return config


def probabilities(opponents):
    from .sampling_config import CONFIG
    config = validate_config(CONFIG)
    if list(opponents) != list(OPPONENTS):
        raise ValueError('Weighted sampler requires the complete, sorted eleven-opponent pool')
    return [config['probabilities'][str(opponent)] for opponent in opponents]
