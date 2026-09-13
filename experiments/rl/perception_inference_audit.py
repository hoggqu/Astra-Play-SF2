"""Reconstruct every recorded screen observation and replay its PPO decision."""
import argparse
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from stable_baselines3 import PPO

from astra_play_sf2.config import atomic_json
from astra_play_sf2.runner import sha256
from .perception_identity import OBSERVATIONS


def audit(source, model, match, output):
    source, model, match, output = map(lambda p: Path(p).resolve(), (source, model, match, output))
    sys.path.insert(0, str(source.parent))
    try:
        encode = importlib.import_module(source.name+'.perception_features').features
        support = importlib.import_module(source.name+'.support')
        support.validate_chain_build(json.loads((source.parent/'build.json').read_text()), source)
        policy = PPO.load(model, device='cpu'); support.validate_model(policy)
        torch.set_num_threads(1)
        data = json.loads(match.read_text()); trace = data['trace']
        rows = {r[0]:dict(zip(trace['columns'], r)) for r in trace['rows']}
        initial = next(e['state'] for e in data['events'] if e['kind']=='match_start')
        history = None; previous_round = None; checked = 0
        for d in trace['decisions']:
            state = {'p1':d['ken'], 'p2':d['cpu'],
                     'timer':rows[d['frame']]['timer'] if d['frame'] else initial['timer']}
            for field in ('visible_feedback', 'fighter_perception', 'visible_projectiles'):
                state[field] = d[field]
            feature = encode(state)
            history = [feature]*4 if d['round']!=previous_round else history[1:]+[feature]
            observation = np.concatenate(history)
            if observation.shape!=(OBSERVATIONS,): raise RuntimeError('Wrong observation dimension')
            action, _ = policy.predict(observation, deterministic=True)
            expected = int(d['reason'].removeprefix('rl_action_'))
            if int(action)!=expected:
                raise RuntimeError(f"Torch/Lua action mismatch at decision {d['id']}: {action} != {expected}")
            checked += 1; previous_round = d['round']
        if not checked: raise RuntimeError('Empty decision evidence')
        result = {'status':'pass', 'diagnostic_only':True, 'decisions':checked,
                  'frames':len(rows), 'observations':OBSERVATIONS, 'actions':85,
                  'all_recorded_decisions_match_torch':True,
                  'score':data['summary']['score'],
                  'valid_continuous':data['summary']['valid_continuous'],
                  'model_sha256':sha256(model), 'match_sha256':sha256(match),
                  'source_build_sha256':sha256(source.parent/'build.json')}
        atomic_json(output, result)
        return result
    finally:
        sys.path.remove(str(source.parent))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'model', 'match', 'output'): p.add_argument('--'+name, type=Path, required=True)
    print(json.dumps(audit(**vars(p.parse_args())), indent=2))


if __name__=='__main__': main()
