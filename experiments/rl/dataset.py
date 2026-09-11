"""Validate natural-opening train/dev/holdout splits before launching MAME."""
import json
from pathlib import Path
from astra_play_sf2.runner import sha256


def load_dataset(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding='utf-8'))
    if data.get('schema') not in ('astra.rl-openings.v1', 'astra.rl-openings.v2') or data.get('status') != 'complete':
        raise ValueError('Need a completed natural-opening collection')
    opponents = data.get('opponents', [data.get('opponent', 2)])
    if not opponents or len(set(opponents)) != len(opponents) or any(type(o) is not int or o not in (0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11) for o in opponents):
        raise ValueError('Invalid opponent set')
    if type(data.get('difficulty')) is not int or data['difficulty'] not in range(3, 8):
        raise ValueError('Invalid difficulty')
    groups = {'train': [], 'dev': [], 'holdout': []}
    seen = set()
    for original in data['openings']:
        sample = dict(original)
        if sample.get('status') != 'accepted' or sample['difficulty'] != data['difficulty'] or sample['opponent'] not in opponents or sample.get('split') not in groups:
            raise ValueError('Checkpoint metadata mismatch')
        filename = (path.parent/sample['path']).resolve()
        if not filename.is_relative_to(path.parent):
            raise ValueError('Checkpoint outside dataset directory')
        actual = sha256(filename)
        if actual != sample['sha256'] or actual in seen:
            raise ValueError('Checkpoint checksum mismatch or duplicate across splits')
        seen.add(actual)
        sample.update(path=str(filename), difficulty=data['difficulty'], opponent=sample['opponent'])
        groups[sample['split']].append(sample)
    if not all(groups.values()):
        raise ValueError('Need nonempty train, dev and holdout splits')
    if any({sample['opponent'] for sample in rows} != set(opponents) for rows in groups.values()):
        raise ValueError('Every opponent needs train, dev and holdout coverage')
    return groups, data['difficulty']

