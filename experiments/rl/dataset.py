"""Validate natural-opening train/dev/holdout splits before launching MAME."""
import json
from pathlib import Path
from astra_play_sf2.runner import sha256


def load_dataset(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding='utf-8'))
    if data.get('schema') != 'astra.rl-openings.v1' or data.get('status') != 'complete':
        raise ValueError('Need a completed natural-opening collection')
    groups = {'train': [], 'dev': [], 'holdout': []}
    seen = set()
    for original in data['openings']:
        sample = dict(original)
        if sample.get('status') != 'accepted' or sample['difficulty'] != data['difficulty'] or sample['opponent'] != data['opponent']:
            raise ValueError('Checkpoint metadata mismatch')
        filename = (path.parent/sample['path']).resolve()
        if not filename.is_relative_to(path.parent):
            raise ValueError('Checkpoint outside dataset directory')
        actual = sha256(filename)
        if actual != sample['sha256'] or actual in seen:
            raise ValueError('Checkpoint checksum mismatch or duplicate across splits')
        seen.add(actual)
        sample.update(path=str(filename), difficulty=data['difficulty'], opponent=data.get('opponent', 2))
        groups[sample['split']].append(sample)
    if not all(groups.values()):
        raise ValueError('Need nonempty train, dev and holdout splits')
    return groups, data['difficulty']

