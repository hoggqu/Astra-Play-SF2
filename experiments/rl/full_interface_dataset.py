"""Full85 input protocol; eight train openings per declared opponent.

Completed v1/v2 collections may declare all eleven opponents or a subset.
Membership never changes. Only the train group enters BatchEnv; development and
holdout checksums are validated without evaluating or training on those states.
"""
import json
from pathlib import Path
from astra_play_sf2.runner import sha256

SCHEMA = 'astra.rl-full85-dataset-protocol.v1'
OPPONENTS = (0,1,2,3,5,6,7,8,9,10,11)


def load_dataset(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding='utf-8'))
    if data.get('schema') == 'astra.rl-specialist-openings.v1':
        from .specialist_dataset import load_dataset as original
        return original(path)
    if (data.get('schema') not in ('astra.rl-openings.v1','astra.rl-openings.v2')
            or data.get('status') != 'complete' or data.get('error')):
        raise ValueError('Expected a completed, valid opening collection')
    opponents = data.get('opponents', [data.get('opponent')])
    if (not opponents or len(set(opponents)) != len(opponents)
            or any(type(o) is not int or o not in OPPONENTS for o in opponents)):
        raise ValueError('Invalid declared opponent set')
    difficulty = data.get('difficulty')
    if type(difficulty) is not int or difficulty not in range(3,8):
        raise ValueError('Invalid difficulty')
    groups = {'train':[], 'dev':[], 'holdout':[]}
    ids, hashes = set(), set()
    for original in data['openings']:
        row = dict(original)
        if (row.get('status') != 'accepted' or row.get('difficulty') != difficulty
                or row.get('opponent') not in opponents or row.get('split') not in groups):
            raise ValueError('Opening metadata/split mismatch')
        if row.get('id') in ids or row.get('sha256') in hashes:
            raise ValueError('Duplicate opening ID/hash across splits')
        ids.add(row['id']); hashes.add(row['sha256'])
        filename = (path.parent/row['path']).resolve()
        if not filename.is_relative_to(path.parent) or sha256(filename) != row['sha256']:
            raise ValueError('Checkpoint path/checksum mismatch')
        row['path'] = str(filename)
        groups[row['split']].append(row)
    for op in opponents:
        if sum(r['opponent']==op for r in groups['train']) != 8:
            raise ValueError('Exactly eight train openings per declared opponent required')
        if any(not any(r['opponent']==op for r in groups[s]) for s in ('dev','holdout')):
            raise ValueError('Every opponent needs separate dev and holdout coverage')
    return groups, difficulty


def metadata(path):
    path = Path(path).resolve(); groups, difficulty = load_dataset(path)
    data = json.loads(path.read_text())
    return {'schema':SCHEMA, 'source_schema':data['schema'], 'difficulty':difficulty,
            'dataset_sha256':sha256(path), 'opponents':sorted({r['opponent'] for r in groups['train']}),
            'counts':{s:len(rows) for s,rows in groups.items()},
            'training_split':'train', 'dev_used_for_training':False,
            'holdout_used_for_training':False, 'selection_performed':False,
            'sampling':'uniform opponent; uniform opening and original train lead'}
