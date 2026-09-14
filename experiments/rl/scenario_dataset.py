"""Replace four train openings for declared weak opponents; preserve all sealed evaluation states."""
import argparse
import copy
import json
from pathlib import Path
import shutil
from .full_interface_dataset import load_dataset
from .versioned_campaign import digest, write_json


def build(baseline, collection, output, opponents=(0,2,5)):
    baseline,collection,output=map(lambda p:Path(p).resolve(),(baseline,collection,output))
    load_dataset(baseline)
    old=json.loads(baseline.read_text()); new=json.loads(collection.read_text())
    if new.get('status')!='complete' or new.get('difficulty')!=3 or new.get('error'):
        raise ValueError('Need a completed Normal natural collection')
    reserved={r['attempt'] for r in new['openings'] if r['split']!='train'}
    known={r['sha256'] for r in old['openings']}
    chosen={}
    for opponent in opponents:
        eligible=[r for r in new['openings'] if r['opponent']==opponent and r['split']=='train'
                  and r['attempt'] not in reserved and r['sha256'] not in known]
        if len(eligible)<4: raise ValueError('Need four new, nonoverlapping train openings for '+str(opponent))
        chosen[opponent]=eligible[:4]
    output.mkdir(parents=True,exist_ok=False)
    def materialize(row, source, prefix, split=None):
        row=copy.deepcopy(row); file=(source.parent/row['path']).resolve()
        if not file.is_relative_to(source.parent) or digest(file)!=row['sha256']:
            raise ValueError('Source checkpoint path/hash mismatch')
        row['id']=prefix+row['id']; name='states/'+row['sha256']+'.sta'
        dest=output/name;dest.parent.mkdir(exist_ok=True)
        if not dest.exists():shutil.copyfile(file,dest)
        row.update(path=name,source_manifest_sha256=digest(source),source_id=row['id'][len(prefix):])
        if split:row['split']=split
        return row
    base=[materialize(r,baseline,'base-') for r in old['openings']]
    keep_ids={}
    for opponent in opponents:
        rows=[r for r in base if r['opponent']==opponent and r['split']=='train']
        keep_ids[opponent]={r['id'] for r in rows[::2]}
    candidate=[r for r in base if r['split']!='train' or r['opponent'] not in opponents or r['id'] in keep_ids[r['opponent']]]
    candidate += [materialize(r,collection,'fresh-') for rs in chosen.values() for r in rs]
    fresh_dev=[materialize(r,collection,'fresh-') for r in new['openings'] if r['split']=='dev' and r['opponent'] in opponents and r['sha256'] not in known]
    if {r['opponent'] for r in fresh_dev}!=set(opponents):raise ValueError('Missing fresh development coverage')
    for name,rows in [('A',base),('B',candidate),('evaluation',fresh_dev+base)]:
        value=copy.deepcopy(old);value.update(openings=rows,scenario_coverage={'protocol':'four-old-four-fresh-v1',
            'arm':name,'source_collection_sha256':digest(collection),'opponents':list(opponents),
            'whole_attempts_excluded_from_training':sorted(reserved),'outcome_based_selection':False})
        write_json(output/(name+'.json'),value);load_dataset(output/(name+'.json'))
    return {name:{'path':str(output/(name+'.json')),'sha256':digest(output/(name+'.json'))} for name in ('A','B','evaluation')}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('baseline','collection','output'):p.add_argument('--'+name,type=Path,required=True)
    result=build(**vars(p.parse_args()));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
