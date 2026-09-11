"""Offline, explicitly development-informed routing; no emulator or PPO updates."""
import argparse
import json
import math
from pathlib import Path
from .opponent_bundle import OPS,build_bundle,digest,encoded

ORDER=('f26','local-c2','remote85','local-c3')
NAMES={0:'Ryu',1:'Honda',2:'Blanka',3:'Guile',5:'Chun-Li',6:'Zangief',7:'Dhalsim',8:'Bison',9:'Sagat',10:'Balrog',11:'Vega'}

def wilson_lower(win,loss,draw,z=1.96):
    if any(type(x) is not int or x<0 for x in (win,loss,draw)):raise ValueError('Invalid W/L/D')
    n=win+loss+draw
    if not n:return -1.0
    p=win/n
    return (p+z*z/(2*n)-z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)))/(1+z*z/n)

def dev_selection(results):
    if tuple(results)!=ORDER:raise ValueError('Fixed model order required')
    aligned=None;dataset=None;stats={}
    for name,d in results.items():
        if (d.get('status')!='complete' or d.get('split')!='dev' or d.get('holdout_opened') is not False
            or d.get('selection')!='deterministic_argmax' or d.get('matches_requested')!=22
            or len(d.get('cases',[]))!=22):raise ValueError('Expected complete dev22 argmax evidence')
        keys=[(c['opponent'],c['checkpoint_sha256'],c['lead']) for c in d['cases']]
        if len(set(keys))!=22 or any(sum(k[0]==op for k in keys)!=2 for op in OPS):raise ValueError('Expected two distinct cases per opponent')
        if aligned is None:aligned=keys;dataset=d['dataset_sha256']
        if keys!=aligned or d['dataset_sha256']!=dataset:raise ValueError('Dev cases differ between candidates')
        stats[name]={str(op):{'matches':0,'match_wins':0,'round_wins':0} for op in OPS}
        for c in d['cases']:
            a=c['audit']
            if c['status']!='complete' or a.get('ok') is not True or a.get('outcome') not in ('win','loss'):raise ValueError('Invalid dev case')
            if any(x not in ('win','loss','draw') for x in a['round_outcomes']):raise ValueError('Invalid dev round')
            wins=a['round_outcomes'].count('win');losses=a['round_outcomes'].count('loss')
            if (max(wins,losses)!=2 or min(wins,losses)>=2 or (a['outcome']=='win')!=(wins==2)
                or a['round_outcomes'][-1]!=a['outcome']
                or a['round_outcomes'][:-1].count(a['outcome'])!=1
                or ('score' in a and a['score']!=[wins,losses])):raise ValueError('Dev outcome and mature round evidence differ')
            s=stats[name][str(c['opponent'])];s['matches']+=1;s['match_wins']+=a['outcome']=='win';s['round_wins']+=a['round_outcomes'].count('win')
    route={op:max(ORDER,key=lambda n:(stats[n][str(op)]['match_wins'],stats[n][str(op)]['round_wins'],-ORDER.index(n))) for op in OPS}
    return route,{'kind':'dev_selected','rule':'highest dev match wins, then round wins, then fixed model order',
                  'model_order':list(ORDER),'dataset_sha256':dataset,'aligned_cases':aligned,'statistics':stats,
                  'retrospective_selected_match_wins':sum(stats[route[op]][str(op)]['match_wins'] for op in OPS),
                  'retrospective_matches':22,'independent_win_rate':False,'formal_data_used':False}

def formal_selection(results):
    if tuple(results)!=ORDER:raise ValueError('Fixed model order required')
    stats={}
    for name,d in results.items():
        # Inputs are complete previous formal batches, now explicitly development data.
        if (d['status']!='complete' or d['attempts']!=20 or d.get('invalid',0)!=0):raise ValueError('Expected one complete previous full20 per model')
        rows=d['rounds'];stats[name]={}
        if set(rows)-set(map(str,OPS)):raise ValueError('Unknown opponent in prior formal evidence')
        for op in OPS:
            s={k:rows.get(str(op),{}).get(k,0) for k in ('win','loss','draw')}
            s['wilson_lower_heuristic']=wilson_lower(**s);stats[name][str(op)]=s
    route={op:max(ORDER,key=lambda n:(stats[n][str(op)]['wilson_lower_heuristic'],-ORDER.index(n))) for op in OPS}
    return route,{'kind':'prior_formal_informed','rule':'maximum Wilson lower-bound heuristic, z=1.96; draw is non-win; zero sample last; exact tie fixed order',
                  'model_order':list(ORDER),'z':1.96,'statistics':stats,'formal_data_used':True,
                  'old_formal_batches_are_development_inputs':True,'independent_win_rate':False,
                  'confidence_guarantee':False,'note':'Rounds and routes are not independent; future full20 must belong to this frozen bundle alone.'}

def load_formal(path,selector):
    d=json.loads(path.read_text())
    if selector in ('1','2'):
        d=d['formal'][selector];r=d['reliability']
        if r.get('errors') or not d.get('native_recomputed'):raise ValueError('Incomplete prior formal audit')
        return {'status':r['status'],'attempts':r['observed_attempts'],'invalid':r['attempt_counts'].get('invalid',0),'model_sha256':r['model_sha256'],'rounds':d['rounds']}
    if selector!='dual' or d['status']!='pass' or d.get('interface_parent_source_reaudit') is not True:raise ValueError('Unsupported or failed prior audit')
    return {'status':'complete','attempts':len(d['attempts']),'invalid':d['invalid'],'model_sha256':d['model_sha256'],
            'rounds':{str(op):d['rounds'][name] for op,name in NAMES.items()}}

def build_selected(config,output):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    if tuple(config['candidates'])!=ORDER:raise ValueError('Candidates must follow fixed declared model order')
    models={};dev={};formal={};sources={}
    for name,row in config['candidates'].items():
        models[name]=Path(row['model']);h=digest(models[name].read_bytes())
        if h!=row['model_sha256']:raise ValueError('Candidate model hash differs')
        dp=Path(row['dev']);fp=Path(row['formal']);dev[name]=json.loads(dp.read_text());formal[name]=load_formal(fp,row['formal_selector'])
        if dev[name]['model_sha256']!=h or formal[name]['model_sha256']!=h:raise ValueError('Evidence model differs')
        sources[name]={'model_sha256':h,'dev_source_sha256':digest(dp.read_bytes()),'prior_formal_audit_sha256':digest(fp.read_bytes()),'prior_formal_selector':row['formal_selector']}
    dr,ds=dev_selection(dev);fr,fs=formal_selection(formal)
    documentation=Path(config['public_formal_summary']);public_sha=digest(documentation.read_bytes())
    output.mkdir(parents=True)
    results={}
    for name,route,selection in [('baseline-all-f26',{op:'f26' for op in OPS},{'kind':'baseline_equivalence','model_order':list(ORDER),'formal_data_used':False,'independent_win_rate':False}),('dev-selected',dr,ds),('prior-formal-informed',fr,fs)]:
        selection['selection_tool_sha256']=digest(Path(__file__).read_bytes())
        selection['sources']=sources if name=='prior-formal-informed' else {n:{k:v for k,v in s.items() if not k.startswith('prior_formal')} for n,s in sources.items()}
        if name=='prior-formal-informed':selection['public_summary_sha256']=public_sha
        result=build_bundle(output/(name+'.zip'),models,route,selection)
        results[name]={'file':name+'.zip','model_sha256':result['model_sha256'],'route_sha256':result['route_sha256'],'route_candidates':{str(k):v for k,v in route.items()},'selection':selection}
    summary={'schema':'astra.rl-opponent-bundle-selection.v1','status':'complete','native_validated':False,'formal_goal_claim':False,'candidates':results}
    (output/'selection.json').write_bytes(encoded(summary));return summary

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args(argv)
    r=build_selected(json.loads(a.config.read_text()),a.output);print(json.dumps({k:{'model_sha256':v['model_sha256'],'route':v['route_candidates']} for k,v in r['candidates'].items()},indent=2))
if __name__=='__main__':main()
