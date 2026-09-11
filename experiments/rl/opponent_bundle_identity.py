"""Source derivation and branch audit for a new frozen opponent-routed policy."""
import ast
import hashlib
import json
from pathlib import Path
import re
from astra_play_sf2.runner import sha256
from .pulsed_normals_identity import validate_pulsed_build

EXTRAS=('opponent_bundle.py','opponent_bundle_identity.py','bundle_base_nn.lua')
def patch(text,old,new):
    if text.count(old)!=1:raise RuntimeError('Ambiguous bundle anchor: '+old[:100])
    return text.replace(old,new)

def derive(captured,nn_source):
    out=dict(captured)
    for name in ('continuous.py','native_continuous.py','support.py'):
        text=out[name].decode().replace('astra.rl-continuous.actions16.v1','astra.rl-continuous.opponent-bundle.v1').replace('mame.rl-continuous-play.actions16.v1','mame.rl-continuous-play.opponent-bundle.v1')
        text=text.replace("policy_kind='ppo'","policy_kind='ppo_opponent_bundle'").replace("get('policy_kind') != 'ppo'","get('policy_kind') != 'ppo_opponent_bundle'").replace("'rl_ppo'","'rl_ppo_bundle'")
        out[name]=text.encode()
    text=out['export.py'].decode();node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='export_policy')
    lines=text.splitlines(keepends=True);lines[node.lineno-1:node.end_lineno]=['def export_policy(model_path):\n    from .opponent_bundle import export_bundle\n    return export_bundle(model_path)\n']
    out['export.py']=''.join(lines).encode()
    text=out['continuous.py'].decode()
    text=patch(text,"[('nn.lua', 'rl_nn.lua'),", "[('bundle_base_nn.lua', 'rl_bundle_base_nn.lua'), ('nn.lua', 'rl_nn.lua'),")
    text=patch(text,"'training/runtime/rl_policy.lua','training/runtime/rl_nn.lua',", "'training/runtime/rl_policy.lua','training/runtime/rl_bundle_base_nn.lua','training/runtime/rl_nn.lua',")
    # Insert fields after the native stage's existing exact replacement anchor remains intact.
    text=patch(text,"model_sha256=Model.model_sha256\")", "model_sha256=Model.model_sha256,bundle_route_sha256=Model.route_sha256,branch_id=Model.route[tostring(r.opponent)],branch_source_model_sha256=Model.branches[Model.route[tostring(r.opponent)]].source_model_sha256\")")
    text=patch(text,"        result['model_sha256'] = payload['model_sha256']", "        result['model_sha256'] = payload['model_sha256']\n        result['policy_kind'] = 'ppo_opponent_bundle'\n        result['bundle_route_sha256'] = payload['route_sha256']\n        result['branch_model_sha256'] = {k:b['source_model_sha256'] for k,b in payload['branches'].items()}")
    out['continuous.py']=text.encode()
    text=out['support.py'].decode()
    text=patch(text,'from .pulsed_normals_identity import validate_pulsed_build as validate_chain_build','from .opponent_bundle_identity import validate_bundle_build as validate_chain_build')
    text=text.replace("'astra.rl-policy.actions16.v1'","'astra.rl-opponent-bundle-policy.v1'")
    text=patch(text,"('rl_actions.lua','actions.lua'),('rl_nn.lua','nn.lua'),", "('rl_actions.lua','actions.lua'),('rl_nn.lua','nn.lua'),('rl_bundle_base_nn.lua','bundle_base_nn.lua'),")
    text=patch(text,"\n    result['action_interface_audit'] = {'ok':False,'reason':'Run did not complete'}", "\n    result['bundle_audit'] = {'ok':False,'reason':'Run did not complete'}\n    result['action_interface_audit'] = {'ok':False,'reason':'Run did not complete'}")
    text=patch(text,"\n        result['action_interface_audit'] = {'ok':False,'reason':'Run did not complete'}", "\n        result['bundle_audit'] = {'ok':False,'reason':'Run did not complete'}\n        result['action_interface_audit'] = {'ok':False,'reason':'Run did not complete'}")
    text=patch(text,"            result['action_interface_audit'] = audit_interface(snapshot, output, result)", "            result['action_interface_audit'] = audit_interface(snapshot, output, result)\n            from .opponent_bundle_identity import audit_branches\n            result['bundle_audit'] = audit_branches(snapshot['payload'], output, result)")
    out['support.py']=text.encode();out['nn.lua']=nn_source;out['bundle_base_nn.lua']=captured['nn.lua']
    return out

def validate_bundle_build(manifest,package):
    package=Path(package).resolve();root=package.parent
    if (manifest.get('schema')!='astra.rl-opponent-bundle-code.v1' or manifest.get('package')!=package.name
        or manifest.get('action_interface')!='ken_actions16_pulsed_normals_v2' or manifest.get('actions')!=16
        or manifest.get('observations')!=344):raise RuntimeError('Invalid routed bundle code identity')
    parent_root=root/'bundle-parent-source';parent=json.loads((parent_root/'build.json').read_text())
    if sha256(parent_root/'build.json')!=manifest['parent_build_sha256']:raise RuntimeError('Bundle code parent manifest changed')
    validate_pulsed_build(parent,parent_root/package.name)
    for name,digest in manifest['frozen_inputs_sha256'].items():
        p=(root/name).resolve()
        if not p.is_relative_to(root) or sha256(p)!=digest:raise RuntimeError('Bundle provenance changed: '+name)
    captured={n:(parent_root/package.name/n).read_bytes() for n in parent['derived_sha256']}
    expected=derive(captured,(package/'nn.lua').read_bytes())
    if set(manifest['derived_sha256'])!=set(expected)|set(EXTRAS):raise RuntimeError('Unexpected bundle executing files')
    for name,body in expected.items():
        if (package/name).read_bytes()!=body:raise RuntimeError('Unexpected bundle source derivation: '+name)
    for name,digest in manifest['derived_sha256'].items():
        if Path(name).name!=name or sha256(package/name)!=digest:raise RuntimeError('Bundle source changed: '+name)
    return parent

def audit_branches(payload,run,result):
    if (result.get('schema')!='astra.rl-continuous.opponent-bundle.v1' or result.get('policy_kind')!='ppo_opponent_bundle'
        or result.get('bundle_route_sha256')!=payload['route_sha256'] or result.get('model_sha256')!=payload['model_sha256']
        or result.get('branch_model_sha256')!={k:b['source_model_sha256'] for k,b in payload['branches'].items()}):raise RuntimeError('Run bundle identity differs')
    count=decisions=0;used={}
    for attempt in result['attempts']:
        for name in attempt['matches']:
            path=(Path(run)/name).resolve()
            if not path.is_relative_to(Path(run).resolve()):raise RuntimeError('Match escaped run')
            match=json.loads(path.read_text());s=match['summary'];op=s['opponent'];bid=payload['route'].get(str(op))
            if bid is None:raise RuntimeError('Unknown opponent in match')
            branch=payload['branches'][bid];h=branch['source_model_sha256']
            if (s.get('branch_id')!=bid or s.get('branch_source_model_sha256')!=h
                or s.get('bundle_route_sha256')!=payload['route_sha256']
                or s.get('model_sha256')!=payload['model_sha256'] or s.get('policy_kind')!='ppo_opponent_bundle'):raise RuntimeError('Match branch identity differs')
            rows=match['trace']['decisions']
            if not rows:raise RuntimeError('Missing branch decision evidence')
            for d in rows:
                reason=d.get('reason','');m=re.fullmatch('rl_bundle_'+re.escape(bid)+'_'+h+r'_action_(\d+)',reason)
                if not m or int(m[1]) not in range(16) or d.get('cpu',{}).get('char')!=op:raise RuntimeError('Decision branch identity differs')
            count+=1;decisions+=len(rows);used[str(op)]=bid
    if count==0:raise RuntimeError('No bundle matches')
    return {'ok':True,'model_sha256':payload['model_sha256'],'route_sha256':payload['route_sha256'],
            'all_branches_frozen':11,'checked_matches':count,'checked_decisions':decisions,'used_routes':used,
            'selection':'fixed opponent route then deterministic argmax','external_fallback':False}
