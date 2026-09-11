"""Bounded port-latch comparison around the known 33-decision batch boundary."""
import argparse
import importlib
import json
from pathlib import Path
import shutil
import sys
from astra_play_sf2.config import atomic_json,load_config
from .dataset import load_dataset

HERE=Path(__file__).resolve().parent


def instrument(text,reference=False):
    if "local held_input=''" not in text:
        text=text.replace('local function input(text)', "local held_input=''\nlocal function input(text)\n held_input=text or ''")
    marker='local function fail(err)' if reference else 'local function answer()'
    helper='''local port_trace={}
local function sample_ports(kind)
 if frames and frames>=0 and frames<=460 and #port_trace<10000 then
  port_trace[#port_trace+1]={kind=kind,frame=frames,time=m.time:as_double(),paused=m.paused,
   in1=m.ioport.ports[':IN1']:read(),in2=m.ioport.ports[':IN2']:read(),held=held_input}
 end
end
local function publish_ports() IO.publish(string.format('training/port-phase-%08d.json',pending.id),json(port_trace)..'\\n') end
'''
    text=text.replace(marker,helper+marker)
    text=text.replace("rl_batch_frame_subscription=emu.add_machine_frame_notifier(function()", "rl_batch_frame_subscription=emu.add_machine_frame_notifier(function()\n sample_ports('machine_before')")
    text=text.replace('local effects=core:tick(s)', "local effects=core:tick(s);sample_ports('machine_after')")
    text=text.replace('rl_batch_rpc_subscription=emu.register_frame_done(function()', "rl_batch_rpc_subscription=emu.register_frame_done(function()\n sample_ports('frame_done_before')")
    text=text.replace("  return\n end\n local f=io.open", "  sample_ports('frame_done_after');return\n end\n local f=io.open")
    text=text.replace('  emu.unpause()','  emu.unpause();sample_ports(\'rpc_unpause\')')
    if reference:
        point="  if core.phase=='complete' then"
        extra="""  if pending.probe_frames and frames==pending.probe_frames then
   publish_ports();release();emu.pause()
   local result={id=pending.id,training_only=true,round_chain=true,state=s,observation={},episode_start=false,
    transitions={},episodes={},trace=pending.trace,chain_metrics={}}
   pending=nil;IO.publish(string.format('training/rl-batch-reply-%08d.json',result.id),json(result)..'\\n');return
  end
"""
        text=text.replace(point,extra+point)
    else:
        text=text.replace(" release();emu.pause()\n local result=", " sample_ports('answer_before');release();emu.pause();sample_ports('answer_after');publish_ports()\n local result=")
    return text


def run(source,plan,dataset,output):
    source=source.resolve();plan=json.loads(plan.read_text());groups,difficulty=load_dataset(dataset)
    options={'checkpoint':plan['checkpoint'],'lead':plan['lead']};output.mkdir(parents=True,exist_ok=False)
    results={};environment=None
    for role in ('fixed','manual','reference'):
        root=output/'sources'/role;shutil.copytree(source.parent,root,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        package=root/source.name
        text=(HERE/'chain_reference_runtime.lua').read_text() if role=='reference' else (HERE/'round_chain_runtime.lua').read_text()
        text=instrument(text,role=='reference')
        if role=='hold':text=text.replace("sample_ports('answer_before');release();emu.pause()", "sample_ports('answer_before');emu.pause()")
        (package/'batch_runtime.lua').write_text(text)
        sys.path.insert(0,str(root))
        try:
            module=importlib.import_module(source.name+'.batch_env')
            environment=module.BatchEnv(load_config(),output/role,difficulty,checkpoints=groups['train'])
            if role=='reference':
                response=environment.batch_rpc({'op':'reference_match','reset':options,'actions':plan['actions'],
                                               'frames':plan['frames'],'probe_frames':600})
                trace=response['trace']
            else:
                trace=[];offset=0
                if role=='manual':environment.batch_rpc({'op':'reset','reset':options})
                for count in (33,17):
                    response=environment.batch_rpc({'op':'reset_rollout' if offset==0 and role!='manual' else 'rollout','reset':options,
                        'count':count,'actions':[a['action'] for a in plan['actions'][offset:offset+count]],
                        'resets':[options]*count,'capture_trace':1})
                    trace+=response['trace'];offset+=count
            results[role]={'runtime_sha256':environment.manifest['runtime_sha256'],'trace':trace}
            atomic_json(output/(role+'.json'),results[role])
        finally:
            if environment:environment.close();environment=None
            sys.path.remove(str(root))
            for name in list(sys.modules):
                if name==source.name or name.startswith(source.name+'.'):sys.modules.pop(name)
    summary={'status':'complete','native_frames':600,'roles':{}}
    port_samples={}
    for role in results:
        files=sorted((output/role/'training').glob('port-phase-*.json'))
        port_samples[role]={r['frame']:(r['time'],r['in1'],r['in2']) for r in json.loads(files[-1].read_text())
                            if r['kind']=='machine_after' and r['frame']>0}
    for role in ('fixed','manual'):
        a=results[role]['trace'];b=results['reference']['trace']
        if len(a)!=600 or len(b)!=600:raise RuntimeError('Incomplete native trace')
        if any(any(x[k]!=y[k] for k in ('frame','state','phase','round','events')) for x,y in zip(a,b)):
            raise RuntimeError('Native trajectory differs')
        common=sorted(set(port_samples[role])&set(port_samples['reference']))
        if common!=list(range(1,461)):raise RuntimeError('Incomplete native port sample')
        if any(port_samples[role][n]!=port_samples['reference'][n] for n in common):
            raise RuntimeError('Native input port latch differs')
        summary['roles'][role]={'matching_native_frames':600,'matching_port_frames':len(common)}
    atomic_json(output/'summary.json',summary);return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source','plan','dataset','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();print(json.dumps(run(a.source,a.plan,a.dataset,a.output),indent=2))
