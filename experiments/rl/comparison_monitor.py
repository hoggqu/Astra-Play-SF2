"""One cheap status check; invoke periodically without an AI polling loop."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

PROBE = '''import json,os,pathlib,sys
p=pathlib.Path(sys.argv[1]); d=json.loads(p.read_text())
if d.get('status') not in ('complete','invalid'):
 try: os.kill(d['pid'],0)
 except ProcessLookupError: d.update(status='invalid',error='Suite process is no longer running')
print(json.dumps(d))
'''


def probe(endpoint):
    command=[endpoint['python'],'-c',PROBE,endpoint['result']]
    if endpoint.get('ssh'):command=endpoint['ssh']+[shlex.join(command)]
    result=subprocess.run(command,capture_output=True,text=True,timeout=30,check=True)
    return json.loads(result.stdout)


def compact(result):
    return {k:result.get(k) for k in ('status','error','device_resolved','wall_seconds','active_arm','arms','baseline_diagnostic') if k in result}


def notify(message):
    if Path('/usr/bin/osascript').exists():
        subprocess.run(['/usr/bin/osascript','-e','on run argv\ndisplay notification (item 1 of argv) with title "Astra SF2 A/B"\nend run',message],
                       capture_output=True,timeout=15)


def check(config):
    config=json.loads(Path(config).read_text());output=Path(config['output']);output.mkdir(parents=True,exist_ok=True)
    with (output/'monitor.lock').open('w') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        state_file=output/'monitor.json'
        state=json.loads(state_file.read_text()) if state_file.exists() else {'checks':0,'errors':{},'alerts':[]}
        if state.get('finished'):return
        state['checks']+=1;state['checked_unix']=time.time();results={}
        for label,endpoint in config['endpoints'].items():
            try:results[label]=compact(probe(endpoint));state['errors'][label]=0
            except Exception as error:
                state['errors'][label]=state['errors'].get(label,0)+1
                results[label]={'status':'unreachable','error':str(error)}
            if results[label]['status']=='invalid' or state['errors'][label]>=3:
                if label not in state['alerts']:
                    notify(label+' 实验异常，请查看报告');state['alerts'].append(label)
        state['results']=results
        finished=all(r['status'] in ('complete','invalid') for r in results.values())
        expired=time.time()>config['deadline_unix']
        if finished or expired:
            state.update(finished=True,deadline_expired=expired and not finished)
            report=output/'summary.json';report.write_text(json.dumps(state,ensure_ascii=False,indent=2))
            # Persist before launching the single completion analysis. A crash cannot
            # cause another AI call on every timer tick.
            state_file.write_text(json.dumps(state,ensure_ascii=False,indent=2))
            prompt=('请仅根据附加 JSON 写中文 SF2 PPO 实验结果分析，不调用工具，不执行任务，不启动训练。'
                    '说明完成或异常情况，不能合并不同权重的中间成绩。自然投币是连续序列，并非独立随机种子。'
                    '硬件差异和小样本限制必须保留，不夸大因果性。输出结果表、失败对手和下一步建议。\n'
                    +config.get('analysis_context','实验设计以结果 JSON 中记录为准。')+'\n'
                    +json.dumps(state,ensure_ascii=False))
            command=config.get('analysis_command')
            if command:
                with (output/'analysis.log').open('w') as log:
                    try:
                        done=subprocess.run(command,input=prompt,text=True,stdout=log,stderr=subprocess.STDOUT,timeout=600)
                        state['analysis_exit_code']=done.returncode
                    except Exception as error:state['analysis_error']=str(error)
            notify('实验完成，结果和分析已保存' if finished else '实验超过检查期限，请查看状态报告')
        state_file.write_text(json.dumps(state,ensure_ascii=False,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    check(p.parse_args().config)


if __name__=='__main__':main()
