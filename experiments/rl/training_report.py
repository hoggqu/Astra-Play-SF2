#!/usr/bin/env python3
"""Portable read-only training summary and standalone HTML; no traces or ZIP reads.

Completed-stage validation retains the original campaign-summary-001 semantics.
"""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import statistics
import html
import os
import tempfile

NAMES = dict(enumerate(('Ryu','Honda','Blanka','Guile','Ken','Chun-Li','Zangief','Dhalsim','Bison','Sagat','Balrog','Vega')))

SCHEMA = 'astra.rl-training-summary.v1'
REPORT_MARKER = '<!-- astra.rl-training-report.v1 -->'

def read(path):
    if path.stat().st_size > 32*1024*1024:raise ValueError('Summary input exceeds 32 MiB; refusing a possible raw trace: '+str(path))
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()

def stats(rows):
    c = collections.Counter(r['outcome'] for r in rows)
    returns = [r['return'] for r in rows if 'return' in r]
    return dict(W=c['win'], L=c['loss'], D=c['draw'], n=len(rows),
                win_rate=c['win']/len(rows) if rows else None,
                mean_return=statistics.mean(returns) if returns else None)

def training(path, result):
    episodes, matches, seen, excluded = [], [], set(), collections.Counter()
    workers = sorted(path.glob('worker-*'))
    if len(workers) != result['workers']:
        raise ValueError('worker directory count differs from completed result')
    for worker in workers:
        with (worker/'episodes.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                if row.get('baseline') or row.get('incomplete') or row.get('phase') != 'train':
                    excluded['not_complete_train'] += 1
                    continue
                key = (worker.name, row['episode'])
                if key in seen:
                    raise ValueError(f'duplicate Python episode {key}')
                seen.add(key)
                native = row['native_round']
                if row['outcome'] not in ('win','loss','draw') or native['outcome'] != row['outcome']:
                    raise ValueError('Python/native round outcome mismatch')
                if not math.isfinite(row['return']):
                    raise ValueError('nonfinite episode return')
                op = row['opponent']
                if op not in NAMES or op == 4:
                    raise ValueError('unsupported opponent')
                e = dict(opponent=op, outcome=row['outcome'], **{'return':row['return']})
                episodes.append(e)
                score = native['score']
                if len(score) != 2 or any(type(v) is not int or v not in (0,1,2) for v in score) or score == [2,2]:
                    raise ValueError('invalid mature score')
                if 2 in score:
                    outcome = 'win' if score[0] == 2 else 'loss'
                    if outcome != row['outcome']:
                        raise ValueError('terminal match/round mismatch')
                    matches.append(dict(opponent=op, outcome=outcome))
    return dict(rounds=stats(episodes), matches=stats(matches), excluded=dict(excluded),
        match_count_basis='terminal native_round.score from Python complete episodes only',
        per_opponent={str(op):dict(name=NAMES[op], rounds=stats([r for r in episodes if r['opponent']==op]),
                    matches=stats([r for r in matches if r['opponent']==op]))
                    for op in sorted({r['opponent'] for r in episodes})})

def evaluation(path, model_sha):
    file = path/'result.json'
    if not file.exists():
        return dict(status='not_available', counted=False)
    result, digest = read(file)
    out = dict(status=result.get('status'), counted=False, result_sha256=digest,
               model_sha256=result.get('model_sha256'), attempts=[])
    if result.get('status') != 'complete':
        return out
    if result.get('model_sha256') != model_sha:
        raise ValueError('evaluation/train model SHA mismatch')
    audits = ['native_timing_audit','action_interface_audit']
    if result.get('selection') == 'categorical' or (path/'sampling-protocol.json').exists():
        audits.append('sampling_audit')
    if not all(result.get(k,{}).get('ok') is True for k in audits):
        raise ValueError('completed evaluation lacks required audit PASS')
    for a in result['attempts']:
        if a.get('audit',{}).get('ok') is not True or a['outcome'] not in ('loss','rl_gameplay_clear'):
            raise ValueError('invalid completed attempt')
        paths = a['matches']; wins = a['match_wins']; lost = a['outcome']=='loss'
        if wins != len(paths)-int(lost) or (not lost and wins != 11):
            raise ValueError('attempt wins/path count mismatch')
        opponent = None
        if lost:
            raw_path = (path/paths[-1]).resolve()
            if not raw_path.is_relative_to(path.resolve()):
                raise ValueError('match path outside run')
            # Small status sidecar only. Do not open raw_path (full all-frame trace).
            status, _ = read(raw_path.with_name(raw_path.stem+'-status.json'))
            if status.get('active') is not False or status.get('result') != 'cpu_win' or status.get('valid_continuous') is not True or status.get('model_sha256') != model_sha:
                raise ValueError('failure status sidecar not mature loss')
            opponent = status['opponent']
        out['attempts'].append(dict(id=a['id'], outcome=a['outcome'], match_wins=wins,
            failure_opponent=opponent, failure_name=NAMES.get(opponent)))
    # Detailed opponent totals come only from small sidecars. Old evidence may
    # lack these: retain audited attempt totals but report detail coverage.
    per_opponent={};details=0;requested=0;seen=set()
    for attempt in result['attempts']:
        for relative in attempt['matches']:
            requested+=1;raw_path=(path/relative).resolve()
            if not raw_path.is_relative_to(path.resolve()) or raw_path in seen:
                raise ValueError('Duplicate/outside evaluation match path')
            seen.add(raw_path);sidecar=raw_path.with_name(raw_path.stem+'-status.json')
            if not sidecar.exists():continue
            status,_=read(sidecar)
            if (status.get('active') is not False or status.get('valid_continuous') is not True
                or status.get('model_sha256')!=model_sha or status.get('result') not in ('ken_win','cpu_win')):
                raise ValueError('evaluation status sidecar not a mature result')
            op=status.get('opponent')
            if op not in NAMES or op==4:raise ValueError('unsupported sidecar opponent')
            target=per_opponent.setdefault(str(op),dict(name=NAMES[op],match_W=0,match_L=0,
                round_W=0,round_L=0,round_D=0,round_details=0,match_details=0))
            target['match_W' if status['result']=='ken_win' else 'match_L']+=1
            target['match_details']+=1;details+=1
            score=status.get('score');rounds=status.get('round')
            if score is None or rounds is None:continue
            if (not isinstance(score,list) or len(score)!=2 or any(type(v) is not int or not 0<=v<=2 for v in score)
                or score==[2,2] or max(score)!=2 or type(rounds) is not int or rounds<sum(score)
                or (score[0]==2)!=(status['result']=='ken_win')):
                raise ValueError('invalid sidecar final score/round count')
            target['round_W']+=score[0];target['round_L']+=score[1]
            target['round_D']+=rounds-sum(score);target['round_details']+=1
    out.update(per_opponent=per_opponent,match_details_available=details,
               match_details_requested=requested,detail_complete=details==requested)
    out.update(counted=True, selection=result.get('selection','deterministic_argmax'),
               clears=sum(a['outcome']=='rl_gameplay_clear' for a in out['attempts']))
    return out

def resolve_campaign(value):
    root=Path(value).resolve()
    if not root.is_dir():raise ValueError('Campaign directory not found')
    nested=root/'run'
    return nested.resolve() if nested.is_dir() else root


def summarize(campaign):
    requested=Path(campaign).resolve();campaign=resolve_campaign(requested)
    out = dict(schema=SCHEMA, campaign=str(campaign), input=str(requested),
               evidence='Summary of existing audited results, not an independent full-trace audit',
               cycles=[], skipped=[], issues=[])
    campaign_file=campaign/'result.json'
    if campaign_file.exists():
        try:
            record,digest=read(campaign_file)
            out.update(campaign_status=record.get('status','unknown'),campaign_result_sha256=digest,
                stop_reason=record.get('stop_reason'),max_duration_seconds=record.get('max_duration_seconds'),
                wall_seconds=record.get('wall_seconds'),cycles_requested=record.get('cycles_requested'),
                latest_model=record.get('latest_model'),latest_model_sha256=record.get('latest_model_sha256'),
                active_stage=record.get('active_stage'),campaign_error=record.get('error'))
        except (ValueError,KeyError,OSError) as e:
            out['issues'].append(dict(stage='campaign',error=str(e)));out['campaign_status']='unverified'
    else:out['campaign_status']='unknown'
    for cycle in sorted(campaign.glob('cycle-*')):
        file = cycle/'train/result.json'
        if not file.exists():
            out['skipped'].append(dict(cycle=cycle.name, reason='training result absent')); continue
        try:result, digest = read(file)
        except (ValueError,KeyError,OSError) as e:
            out['issues'].append(dict(cycle=cycle.name,stage='training',error=str(e)))
            out['skipped'].append(dict(cycle=cycle.name,reason='training result unreadable',status='unverified'));continue
        if result.get('status') != 'complete':
            out['skipped'].append(dict(cycle=cycle.name, reason='training not complete', status=result.get('status'),
                actual_steps=result.get('actual_steps',result.get('completed_update_steps')),requested_steps=result.get('requested_steps'),
                last_checkpoint=result.get('last_checkpoint'),error=result.get('error'))); continue
        try:
            row = dict(cycle=cycle.name, model_sha256=result['model_sha256'], steps=result['actual_steps'],
                       requested_steps=result.get('requested_steps'),budget_stop=result.get('budget_stop',False),
                       stop_reason=result.get('stop_reason'),model_path=str(cycle/'train/ppo-batch.zip'),
                       seed=result.get('seed'), dataset_sha256=result.get('dataset_sha256'),
                       result_sha256=digest, ppo={})
            row['parameters'] = dict(workers=result.get('workers'), rollout_steps=result.get('rollout_steps'), minibatch_size=result.get('minibatch_size',result.get('effective_ppo',{}).get('batch_size')), epochs=result.get('effective_ppo',{}).get('n_epochs'), update_cycles=len(result.get('iterations',[])))
            row['training'] = training(cycle/'train', result)
            if 'opponent_sampling' in result:
                row['opponent_sampling'] = sampling_summary(result['opponent_sampling'],row['steps'])
            for key in ('policy_entropy','approx_kl','explained_variance'):
                values=[i.get('ppo',{}).get(key) for i in result.get('iterations',[])]
                values=[v for v in values if isinstance(v,(int,float)) and math.isfinite(v)]
                row['ppo'][key]=dict(mean=statistics.mean(values),first=values[0],last=values[-1]) if values else None
            try:
                row['evaluation']=evaluation(cycle/'natural-coins', row['model_sha256'])
            except (ValueError,KeyError,OSError) as e:
                row['evaluation']=dict(status='unverified', counted=False, error=str(e))
                out['issues'].append(dict(cycle=cycle.name, stage='evaluation', error=str(e)))
            if read(file)[1] != digest:
                raise ValueError('completed training result changed during reading')
            out['cycles'].append(row)
        except (ValueError,KeyError,OSError) as e:
            out['issues'].append(dict(cycle=cycle.name,error=str(e)))
    if out['cycles'] and not out.get('latest_model'):
        last=out['cycles'][-1];out.update(latest_model=last['model_path'],latest_model_sha256=last['model_sha256'])
    out['completed_training_steps']=sum(row['steps'] for row in out['cycles'])
    return out

STATUS = {'running':'运行中','sampling':'采样中','updating':'更新中','complete':'完成',
          'clear':'已通关','exhausted':'训练轮数完成','budget_stopped':'时长预算结束',
          'cancelled':'用户安全停止','invalid':'无效运行','unknown':'未知',
          'unverified':'尚未核验','not_available':'尚无结果'}

def esc(value):
    return html.escape('—' if value is None else str(value),quote=True)

def percent(value):
    return '—' if value is None else f'{value*100:.1f}%'

def number(value,digits=3):
    return '—' if value is None else f'{value:.{digits}f}'

def wld(value):
    return '—' if value is None else f"{value['W']} / {value['L']} / {value['D']}"

def table(headers,rows):
    if not rows:return '<p class="muted">暂无可计入的数据。</p>'
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+esc(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'

def sampling_summary(value, steps):
    from .adaptive_sampling import PROTOCOL, probability_map
    if value.get('protocol') != PROTOCOL or value.get('mode') not in ('adaptive','uniform'):
        raise ValueError('Unknown opponent sampler summary')
    rows=value['per_opponent'];ops=sorted(int(k) for k in rows)
    probability_map(ops,{k:r['next_probability'] for k,r in rows.items()})
    total=0
    for key,r in rows.items():
        if int(key) not in NAMES or int(key)==4:raise ValueError('Unknown sampler opponent')
        for name in ('decisions','rounds','wins','losses','draws','round_decisions','recent_rounds'):
            if type(r[name]) is not int or r[name]<0:raise ValueError('Invalid sampler counts')
        if r['rounds'] != r['wins']+r['losses']+r['draws']:raise ValueError('Sampler round total mismatch')
        total+=r['decisions']
        for name in ('decision_share','smoothed_win_rate'):
            if type(r[name]) not in (float,int) or not math.isfinite(r[name]) or not 0<=r[name]<=1:
                raise ValueError('Invalid sampler rate')
        if not math.isclose(r['decision_share'],r['decisions']/steps,abs_tol=1e-10):
            raise ValueError('Sampler decision share mismatch')
    if total!=steps or value['decisions']!=steps:raise ValueError('Sampler decision total mismatch')
    return value


def render_html(summary):
    """Every dynamic value is escaped; the document loads no external assets."""
    status=summary.get('campaign_status','unknown')
    status_name=STATUS.get(status,status)
    cycles=summary['cycles'];attempts=[a for c in cycles if c.get('evaluation',{}).get('counted') for a in c['evaluation']['attempts']]
    clear_count=sum(a['outcome']=='rl_gameplay_clear' for a in attempts)
    color='bad' if status in ('invalid','unverified') or summary['issues'] else 'good' if status in ('clear','complete','exhausted','budget_stopped') else 'neutral'
    chunks=[REPORT_MARKER,'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Astra · 训练报告</title><style>
:root{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;color:#202336;background:#f4f5fa}body{margin:0}main{max-width:1180px;margin:auto;padding:36px 24px 64px}h1{margin:8px 0;font-size:32px;letter-spacing:-1px}h2{font-size:20px;margin:0 0 16px}h3{font-size:16px}.eyebrow{color:#6153b2;font-weight:700;letter-spacing:2px;font-size:12px}.muted{color:#666b80;line-height:1.7;font-size:14px}.badge{display:inline-block;padding:5px 11px;border-radius:20px;font-size:13px;font-weight:650}.good{background:#e0f4ea;color:#176a49}.bad{background:#ffe8e7;color:#a52d2b}.neutral{background:#ebe7ff;color:#544399}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px;margin:24px 0}.card,section{background:white;border:1px solid #e3e5ef;border-radius:14px;padding:20px}.card b{display:block;font-size:27px;margin:8px 0}.card span{font-size:13px;color:#656b80}section{margin:18px 0;box-shadow:0 3px 15px #20233603}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px;text-align:left}th{color:#656b80;font-weight:600;background:#f8f9fc;white-space:nowrap}td,th{padding:11px 12px;border-bottom:1px solid #edf0f6;vertical-align:top}td{font-variant-numeric:tabular-nums}code{font-size:12px;overflow-wrap:anywhere}details{border-top:1px solid #e8eaf2;margin-top:16px;padding-top:14px}summary{cursor:pointer;font-weight:650}.note{border-left:3px solid #8c7ddd;padding:10px 14px;background:#f7f5ff;font-size:14px;line-height:1.65}footer{font-size:12px;color:#7a8092;margin-top:22px;line-height:1.7}
</style><main><div class="eyebrow">ASTRA PLAY SF2</div><h1>强化学习训练报告</h1>''',
        '<span class="badge '+color+'">'+esc(status_name)+'</span>',
        '<p class="muted">离线读取已落盘记录。训练成绩与自然投币验证分开；不同模型不合并计算通关率。</p>',
        '<div class="cards"><div class="card"><span>已汇总完整训练轮</span><b>'+esc(len(cycles))+'</b></div>',
        '<div class="card"><span>完整训练决策数</span><b>'+esc(f"{summary.get('completed_training_steps',0):,}")+'</b></div>',
        '<div class="card"><span>已审计自然投币次数</span><b>'+esc(len(attempts))+'</b></div>',
        '<div class="card"><span>已审计通关记录数 · 不同模型</span><b>'+esc(clear_count)+'</b></div></div>']
    reason=summary.get('stop_reason')
    if reason:
        message=('时长预算已到。保存完整 PPO 更新后停止；收尾更新与最后一次评估可能超过设定时长。' if reason=='max_duration' else '用户请求安全停止。已保存的模型保留；取消不是游戏败局。')
        chunks.append('<p class="note">'+esc(message)+'</p>')
    if summary.get('campaign_error'):chunks.append('<p class="note">运行异常：'+esc(summary['campaign_error'])+'</p>')
    chunks.append('<section><h2>最新可恢复模型</h2><p class="muted">下列路径与 SHA 来自训练结果；报告不重新读取或审计模型 ZIP。</p><p><code>'+esc(summary.get('latest_model'))+'</code></p><p><code>SHA256 '+esc(summary.get('latest_model_sha256'))+'</code></p></section>')
    rows=[]
    for c in cycles:
        train=c['training'];ev=c.get('evaluation',{})
        ev_text=f"{ev['clears']} / {len(ev['attempts'])}" if ev.get('counted') else STATUS.get(ev.get('status'),ev.get('status','未知'))
        rows.append([c['cycle'],c['steps'],percent(train['rounds']['win_rate']),percent(train['matches']['win_rate']),number(train['rounds']['mean_return']),ev_text,c['model_sha256'][:12]])
    chunks.append('<section><h2>逐轮成绩</h2><p class="muted">一轮指一段训练加评估，不是游戏小局。训练统计覆盖该轮采样过程中的策略；自然投币使用该轮末固定模型。</p>'+table(['训练轮','完成决策','训练小局胜率','训练整场胜率','平均回报','该模型通关 / 投币','末模型 SHA'],rows)+'</section>')
    for c in cycles:
        ev=c.get('evaluation',{});tr=c['training']['per_opponent']
        chunks.append('<section><h2>'+esc(c['cycle'])+'</h2><p class="muted">模型 <code>'+esc(c['model_sha256'])+'</code></p>')
        params=c.get('parameters',{})
        chunks.append(table(['Workers','每次更新决策数','Mini-batch','Epochs','完成采样更新次数'],[[params.get(k) for k in ('workers','rollout_steps','minibatch_size','epochs','update_cycles')]]))
        if ev.get('counted'):
            coin_rows=[[a['id'],'通关' if a['outcome']=='rl_gameplay_clear' else '败局',a['match_wins'],a.get('failure_name') or '—'] for a in ev['attempts']]
            chunks.append(table(['自然投币','结果','击败对手数','失败对手'],coin_rows))
        else:chunks.append('<p class="muted">自然投币：'+esc(STATUS.get(ev.get('status'),ev.get('status','尚无结果')))+'；未计入成绩。'+esc(ev.get('error',''))+'</p>')
        opponent_rows=[];per=ev.get('per_opponent',{}) if ev.get('counted') else {}
        for op in sorted(set(tr)|set(per),key=int):
            t=tr.get(op);e=per.get(op);round_detail='—';match_detail='—'
            if e:
                match_detail=f"{e['match_W']} / {e['match_L']}"
                if e['round_details']:
                    round_detail=f"{e['round_W']} / {e['round_L']} / {e['round_D']}"
                    if e['round_details']!=e['match_details']:round_detail+=f"（覆盖 {e['round_details']}/{e['match_details']} 场）"
            opponent_rows.append([NAMES[int(op)],wld(t['rounds']) if t else '—',percent(t['rounds']['win_rate']) if t else '—',
                f"{t['matches']['W']} / {t['matches']['L']}" if t else '—',number(t['rounds']['mean_return']) if t else '—',round_detail,match_detail])
        chunks.append('<details><summary>逐对手明细 · 胜 / 负 / 平</summary>'+table(['对手','训练小局 W/L/D','训练小局胜率','训练整场 W/L','平均回报','自然投币小局 W/L/D','自然投币整场 W/L'],opponent_rows))
        chunks.append('<p class="muted">正式明细从小型状态文件汇总；缺失记为未知，不补零。单局平局数在原生 round 与最终比分完整时由两者差值计算。')
        if ev.get('counted'):chunks.append(' 状态文件覆盖 '+esc(ev.get('match_details_available'))+' / '+esc(ev.get('match_details_requested'))+' 场。')
        chunks.append('</p></details>')
        sampling=c.get('opponent_sampling')
        if sampling:
            mode='自动增加弱项练习' if sampling['mode']=='adaptive' else '均匀采样对照'
            chunks.append('<h3>训练采样分布 · '+esc(mode)+'</h3><p class="muted">近期胜率经过平滑，取每个对手最近128个已结束小局；下一场概率是本轮末设置，实际决策占比统计本轮所有采样，包括未结束小局。抽样概率与决策占比不必相等。'+(' 已继承检查点中的近期统计。' if sampling['restored'] else ' 本次从空白近期统计开始。')+'</p>')
            sample_rows=[[NAMES[int(k)],r['decisions'],percent(r['decision_share']),number(r['mean_round_decisions'],1),r['recent_rounds'],percent(r['smoothed_win_rate']),percent(r['next_probability'])] for k,r in sorted(sampling['per_opponent'].items(),key=lambda item:int(item[0]))]
            chunks.append(table(['对手','实际决策数','实际决策占比','平均小局决策数','近期小局数','平滑小局胜率','下一场抽样概率'],sample_rows))
        chunks.append('</section>')
    if summary['skipped']:
        chunks.append('<section><h2>未完成或未纳入的训练轮</h2>'+table(['训练轮','状态','已有决策','预算决策','说明'],[[r['cycle'],STATUS.get(r.get('status'),r.get('status','未知')),r.get('actual_steps'),r.get('requested_steps'),r.get('error') or r['reason']] for r in summary['skipped']])+'</section>')
    if summary['issues']:
        chunks.append('<section><h2>记录核对问题</h2>'+table(['训练轮 / 阶段','问题'],[[str(r.get('cycle',''))+' / '+str(r.get('stage','training')),r['error']] for r in summary['issues']])+'</section>')
    chunks.append('<footer>输入：<code>'+esc(summary['campaign'])+'</code><br>本报告不读取逐帧轨迹、存档或模型，不替代原生全轨迹审计。重新生成报告可刷新运行中的状态。</footer></main></html>')
    return ''.join(chunks)


def write_report(campaign, output_dir):
    """Write derived files outside the actual campaign; never alter evidence."""
    actual=resolve_campaign(campaign);destination=Path(output_dir).resolve()
    if destination.is_relative_to(actual):raise ValueError('Report output must be outside the read-only campaign directory')
    destination.mkdir(parents=True,exist_ok=True)
    files={'summary.json':None,'report.html':None}
    for name in files:
        target=destination/name
        if target.is_symlink():raise ValueError('Refusing symlink report output: '+str(target))
        if target.exists():
            owned=(read(target)[0].get('schema')==SCHEMA if name.endswith('.json') else target.read_text(encoding='utf-8').startswith(REPORT_MARKER))
            if not owned:raise ValueError('Refusing to replace an unrelated file: '+str(target))
    summary=summarize(campaign)
    files['summary.json']=json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    files['report.html']=render_html(summary)
    for name,text in files.items():
        temporary=None
        try:
            with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=destination,delete=False) as stream:
                temporary=Path(stream.name);stream.write(text)
            os.replace(temporary,destination/name)
        finally:
            if temporary and temporary.exists():temporary.unlink()
    return {'summary':summary,'summary_path':str(destination/'summary.json'),'report_path':str(destination/'report.html')}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign',type=Path)
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args(argv);root=args.campaign.resolve();actual=resolve_campaign(root)
    output=args.output_dir or (root if actual!=root else root.with_name(root.name+'-report'))
    result=write_report(root,output)
    print(json.dumps({k:v for k,v in result.items() if k!='summary'},ensure_ascii=False))
    return 2 if result['summary']['issues'] else 0


if __name__=='__main__':raise SystemExit(main())
