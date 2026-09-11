"""Read-only campaign results: preserve identities, losses, and incomplete evidence.

Supports legacy/native Normal campaigns and standalone training/native-play stages.
This derives statistics from existing evidence; it does not certify, replay, or control MAME.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

NAMES = {0: 'Ryu', 1: 'Honda', 2: 'Blanka', 3: 'Guile', 5: 'Chun-Li', 6: 'Zangief',
         7: 'Dhalsim', 8: 'Bison', 9: 'Sagat', 10: 'Balrog', 11: 'Vega'}
SCHEMAS = {'astra.rl-campaign.v1': 'legacy_rpc', 'astra.rl-native-campaign.v1': 'native_batch',
           'astra.rl-native-campaign.actions16.v1': 'native_batch_actions16',
           'astra.rl-reliability-campaign.v1': 'reliability_actions16'}
TRAINING_SCHEMAS = ('astra.rl-scaled.v1', 'astra.rl-batch-prototype.v1',
                    'astra.rl-batch-prototype.actions16.v1')
CONTINUOUS_SCHEMAS = ('astra.rl-continuous.v1', 'astra.rl-continuous.actions16.v1')
OUTCOMES = ('win', 'loss', 'draw')


def action_identity(result):
    """Keep declared identity and schema family separate; do not invent an interface."""
    schema = result.get('schema') or ''
    is16 = ('.actions16.' in schema or result.get('actions') == 16 or
            result.get('action_interface') == 'ken_actions16_lp_mp_uppercut_v1')
    return {'action_interface': result.get('action_interface'), 'actions': result.get('actions'),
            'action_schema_family': ('actions16' if is16 else 'legacy_actions15') if schema or is16 else None}


def read_object(path, issues, required=False):
    try:
        raw = path.read_bytes()
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError('Expected JSON object')
        return data, hashlib.sha256(raw).hexdigest()
    except (OSError, ValueError) as error:
        if required or path.exists():
            issues.append({'path': str(path), 'error': str(error)})
        return {}, None


def read_lines(path, issues):
    if not path.is_file():
        return []
    rows = []
    with path.open(encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError('Expected JSON object')
                rows.append(row)
            except ValueError as error:
                issues.append({'path': str(path), 'line': number, 'error': str(error),
                               'classification': 'unparsed; possibly in-progress tail'})
    return rows


def rounds_table(rows):
    table = {}
    for row in rows:
        opponent, outcome = row.get('opponent'), row.get('outcome')
        if opponent not in NAMES or outcome not in OUTCOMES:
            continue
        entry = table.setdefault(str(opponent), {'name': NAMES[opponent], 'win': 0, 'loss': 0, 'draw': 0})
        entry[outcome] += 1
    for row in table.values():
        row['rounds'] = sum(row[k] for k in OUTCOMES)
        row['win_rate'] = row['win']/row['rounds']
    return table


def unique_episodes(rows, path, issues, python=True):
    accepted, conflicted, duplicates = {}, set(), 0
    for row in rows:
        episode = row.get('episode')
        if (type(episode) is not int or row.get('opponent') not in NAMES or
                row.get('outcome') not in OUTCOMES or
                (python and (row.get('phase') != 'train' or row.get('baseline') is not False))):
            issues.append({'path': str(path), 'episode': episode, 'error': 'Unrecognized training episode; excluded'})
            continue
        if episode in accepted:
            if accepted[episode] == row:
                duplicates += 1
            else:
                conflicted.add(episode)
                issues.append({'path': str(path), 'episode': episode, 'error': 'Conflicting duplicate episode; all copies excluded'})
        else:
            accepted[episode] = row
    for episode in conflicted:
        accepted.pop(episode, None)
    return accepted, duplicates


def training_summary(folder, issues):
    result, digest = read_object(folder/'result.json', issues)
    full, pending, partials, workers = [], [], [], []
    duplicates = 0
    for worker in sorted(folder.glob('worker-*')):
        if not worker.is_dir():
            continue
        path = worker/'episodes.jsonl'
        primary, repeat = unique_episodes(read_lines(path, issues), path, issues)
        duplicates += repeat
        native_path = worker/'training/rl-batch-episodes.jsonl'
        native, native_repeat = unique_episodes(read_lines(native_path, issues), native_path, issues, python=False)
        duplicates += native_repeat
        unconfirmed = []
        for episode, row in native.items():
            copy = primary.get(episode)
            if copy is not None:
                keys = ('outcome', 'opponent', 'checkpoint', 'steps', 'frames')
                if any(copy.get(key) != row.get(key) for key in keys):
                    issues.append({'path': str(worker), 'episode': episode, 'error': 'Python/native episode mismatch; moved to pending'})
                    primary.pop(episode)
                    unconfirmed.append(row)
            else:
                unconfirmed.append(row)
        worker_partials = read_lines(worker/'partial-episodes.jsonl', issues)
        native_partials = read_lines(worker/'training/rl-batch-partials.jsonl', issues)
        full.extend(primary.values())
        pending.extend(unconfirmed)
        partials.extend(worker_partials)
        workers.append({'worker': worker.name, 'python_rounds': len(primary),
                        'python_steps_in_completed_rounds': sum(row.get('steps', 0) for row in primary.values()),
                        'native_unconfirmed_rounds': len(unconfirmed),
                        'partial_records': len(worker_partials), 'native_error_partial_records': len(native_partials)})
    indexed, unindexed = {}, []
    for row in full:
        number = row.get('round')
        if type(number) is int and number >= 1:
            indexed.setdefault(number, []).append(row)
        else:
            unindexed.append(row)
            if 'round' in row:
                issues.append({'path': str(folder), 'episode': row.get('episode'),
                               'error': 'Invalid training round index; retained only in unindexed and total counts'})
    return {**action_identity(result), 'status': result.get('status', 'not_started'), 'schema': result.get('schema'),
            'round_chain': result.get('round_chain'), 'training_protocol': result.get('training_protocol'),
            'opponent_sampling': result.get('opponent_sampling'),
            'difficulty': result.get('difficulty'), 'seed': result.get('seed'),
            'dataset_sha256': result.get('dataset_sha256'), 'result_sha256': digest,
            'model_sha256': result.get('model_sha256'), 'init_model_sha256': result.get('init_model_sha256'),
            'actual_steps_reported': result.get('actual_steps'), 'error': result.get('error'),
            'python_completed_rounds': len(full), 'rounds': rounds_table(full),
            'rounds_by_index': {str(number): rounds_table(rows) for number, rows in sorted(indexed.items())},
            'rounds_without_index': rounds_table(unindexed), 'rounds_without_index_count': len(unindexed),
            'native_unconfirmed_rounds': len(pending), 'native_unconfirmed_by_opponent': rounds_table(pending),
            'partial_records': len(partials), 'duplicate_lines_excluded': duplicates, 'workers': workers}


def child_path(base, relative):
    if not isinstance(relative, str):
        raise ValueError('Expected relative evidence path')
    path = (base/relative).resolve()
    if not path.is_relative_to(base.resolve()):
        raise ValueError('Evidence path escapes continuous run')
    return path


def continuous_summary(folder, native_required, issues, interface_required=False):
    result, digest = read_object(folder/'result.json', issues)
    output = {**action_identity(result), 'status': result.get('status', 'not_started'), 'schema': result.get('schema'),
              'difficulty': result.get('difficulty'), 'result_sha256': digest,
              'model_sha256': result.get('model_sha256'), 'error': result.get('error'),
              'native_timing': result.get('native_timing', False), 'attempts': [],
              'native_timing_audit': result.get('native_timing_audit'),
              'attempts_requested': result.get('attempts_requested'),
              'stop_on_first_clear': result.get('stop_on_first_clear'),
              'rounds': {}, 'unverified_observed_rounds': {}}
    protocol_path = folder/'sampling-protocol.json'
    protocol, protocol_hash = read_object(protocol_path, issues)
    identity_keys = ('selection', 'policy_seed', 'policy_prng', 'evaluation_kind')
    output.update({key: result.get(key, protocol.get(key)) for key in identity_keys})
    output['sampling_audit'] = result.get('sampling_audit')
    output['sampling_protocol'] = protocol or None
    output['sampling_protocol_sha256'] = protocol_hash
    sampling_required = protocol_path.exists() or str(output.get('selection', '')).startswith('categorical')
    output['sampling_audit_required'] = sampling_required
    interface_path = folder/'action-interface.json'
    interface, interface_hash = read_object(interface_path, issues)
    for key in ('action_interface', 'variant'):
        output[key] = result.get(key, interface.get(key))
    output['action_interface_audit'] = result.get('action_interface_audit')
    output['action_interface_protocol'] = interface or None
    output['action_interface_protocol_sha256'] = interface_hash
    interface_required = (interface_required or output['action_schema_family'] == 'actions16' or
                          interface_path.exists() or 'action_interface' in result)
    native_required = native_required or output['action_schema_family'] == 'actions16'
    output['action_interface_audit_required'] = interface_required
    audits = [(native_required, result.get('native_timing_audit')),
              (sampling_required, result.get('sampling_audit')),
              (interface_required, result.get('action_interface_audit'))]
    required_audits = [value for required, value in audits if required]
    status = result.get('status')
    audit_failed = status == 'invalid' or (status == 'complete' and any(
        (value or {}).get('ok') is False for value in required_audits))
    pending_run = not audit_failed and (status in ('initializing', 'running') or (
        status == 'complete' and any(value is None for value in required_audits)))
    # Outer experiment audits publish after underlying native/continuous results.
    # Missing required final audits in this short window remain pending.
    output['audit_state'] = 'pending' if pending_run else ('invalid' if audit_failed else 'available')
    valid_run = status == 'complete' and not pending_run and all(
        (value or {}).get('ok') is True for value in required_audits)
    if native_required:
        valid_run = valid_run and result.get('native_timing') is True
    verified_rounds, other_rounds = [], []
    for attempt in result.get('attempts', []):
        reported = attempt.get('outcome', 'invalid')
        valid = valid_run and attempt.get('audit', {}).get('ok') is True and reported in ('loss', 'rl_gameplay_clear')
        matches, rows = [], []
        for relative in attempt.get('matches', []):
            try:
                path = child_path(folder, relative)
                match, match_hash = read_object(path, issues, required=not pending_run)
            except ValueError as error:
                issues.append({'path': str(folder), 'error': str(error)})
                match, match_hash = {}, None
            summary = match.get('summary', {})
            opponent = summary.get('opponent')
            match_valid = (summary.get('status') == 'complete' and summary.get('valid_continuous') is True and
                           summary.get('model_sha256') == result.get('model_sha256') and opponent in NAMES and
                           summary.get('result') in ('ken_win', 'cpu_win'))
            if not match_valid:
                valid = False
            for row in match.get('rounds', []):
                rows.append({'opponent': opponent, 'outcome': row.get('outcome')})
            matches.append({'path': relative, 'sha256': match_hash, 'opponent': opponent,
                            'name': NAMES.get(opponent, 'unknown'), 'result': summary.get('result'),
                            'score': summary.get('score'), 'valid': match_valid})
        native_wins = sum(match['valid'] and match['result'] == 'ken_win' for match in matches)
        if reported == 'rl_gameplay_clear':
            valid = valid and len(matches) == 11 and native_wins == 11 and len({m['opponent'] for m in matches}) == 11
        elif reported == 'loss':
            valid = valid and bool(matches) and matches[-1]['result'] == 'cpu_win' and all(m['result'] == 'ken_win' for m in matches[:-1])
        outcome = 'pending' if pending_run else (reported if valid else 'invalid')
        (verified_rounds if valid else other_rounds).extend(rows)
        failed = next((m for m in reversed(matches) if m['result'] == 'cpu_win'), None)
        output['attempts'].append({'id': attempt.get('id'), 'outcome': outcome, 'reported_outcome': reported,
                                   'pending_reason': 'Run or a required native/sampling/action-interface audit has not finalized' if pending_run else None,
                                   'match_wins': native_wins, 'failed_opponent': failed['opponent'] if failed else None,
                                   'failed_opponent_name': failed['name'] if failed else None, 'matches': matches})
    output['rounds'] = rounds_table(verified_rounds)
    output['unverified_observed_rounds'] = rounds_table(other_rounds)
    output['attempt_counts'] = dict(Counter(a['outcome'] for a in output['attempts']))
    return output


def reliability_candidate(cycle, play):
    """Describe one full-20 candidate only; never pool different model outcomes."""
    attempts = play['attempts']
    counts = play['attempt_counts']
    matches = [m['path'] for a in attempts for m in a['matches']]
    errors = []
    if play['status'] == 'invalid' or cycle.get('status') == 'invalid' or counts.get('invalid'):
        state = 'invalid'
    elif play['status'] != 'complete' or play['audit_state'] == 'pending' or cycle.get('status') != 'complete':
        state = 'pending'
    else:
        state = 'complete'
        if (play['difficulty'] != 3 or play['stop_on_first_clear'] is not False or
                play['attempts_requested'] != 20 or len(attempts) != 20 or
                [a['id'] for a in attempts] != [f'l3-{i:03d}' for i in range(1,21)]):
            errors.append('Requires all twenty ordered Normal natural-coin attempts without early stop')
        if not cycle.get('model_sha256') or play['model_sha256'] != cycle['model_sha256']:
            errors.append('Candidate/evaluation model identities differ')
        if len(matches) != len(set(matches)):
            errors.append('Reused match evidence within this candidate')
        for field, key in (('native_timing_audit','matches'), ('action_interface_audit','checked_matches')):
            audit = play.get(field) or {}
            if audit.get('ok') is not True or audit.get(key) != len(matches):
                errors.append('Missing or incomplete '+field)
        code = cycle.get('exit_codes', {}).get('continuous')
        if code != (0 if counts.get('rl_gameplay_clear') else 1):
            errors.append('Evaluation exit code differs from outcomes or is missing')
        if sum(counts.get(k, 0) for k in ('rl_gameplay_clear','loss')) != 20:
            errors.append('Invalid or pending attempts cannot be replaced or counted as complete')
        if errors:
            state = 'invalid'
    clears = counts.get('rl_gameplay_clear', 0)
    return {'status': state, 'required_attempts': 20, 'required_clears': 10,
            'model_sha256': cycle.get('model_sha256'), 'observed_attempts': len(attempts),
            'attempt_counts': counts, 'clears': clears, 'losses': counts.get('loss', 0),
            'clear_rate': clears/20 if state == 'complete' else None,
            'goal_achieved': state == 'complete' and clears >= 10, 'errors': errors,
            'failed_opponents': dict(Counter(a['failed_opponent_name'] for a in attempts
                if a['outcome'] == 'loss' and a['failed_opponent_name']))}


def summarize_campaign(path):
    path = Path(path).resolve()
    issues = []
    result, digest = read_object(path/'result.json', issues, required=True)
    if result.get('schema') not in SCHEMAS:
        raise ValueError(f'Unsupported/missing campaign schema: {path}')
    report = {**action_identity(result), 'path': str(path), 'name': path.name, 'schema': result['schema'], 'kind': SCHEMAS[result['schema']],
              'round_chain': result.get('round_chain'), 'training_protocol': result.get('training_protocol'),
              'opponent_sampling': result.get('opponent_sampling'),
              'result_sha256': digest, 'status': result.get('status'), 'error': result.get('error'),
              'difficulty': result.get('difficulty'), 'seed': result.get('seed'),
              'dataset_sha256': result.get('dataset_sha256'), 'initial_model_sha256': result.get('initial_model_sha256'),
              'completed_training_steps_reported': result.get('completed_training_steps'),
              'goal_achieved_reported': result.get('goal_achieved'), 'cycles': [], 'issues': issues}
    ordinals = {row['ordinal']: row for row in result.get('cycles', [])}
    folders = {p.name: p for p in path.glob('cycle-*') if p.is_dir()}
    folders.update({f'cycle-{n:03d}': path/f'cycle-{n:03d}' for n in ordinals})
    for name, folder in sorted(folders.items()):
        try:
            ordinal = int(name.removeprefix('cycle-'))
        except ValueError:
            issues.append({'path': str(folder), 'error': 'Unexpected cycle directory name'})
            continue
        cycle = ordinals.get(ordinal, {})
        item = {'cycle': ordinal, 'status': cycle.get('status', 'unlisted'),
                                 'model_sha256': cycle.get('model_sha256'), 'stage': cycle.get('stage'),
                                 'training': training_summary(folder/'train', issues),
                                 'continuous': continuous_summary(folder/'continuous', report['kind'] != 'legacy_rpc', issues,
                                                                  interface_required=report['action_schema_family'] == 'actions16')}
        if report['kind'] == 'reliability_actions16':
            if ordinal == 1 and not (folder/'train').exists():
                item['training']['status'] = 'not_applicable'
                item['training']['note'] = 'Initial frozen candidate is evaluated before any new training'
            item['reliability'] = reliability_candidate(cycle, item['continuous'])
        report['cycles'].append(item)
    def combine(tables):
        combined = {}
        for table in tables:
            for opponent, values in table.items():
                target = combined.setdefault(opponent, {'name': values['name'], 'win': 0, 'loss': 0, 'draw': 0})
                for outcome in OUTCOMES:
                    target[outcome] += values[outcome]
        for values in combined.values():
            values['rounds'] = sum(values[k] for k in OUTCOMES)
            values['win_rate'] = values['win']/values['rounds']
        return combined
    attempts = [a for cycle in report['cycles'] for a in cycle['continuous']['attempts']]
    report['aggregate'] = {
        'training_rounds': combine(c['training']['rounds'] for c in report['cycles']),
        'completed_stage_training_rounds': combine(c['training']['rounds'] for c in report['cycles'] if c['training']['status'] == 'complete'),
        'unfinished_stage_training_rounds': combine(c['training']['rounds'] for c in report['cycles'] if c['training']['status'] != 'complete'),
        'continuous_rounds': combine(c['continuous']['rounds'] for c in report['cycles']),
        'native_unconfirmed_rounds': sum(c['training']['native_unconfirmed_rounds'] for c in report['cycles']),
        'attempt_counts': dict(Counter(a['outcome'] for a in attempts)),
        'failed_opponents': dict(Counter(a['failed_opponent_name'] for a in attempts
                                         if a['outcome'] == 'loss' and a['failed_opponent_name'])),
        'invalid_stages': [{'cycle': c['cycle'], 'stage': name}
                           for c in report['cycles'] for name in ('training', 'continuous')
                           if c[name]['status'] == 'invalid']}
    if report['kind'] == 'reliability_actions16':
        qualified = [c['cycle'] for c in report['cycles'] if c['reliability']['goal_achieved']]
        report['reliability'] = {'qualifying_candidates': qualified,
            'goal_achieved': result.get('status') == 'complete' and bool(qualified),
            'aggregation_rule': 'Candidate-specific full20 only; aggregate totals never establish success'}
    return report


def sampling_markdown(summary):
    metadata = summary.get('opponent_sampling')
    if metadata is None:
        return ['对手采样：未声明；不推断为均匀采样。', '']
    return ['对手采样元数据（运行记录原样保留，未重新验证）：', '',
            '```json', json.dumps(metadata, ensure_ascii=False, indent=2), '```', '']


def training_round_markdown(summary):
    lines = [f"训练协议：{summary.get('training_protocol') or '未声明'}；round_chain：{summary.get('round_chain')}。", '',
             '以下为完整训练小局的轮次分层，已包含在训练总数中，不另行累加；旧记录不推断轮次。', '',
             '| 轮次 | 对手 | 胜/负/平 |', '|---|---|---:|']
    groups = [(f'R{number}', table) for number, table in summary['rounds_by_index'].items()]
    if summary['rounds_without_index']:
        groups.append(('未记录有效轮次', summary['rounds_without_index']))
    for label, table in groups:
        for opponent in sorted(table, key=int):
            values = '/'.join(str(table[opponent][key]) for key in OUTCOMES)
            lines.append(f"| {label} | {NAMES[int(opponent)]} | {values} |")
    return sampling_markdown(summary)+lines+['']


def render_markdown(report):
    lines = ['# Normal 训练与连续验证汇总', '',
             '只读派生统计；不重新认证。训练小局、连续游玩小局与整路线尝试分开。运行中快照可能尚未完整。', '',
             'Python 逐局记录为训练主计数；Lua 原生副本不重复计数，尚未确认的记录单列待核对。', '']
    for run in report['campaigns']:
        lines += [f"## {run['name']}", '', f"类型：{run['kind']}；状态：{run['status']}；难度：{run['difficulty']}；种子：{run['seed']}。",
                  f"动作族：{run['action_schema_family']}；接口：{run.get('action_interface') or '未声明'}；动作数：{run.get('actions')}。",
                  f"运行声明训练协议：{run.get('training_protocol') or '未声明'}；round_chain：{run.get('round_chain')}。",
                  f"路径：`{run['path']}`", f"结果快照 SHA-256：`{run['result_sha256']}`", '']
        lines += sampling_markdown(run)
        totals = run['aggregate']
        if 'reliability' in run:
            lines += ['以下合计仅为跨候选工作量；不能合并成某一冻结模型的通关率或达标成绩。', '',
                      f"完整 20 币且至少 10 通关的候选：{run['reliability']['qualifying_candidates']}；派生达标：{run['reliability']['goal_achieved']}。", '']
        lines += [f"全路线结果：{totals['attempt_counts'].get('rl_gameplay_clear', 0)} 通关 / {totals['attempt_counts'].get('loss', 0)} 失败 / {totals['attempt_counts'].get('invalid', 0)} 无效 / {totals['attempt_counts'].get('pending', 0)} 待定；失败对手：{json.dumps(totals['failed_opponents'], ensure_ascii=False)}。", '']
        if run.get('error'):
            lines += [f"运行错误：{run['error']}", '']
        for cycle in run['cycles']:
            train, play = cycle['training'], cycle['continuous']
            if 'reliability' in cycle:
                measured = cycle['reliability']
                rate = f"{measured['clear_rate']:.1%}" if measured['clear_rate'] is not None else '未完成/无效，不计算'
                lines += [f"候选 {cycle['cycle']} 完整 20 币检查：{measured['status']}；已观察 {measured['observed_attempts']} 币；"
                          f"通关 {measured['clears']} / 失败 {measured['losses']}；完整批次通关率：{rate}；达标：{measured['goal_achieved']}。",
                          f"本候选失败对手：{json.dumps(measured['failed_opponents'], ensure_ascii=False)}。", '']
                if measured['errors']:
                    lines += ['完整批次问题：'+'；'.join(measured['errors']), '']
                if train['status'] == 'not_applicable':
                    lines += ['首候选直接评估初始冻结模型，没有新训练阶段。', '']
            lines += [f"### Cycle {cycle['cycle']}（{cycle['status']}）", '',
                      f"模型：`{cycle.get('model_sha256') or train.get('model_sha256') or '尚无最终模型'}`", '',
                      f"游玩选择方式：{play.get('selection') or '未声明'}；策略种子：{play.get('policy_seed')}；PRNG：{play.get('policy_prng')}；采样审计：{json.dumps(play.get('sampling_audit'), ensure_ascii=False)}。", '',
                      f"动作接口：{play.get('action_interface') or '未声明'}；变体：{play.get('variant')}；接口审计：{json.dumps(play.get('action_interface_audit'), ensure_ascii=False)}。", '',
                      f"训练动作族：{train['action_schema_family']}；接口：{train.get('action_interface') or '未声明'}；动作数：{train.get('actions')}。",
                      f"训练：{train['status']}；主记录完整小局 {train['python_completed_rounds']}；Lua 待核对 {train['native_unconfirmed_rounds']}；未完成片段记录 {train['partial_records']}。", '',
                      '| 对手 | 训练胜/负/平 | 连续验证小局胜/负/平 | 未通过审计的游玩观察胜/负/平 |',
                      '|---|---:|---:|---:|']
            tables = [train['rounds'], play['rounds'], play['unverified_observed_rounds']]
            for opponent in sorted(set().union(*(set(table) for table in tables)), key=int):
                counts = ['/'.join(str(table.get(opponent, {}).get(k, 0)) for k in OUTCOMES) for table in tables]
                if 'reliability' in cycle and opponent in play['rounds']:
                    counts[1] += f"（胜率 {play['rounds'][opponent]['win_rate']:.1%}）"
                lines.append(f"| {NAMES[int(opponent)]} | {' | '.join(counts)} |")
            lines += ['']+training_round_markdown(train)
            lines += ['', '整路线尝试：', '', '| 尝试 | 结果 | 已赢对手数 | 失利对手 |', '|---|---|---:|---|']
            if not play['attempts']:
                lines.append(f"| — | {play['status']}；尚无尝试结果 | — | — |")
            for attempt in play['attempts']:
                outcome = {'rl_gameplay_clear': '通关', 'loss': '失败', 'invalid': '无效', 'pending': '待定（尚未终审）'}[attempt['outcome']]
                lines.append(f"| {attempt['id']} | {outcome} | {attempt['match_wins']} | {attempt['failed_opponent_name'] or '—'} |")
            if play.get('error'):
                lines += ['', f"连续验证错误：{play['error']}"]
            if train['native_unconfirmed_rounds']:
                lines += ['', '待核对 Lua 原生小局（未加入训练成绩）：`'+json.dumps(train['native_unconfirmed_by_opponent'], ensure_ascii=False)+'`']
            lines.append('')
        lines += [f"数据问题/未完成尾行：{len(run['issues'])}（详见 JSON）。", '']
    for item in report.get('standalone', []):
        summary = item['summary']
        lines += [f"## 独立{'训练' if item['kind'] == 'training' else '连续验证'}：{item['name']}", '',
                  f"状态：{summary['status']}；分类：{item['classification']}；难度：{summary.get('difficulty')}。",
                  f"动作族：{summary['action_schema_family']}；接口：{summary.get('action_interface') or '未声明'}；动作数：{summary.get('actions')}。",
                  f"路径：`{item['path']}`", f"结果快照 SHA-256：`{summary['result_sha256']}`",
                  f"模型 SHA-256：`{summary.get('model_sha256') or '尚无最终模型'}`", '']
        if item['kind'] == 'training':
            lines += [f"完整训练小局 {summary['python_completed_rounds']}；Lua 待核对 {summary['native_unconfirmed_rounds']}；未完成片段记录 {summary['partial_records']}。", '',
                      '| 对手 | 训练胜/负/平 | Lua 待核对胜/负/平 |', '|---|---:|---:|']
            tables = [summary['rounds'], summary['native_unconfirmed_by_opponent']]
        else:
            lines += [f"选择方式：{summary.get('selection') or '未声明'}；策略种子：{summary.get('policy_seed')}；PRNG：{summary.get('policy_prng')}。",
                      f"评估类型：{summary.get('evaluation_kind') or '未声明'}；采样审计：{json.dumps(summary.get('sampling_audit'), ensure_ascii=False)}。",
                      f"动作接口：{summary.get('action_interface') or '未声明'}；变体：{summary.get('variant')}；接口审计：{json.dumps(summary.get('action_interface_audit'), ensure_ascii=False)}。", '']
            lines += ['| 对手 | 已审计游玩小局胜/负/平 | 未终审/无效观察胜/负/平 |', '|---|---:|---:|']
            tables = [summary['rounds'], summary['unverified_observed_rounds']]
        for opponent in sorted(set().union(*(set(table) for table in tables)), key=int):
            counts = ['/'.join(str(table.get(opponent, {}).get(k, 0)) for k in OUTCOMES) for table in tables]
            lines.append(f"| {NAMES[int(opponent)]} | {' | '.join(counts)} |")
        if item['kind'] == 'training':
            lines += ['']+training_round_markdown(summary)
        if item['kind'] == 'continuous':
            lines += ['', '| 整路线尝试 | 结果 | 已赢对手数 | 失利对手 |', '|---|---|---:|---|']
            for attempt in summary['attempts']:
                outcome = {'rl_gameplay_clear': '通关', 'loss': '失败', 'invalid': '无效', 'pending': '待定（尚未终审）'}[attempt['outcome']]
                lines.append(f"| {attempt['id']} | {outcome} | {attempt['match_wins']} | {attempt['failed_opponent_name'] or '—'} |")
        if summary.get('error'):
            lines += ['', f"执行错误：{summary['error']}"]
        lines += ['', f"数据问题/未完成尾行：{len(item['issues'])}（详见 JSON）。", '']
    if report.get('duplicate_input_paths_ignored'):
        lines += ['去重输入（未再次统计）：`'+json.dumps(report['duplicate_input_paths_ignored'], ensure_ascii=False)+'`', '']
    if report['duplicate_campaign_paths_ignored']:
        lines += ['重复传入的同一路径已忽略：`'+', '.join(report['duplicate_campaign_paths_ignored'])+'`', '']
    return '\n'.join(lines)


def summarize_standalone(path, kind):
    issues = []
    raw, _ = read_object(path/'result.json', issues, required=True)
    if kind == 'training':
        if raw.get('schema') not in TRAINING_SCHEMAS:
            raise ValueError(f'Unsupported standalone training schema: {path}')
        if any(raw.get(key) is True for key in ('benchmark', 'parity', 'native_parity')):
            raise ValueError('Benchmark/parity runs are not training stages')
        summary = training_summary(path, issues)
    else:
        if raw.get('schema') not in CONTINUOUS_SCHEMAS:
            raise ValueError(f'Unsupported standalone continuous schema: {path}')
        summary = continuous_summary(path, True, issues)
    state = summary['status']
    if state not in ('complete', 'invalid', 'not_started') or summary.get('audit_state') == 'pending':
        state = 'pending'
    if kind == 'continuous' and summary.get('attempt_counts', {}).get('invalid', 0):
        state = 'invalid'
    return {'path': str(path), 'name': path.name, 'kind': kind,
            'classification': state, 'summary': summary, 'issues': issues}


def summarize(paths=(), training_paths=(), continuous_paths=()):
    if not paths and not training_paths and not continuous_paths:
        raise ValueError('At least one campaign, training, or continuous path is required')
    seen, reports, repeated, ignored = set(), [], [], []
    for path in paths:
        resolved = Path(path).resolve()
        if resolved in seen:
            repeated.append(str(resolved))
            ignored.append({'path': str(resolved), 'kind': 'campaign', 'reason': 'repeated_input'})
            continue
        seen.add(resolved)
        reports.append(summarize_campaign(resolved))
    covered = set()
    for run in reports:
        for cycle in run['cycles']:
            folder = Path(run['path'])/f"cycle-{cycle['cycle']:03d}"
            covered.update((folder/name).resolve() for name in ('train', 'continuous'))
    standalone = []
    for kind, items in (('training', training_paths), ('continuous', continuous_paths)):
        for path in items:
            resolved = Path(path).resolve()
            if resolved in covered or resolved in seen:
                ignored.append({'path': str(resolved), 'kind': kind,
                                'reason': 'already_reported_campaign_stage' if resolved in covered else 'repeated_input'})
                continue
            standalone.append(summarize_standalone(resolved, kind))
            seen.add(resolved)
    return {'schema': 'astra.rl-normal-report.v1', 'generated_utc': datetime.now(timezone.utc).isoformat(),
            'read_only_derived_statistics': True, 'campaigns': reports, 'standalone': standalone,
            'duplicate_campaign_paths_ignored': repeated, 'duplicate_input_paths_ignored': ignored}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, action='append', default=[], help='Repeat for each campaign directory')
    parser.add_argument('--training', type=Path, action='append', default=[], help='Repeat for standalone training directories')
    parser.add_argument('--continuous', type=Path, action='append', default=[], help='Repeat for standalone continuous directories; native audit required')
    parser.add_argument('--output', type=Path, required=True, help='Fresh report directory outside source campaigns')
    args = parser.parse_args()
    sources = args.campaign+args.training+args.continuous
    if not sources:
        parser.error('At least one --campaign, --training, or --continuous is required')
    output = args.output.resolve()
    if any(output.is_relative_to(path.resolve()) for path in sources):
        parser.error('Write derived report outside all input directories')
    report = summarize(args.campaign, args.training, args.continuous)
    output.mkdir(parents=True, exist_ok=False)
    (output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (output/'report.md').write_text(render_markdown(report), encoding='utf-8')
    print(json.dumps({'campaigns': len(report['campaigns']), 'standalone': len(report['standalone']), 'output': str(output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
