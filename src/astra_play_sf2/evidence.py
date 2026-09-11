"""Portable, local integrity checks and explicit human/agent review attestations.

Checks detect missing, inconsistent or changed local evidence. They are not an
adversarial signature scheme, and a named review is not an authenticated identity.
"""
from __future__ import annotations

import collections
import copy
import datetime
import hashlib
import html
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import quote

OPS = {0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11}
NAMES = {0: 'Ryu', 1: 'E. Honda', 2: 'Blanka', 3: 'Guile', 5: 'Chun-Li',
         6: 'Zangief', 7: 'Dhalsim', 8: 'M. Bison', 9: 'Sagat',
         10: 'Balrog', 11: 'Vega'}
KEYS = {'U', 'D', 'L', 'R', 'LP', 'MP', 'HP', 'LK', 'MK', 'HK'}
OUTCOMES = ('win', 'loss', 'draw')
DIFFICULTY_COLUMNS = {'difficulty_bits', 'difficulty_mirror', 'effective_difficulty'}
TRACE_COLUMNS = DIFFICULTY_COLUMNS | {'frame', 'round', 'phase', 'timer', 'timer_raw', 'emulated_seconds',
                 'preceding_input', 'preceding_decision', 'issued_input'} | {
                     f'p{i}_{field}' for i in (1, 2) for field in
                     ('x', 'y', 'hp', 'displayed_hp', 'timeout_hp', 'action', 'anim', 'wins')}
FORBIDDEN = re.compile(r'\b(?:restore|checkpoint|formal_continue|train_match|soft_reset|hard_reset)\s*\(|(?:machine\s*[:.]\s*(?:load|save)|emu\s*\.\s*pause)\s*\(')


def _read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require(value, message):
    if not value:
        raise ValueError(message)


def _path(run, relative):
    _require(isinstance(relative, str) and relative and '\\' not in relative,
             'Evidence path must be a relative POSIX path')
    p = Path(relative)
    _require(not p.is_absolute() and '..' not in p.parts and ':' not in relative,
             'Evidence path escapes run')
    resolved = (run / p).resolve()
    _require(resolved.is_relative_to(run.resolve()), 'Evidence symlink escapes run')
    return resolved


def _pose(p, timer):
    ko = (p['a'] == 0 or (p['char'] == 4 and p['anim'] == 388594)
          or (p['char'], p['a'], p['anim']) in ((6, 2, 328980), (8, 2, 281952)))
    return p['y'] == 40 and ((p['hp'] < 0 and p['displayed_hp'] < 0 and ko)
                            or (timer == 0 and p['a'] == 18))


def _winner(w, loser, state, stop):
    time_hp = lambda p: type(p['timeout_hp']) is int and 0 <= p['timeout_hp'] <= 144 and p['displayed_hp'] == p['timeout_hp']
    late = (stop['timer'] == state['timer'] == 0
            and (loser['char'], loser['a'], loser['anim']) in
            ((1, 0, 462306), (4, 0, 388594), (4, 12, 388594),
             (8, 0, 281952), (0, 0, 493744), (3, 0, 407552))
            and loser['y'] == 40 and loser['hp'] < 0
            and time_hp(w) and time_hp(loser) and w['timeout_hp'] > loser['timeout_hp'])
    return w['y'] == 40 and w['a'] == 16 and (_pose(loser, state['timer']) or late)


def _draw(s, stop, trace, previous, stop_frame, end_frame):
    a, b = s['p1'], s['p2']
    double = (a['hp'] < 0 and b['hp'] < 0 and a['displayed_hp'] < 0 and b['displayed_hp'] < 0
              and a['y'] == b['y'] == 40 and b['a'] == 0
              and (a['a'] == 0 or (a['char'], a['a'], a['anim']) == (4, 12, 388594)))
    hp = a['timeout_hp']
    timed = (stop['timer'] == s['timer'] == 0 and a['a'] == b['a'] == 18
             and a['y'] == b['y'] == 40 and type(hp) is int and 0 <= hp <= 144
             and b['timeout_hp'] == a['displayed_hp'] == b['displayed_hp'] == hp)
    if double or timed:
        return True
    # Frozen Core's narrow Ken/Chun-Li equal-time latch, including pre-KO chronology.
    late = (stop['timer'] == s['timer'] == 0 and a['char'] == 4 and b['char'] == 5
            and a['y'] == b['y'] == 40 and a['a'] == 0 and a['anim'] == 388594 and a['hp'] < 0
            and b['a'] == 18 and type(hp) is int and 0 < hp <= 144
            and b['hp'] == b['timeout_hp'] == a['displayed_hp'] == b['displayed_hp'] == hp
            and stop['p1']['timeout_hp'] == stop['p2']['timeout_hp'] == 0)
    if not late or stop['p1']['hp'] < 0 or stop['p2']['hp'] < 0:
        return False
    for row in trace[stop_frame:end_frame]:
        if row['p1_hp'] < 0 or row['p2_hp'] < 0:
            return False
        if (row['timer'] == 0 and row['p1_timeout_hp'] == row['p2_timeout_hp'] == hp
                and row['p1_hp'] == row['p2_hp'] == hp
                and [row['p1_wins'], row['p2_wins']] == previous):
            return True
    return False


def _difficulty_state(state, level):
    _require(all(type(state.get(k)) is int and state[k] == value for k, value in
                 (('difficulty_bits', 7-level), ('difficulty_mirror', level),
                  ('effective_difficulty', level))), 'Difficulty unverified: internal/DIP reading differs')


def _snapshot_difficulties(value, level):
    if isinstance(value, dict):
        if 'p1' in value and 'p2' in value:
            _difficulty_state(value, level)
        for child in value.values():
            _snapshot_difficulties(child, level)
    elif isinstance(value, list):
        for child in value:
            if isinstance(child, (dict, list)):
                _snapshot_difficulties(child, level)


def _match(run, relative, level, selection, speed, needed, verify_difficulty=True):
    path = _path(run, relative)
    status_rel = str(Path(relative).with_name(Path(relative).stem + '-status.json')).replace('\\', '/')
    policy_rel = str(Path(relative).with_name(Path(relative).stem + '-policy.lua')).replace('\\', '/')
    needed.update((relative, status_rel, policy_rel))
    raw = _read(path)
    s = raw['summary']
    _require(_read(_path(run, status_rel)) == s, 'Terminal status differs from raw summary')
    _require(s['valid_continuous'] is True and s['status'] == s['phase'] == 'complete'
             and s['active'] is False and s['training_validation'] is False, 'Not a valid continuous formal match')
    _require(s['loads'] == s['saves'] == s['pauses_during_match'] == 0, 'Match lifecycle violation')
    _require(s.get('resets', 0) == 0 and s.get('continues', 0) == 0, 'Match reset/continue')
    _require(s['difficulty_bits'] == 7-level and s['difficulty_checks'] == s['frame'], 'Difficulty/frame checks differ')
    if verify_difficulty:
        _require(s.get('effective_difficulty') == level
                 and s.get('effective_difficulty_checks') == s['frame'],
                 'Difficulty unverified: internal frame checks differ')
        _snapshot_difficulties({key: value for key, value in raw.items() if key != 'trace'}, level)
    _require(s['speed'] == speed, 'Speed differs from manifest')
    op = s['opponent']
    _require(op in OPS and s['mode'] == selection[str(op)], 'Wrong selected mode/opponent')
    _require(s['timeout_guard'] == (op in (1, 5, 8, 11)), 'Wrong frozen timeout guard')
    _require(_sha(_path(run, policy_rel)) == _sha(run/'training/runtime/fighter.lua'), 'Per-match policy copy differs')
    trace = raw['trace']
    _require(not raw['telemetry_error'] and not s['telemetry_error'] and s['diagnostic_level'] == 'full', 'Telemetry incomplete')
    _require(trace['all_frames'] is True and trace['all_decisions'] is True, 'Full trace flags missing')
    _require(len(trace['columns']) == len(set(trace['columns'])), 'Duplicate trace columns')
    required_columns = TRACE_COLUMNS if verify_difficulty else TRACE_COLUMNS-DIFFICULTY_COLUMNS
    _require(required_columns <= set(trace['columns']), 'Required trace columns missing')
    _require(len(trace['rows']) == s['frame'] == s['trace_frames'] == trace['observed_frames'], 'Trace frame count differs')
    rows = []
    for frame, values in enumerate(trace['rows'], 1):
        _require(len(values) == len(trace['columns']), 'Truncated trace row')
        r = dict(zip(trace['columns'], values))
        _require(r['frame'] == frame, 'Noncontiguous trace')
        if verify_difficulty:
            _difficulty_state(r, level)
        for key in ('preceding_input', 'issued_input'):
            text = r[key]
            _require(text is False or (isinstance(text, str) and set(text.split()) <= KEYS), 'Forbidden trace input')
        rows.append(r)
    for i, d in enumerate(trace['decisions'], 1):
        _require(d['id'] == i and 1 <= d['frame'] <= s['frame'] and d['mode'] == s['mode'], 'Invalid decision identity')
        for frames, keys in d['sequence']:
            _require(type(frames) is int and frames > 0 and isinstance(keys, str) and set(keys.split()) <= KEYS, 'Forbidden policy input')
    events = raw['events']
    starts = [e for e in events if e['kind'] == 'match_start']
    _require(len(starts) == 1 and starts[0]['lead_frames'] == (2 if op == 8 else 0), 'Wrong opening lead')
    opening = starts[0]['state']
    _require(opening['p1']['char'] == 4 and opening['p2']['char'] == op and opening['timer'] == 99, 'Not Ken R1')
    _require(all(p['hp'] == 144 and p['displayed_hp'] == 144 and p['wins'] == 0 and p['y'] == 40 and p['x'] > 0 and p['anim'] > 0 for p in (opening['p1'], opening['p2'])), 'Uninitialized/damaged R1')
    finishes = [e for e in events if e['kind'] == 'match_finished']
    _require(len(finishes) == 1 and finishes[0]['valid'] is True and finishes[0]['result'] == s['result'], 'Missing valid final event')
    previous = [0, 0]
    counts = collections.Counter({o: 0 for o in OUTCOMES})
    for n, r in enumerate(raw['rounds'], 1):
        _require(r['round'] == n and r['outcome'] in OUTCOMES, 'Invalid round sequence')
        stop_events = [e for e in events if e['kind'] == 'round_stop' and e['round'] == n]
        result_events = [e for e in events if e['kind'] == 'round_result' and e['round'] == n]
        _require(len(stop_events) == len(result_events) == 1, 'Missing round stop/result')
        stop = stop_events[0]
        _require(r['frame'] - stop['frame'] >= 360 and stop['state'] == r['stop']
                 and result_events[0]['result'] == r and result_events[0]['frame'] == r['frame'], 'Immature/inconsistent round result')
        settled = r['settled']
        a, b = settled['p1'], settled['p2']
        _require(a['char'] == 4 and b['char'] == op, 'Round actors changed')
        current = [a['wins'], b['wins']]
        delta = [current[i]-previous[i] for i in (0, 1)]
        outcome = r['outcome']
        _require(delta == ({'win': [1, 0], 'loss': [0, 1], 'draw': [0, 0]}[outcome]) and r['score'] == current, 'Wrong native pip transition')
        mature = (_winner(a, b, settled, r['stop']) if outcome == 'win' else
                  _winner(b, a, settled, r['stop']) if outcome == 'loss' else
                  _draw(settled, r['stop'], rows, previous, stop['frame'], r['frame']))
        _require(mature, 'Winner/draw pose not supported by frozen settlement rules')
        counts[outcome] += 1
        previous = current
    _require(previous == s['score'] and len(raw['rounds']) > 0, 'Final score differs')
    _require(s['latest'] == raw['rounds'][-1]['settled'] and s['frame'] == raw['rounds'][-1]['frame'], 'Final summary state/frame differs from mature result')
    _require((s['result'] == 'ken_win' and previous[0] == 2 and previous[1] < 2)
             or (s['result'] == 'cpu_win' and previous[1] == 2 and previous[0] < 2), 'No whole-match winner')
    return {'opponent': op, 'result': s['result'], 'rounds': dict(counts), 'log': relative}


def _inspect(run):
    run = Path(run).resolve()
    errors, audited = [], []
    needed = {'run.json'}
    manifest = _read(run/'run.json')
    if manifest.get('schema') == 'astra.batch.v1':
        return _inspect_batch(run, manifest)
    schema = manifest.get('schema')
    _require(schema in ('astra.run.v1', 'astra.run.v2'), 'Unsupported run schema')
    verify_difficulty = schema == 'astra.run.v2'
    if not verify_difficulty:
        errors.append('Difficulty unverified: legacy astra.run.v1 lacks game-internal difficulty evidence')
    if manifest['status'] != 'complete':
        errors.append('Run is not complete')
    runtime = manifest['runtime_sha256']
    _require(all(n in runtime for n in ('fighter.lua', 'play_core.lua', 'play.lua', 'selection.json')), 'Runtime identity incomplete')
    _require({p.name for p in (run/'training/runtime').iterdir() if p.is_file()} == set(runtime), 'Unmanifested/missing runtime file')
    for name, digest in runtime.items():
        relative = 'training/runtime/' + name
        needed.add(relative)
        _require(_sha(_path(run, relative)) == digest, 'Runtime SHA differs: '+name)
    _require(manifest['policy_sha256'] == runtime['fighter.lua'], 'Policy identity differs')
    selection = _read(run/'training/runtime/selection.json')
    _require(set(map(int, selection)) == OPS, 'Selection must cover eleven opponents')
    levels = manifest['difficulty']
    _require(isinstance(levels, list) and levels and len(set(levels)) == len(levels) and all(type(n) is int and 3 <= n <= 7 for n in levels), 'Invalid difficulty list')
    if verify_difficulty:
        _require(len(levels) == 1, 'A v2 session must contain exactly one fixed difficulty')
        try:
            _boot_difficulty(run, manifest, levels[0], needed)
        except (KeyError, ValueError, TypeError, OSError, ET.ParseError) as exc:
            errors.append('Difficulty unverified: '+str(exc))
    attempts = manifest['attempts']
    _require(len({a['id'] for a in attempts}) == len(attempts), 'Duplicate attempt IDs')
    requested = manifest['attempts_requested']
    _require(type(requested) is int and requested > 0, 'Invalid attempts_requested')
    target = manifest.get('consecutive')
    _require(target is None or (type(target) is int and target > 0), 'Invalid consecutive target')
    for level in levels:
        level_attempts = [a for a in attempts if a['difficulty'] == level]
        actual_ids = [a['id'] for a in level_attempts]
        _require(actual_ids == [f'l{level}-{i:03d}' for i in range(1, len(actual_ids)+1)]
                 and len(actual_ids) <= requested, 'Skipped/unplanned attempt ID')
        streak = 0
        for i, a in enumerate(level_attempts):
            streak = streak+1 if a['outcome'] == 'gameplay_clear' else 0
            _require(not target or streak < target or i == len(level_attempts)-1,
                     'Continued after requested consecutive target')
        if len(actual_ids) < requested and not (target and streak >= target):
            errors.append(f'Incomplete requested attempt count for level {level}')
    for attempt in attempts:
        row = {'id': attempt['id'], 'difficulty': attempt['difficulty'], 'outcome': attempt['outcome'], 'matches': [], 'valid': False}
        audited.append(row)
        try:
            level = attempt['difficulty']
            _require(level in levels and re.fullmatch(rf'l{level}-[0-9]{{3,}}', attempt['id']), 'Attempt identity/difficulty')
            lifecycle = attempt['lifecycle']; needed.add(lifecycle)
            life = _read(_path(run, lifecycle))
            _require(all(life[k] == 0 for k in ('resets', 'loads', 'saves')) and life['active'] is False and not life['violation'], 'Attempt lifecycle violation')
            _require(attempt['outcome'] in ('gameplay_clear', 'loss'), 'Invalid attempt cannot certify')
            _require(len(set(attempt['matches'])) == len(attempt['matches']), 'Duplicate match paths')
            for relative in attempt['matches']:
                row['matches'].append(_match(run, relative, level, selection, manifest['speed'], needed, verify_difficulty))
            matches = row['matches']; route = [m['opponent'] for m in matches]
            _require(route and len(route) == len(set(route)) and len(route) <= 11, 'Invalid route')
            _require(set(route[:7]) <= (OPS-{8, 9, 10, 11}) and route[7:] == [10, 11, 9, 8][:max(0, len(route)-7)], 'Route is not native fighters then bosses')
            _require(all(m['result'] == 'ken_win' for m in matches[:-1]), 'Continued after whole loss')
            clear = attempt['outcome'] == 'gameplay_clear'
            _require((clear and len(route) == 11 and set(route) == OPS and matches[-1]['result'] == 'ken_win')
                     or (not clear and matches[-1]['result'] == 'cpu_win'), 'Attempt outcome differs from matches')
            images = attempt['images']; image_paths = [images['selection']]
            if clear:
                _require(len(images['ending']) == 3, 'Three ending images required')
                image_paths += [images['bison'], *images['ending']]
                _require(len(set(image_paths)) == 5, 'Five distinct review images required')
            for rel in image_paths:
                needed.add(rel); _require(_path(run, rel).is_file(), 'Missing review image')
            row['images'] = images; row['valid'] = True
        except (KeyError, ValueError, TypeError, OSError) as exc:
            errors.append(attempt['id']+': '+str(exc))
    global_lifecycle = run/'training/lifecycle.json'
    if global_lifecycle.exists():
        needed.add('training/lifecycle.json')
        life = _read(global_lifecycle)
        if not (all(life[k] == 0 for k in ('resets', 'loads', 'saves'))
                and life['active'] is False and not life['violation']):
            errors.append('Global lifecycle violation or active session')
    commands = run/'training/commands.jsonl'
    if commands.exists():
        needed.add('training/commands.jsonl')
        groups = {}
        for line in commands.read_text(encoding='utf-8').splitlines():
            event = json.loads(line)
            _require(event['event'] in ('prepared', 'accepted', 'completed'), 'Unknown command audit event')
            _require(not FORBIDDEN.search(event['command']), 'Forbidden command recorded')
            sequence = groups.setdefault(str(event['id']), [])
            _require(not sequence or sequence[0]['command'] == event['command'], 'Command content changed under same ID')
            sequence.append(event)
        if not all([e['event'] for e in seq] == ['prepared', 'accepted', 'completed'] for seq in groups.values()):
            errors.append('Unfinished/duplicate command event sequence')
    _check_seal(run, needed, errors)
    return manifest, audited, errors


def _check_seal(run, needed, errors):
    try:
        seal = _read(run/'evidence-sha256.json')['files']
        _require(needed <= set(seal), 'Evidence seal omits required files: '+', '.join(sorted(needed-set(seal))))
        for rel, digest in seal.items():
            _require(_sha(_path(run, rel)) == digest, 'Sealed evidence changed: '+rel)
    except (KeyError, ValueError, TypeError, OSError) as exc:
        errors.append(str(exc))


def _boot_difficulty(run, manifest, level, needed):
    settings = 'training/runtime/settings.lua'
    needed.add(settings)
    _require('settings.lua' in manifest['runtime_sha256'], 'Runtime identity omits settings.lua')
    expected = f"astra_difficulty_bits={7-level}\nastra_difficulty_label='{level}'\n"
    _require(_path(run, settings).read_text(encoding='utf-8') == expected,
             'Runtime settings.lua difficulty differs')
    relative = manifest['boot_observation']
    _require(relative == 'training/boot-observation.json', 'Unexpected boot observation path')
    needed.update(('boot-config.xml', relative))
    config = run/'boot-config.xml'
    _require(_sha(config) == manifest['boot_config_sha256'], 'Boot configuration SHA differs')
    root = ET.parse(config).getroot()
    _require(root.tag == 'mameconfig' and root.get('version') == '10', 'Invalid boot configuration root')
    systems = root.findall('system')
    _require(len(systems) == 1 and systems[0].get('name') == 'sf2', 'Wrong boot configuration system')
    ports = [p for p in systems[0].findall('input/port') if p.get('tag') == ':DSWB'
             and p.get('type') == 'DIPSWITCH' and int(p.get('mask', '0'), 0) & 7]
    _require(len(ports) == 1 and int(ports[0].get('mask', '0'), 0) == 7
             and int(ports[0].get('defvalue', '-1'), 0) == 4
             and int(ports[0].get('value', '-1'), 0) == 7-level,
             'Boot configuration Difficulty port differs')
    _difficulty_state(_read(_path(run, relative)), level)


def _rebase_attempt(attempt, prefix):
    result = copy.deepcopy(attempt)
    result['matches'] = [prefix+'/'+p for p in result.get('matches', [])]
    if result.get('lifecycle'):
        result['lifecycle'] = prefix+'/'+result['lifecycle']
    images = result.get('images', {})
    for name in ('selection', 'bison'):
        if images.get(name):
            images[name] = prefix+'/'+images[name]
    if 'ending' in images:
        images['ending'] = [prefix+'/'+p for p in images['ending']]
    return result


def _inspect_batch(run, manifest):
    errors, audited, combined = [], [], []
    needed = {'run.json'}
    if manifest['status'] != 'complete':
        errors.append('Batch is not complete')
    levels = manifest['difficulty']
    _require(isinstance(levels, list) and len(levels) > 1 and len(set(levels)) == len(levels)
             and all(type(n) is int and 3 <= n <= 7 for n in levels), 'Invalid batch difficulty list')
    sessions = manifest['sessions']
    _require(isinstance(sessions, list), 'Invalid session list')
    seen = []
    shared_runtime = None
    for session in sessions:
        level, relative = session['difficulty'], session['path']
        _require(level in levels and level not in seen and relative == f'sessions/l{level}',
                 'Duplicate/unplanned batch session or path')
        seen.append(level)
        needed.update((relative+'/run.json', relative+'/evidence-sha256.json'))
        try:
            child_dir = _path(run, relative)
            child = _read(child_dir/'run.json')
            _require(child.get('schema') == 'astra.run.v2', 'Batch child is not a v2 single session')
            _require(child['difficulty'] == [level], 'Batch session difficulty differs')
            for key in ('version', 'mame_version', 'rom', 'policy_sha256', 'attempts_requested', 'consecutive', 'speed'):
                _require(key in child and key in manifest and child[key] == manifest[key],
                         'Batch session '+key+' differs or is missing')
            runtime = {name: digest for name, digest in child['runtime_sha256'].items()
                       if name != 'settings.lua'}
            if shared_runtime is None:
                shared_runtime = runtime
            else:
                _require(runtime == shared_runtime,
                         'Batch session runtime identity differs outside settings.lua')
            _, child_attempts, child_errors = _inspect(child_dir)
            errors.extend(relative+': '+e for e in child_errors)
            combined.extend(_rebase_attempt(a, relative) for a in child['attempts'])
            for attempt in child_attempts:
                row = copy.deepcopy(attempt)
                for match in row['matches']:
                    match['log'] = relative+'/'+match['log']
                if 'images' in row:
                    row['images'] = _rebase_attempt({'images': row['images']}, relative)['images']
                audited.append(row)
        except (KeyError, ValueError, TypeError, OSError) as exc:
            errors.append(relative+': '+str(exc))
    if seen != levels:
        errors.append('Incomplete or out-of-order difficulty sessions')
    if manifest['attempts'] != combined:
        errors.append('Batch attempts differ from complete rebased child manifests')
    _check_seal(run, needed, errors)
    return manifest, audited, errors


def audit_run(run: Path) -> dict:
    """Return structured audit failures; never modifies a run."""
    try:
        _, attempts, errors = _inspect(run)
        return {'ok': not errors, 'errors': errors, 'attempts': attempts,
                'difficulty_verified': not errors}
    except (KeyError, ValueError, TypeError, OSError) as exc:
        return {'ok': False, 'errors': [str(exc)], 'attempts': [], 'difficulty_verified': False}


def record_review(run: Path, attempt_id: str, reviewer: str, decision: str) -> dict:
    """Append a named assertion after strict audit; does not itself view images."""
    run = Path(run)
    _require(isinstance(reviewer, str) and reviewer.strip() and len(reviewer) <= 200, 'Reviewer name required')
    _require(decision in ('approve', 'reject'), 'Decision must be approve or reject')
    audit = audit_run(run)
    _require(audit['ok'], 'Evidence audit failed: '+'; '.join(audit['errors']))
    attempt = next((a for a in audit['attempts'] if a['id'] == attempt_id), None)
    _require(attempt and attempt['outcome'] == 'gameplay_clear', 'Review requires a gameplay-clear attempt')
    row = {'attempt_id': attempt_id, 'reviewer': reviewer.strip(), 'decision': decision,
           'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'evidence_manifest_sha256': _sha(run/'evidence-sha256.json'),
           'images': attempt['images'],
           'assertion': 'Named reviewer asserts actual inspection of Ken selection, Bison victory and all three ending images.',
           'identity_note': 'Self-declared name; not a cryptographic signature or authenticated identity.'}
    with (run/'reviews.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False)+'\n')
    return row


def _difficulty_totals(manifest, audit):
    """Presentation only: recorded outcomes stay distinct from audited clears."""
    audited = {a['id'] for a in audit['attempts']
               if audit['ok'] and a['valid'] and a['outcome'] == 'gameplay_clear'}
    result = {}
    for level in manifest['difficulty']:
        attempts = [a for a in manifest['attempts'] if a['difficulty'] == level]
        counts = collections.Counter(a['outcome'] for a in attempts)
        streak = best = 0
        for a in attempts:
            streak = streak+1 if a['outcome'] == 'gameplay_clear' else 0
            best = max(best, streak)
        count = len(attempts)
        result[str(level)] = {
            'difficulty': level, 'difficulty_verification': 'verified' if audit['ok'] else 'unverified',
            'attempts_started': count,
            'gameplay_clears': counts['gameplay_clear'], 'losses': counts['loss'],
            'invalids': counts['invalid'],
            'pending_or_other': count-counts['gameplay_clear']-counts['loss']-counts['invalid'],
            'clear_rate': counts['gameplay_clear']/count if count else None,
            'clear_rate_denominator': count,
            'rate_definition': 'recorded gameplay clears / all started attempts at this difficulty, including invalid and pending',
            'max_consecutive_streak': best, 'final_consecutive_streak': streak,
            'gameplay_clears_audited': sum(a['id'] in audited for a in attempts),
            'basis': 'sealed audited outcomes' if audit['ok'] else 'recorded outcomes; provisional because run audit did not pass',
        }
    return result


def write_report(run: Path) -> dict:
    """Write portable JSON/Markdown/HTML reports; never invent visual approval."""
    run = Path(run)
    audit = audit_run(run)
    manifest = _read(run/'run.json')
    stats = collections.Counter({o: 0 for o in OUTCOMES})
    opponents = {}
    for a in audit['attempts']:
        if not a['valid']:
            continue
        for m in a['matches']:
            stats.update(m['rounds'])
            bucket = opponents.setdefault(str(m['opponent']), collections.Counter({o: 0 for o in OUTCOMES}))
            bucket.update(m['rounds'])
            bucket['matches'] += 1; bucket['match_wins'] += m['result'] == 'ken_win'
    reviews = []
    if (run/'reviews.jsonl').exists():
        reviews = [json.loads(line) for line in (run/'reviews.jsonl').read_text(encoding='utf-8').splitlines()]
    seal_sha = _sha(run/'evidence-sha256.json') if (run/'evidence-sha256.json').exists() else None
    latest = {r['attempt_id']: r for r in reviews if r.get('evidence_manifest_sha256') == seal_sha}
    valid_clears = sum(a['valid'] and a['outcome'] == 'gameplay_clear' for a in audit['attempts']) if audit['ok'] else 0
    approved = sum(a['valid'] and a['outcome'] == 'gameplay_clear' and latest.get(a['id'], {}).get('decision') == 'approve' for a in audit['attempts']) if audit['ok'] else 0
    for op, bucket in opponents.items():
        bucket['round_win_rate'] = bucket['win']/sum(bucket[o] for o in OUTCOMES)
        bucket['name'] = NAMES[int(op)]
    count = len(manifest['attempts'])
    difficulties = _difficulty_totals(manifest, audit)
    report = {'schema': 'astra-play-sf2.report.v1', 'status': manifest['status'], 'audit': audit,
              'attempts_started': count, 'gameplay_clears_audited': valid_clears, 'reviewer_approved_clears': approved,
              'clear_rate': valid_clears/count if count else None, 'rounds': dict(stats), 'opponents': opponents,
              'clear_rate_denominator': count, 'difficulties': difficulties,
              'difficulty_verification': 'verified' if audit['difficulty_verified'] else 'unverified',
              'round_stats_basis': 'sealed audited results' if audit['ok'] else 'provisional completed-match statistics; run integrity not certified',
              'reviews': reviews, 'limitations': 'Local integrity checks, not adversarial proof. Gameplay result is separate from a named visual-review assertion; finite success is not a true win-rate guarantee.'}
    lines = ['# Astra-Play-SF2 report', '', f"Audit: {'PASS' if audit['ok'] else 'NOT VERIFIED'}", '',
             f'Audited gameplay clears: {valid_clears}/{count}. Reviewer-approved clears: {approved}.',
             'Difficulty: '+report['difficulty_verification']+'.',
             f"Rounds: {stats['win']}W {stats['loss']}L {stats['draw']}D.", '']
    level_lines = ['| Difficulty | Clears/started | Losses | Invalid | Pending | Clear rate | Max/final streak |',
                   '|---|---:|---:|---:|---:|---:|---:|']
    for d in difficulties.values():
        rate = f"{d['clear_rate']:.1%}" if d['clear_rate'] is not None else '—'
        level_lines.append(f"| {d['difficulty']} | {d['gameplay_clears']}/{d['attempts_started']} | {d['losses']} | {d['invalids']} | {d['pending_or_other']} | {rate} | {d['max_consecutive_streak']}/{d['final_consecutive_streak']} |")
    denominator = 'Per-difficulty denominator: all started attempts, including invalid and pending. Streaks use recorded outcomes and are provisional unless the run audit passes.'
    lines += ['## Per difficulty', '', *level_lines, '', denominator, '']
    opponent_lines = ['| Opponent | Matches won/played | W/L/D | Round win rate |',
                      '|---|---:|---:|---:|']
    for op, bucket in sorted(opponents.items(), key=lambda item: int(item[0])):
        opponent_lines.append(f"| {bucket['name']} ({op}) | {bucket['match_wins']}/{bucket['matches']} | {bucket['win']}/{bucket['loss']}/{bucket['draw']} | {bucket['round_win_rate']:.1%} |")
    lines += ['## Opponents', '', *opponent_lines, '']
    for a in manifest['attempts']:
        lines += [f"## {a['id']} — {a['outcome']}", '']
        images = a.get('images', {})
        for label, rel in [('Ken selection', images.get('selection')), ('Bison', images.get('bison'))] + [(f'Ending {i}', p) for i, p in enumerate(images.get('ending', []), 1)]:
            if rel:
                try: _path(run, rel)
                except ValueError: continue
                lines.append(f'[{label}]({quote(rel, safe="/")})')
        lines.append('')
    lines += ['## Audit errors', '', *('- '+e for e in audit['errors']), '', report['limitations']]
    md = '\n'.join(lines)+'\n'
    body = ['<!doctype html><meta charset="utf-8"><title>Astra-Play-SF2 report</title>', '<h1>Astra-Play-SF2 report</h1>',
            '<pre>'+html.escape('\n'.join(lines[:7]))+'</pre>']
    body.append('<h2>Per difficulty</h2><pre>'+html.escape('\n'.join(level_lines))+'</pre><p>'+html.escape(denominator)+'</p>')
    body.append('<h2>Opponent round statistics</h2><pre>'+html.escape('\n'.join(opponent_lines))+'</pre>')
    for a in manifest['attempts']:
        body.append('<h2>'+html.escape(a['id']+' — '+a['outcome'])+'</h2>')
        images = a.get('images', {})
        for rel in [images.get('selection'), images.get('bison'), *images.get('ending', [])]:
            if not rel: continue
            try: _path(run, rel)
            except ValueError: continue
            url = html.escape(quote(rel, safe='/'), quote=True)
            body.append(f'<a href="{url}"><img src="{url}" alt="Evidence image" style="max-width:480px"></a>')
    body += ['<pre>'+html.escape('\n'.join(audit['errors']))+'</pre>', '<p>'+html.escape(report['limitations'])+'</p>']
    for name, content in [('report.json', json.dumps(report, ensure_ascii=False, indent=2)+'\n'), ('report.md', md), ('report.html', '\n'.join(body))]:
        (run/name).write_text(content, encoding='utf-8')
    return report
