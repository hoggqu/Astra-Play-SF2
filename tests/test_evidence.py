import copy
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from astra_play_sf2.evidence import TRACE_COLUMNS, _difficulty_totals, _draw, _winner, audit_run, record_review, write_report


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def player(char, hp=144, wins=0, action=0, anim=100):
    return dict(char=char, hp=hp, displayed_hp=hp, timeout_hp=0, wins=wins,
                x=100 if char == 4 else 200, y=40, a=action, anim=anim)


def match(op, mode, lose=False, level=3):
    opening = dict(difficulty_bits=7-level, difficulty_mirror=level, effective_difficulty=level, timer=99, p1=player(4), p2=player(op))
    events = [dict(kind='match_start', frame=0, state=opening, lead_frames=2 if op == 8 else 0)]
    rounds = []
    for n in (1, 2):
        stop = dict(difficulty_bits=7-level, difficulty_mirror=level, effective_difficulty=level, timer=70, p1=player(4, hp=-1 if lose else 100, wins=0 if lose else n-1),
                    p2=player(op, hp=100 if lose else -1, wins=n-1 if lose else 0))
        settled = copy.deepcopy(stop)
        winner = settled['p2' if lose else 'p1']
        winner.update(wins=n, a=16)
        score = [0, n] if lose else [n, 0]
        row = dict(round=n, outcome='loss' if lose else 'win', frame=n*361,
                   opening=opening, stop=stop, settled=settled, score=score)
        events += [dict(kind='round_stop', round=n, frame=(n-1)*361+1, state=stop),
                   dict(kind='round_result', round=n, frame=n*361, result=row)]
        rounds.append(row)
    result = 'cpu_win' if lose else 'ken_win'
    events.append(dict(kind='match_finished', frame=722, valid=True, result=result))
    summary = dict(valid_continuous=True, status='complete', phase='complete', active=False,
                   training_validation=False, loads=0, saves=0, pauses_during_match=0,
                   difficulty_bits=7-level, difficulty_checks=722, effective_difficulty=level, effective_difficulty_checks=722, frame=722, speed='fast', opponent=op,
                   mode=mode, timeout_guard=op in (1, 5, 8, 11), telemetry_error=False,
                   diagnostic_level='full', trace_frames=722, score=rounds[-1]['score'], result=result,
                   latest=rounds[-1]['settled'])
    columns=['frame', 'preceding_input', 'issued_input']+sorted(TRACE_COLUMNS-{'frame','preceding_input','issued_input'})
    trace = dict(columns=columns,
                 rows=[[i, '', False]+[0]*(len(columns)-3) for i in range(1, 723)],
                 decisions=[], all_frames=True, all_decisions=True, observed_frames=722)
    for row in trace['rows']:
        for key, value in [('difficulty_bits', 7-level), ('difficulty_mirror', level), ('effective_difficulty', level)]:
            row[columns.index(key)] = value
    return dict(summary=summary, trace=trace, rounds=rounds, events=events, telemetry_error=False)


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run = Path(self.tmp.name)
        self.route = [7, 5, 6, 0, 1, 3, 2, 10, 11, 9, 8]
        selection = {str(op): 'mode_'+str(op) for op in self.route}
        runtime = self.run/'training/runtime'; runtime.mkdir(parents=True)
        for name in ('fighter.lua', 'play_core.lua', 'play.lua', 'bootstrap.lua'):
            (runtime/name).write_text('-- fixture '+name)
        write(runtime/'selection.json', selection)
        paths = []
        for i, op in enumerate(self.route):
            rel = f'training/l3-001/m{i:02d}.json'; paths.append(rel)
            self.put_match(rel, match(op, selection[str(op)]))
        images = dict(selection='training/selection.png', bison='training/bison.png',
                      ending=[f'training/ending-{i}.png' for i in range(3)])
        for rel in [images['selection'], images['bison'], *images['ending']]:
            (self.run/rel).write_bytes(b'fixture-image')
        lifecycle = 'training/l3-001/session-lifecycle.json'
        write(self.run/lifecycle, dict(resets=0, loads=0, saves=0, active=False, violation=False))
        self.attempt = dict(id='l3-001', difficulty=3, outcome='gameplay_clear', matches=paths, images=images, lifecycle=lifecycle)
        self.manifest = dict(schema='astra.run.v2', status='complete', difficulty=[3], attempts_requested=1,
                             consecutive=None, version='0.1.0', mame_version='0.288', rom='sf2',
                             speed='fast', policy_sha256=digest(runtime/'fighter.lua'),
                             runtime_sha256={p.name: digest(p) for p in runtime.iterdir()}, attempts=[self.attempt])
        self.set_boot(3)
        command = "play_match('training/l3-001/m00',7,{training_validation=false,speed='fast'})"
        (self.run/'training/commands.jsonl').write_text('\n'.join(json.dumps(dict(id=1, command=command, event=e)) for e in ('prepared', 'accepted', 'completed'))+'\n')
        self.seal()

    def set_boot(self, level):
        settings = self.run/'training/runtime/settings.lua'
        settings.write_text(f"astra_difficulty_bits={7-level}\nastra_difficulty_label='{level}'\n", encoding='utf-8')
        self.manifest['runtime_sha256']['settings.lua'] = digest(settings)
        (self.run/'boot-config.xml').write_text(
            '<mameconfig version="10"><system name="sf2"><input>'
            f'<port tag=":DSWB" type="DIPSWITCH" mask="7" defvalue="4" value="{7-level}" />'
            '</input></system></mameconfig>', encoding='utf-8')
        self.manifest['boot_config_sha256'] = digest(self.run/'boot-config.xml')
        self.manifest['boot_observation'] = 'training/boot-observation.json'
        write(self.run/'training/boot-observation.json',
              dict(difficulty_bits=7-level, difficulty_mirror=level, effective_difficulty=level))

    def tearDown(self):
        self.tmp.cleanup()

    def put_match(self, rel, raw):
        p = self.run/rel; write(p, raw)
        write(p.with_name(p.stem+'-status.json'), raw['summary'])
        p.with_name(p.stem+'-policy.lua').write_bytes((self.run/'training/runtime/fighter.lua').read_bytes())

    def seal(self, include_child_seals=False):
        write(self.run/'run.json', self.manifest)
        files = {p.relative_to(self.run).as_posix(): digest(p) for p in self.run.rglob('*')
                 if p.is_file() and (p.name not in ('evidence-sha256.json', 'report.json', 'report.md', 'report.html', 'reviews.jsonl')
                                    or (include_child_seals and p.name == 'evidence-sha256.json' and p.parent != self.run))}
        write(self.run/'evidence-sha256.json', dict(files=files))

    def test_complete_report_and_explicit_review(self):
        self.assertTrue(audit_run(self.run)['ok'])
        report = write_report(self.run)
        self.assertEqual(report['rounds'], dict(win=22, loss=0, draw=0))
        self.assertEqual(report['reviewer_approved_clears'], 0)
        review = record_review(self.run, 'l3-001', 'Human reviewer', 'approve')
        self.assertIn('not a cryptographic', review['identity_note'])
        self.assertEqual(write_report(self.run)['reviewer_approved_clears'], 1)
        record_review(self.run, 'l3-001', 'Human reviewer', 'reject')
        self.assertEqual(write_report(self.run)['reviewer_approved_clears'], 0)
        self.assertIn('training/ending-0.png', (self.run/'report.html').read_text())
        self.assertNotIn(str(self.run), (self.run/'report.md').read_text())
        self.assertEqual(report['difficulties']['3']['gameplay_clears'], 1)
        self.assertEqual(report['difficulties']['3']['final_consecutive_streak'], 1)
        self.assertEqual(report['opponents']['8']['name'], 'M. Bison')
        self.assertIn('Dhalsim (7)', (self.run/'report.html').read_text())
        self.assertIn('all started attempts', (self.run/'report.md').read_text())

    def test_per_difficulty_denominators_and_streaks_are_separate(self):
        attempts = []
        for level, outcomes in [(3, ['gameplay_clear','gameplay_clear','invalid','loss','gameplay_clear']),
                                (7, ['gameplay_clear','gameplay_clear','pending'])]:
            attempts += [dict(id=f'l{level}-{i:03d}', difficulty=level, outcome=o)
                         for i, o in enumerate(outcomes, 1)]
        result = _difficulty_totals(dict(difficulty=[3, 7, 5], attempts=attempts), dict(ok=False, attempts=[]))
        normal = result['3']
        self.assertEqual((normal['gameplay_clears'], normal['losses'], normal['invalids']), (3, 1, 1))
        self.assertEqual((normal['clear_rate_denominator'], normal['clear_rate']), (5, 3/5))
        self.assertEqual((normal['max_consecutive_streak'], normal['final_consecutive_streak']), (2, 1))
        self.assertEqual(result['7']['final_consecutive_streak'], 0)
        self.assertEqual(result['7']['pending_or_other'], 1)
        self.assertEqual(result['7']['clear_rate'], 2/3)
        self.assertIsNone(result['5']['clear_rate'])
        self.assertEqual(normal['gameplay_clears_audited'], 0)
        self.assertIn('provisional', normal['basis'])

    def test_valid_loss_preserved_not_reviewable_clear(self):
        rel = self.attempt['matches'][0]
        self.put_match(rel, match(7, 'mode_7', lose=True))
        self.attempt['matches'] = [rel]; self.attempt['outcome'] = 'loss'; self.seal()
        self.assertTrue(audit_run(self.run)['ok'])
        self.assertEqual(write_report(self.run)['rounds'], dict(win=0, loss=2, draw=0))
        with self.assertRaises(ValueError): record_review(self.run, 'l3-001', 'name', 'approve')

    def test_sealed_mutation_fails(self):
        (self.run/self.attempt['images']['selection']).write_bytes(b'changed')
        self.assertFalse(audit_run(self.run)['ok'])

    def test_resealed_runtime_change_still_fails_identity(self):
        (self.run/'training/runtime/fighter.lua').write_text('changed')
        self.seal(); self.assertFalse(audit_run(self.run)['ok'])

    def test_seal_missing_required_file_fails(self):
        p=self.run/'evidence-sha256.json'; data=json.loads(p.read_text())
        del data['files']['run.json']; write(p,data)
        self.assertFalse(audit_run(self.run)['ok'])

    def test_global_lifecycle_is_required_in_seal_when_present(self):
        life=self.run/'training/lifecycle.json'
        write(life,dict(resets=0,loads=0,saves=0,active=False,violation=False))
        self.assertFalse(audit_run(self.run)['ok'])
        self.seal(); self.assertTrue(audit_run(self.run)['ok'])
        write(life,dict(resets=1,loads=0,saves=0,active=False,violation=True))
        self.seal(); self.assertFalse(audit_run(self.run)['ok'])

    def test_unmanifested_runtime_file_refused(self):
        (self.run/'training/runtime/extra.lua').write_text('return {}')
        self.seal(); self.assertFalse(audit_run(self.run)['ok'])

    def test_trace_lifecycle_status_and_maturity_failures(self):
        rel=self.attempt['matches'][0]; original=json.loads((self.run/rel).read_text())
        for mutate in (lambda r:r['trace']['rows'].pop(),
                       lambda r:r['summary'].update(loads=1),
                       lambda r:r['trace']['rows'][0].__setitem__(2,'C'),
                       lambda r:r['rounds'][0]['settled']['p1'].update(a=0),
                       lambda r:r['events'][1].update(frame=2)):
            with self.subTest(mutate=mutate):
                raw=copy.deepcopy(original); mutate(raw); self.put_match(rel,raw); self.seal()
                self.assertFalse(audit_run(self.run)['ok'])

    def test_command_forbidden_or_incomplete(self):
        p=self.run/'training/commands.jsonl'
        for command, events in [("restore('file',1)",('prepared','accepted','completed')),
                                ('observe()',('prepared','accepted'))]:
            p.write_text('\n'.join(json.dumps(dict(command=command,event=e,id=1))for e in events))
            self.seal(); self.assertFalse(audit_run(self.run)['ok'])

    def test_invalid_pending_and_skipped_attempts(self):
        for state in ('running','invalid'):
            self.manifest['status']=state; self.seal()
            self.assertFalse(audit_run(self.run)['ok'])
        self.manifest['status']='complete'; self.manifest['attempts_requested']=2; self.seal()
        self.assertFalse(audit_run(self.run)['ok'])

    def test_pending_report_keeps_provisional_completed_rounds(self):
        self.manifest['status']='running'; self.seal()
        (self.run/'evidence-sha256.json').unlink()
        report=write_report(self.run)
        self.assertEqual(report['rounds']['win'],22)
        self.assertEqual(report['gameplay_clears_audited'],0)
        self.assertIn('provisional',report['round_stats_basis'])
        with self.assertRaises(ValueError): record_review(self.run,'l3-001','reviewer','approve')

    def enable_entry(self):
        entry = self.run/'training/runtime/entry.lua'
        entry.write_text('-- fixture native entry controller', encoding='utf-8')
        self.manifest['runtime_sha256']['entry.lua'] = digest(entry)
        idle = dict(task_boot=0, task_attract=8, task_credit=0, task_input=0,
                    task_game=0, mode=10, playback=1, ending=0, fade_busy=0)
        start = dict(idle, task_attract=0, task_credit=8, mode=4)
        self.attempt['readiness'] = {
            kind: dict(kind=kind, status='ready', frames=20, max_frames=9000,
                       stable_frames=2, required_stable_frames=2, state=state)
            for kind, state in [('coin', idle), ('start', start)]}
        self.seal()

    def test_native_entry_evidence_required_only_for_entry_runtime(self):
        self.assertTrue(audit_run(self.run)['ok'])  # Original 0.1.1 has no entry.lua.
        self.enable_entry()
        self.assertTrue(audit_run(self.run)['ok'])
        del self.attempt['readiness']
        self.seal()
        self.assertFalse(audit_run(self.run)['ok'])

    def test_entry_refuses_continue_ending_boot_and_unready_start(self):
        self.enable_entry()
        original = copy.deepcopy(self.attempt['readiness'])
        for kind, changes in [('coin', {'task_boot': 8}),
                              ('coin', {'task_input': 1, 'task_game': 8}),
                              ('coin', {'task_game': 8, 'ending': 1}),
                              ('coin', {'task_attract': 0}),
                              ('coin', {'task_credit': 8}),
                              ('coin', {'playback': 0}),
                              ('start', {'task_game': 1}),
                              ('start', {'task_credit': 0}),
                              ('start', {'task_attract': 8}),
                              ('start', {'mode': 2}),
                              ('start', {'fade_busy': 1}),
                              ('start', {'playback': 0})]:
            with self.subTest(kind=kind, changes=changes):
                self.attempt['readiness'] = copy.deepcopy(original)
                self.attempt['readiness'][kind]['state'].update(changes)
                self.seal()
                self.assertFalse(audit_run(self.run)['ok'])
        # Real ending cleanup leaves this flag stale until later attract init.
        self.attempt['readiness'] = copy.deepcopy(original)
        for result in self.attempt['readiness'].values():
            result['state']['ending'] = 1
        self.attempt['readiness']['coin']['state']['fade_busy'] = 1
        self.seal()
        self.assertTrue(audit_run(self.run)['ok'])

    def test_malformed_entry_evidence_is_audit_failure_without_crash(self):
        self.enable_entry()
        original = copy.deepcopy(self.attempt['readiness'])
        for location in ('readiness', 'result', 'state'):
            for value in (None, [], ['bad']):
                with self.subTest(location=location, value=value):
                    self.attempt['readiness'] = copy.deepcopy(original)
                    if location == 'readiness':
                        self.attempt['readiness'] = value
                    elif location == 'result':
                        self.attempt['readiness']['coin'] = value
                    else:
                        self.attempt['readiness']['coin']['state'] = value
                    self.seal()
                    self.assertFalse(audit_run(self.run)['ok'])
        for location in ('readiness', 'result', 'state'):
            with self.subTest(missing=location):
                self.attempt['readiness'] = copy.deepcopy(original)
                if location == 'readiness':
                    del self.attempt['readiness']
                elif location == 'result':
                    del self.attempt['readiness']['coin']
                else:
                    del self.attempt['readiness']['coin']['state']
                self.seal()
                self.assertFalse(audit_run(self.run)['ok'])
        self.attempt['readiness'] = copy.deepcopy(original)
        self.attempt['readiness']['coin']['error'] = 'native failure'
        self.seal()
        self.assertFalse(audit_run(self.run)['ok'])
        self.attempt['readiness']['coin']['error'] = ''
        self.seal()
        self.assertTrue(audit_run(self.run)['ok'])

    def test_entry_timeout_and_inconsistent_readiness_counters_refused(self):
        self.enable_entry()
        original = copy.deepcopy(self.attempt['readiness'])
        for updates in [dict(status='timeout'), dict(kind='start'), dict(frames=1),
                        dict(frames=9001), dict(frames=True), dict(max_frames=12000),
                        dict(stable_frames=1), dict(required_stable_frames=1)]:
            with self.subTest(updates=updates):
                self.attempt['readiness'] = copy.deepcopy(original)
                self.attempt['readiness']['coin'].update(updates)
                self.seal()
                self.assertFalse(audit_run(self.run)['ok'])

    def test_boot_configuration_and_internal_reading_must_match(self):
        original = (self.run/'boot-config.xml').read_text()
        for content in (original.replace('value="4" />', 'value="0" />'),
                        original.replace('tag=":DSWB"', 'tag=":DSWA"'),
                        '<malformed'):
            with self.subTest(content=content):
                (self.run/'boot-config.xml').write_text(content)
                self.manifest['boot_config_sha256'] = digest(self.run/'boot-config.xml')
                self.seal()
                self.assertFalse(audit_run(self.run)['ok'])
        self.set_boot(3)
        self.manifest['boot_config_sha256'] = '0'*64
        self.seal()
        self.assertFalse(audit_run(self.run)['ok'])
        self.set_boot(3)
        write(self.run/'training/boot-observation.json',
              dict(difficulty_bits=4, difficulty_mirror=3, effective_difficulty=7))
        self.seal()
        self.assertFalse(audit_run(self.run)['ok'])
        self.assertEqual(write_report(self.run)['difficulty_verification'], 'unverified')

    def test_internal_summary_snapshot_and_trace_tampering(self):
        rel = self.attempt['matches'][0]
        original = json.loads((self.run/rel).read_text())
        for mutate in (lambda r: r['summary'].update(effective_difficulty=7),
                       lambda r: r['summary'].update(effective_difficulty_checks=721),
                       lambda r: r['events'][0]['state'].update(difficulty_mirror=7),
                       lambda r: r['rounds'][0]['settled'].update(effective_difficulty=7),
                       lambda r: r['trace']['rows'][19].__setitem__(r['trace']['columns'].index('effective_difficulty'), 7)):
            with self.subTest(mutate=mutate):
                raw = copy.deepcopy(original)
                mutate(raw)
                self.put_match(rel, raw)
                self.seal()
                self.assertFalse(audit_run(self.run)['ok'])

    def test_legacy_run_remains_readable_but_difficulty_unverified(self):
        self.manifest['schema'] = 'astra.run.v1'
        del self.manifest['boot_config_sha256']
        del self.manifest['boot_observation']
        for rel in self.attempt['matches']:
            raw = json.loads((self.run/rel).read_text())
            raw['summary'].pop('effective_difficulty')
            raw['summary'].pop('effective_difficulty_checks')
            indices = [i for i, key in enumerate(raw['trace']['columns']) if key not in
                       ('difficulty_bits', 'difficulty_mirror', 'effective_difficulty')]
            raw['trace']['columns'] = [raw['trace']['columns'][i] for i in indices]
            raw['trace']['rows'] = [[row[i] for i in indices] for row in raw['trace']['rows']]
            self.put_match(rel, raw)
        self.seal()
        report = write_report(self.run)
        self.assertFalse(report['audit']['ok'])
        self.assertEqual(report['rounds']['win'], 22)
        self.assertEqual(report['difficulties']['3']['gameplay_clears'], 1)
        self.assertEqual(report['gameplay_clears_audited'], 0)
        self.assertIn('legacy', ' '.join(report['audit']['errors']))
        with self.assertRaises(ValueError):
            record_review(self.run, 'l3-001', 'name', 'approve')

    def make_batch(self):
        from astra_play_sf2.evidence import _rebase_attempt
        batch = self.run/'batch'
        original = self.run
        original_manifest = copy.deepcopy(self.manifest)
        for level in (3, 7):
            child = batch/f'sessions/l{level}'
            child.mkdir(parents=True)
            shutil.copytree(original/'training', child/'training')
            self.run = child
            self.manifest = copy.deepcopy(original_manifest)
            self.manifest['difficulty'] = [level]
            self.manifest['attempts'][0]['difficulty'] = level
            self.manifest['attempts'][0]['id'] = f'l{level}-001'
            for relative, op in zip(self.manifest['attempts'][0]['matches'], self.route):
                self.put_match(relative, match(op, f'mode_{op}', level=level))
            self.set_boot(level)
            self.seal()
        self.run = batch
        self.manifest = dict(schema='astra.batch.v1', status='complete', difficulty=[3, 7],
                             attempts_requested=1, consecutive=None, speed='fast',
                             version='0.1.0', mame_version='0.288', rom='sf2',
                             policy_sha256=original_manifest['policy_sha256'], attempts=[],
                             sessions=[dict(difficulty=n, path=f'sessions/l{n}') for n in (3, 7)])
        for session in self.manifest['sessions']:
            child = json.loads((batch/session['path']/'run.json').read_text())
            self.manifest['attempts'] += [_rebase_attempt(a, session['path']) for a in child['attempts']]
        self.seal(include_child_seals=True)
        return batch

    def test_batch_recursively_verifies_and_rebases(self):
        batch = self.make_batch()
        audited = audit_run(batch)
        self.assertTrue(audited['ok'], audited['errors'])
        self.assertEqual(len(audited['attempts']), 2)
        self.assertTrue(audited['attempts'][1]['matches'][0]['log'].startswith('sessions/l7/'))
        report = write_report(batch)
        self.assertEqual(report['rounds']['win'], 44)
        self.assertEqual(report['gameplay_clears_audited'], 2)
        record_review(batch, 'l7-001', 'reviewer', 'approve')
        self.assertEqual(write_report(batch)['reviewer_approved_clears'], 1)

    def test_batch_cross_session_splice_and_missing_child_seal_rejected(self):
        batch = self.make_batch()
        self.manifest['attempts'][1]['matches'][0] = self.manifest['attempts'][0]['matches'][0]
        self.seal()
        self.assertFalse(audit_run(batch)['ok'])
        self.assertIn('rebased', ' '.join(audit_run(batch)['errors']))
        self.assertIn('seal omits', ' '.join(audit_run(batch)['errors']))

    def test_resealed_wrong_runtime_settings_refused(self):
        settings = self.run/'training/runtime/settings.lua'
        settings.write_text("astra_difficulty_bits=0\nastra_difficulty_label='7'\n")
        self.manifest['runtime_sha256']['settings.lua'] = digest(settings)
        self.seal()
        result = audit_run(self.run)
        self.assertFalse(result['ok'])
        self.assertIn('settings.lua difficulty differs', ' '.join(result['errors']))

    def test_batch_resealed_core_or_selection_swap_refused(self):
        batch = self.make_batch()
        parent_manifest = self.manifest
        child = batch/'sessions/l7'
        child_manifest = json.loads((child/'run.json').read_text())
        originals = {name: (child/'training/runtime'/name).read_bytes()
                     for name in ('play_core.lua', 'selection.json')}
        for name in originals:
            with self.subTest(name=name):
                path = child/'training/runtime'/name
                # Both altered leaves remain individually valid under their own identity.
                # The batch additionally prohibits switching that identity between levels.
                path.write_bytes(originals[name]+b'\n ')
                self.run, self.manifest = child, copy.deepcopy(child_manifest)
                self.manifest['runtime_sha256'][name] = digest(path)
                self.seal()
                self.assertTrue(audit_run(child)['ok'])
                self.run, self.manifest = batch, parent_manifest
                self.seal(include_child_seals=True)
                result = audit_run(batch)
                self.assertFalse(result['ok'])
                self.assertIn('runtime identity differs', ' '.join(result['errors']))
                self.assertNotIn('Sealed evidence changed', ' '.join(result['errors']))
                path.write_bytes(originals[name])
                self.run, self.manifest = child, child_manifest
                self.seal()
                self.run, self.manifest = batch, parent_manifest
                self.seal(include_child_seals=True)
        self.assertTrue(audit_run(batch)['ok'])

    def test_batch_partial_and_wrong_session_settings_not_verified(self):
        batch = self.make_batch()
        child = batch/'sessions/l7/run.json'
        data = json.loads(child.read_text())
        data['speed'] = 'normal'
        write(child, data)
        result = audit_run(batch)
        self.assertFalse(result['ok'])
        self.assertIn('speed differs', ' '.join(result['errors']))
        self.manifest['status'] = 'invalid'
        self.manifest['sessions'].pop()
        self.manifest['attempts'].pop()
        self.seal()
        report = write_report(batch)
        self.assertFalse(report['audit']['ok'])
        self.assertEqual(report['difficulties']['7']['attempts_started'], 0)

    def test_draw_pose_and_native_time_latch_boundaries(self):
        state=dict(timer=0,p1=player(4,5,action=18),p2=player(5,5,action=18))
        for p in (state['p1'],state['p2']):p['timeout_hp']=5
        self.assertTrue(_draw(state,state,[],[0,0],0,0))
        bad=copy.deepcopy(state);bad['p2']['timeout_hp']=4
        self.assertFalse(_draw(bad,bad,[],[0,0],0,0))
        state=dict(timer=33,p1=player(4,-1,action=12,anim=388594),p2=player(8,-1))
        self.assertTrue(_draw(state,state,[],[1,0],0,0))
        state['p1']['a']=8
        self.assertFalse(_draw(state,state,[],[1,0],0,0))
        winner=player(4,34,1,16);winner['timeout_hp']=34
        loser=player(8,-1,0,0,281952);loser.update(timeout_hp=1,displayed_hp=1)
        self.assertTrue(_winner(winner,loser,{'timer':0},{'timer':0}))
        loser['anim']=281953
        self.assertFalse(_winner(winner,loser,{'timer':0},{'timer':0}))

    def test_wrong_route_images_and_path_escape(self):
        self.attempt['images']['ending']=self.attempt['images']['ending'][:2]; self.seal()
        self.assertFalse(audit_run(self.run)['ok'])

    def test_requested_consecutive_early_stop(self):
        self.manifest['attempts_requested']=5
        self.manifest['consecutive']=1
        self.seal(); self.assertTrue(audit_run(self.run)['ok'])
        self.manifest['consecutive']=2
        self.seal(); self.assertFalse(audit_run(self.run)['ok'])
        self.attempt['images']['selection']='../outside'; self.seal()
        self.assertFalse(audit_run(self.run)['ok'])


if __name__ == '__main__':
    unittest.main()
