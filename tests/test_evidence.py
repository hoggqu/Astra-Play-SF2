import copy
import hashlib
import json
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


def match(op, mode, lose=False):
    opening = dict(timer=99, p1=player(4), p2=player(op))
    events = [dict(kind='match_start', frame=0, state=opening, lead_frames=2 if op == 8 else 0)]
    rounds = []
    for n in (1, 2):
        stop = dict(timer=70, p1=player(4, hp=-1 if lose else 100, wins=0 if lose else n-1),
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
                   difficulty_bits=4, difficulty_checks=722, frame=722, speed='fast', opponent=op,
                   mode=mode, timeout_guard=op in (1, 5, 8, 11), telemetry_error=False,
                   diagnostic_level='full', trace_frames=722, score=rounds[-1]['score'], result=result,
                   latest=rounds[-1]['settled'])
    columns=['frame', 'preceding_input', 'issued_input']+sorted(TRACE_COLUMNS-{'frame','preceding_input','issued_input'})
    trace = dict(columns=columns,
                 rows=[[i, '', False]+[0]*(len(columns)-3) for i in range(1, 723)],
                 decisions=[], all_frames=True, all_decisions=True, observed_frames=722)
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
        self.manifest = dict(schema='test.v1', status='complete', difficulty=[3], attempts_requested=1,
                             speed='fast', policy_sha256=digest(runtime/'fighter.lua'),
                             runtime_sha256={p.name: digest(p) for p in runtime.iterdir()}, attempts=[self.attempt])
        command = "play_match('training/l3-001/m00',7,{training_validation=false,speed='fast'})"
        (self.run/'training/commands.jsonl').write_text('\n'.join(json.dumps(dict(id=1, command=command, event=e)) for e in ('prepared', 'accepted', 'completed'))+'\n')
        self.seal()

    def tearDown(self):
        self.tmp.cleanup()

    def put_match(self, rel, raw):
        p = self.run/rel; write(p, raw)
        write(p.with_name(p.stem+'-status.json'), raw['summary'])
        p.with_name(p.stem+'-policy.lua').write_bytes((self.run/'training/runtime/fighter.lua').read_bytes())

    def seal(self):
        write(self.run/'run.json', self.manifest)
        files = {p.relative_to(self.run).as_posix(): digest(p) for p in self.run.rglob('*')
                 if p.is_file() and p.name not in ('evidence-sha256.json', 'report.json', 'report.md', 'report.html', 'reviews.jsonl')}
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
