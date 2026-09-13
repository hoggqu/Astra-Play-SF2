import json
from pathlib import Path
import tempfile
import unittest
from . import training_report as summary

class SummaryTest(unittest.TestCase):
    def fixture(self, root):
        c=root/'cycle-001';t=c/'train';w=t/'worker-00';w.mkdir(parents=True)
        (t/'result.json').write_text(json.dumps(dict(status='complete',workers=1,model_sha256='abc',actual_steps=12,iterations=[{'ppo':{'policy_entropy':2.,'approx_kl':.01,'explained_variance':.8}}])))
        rows=[dict(episode=i,opponent=0,phase='train',outcome='win',native_round={'outcome':'win','score':[i,0]},**{'return':1.}) for i in (1,2)]
        (w/'episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        # Raw Lua copy is deliberately invalid: it must never be read/double-counted.
        (w/'training').mkdir();(w/'training/rl-chain-match-1.json').write_text('DO NOT READ')
        return c,w

    def test_sampler_report_shows_exposure_and_detects_mismatched_totals(self):
        from .adaptive_sampling import OpponentSampler
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);c,w=self.fixture(root)
            sampler=OpponentSampler([0,1])
            sampler.observe([dict(transitions=[{'state':{'p2':{'char':0}}}]*12,
                episodes=[dict(opponent=0,outcome='win',steps=6)]*2)])
            p=c/'train/result.json';r=json.loads(p.read_text());r['opponent_sampling']=sampler.snapshot();p.write_text(json.dumps(r))
            got=summary.summarize(root);self.assertFalse(got['issues'])
            page=summary.render_html(got)
            self.assertIn('实际决策占比',page);self.assertIn('下一场抽样概率',page)
            self.assertEqual(got['cycles'][0]['opponent_sampling']['per_opponent']['0']['decisions'],12)
            r['opponent_sampling']['decisions']=13;p.write_text(json.dumps(r))
            self.assertTrue(summary.summarize(root)['issues'])

    def test_actual_update_parameters_in_report(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);c,w=self.fixture(root)
            p=c/'train/result.json';r=json.loads(p.read_text())
            r.update(rollout_steps=12,minibatch_size=4,effective_ppo={'batch_size':4,'n_epochs':4})
            p.write_text(json.dumps(r))
            got=summary.summarize(root)
            self.assertEqual(got['cycles'][0]['parameters'],dict(workers=1,rollout_steps=12,minibatch_size=4,epochs=4,update_cycles=1))
            self.assertIn('每次更新决策数',summary.render_html(got))

    def test_native_terminal_score_only_no_trace_read(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);c,w=self.fixture(root)
            got=summary.summarize(root);self.assertFalse(got['issues'])
            self.assertEqual(got['cycles'][0]['training']['rounds']['W'],2)
            self.assertEqual(got['cycles'][0]['training']['matches']['W'],1)
            self.assertEqual(got['cycles'][0]['ppo']['policy_entropy']['mean'],2.)

    def test_incomplete_skips_workers_and_duplicate_fails(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);c,w=self.fixture(root)
            p=root/'cycle-002/train';p.mkdir(parents=True);(p/'result.json').write_text('{"status":"sampling"}')
            self.assertEqual(len(summary.summarize(root)['skipped']),1)
            p=w/'episodes.jsonl';s=p.read_text();p.write_text(s+s)
            self.assertIn('duplicate',summary.summarize(root)['issues'][0]['error'])

    def test_formal_failure_sidecar_and_clear_no_raw(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);c,w=self.fixture(root);p=c/'natural-coins';p.mkdir()
            loss=dict(id='1',outcome='loss',matches=['m01-ryu.json'],match_wins=0,audit={'ok':True})
            clear=dict(id='2',outcome='rl_gameplay_clear',matches=[str(i)+'.json' for i in range(11)],match_wins=11,audit={'ok':True})
            e=dict(status='complete',model_sha256='abc',native_timing_audit={'ok':True},action_interface_audit={'ok':True},attempts=[loss,clear])
            (p/'result.json').write_text(json.dumps(e));(p/'m01-ryu.json').write_text('GIANT TRACE MUST NOT BE READ')
            (p/'m01-ryu-status.json').write_text(json.dumps(dict(active=False,valid_continuous=True,model_sha256='abc',result='cpu_win',opponent=0)))
            got=summary.summarize(root);self.assertFalse(got['issues']);ev=got['cycles'][0]['evaluation'];self.assertEqual(ev['clears'],1);self.assertEqual(ev['attempts'][0]['failure_name'],'Ryu')
            e['action_interface_audit']['ok']=False;(p/'result.json').write_text(json.dumps(e))
            self.assertTrue(summary.summarize(root)['issues'])


    def test_top_level_reports_are_escaped_and_leave_evidence_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            top=Path(folder);root=top/'run';root.mkdir();self.fixture(root)
            model='</code><script>alert("x")</script>'
            (root/'result.json').write_text(json.dumps(dict(status='budget_stopped',stop_reason='max_duration',latest_model=model,latest_model_sha256='abc')))
            before={str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}
            result=summary.write_report(top,top)
            self.assertEqual(result['summary']['campaign'],str(root.resolve()))
            self.assertEqual(result['report_path'],str((top/'report.html').resolve()))
            page=(top/'report.html').read_text()
            self.assertIn('&lt;script&gt;',page)
            self.assertNotIn('<script>',page)
            self.assertNotIn('https://',page)
            self.assertIn('时长预算已到',page)
            self.assertEqual(before,{str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()})
            summary.write_report(top,top) # Updating our own generated files is allowed.

    def test_report_cannot_overwrite_campaign_or_follow_output_symlink(self):
        with tempfile.TemporaryDirectory() as folder:
            top=Path(folder);root=top/'run';root.mkdir();self.fixture(root)
            with self.assertRaises(ValueError):summary.write_report(root,root)
            with self.assertRaises(ValueError):summary.write_report(root,root/'reports')
            target=top/'summary.json';target.symlink_to(root/'cycle-001/train/result.json')
            with self.assertRaisesRegex(ValueError,'symlink'):summary.write_report(top,top)
            target.unlink();(top/'report.html').write_text('user-owned content')
            with self.assertRaisesRegex(ValueError,'unrelated'):summary.write_report(top,top)

    def test_native_opponent_detail_uses_score_and_round_sidecar_without_trace(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);c,w=self.fixture(root);p=c/'natural-coins';p.mkdir()
            a=dict(id='1',outcome='loss',matches=['one.json','two.json'],match_wins=1,audit={'ok':True})
            result=dict(status='complete',model_sha256='abc',native_timing_audit={'ok':True},action_interface_audit={'ok':True},attempts=[a])
            (p/'result.json').write_text(json.dumps(result))
            for name,op,score,rnd in [('one',0,[2,1],4),('two',3,[0,2],2)]:
                (p/(name+'.json')).write_text('NEVER READ THIS TRACE')
                status=dict(active=False,valid_continuous=True,model_sha256='abc',result='ken_win' if score[0]==2 else 'cpu_win',opponent=op,score=score,round=rnd)
                (p/(name+'-status.json')).write_text(json.dumps(status))
            result=summary.summarize(root);self.assertFalse(result['issues'])
            e=result['cycles'][0]['evaluation'];self.assertTrue(e['detail_complete'])
            self.assertEqual(e['per_opponent']['0']['round_D'],1)
            self.assertEqual(e['per_opponent']['0']['match_W'],1)
            self.assertEqual(e['per_opponent']['3']['round_L'],2)
            self.assertEqual(e['attempts'][0]['failure_name'],'Guile')

    def test_invalid_and_running_stages_are_visible_but_not_counted(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.fixture(root)
            for n,status in ((2,'sampling'),(3,'invalid')):
                p=root/f'cycle-{n:03d}/train';p.mkdir(parents=True)
                (p/'result.json').write_text(json.dumps(dict(status=status,actual_steps=1024,requested_steps=4096)))
            got=summary.summarize(root)
            self.assertEqual(len(got['cycles']),1);self.assertEqual(len(got['skipped']),2)
            self.assertEqual(got['completed_training_steps'],12)
            page=summary.render_html(got)
            self.assertIn('采样中',page);self.assertIn('无效运行',page)
            self.assertEqual(got['latest_model_sha256'],'abc')

    def test_malformed_result_generates_issue_not_fabricated_statistics(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);c,w=self.fixture(root)
            (c/'train/result.json').write_text('{broken')
            got=summary.summarize(root)
            self.assertTrue(got['issues']);self.assertFalse(got['cycles'])

if __name__=='__main__':unittest.main()
