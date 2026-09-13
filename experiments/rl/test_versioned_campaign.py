"""Campaign control/evidence tests with fake child CLIs; no Torch/MAME required."""
import json
import os
import sys
import time
from unittest.mock import Mock, patch
from pathlib import Path
import tempfile
import unittest
import zipfile
from .versioned_campaign import campaign,digest,write_json,request_stop,read_stop,run_child,STOP_PROTOCOL,failure_diagnostics

class VersionedCampaignTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name).resolve()
        self.code=self.root/'code';self.code.mkdir();(self.code/'launch.py').write_text('# fake pinned launcher\n')
        (self.code/'pkg').mkdir();(self.code/'pkg/runtime.lua').write_text('return {}\n')
        write_json(self.code/'build.json',{'package':'pkg','action_interface':'actions-test','observation_interface':'observations-test',
            'frozen_files_sha256':{'launch.py':digest(self.code/'launch.py')},'derived_sha256':{'runtime.lua':digest(self.code/'pkg/runtime.lua')}})
        self.dataset=self.root/'dataset.json';write_json(self.dataset,{'status':'complete','difficulty':3})
        self.config=self.root/'config.json';write_json(self.config,{'mame':'unused'})
        self.calls=[];self.updates=0;self.outcomes=[False,True];self.eval_code=None;self.bad_adam=False;self.bad_audit=False
    def zip(self,path):
        path.parent.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(path,'w') as z:
            for key in ('data','policy.pth','policy.optimizer.pth'):z.writestr(key,f'{key}:{self.updates}')
    def runner(self,command,log,environment,timeout):
        self.calls.append(command);Path(log).write_text('fakechild\n');self.assertEqual(environment['ASTRA_SF2_CONFIG'],str(self.config))
        module=command[2];args=command[3:];opts=dict(zip(args[::2],args[1::2]));target=Path(opts['--output']);target.mkdir()
        if module=='initialize':
            model=target/'ppo-initial.zip';self.zip(model);result={'status':'complete','model_sha256':digest(model)};status=0
        elif module=='batch_train':
            steps=int(opts['--steps']);workers=int(opts['--workers'])
            old=self.updates;self.updates+=4*(steps//int(opts.get('--rollout-steps',workers*256)));model=target/'ppo-batch.zip';self.zip(model)
            result={'rollout_steps':int(opts['--rollout-steps']) if '--rollout-steps' in opts else None,
                'minibatch_size':int(opts.get('--minibatch-size',64)), 'effective_ppo':{'batch_size':int(opts.get('--minibatch-size',64))},
                'status':'complete','actual_steps':steps,'completed_update_steps':steps,'difficulty':3,
                'dataset_sha256':digest(self.dataset),'init_model_sha256':digest(opts['--init-model']),
                'checkpoint_every_requested':int(opts['--checkpoint-every']),
                'checkpoint_every_effective':int(opts['--checkpoint-every']),
                'parameters_changed':True,'benchmark':False,'parity':False,'native_parity':False,
                'optimizer_initialization':{'loaded_from_init_model':True,'initial_updates':0 if self.bad_adam else old,'initial_state_entries':16 if old else 0},
                'final_checkpoint':{'complete_update':True,'optimizer_updates':self.updates,'sha256':digest(model)},'model_sha256':digest(model)};status=0
        else:
            cleared=self.outcomes.pop(0);attempts=[]
            for i in range(1 if cleared else 3):
                attempts.append({'id':f'l3-{i+1:03d}','outcome':'rl_gameplay_clear' if cleared else 'loss',
                    'match_wins':11 if cleared else 2,'matches':[f'a{i}-m{j}.json' for j in range(11 if cleared else 3)],'audit':{'ok':True}})
            for attempt in attempts:
                for j,name in enumerate(attempt['matches']):
                    win=cleared or j<2
                    write_json(target/name,{'summary':{'status':'complete','valid_continuous':True,'opponent':j,'score':[2,0] if win else [0,2],'result':'ken_win' if win else 'cpu_win'},'rounds':[{'outcome':'win' if win else 'loss'}]*2})
            result={'status':'complete','difficulty':3,'model_sha256':digest(opts['--model']),
                'action_interface':'actions-test','observation_interface':'observations-test','native_timing':True,
                'native_timing_audit':{'ok':True},'action_interface_audit':{'ok':not self.bad_audit},'attempts':attempts}
            status=(0 if cleared else 1) if self.eval_code is None else self.eval_code
        write_json(target/'result.json',result);return status
    def run_campaign(self,**extra):
        args=dict(code=self.code,package='pkg',dataset=self.dataset,output=self.root/'output',config=self.config,
            cycles=3,steps=409600,workers=16,runner=self.runner)
        args.update(extra)
        return campaign(**args)
    def test_exact_rollout_and_minibatch_pass_to_trainer(self):
        self.outcomes=[True]
        result=self.run_campaign(workers=3,steps=4096,checkpoint_every=4096,rollout_steps=1024,minibatch_size=128)
        self.assertEqual(result['decisions_per_update'],1024)
        self.assertEqual(result['minibatch_size'],128)
        self.assertEqual(result['cycles'][0]['optimizer_updates'],16)
        train=next(c for c in self.calls if c[2]=='batch_train')
        self.assertEqual(train[train.index('--rollout-steps')+1],'1024')
        self.assertEqual(train[train.index('--minibatch-size')+1],'128')

    def test_custom_worker_count_passes_through_all_training_stages(self):
        self.outcomes=[True]
        result=self.run_campaign(workers=3,steps=768,checkpoint_every=768)
        self.assertEqual(result['workers'],3)
        train=next(c for c in self.calls if c[2]=='batch_train')
        self.assertEqual(train[train.index('--workers')+1],'3')
        self.assertEqual(result['cycles'][0]['training_steps'],768)

    def test_latest_adam_continues_after_three_losses_and_stops_on_first_clear(self):
        result=self.run_campaign();self.assertEqual(result['status'],'clear');self.assertEqual(len(result['cycles']),2)
        self.assertEqual(len(result['cycles'][0]['attempts']),3)
        train=[c for c in self.calls if c[2]=='batch_train'];self.assertIn(str(self.root/'output/cycle-001/train/ppo-batch.zip'),train[1])
        self.assertEqual(result['cycles'][1]['optimizer_updates'],800)
        self.assertEqual(result['block'],128)
        self.assertIsNone(result['first_cycle_steps'])
        self.assertEqual(result['first_cycle_steps_effective'],409600)
        self.assertEqual(result['checkpoint_every'],20480)
        self.assertEqual(train[0][train[0].index('--checkpoint-every')+1],'20480')
        self.assertEqual(train[0][train[0].index('--block')+1],'128')
        self.assertEqual(result['evaluation_matchups']['2']['round_losses'],6)
        self.assertEqual(result['evaluation_matchups']['2']['match_losses'],3)
        self.assertTrue((self.root/'output/cycle-001/cycle.json').is_file())
        self.assertNotIn('--all-attempts',sum(self.calls,[]))
    def test_invalid_child_exit_two_stops_immediately_preserving_completed_model(self):
        self.eval_code=2
        with self.assertRaises(RuntimeError):self.run_campaign()
        result=json.loads((self.root/'output/result.json').read_text());self.assertEqual(result['status'],'invalid')
        self.assertEqual(len(self.calls),3);self.assertTrue(Path(result['latest_model']).is_file())
    def test_positive_without_interface_audit_rejected(self):
        self.outcomes=[True];self.bad_audit=True
        with self.assertRaisesRegex(RuntimeError,'action_interface_audit'):self.run_campaign()
    def test_optimizer_reset_rejected_before_second_evaluation(self):
        self.bad_adam=True
        with self.assertRaisesRegex(RuntimeError,'Adam'):self.run_campaign()
        self.assertEqual([c[2] for c in self.calls].count('native_continuous'),1)
    def test_source_mutation_after_child_is_invalid(self):
        original=self.runner
        def changed(*args):
            code=original(*args);(self.code/'pkg/runtime.lua').write_text('changed');return code
        self.runner=changed
        with self.assertRaisesRegex(RuntimeError,'Pinned source changed'):self.run_campaign()
    def test_invalid_initial_zip_or_existing_output_not_reused(self):
        bad=self.root/'bad.zip'
        with zipfile.ZipFile(bad,'w') as z:z.writestr('policy.pth','weights only')
        with self.assertRaisesRegex(RuntimeError,'complete PPO'):self.run_campaign(init_model=bad)
        (self.root/'output').mkdir()
        with self.assertRaises(FileExistsError):self.run_campaign()
        self.assertFalse(self.calls)
    def test_explicit_initial_full_zip_skips_initializer(self):
        initial=self.root/'initial.zip';self.zip(initial);self.outcomes=[True]
        result=self.run_campaign(init_model=initial);self.assertEqual(result['status'],'clear')
        self.assertNotIn('initialize',[c[2] for c in self.calls])
    def test_explicit_block_is_forwarded_and_invalid_block_rejected(self):
        with self.assertRaises(ValueError):self.run_campaign(block=7)
        self.outcomes=[True];result=self.run_campaign(block=256)
        train=next(c for c in self.calls if c[2]=='batch_train')
        self.assertEqual(train[train.index('--block')+1],'256')
        self.assertEqual(result['block'],256)
    def test_checkpoint_interval_is_validated_forwarded_and_recorded(self):
        for invalid in (0,-4096,4097,True):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.run_campaign(checkpoint_every=invalid)
        self.outcomes=[True];result=self.run_campaign(checkpoint_every=4096)
        train=next(c for c in self.calls if c[2]=='batch_train')
        self.assertEqual(train[train.index('--checkpoint-every')+1],'4096')
        self.assertEqual(result['checkpoint_every'],4096)
        self.assertEqual(result['cycles'][0]['checkpoint_every'],4096)

    def test_four_workers_forwarded_with_complete_1024_decision_update(self):
        self.outcomes=[True]
        result=self.run_campaign(workers=4,steps=1024,checkpoint_every=1024)
        train=next(c for c in self.calls if c[2]=='batch_train')
        for name,value in (('--workers','4'),('--steps','1024'),('--checkpoint-every','1024')):
            self.assertEqual(train[train.index(name)+1],value)
        self.assertEqual(result['workers'],4)
        self.assertEqual(result['steps_per_cycle'],1024)
        self.assertEqual(result['cycles'][0]['optimizer_updates'],4)
        self.assertEqual(result['checkpoint_every'],1024)

    def test_worker_count_and_per_worker_update_multiples_rejected(self):
        for workers in (0,-1,True,4.0):
            with self.subTest(workers=workers), self.assertRaises(ValueError):
                self.run_campaign(workers=workers)
        for steps in (256,512,1025,True,1024.0):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                self.run_campaign(workers=4,steps=steps)
        for workers,interval in ((4,512),(8,1024),(16,2048)):
            with self.subTest(workers=workers,interval=interval), self.assertRaises(ValueError):
                self.run_campaign(workers=workers,checkpoint_every=interval)
        self.assertFalse(self.calls)
        self.assertFalse((self.root/'output').exists())

    def test_first_cycle_override_only_applies_once_with_actual_budget_recorded(self):
        result=self.run_campaign(workers=8,first_cycle_steps=114688)
        train=[c for c in self.calls if c[2]=='batch_train']
        self.assertEqual([c[c.index('--steps')+1] for c in train],['114688','409600'])
        self.assertEqual(result['steps_per_cycle'],409600)
        self.assertEqual(result['first_cycle_steps'],114688)
        self.assertEqual(result['first_cycle_steps_effective'],114688)
        self.assertEqual([c['training_steps_requested'] for c in result['cycles']],[114688,409600])
        self.assertEqual([c['training_steps'] for c in result['cycles']],[114688,409600])
        self.assertEqual(result['cycles'][1]['optimizer_updates'],1024)

    def test_first_cycle_budget_rejects_noninteger_or_incomplete_update(self):
        for value in (0,-2048,1024,2049,True,2048.0):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.run_campaign(workers=8,first_cycle_steps=value)
        self.assertFalse(self.calls)
        self.assertFalse((self.root/'output').exists())

    def test_child_reporting_default_steps_for_short_first_cycle_is_invalid(self):
        original=self.runner
        def wrong_budget(command,*args):
            status=original(command,*args)
            if command[2]=='batch_train':
                target=Path(command[command.index('--output')+1])/'result.json'
                data=json.loads(target.read_text())
                data.update(actual_steps=409600,completed_update_steps=409600)
                write_json(target,data)
            return status
        self.runner=wrong_budget
        with self.assertRaisesRegex(RuntimeError,'Training budget'):
            self.run_campaign(workers=8,first_cycle_steps=114688)
        self.assertNotIn('native_continuous',[c[2] for c in self.calls])
        result=json.loads((self.root/'output/result.json').read_text())
        self.assertEqual(result['status'],'invalid')
        self.assertEqual(result['cycles'][0]['training_steps_requested'],114688)

    def test_no_stop_on_clear_runs_all_cycles_and_retains_each_result(self):
        self.outcomes=[True]*3
        result=self.run_campaign(stop_on_clear=False)
        self.assertEqual(result['status'],'exhausted')
        self.assertTrue(result['any_clear'])
        self.assertEqual(len(result['cycles']),3)
        self.assertEqual([c[2] for c in self.calls].count('native_continuous'),3)

    def test_duration_budget_validation_and_at_least_one_limit(self):
        for value in (0,-1,True,float('nan'),float('inf'),'2'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.run_campaign(max_duration=value)
        with self.assertRaises(ValueError):self.run_campaign(cycles=None)
        with self.assertRaises(ValueError):self.run_campaign(stop_on_clear=1)
        self.assertFalse(self.calls)

    def test_duration_during_evaluation_finishes_and_counts_native_results(self):
        now=[0.];original=self.runner
        def advance_after_evaluation(command,*args):
            status=original(command,*args)
            if command[2]=='native_continuous':now[0]=11.
            return status
        self.runner=advance_after_evaluation
        result=self.run_campaign(cycles=None,max_duration=10,clock=lambda:now[0])
        self.assertEqual(result['status'],'budget_stopped')
        self.assertEqual(result['stop_reason'],'max_duration')
        self.assertEqual(len(result['cycles']),1)
        self.assertEqual(result['cycles'][0]['status'],'complete')
        self.assertEqual(len(result['cycles'][0]['attempts']),3)
        self.assertTrue(Path(result['latest_model']).is_file())

    def stopped_runner(self,reason='max_duration',supported=True,bad_checkpoint=False):
        manifest=json.loads((self.code/'build.json').read_text())
        if supported:manifest['cooperative_stop_protocol']=STOP_PROTOCOL
        write_json(self.code/'build.json',manifest)
        original=self.runner
        def stopped(command,*args):
            status=original(command,*args)
            if command[2]=='batch_train':
                request_stop(self.root/'output/stop-request.json',reason)
                target=Path(command[command.index('--output')+1])/'result.json'
                data=json.loads(target.read_text())
                data.update(actual_steps=4096,completed_update_steps=4096,budget_stop=True,
                    stop_reason=reason,cooperative_stop_protocol=STOP_PROTOCOL)
                data['final_checkpoint']['steps']=8192 if bad_checkpoint else 4096
                write_json(target,data)
            return status
        self.runner=stopped

    def test_cooperative_duration_stop_keeps_final_full_zip_without_false_evaluation(self):
        self.stopped_runner()
        result=self.run_campaign(max_duration=60)
        self.assertEqual(result['status'],'budget_stopped')
        self.assertEqual(result['cycles'][0]['training_steps'],4096)
        self.assertEqual(result['cycles'][0]['training_steps_requested'],409600)
        self.assertTrue(result['cycles'][0]['training_budget_stopped'])
        self.assertEqual(result['cycles'][0]['evaluation_skipped_reason'],'max_duration')
        self.assertNotIn('native_continuous',[c[2] for c in self.calls])
        self.assertTrue(Path(result['latest_model']).is_file())

    def test_duration_can_finish_with_one_final_evaluation_without_new_training(self):
        self.stopped_runner();self.outcomes=[True]
        result=self.run_campaign(max_duration=60,evaluate_on_stop=True)
        self.assertEqual(result['status'],'budget_stopped')
        self.assertTrue(result['any_clear'])
        self.assertTrue(result['cycles'][0]['final_evaluation_after_budget'])
        self.assertEqual([c[2] for c in self.calls],['initialize','batch_train','native_continuous'])
        self.assertEqual(len(result['cycles'][0]['attempts']),1)

    def test_cancel_does_not_start_extra_evaluation_even_when_enabled(self):
        self.stopped_runner(reason='user_cancelled')
        result=self.run_campaign(evaluate_on_stop=True)
        self.assertEqual(result['status'],'cancelled')
        self.assertNotIn('native_continuous',[c[2] for c in self.calls])

    def test_cooperative_user_cancellation_is_not_invalid(self):
        self.stopped_runner(reason='user_cancelled')
        result=self.run_campaign()
        self.assertEqual(result['status'],'cancelled')
        self.assertEqual(result['stop_reason'],'user_cancelled')
        self.assertEqual(result['cycles'][0]['training_steps'],4096)

    def test_shortened_training_requires_supported_protocol_and_exact_checkpoint(self):
        self.stopped_runner(supported=False)
        with self.assertRaisesRegex(RuntimeError,'unsupported training budget stop'):
            self.run_campaign(max_duration=60)

    def test_budget_stop_wrong_checkpoint_step_rejected(self):
        self.stopped_runner(bad_checkpoint=True)
        with self.assertRaisesRegex(RuntimeError,'checkpoint step mismatch'):
            self.run_campaign(max_duration=60)

    def test_real_child_deadline_uses_cooperative_file_without_termination(self):
        stop=self.root/'stop.json'
        env=dict(os.environ,ASTRA_RL_STOP_FILE=str(stop),ASTRA_CAMPAIGN_DEADLINE_MONOTONIC=str(time.monotonic()-1))
        code="import os,time;from pathlib import Path;p=Path(os.environ['ASTRA_RL_STOP_FILE']);deadline=time.monotonic()+3\nwhile not p.exists() and time.monotonic()<deadline:time.sleep(.005)\nassert p.exists()"
        result=run_child([sys.executable,'-c',code],self.root/'child.log',env,5)
        self.assertEqual(result,0)
        self.assertEqual(read_stop(stop),'max_duration')

    def test_ctrl_c_requests_safe_stop_and_waits_for_owned_child(self):
        child=Mock();child.wait.side_effect=[KeyboardInterrupt(),0];child.poll.return_value=0
        stop=self.root/'stop.json'
        with patch('experiments.rl.versioned_campaign.subprocess.Popen',return_value=child):
            result=run_child(['unused'],self.root/'ctrl-c.log',{'ASTRA_RL_STOP_FILE':str(stop)},5)
        self.assertEqual(result,0)
        self.assertEqual(read_stop(stop),'user_cancelled')
        child.terminate.assert_not_called()
        request_stop(stop,'max_duration')
        self.assertEqual(read_stop(stop),'user_cancelled')

    def failing_training(self,checkpoint=True,bad_path=None,bad_hash=False,empty_adam=False):
        original=self.runner
        def fail(command,log,*args):
            status=original(command,log,*args)
            if command[2]=='batch_train':
                target=Path(command[command.index('--output')+1]);result=json.loads((target/'result.json').read_text())
                result.update(status='invalid',error='EOFError: ',actual_steps=4096,completed_update_steps=4096)
                if checkpoint:
                    model=target/'checkpoint-000004096.zip'
                    if empty_adam:
                        with zipfile.ZipFile(model,'w') as archive:
                            archive.writestr('data','data');archive.writestr('policy.pth','policy');archive.writestr('policy.optimizer.pth','')
                    else:self.zip(model)
                    result['last_checkpoint']={'path':bad_path or model.name,'sha256':'0'*64 if bad_hash else digest(model),
                        'steps':4096,'optimizer_updates':4,'complete_update':True}
                write_json(target/'result.json',result)
                Path(log).write_text('Process SpawnProcess-13:\nRuntimeError: new round arrived before previous result was resolved\nEOFError: \nBrokenPipeError: broken pipe\n')
                return 1
            return status
        self.runner=fail

    def test_failed_training_reports_worker_cause_and_preserves_complete_update(self):
        self.failing_training()
        with self.assertRaisesRegex(RuntimeError,'new round arrived'):
            self.run_campaign()
        result=json.loads((self.root/'output/result.json').read_text())
        self.assertEqual(result['status'],'invalid')
        self.assertTrue(result['latest_model_from_incomplete_stage'])
        self.assertEqual(result['recovery_checkpoint']['steps'],4096)
        self.assertEqual(result['latest_model'],result['recovery_checkpoint']['path'])
        self.assertEqual(digest(result['latest_model']),result['latest_model_sha256'])
        self.assertEqual(result['cycles'][0]['status'],'invalid')
        self.assertEqual(result['cycles'][0]['failure_diagnostics']['exception_type'],'RuntimeError')
        self.assertNotIn('native_continuous',[c[2] for c in self.calls])

    def test_no_recovery_checkpoint_keeps_previous_full_model(self):
        self.failing_training(checkpoint=False)
        with self.assertRaisesRegex(RuntimeError,'new round arrived'):self.run_campaign()
        result=json.loads((self.root/'output/result.json').read_text())
        self.assertEqual(result['latest_model'],str(self.root/'output/initial/ppo-initial.zip'))
        self.assertNotIn('recovery_checkpoint',result)

    def test_escaping_recovery_path_is_rejected_without_overwriting_previous_model(self):
        self.failing_training(bad_path='../outside.zip')
        with self.assertRaisesRegex(RuntimeError,'recovery rejected'):self.run_campaign()
        result=json.loads((self.root/'output/result.json').read_text())
        self.assertNotIn('recovery_checkpoint',result)
        self.assertEqual(result['latest_model'],str(self.root/'output/initial/ppo-initial.zip'))
        self.assertIn('path must be relative',result['failure_diagnostics']['recovery_rejected'])

    def test_wrong_recovery_hash_is_rejected_without_hiding_worker_error(self):
        self.failing_training(bad_hash=True)
        with self.assertRaisesRegex(RuntimeError,'new round arrived'):self.run_campaign()
        result=json.loads((self.root/'output/result.json').read_text())
        self.assertNotIn('recovery_checkpoint',result)
        self.assertIn('SHA mismatch',result['failure_diagnostics']['recovery_rejected'])

    def test_empty_adam_cannot_be_a_recovery_checkpoint(self):
        self.failing_training(empty_adam=True)
        with self.assertRaisesRegex(RuntimeError,'empty model/Adam'):self.run_campaign()
        result=json.loads((self.root/'output/result.json').read_text())
        self.assertNotIn('recovery_checkpoint',result)

    def test_large_log_prefix_preserves_worker_cause_displaced_from_tail(self):
        log=self.root/'large.log'
        with log.open('w') as stream:
            stream.write('RuntimeError: original worker cause\n')
            stream.write('x'*(1024*1024+100))
            stream.write('\nEOFError: parent pipe closed\n')
        result=failure_diagnostics(log,'batch_train',1)
        self.assertEqual(result['message'],'original worker cause')
        self.assertEqual(result['selection'],'first_specific_exception_in_bounded_prefix')
        self.assertLessEqual(result['tail_bytes_read']+result['prefix_scan_bytes'],32*1024*1024)
        self.assertTrue(result['tail_truncated'])

    def test_exhausted_valid_losses_exit_result_not_invalid(self):
        self.outcomes=[False]*3;result=self.run_campaign();self.assertEqual(result['status'],'exhausted')
        self.assertEqual(sum(len(c['attempts']) for c in result['cycles']),9)

if __name__=='__main__':unittest.main()
