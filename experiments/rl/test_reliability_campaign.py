"""Fixed-budget reliability orchestration fixtures; no emulator processes."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from astra_play_sf2.config import atomic_json
from astra_play_sf2.runner import sha256
from .reliability_campaign import campaign, audit_full_twenty, execution_sources, run_isolated, INTERFACE


def verification(clears,model_hash):
    attempts=[]
    for i in range(1,21):
        clear=i<=clears
        attempts.append({'id':f'l3-{i:03d}','outcome':'rl_gameplay_clear' if clear else 'loss',
            'match_wins':11 if clear else 0,'matches':[f'training/l3-{i:03d}/m{j:02d}.json' for j in range(1,12 if clear else 2)],
            'audit':{'ok':True},'images':{'ending':f'ending-{i}.png'} if clear else {}})
    matches=sum(len(a['matches']) for a in attempts)
    return {'schema':'astra.rl-continuous.actions16.v1','status':'complete','difficulty':3,
        'model_sha256':model_hash,'action_interface':INTERFACE,'actions':16,
        'native_timing':True,'stop_on_first_clear':False,'attempts_requested':20,'attempts':attempts,
        'native_timing_audit':{'ok':True,'matches':matches},
        'action_interface_audit':{'ok':True,'checked_matches':matches}}


class ReliabilityTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.model=self.root/'initial.zip';self.model.write_bytes(b'initial')
        self.dataset=self.root/'dataset.json';atomic_json(self.dataset,{})
        self.source=self.root/'source.py';self.source.write_text('frozen')
        self.calls=[];self.scores=[9,10];self.verifies=0;self.mutation=None

    def stage(self,code,module,args,log,timeout):
        args=list(map(str,args));self.calls.append((code,module,args))
        log.write_text('retained stage log')
        folder=Path(args[args.index('--output')+1]);folder.mkdir()
        if self.mutation=='interrupt':raise KeyboardInterrupt('owner cancelled')
        if module.endswith('batch_train'):
            model=folder/'ppo-batch.zip';model.write_bytes(b'updated')
            initial=Path(args[args.index('--init-model')+1])
            atomic_json(folder/'result.json',{'schema':'astra.rl-batch-prototype.actions16.v1','status':'complete',
                'actual_steps':2048,'difficulty':3,'benchmark':False,'parity':False,'native_parity':False,
                'parameters_changed':True,'init_model_sha256':sha256(initial),'dataset_sha256':sha256(self.dataset),
                'model_sha256':sha256(model),'actions':16,'action_interface':INTERFACE,
                'optimizer_initialization':{'loaded_from_init_model':True},
                'opponent_sampling':{'identity':'fixed_weights'}})
            return 0
        self.assertIn('--all-attempts',args);self.assertEqual(args[args.index('--attempts')+1],'20')
        model=Path(args[args.index('--model')+1]);clears=self.scores[min(self.verifies,len(self.scores)-1)];self.verifies+=1
        result=verification(clears,sha256(model))
        if self.mutation=='short':result['attempts']=result['attempts'][:10]
        if self.mutation=='source':self.source.write_text('changed')
        if self.mutation=='model':model.write_bytes(b'changed')
        atomic_json(folder/'result.json',result)
        return 0 if clears else 1

    def run_campaign(self,cycles=3):
        with patch('experiments.rl.reliability_campaign.load_dataset',return_value=({},3)), \
             patch('experiments.rl.reliability_campaign.execution_sources',return_value={'source.py':self.source}):
            return campaign(self.dataset,self.root/'run',self.model,self.root/'train-code','train_package',
                self.root/'verify-code','verify_package',cycles=cycles,steps_per_cycle=2048,
                initial_budget_steps=409600,stage_runner=self.stage)

    def test_initial_full20_before_training_then_complete10_of20_stops(self):
        result=self.run_campaign()
        self.assertTrue(result['goal_achieved']);self.assertEqual(result['success_cycle'],2)
        self.assertEqual([m for _,m,_ in self.calls],['verify_package.native_continuous','train_package.batch_train','verify_package.native_continuous'])
        self.assertEqual([c['clears'] for c in result['cycles']],[9,10])
        self.assertEqual(result['completed_new_training_steps'],2048)
        self.assertEqual(result['reported_initial_budget_steps'],409600)
        self.assertEqual(result['cycles'][1]['opponent_sampling'],{'identity':'fixed_weights'})
        self.assertEqual(self.calls[0][0],self.root/'verify-code')
        self.assertEqual(self.calls[1][0],self.root/'train-code')
        args=self.calls[1][2];self.assertEqual(args[args.index('--init-model')+1],str(self.model))

    def test_two_nine_clear_candidates_never_pool_into_success(self):
        self.scores=[9,9];result=self.run_campaign(cycles=2)
        self.assertEqual(result['status'],'complete');self.assertFalse(result['goal_achieved'])
        self.assertEqual(result['completed_candidate_evaluations'],2)
        self.assertEqual([c['clears'] for c in result['cycles']],[9,9])

    def test_first_candidate_ten_clears_skips_further_training(self):
        self.scores=[10];result=self.run_campaign()
        self.assertTrue(result['goal_achieved']);self.assertEqual(len(self.calls),1)
        self.assertEqual(result['completed_new_training_steps'],0)

    def test_incomplete_twenty_or_mutated_inputs_stop_without_replay(self):
        for mutation in ('short','source','model','interrupt'):
            with self.subTest(mutation=mutation):
                self.mutation=mutation
                result=self.run_campaign()
                self.assertEqual(result['status'],'invalid');self.assertFalse(result['goal_achieved'])
                self.assertEqual(len(self.calls),1)
                self.assertTrue((self.root/'run/result.json').is_file())
                # New independent fixture, never overwrite a retained invalid output.
                self.setUp()

    def test_full20_audit_rejects_shortcuts_and_false_success_exit(self):
        cases=('early_stop','bad_native','bad_interface','duplicate_id','duplicate_match','exit','no_full_win','missing_coverage','wrong_mode')
        for case in cases:
            with self.subTest(case=case):
                d=verification(10,'hash');code=0
                if case=='early_stop':d['stop_on_first_clear']=True
                elif case=='bad_native':d['native_timing_audit']['ok']=False
                elif case=='bad_interface':d['action_interface_audit']['ok']=False
                elif case=='duplicate_id':d['attempts'][1]['id']=d['attempts'][0]['id']
                elif case=='duplicate_match':d['attempts'][1]['matches']=d['attempts'][0]['matches']
                elif case=='exit':code=1
                elif case=='no_full_win':d['attempts'][0]['match_wins']=10
                elif case=='missing_coverage':d['native_timing_audit']['matches']=0
                elif case=='wrong_mode':d['selection']='categorical_softmax'
                with self.assertRaises(RuntimeError):audit_full_twenty(d,'hash',code)

    def test_execution_freezer_requires_manifest_and_copied_production(self):
        code=self.root/'isolated';package=code/'candidate';package.mkdir(parents=True)
        source=package/'__init__.py';source.write_text('')
        atomic_json(code/'build.json',{'package':'candidate','action_interface':INTERFACE,'actions':16,
            'observations':344,'derived_sha256':{'__init__.py':sha256(source)}})
        with self.assertRaisesRegex(ValueError,'production src'):execution_sources(code,'candidate')
        production=code/'src/astra_play_sf2';production.mkdir(parents=True)
        (production/'__init__.py').write_text('')
        paths=execution_sources(code,'candidate')
        self.assertIn('src/astra_play_sf2/__init__.py',paths)
        self.assertIn('build.json',paths)
        source.write_text('changed')
        with self.assertRaisesRegex(ValueError,'differs'):execution_sources(code,'candidate')

    def test_subprocess_uses_own_code_src_and_single_thread_environment(self):
        import os
        code=self.root/'code';code.mkdir()
        (code/'probe.py').write_text("import os,json;print(json.dumps({'cwd':os.getcwd(),'path':os.environ['PYTHONPATH'],'omp':os.environ['OMP_NUM_THREADS']}))")
        log=self.root/'probe.log'
        with patch.dict(os.environ,{'PYTHONPATH':'unrelated-code','OMP_NUM_THREADS':'99'}):
            exit_code=run_isolated(code,'probe',[],log,10)
        self.assertEqual(exit_code,0)
        value=json.loads(log.read_text())
        self.assertEqual(value,{'cwd':str(code),'path':str(code/'src')+os.pathsep+str(code),'omp':'1'})


if __name__=='__main__':unittest.main()
