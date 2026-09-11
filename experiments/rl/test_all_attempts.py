"""Control-flow tests: retain clears/losses and never replace invalid attempts."""
from contextlib import ExitStack, nullcontext
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from . import continuous, native_continuous


class AllAttemptsTests(unittest.TestCase):
    def evaluate(self, outcomes, attempts=20, all_attempts=True):
        context=tempfile.TemporaryDirectory();self.addCleanup(context.cleanup)
        output=Path(context.name)/'run'
        calls=[];process=Mock();process.poll.return_value=None
        bridge=Mock();bridge.send.return_value={}
        def attempt(run, bridge, row, speed, save):
            calls.append(row['id'])
            outcome=outcomes[len(calls)-1]
            if isinstance(outcome,Exception):raise outcome
            row.update(outcome=outcome,match_wins=11 if outcome=='rl_gameplay_clear' else 0)
        with ExitStack() as stack:
            for name,value in {'doctor':{'ok':True},'export_policy':{'model_sha256':'same-model'},
                'stage_policy':{},'boot_config':{},'mame_command':['mame'],'Bridge':bridge,
                'require_difficulty':None,'audit_attempt':None,'seal_run':None}.items():
                stack.enter_context(patch.object(continuous,name,return_value=value))
            stack.enter_context(patch.object(continuous,'session_lock',return_value=nullcontext()))
            stack.enter_context(patch.object(continuous,'_attempt',side_effect=attempt))
            spawn=stack.enter_context(patch.object(continuous.subprocess,'Popen',return_value=process))
            kwargs={'stop_on_first_clear':False} if all_attempts else {}
            result=continuous.evaluate({},Path('frozen.zip'),output,attempts=attempts,**kwargs)
        self.assertEqual(spawn.call_count,1)  # Every attempt shares one native process.
        self.assertEqual(process.terminate.call_count,1)
        return result,calls

    def test_default_still_stops_on_first_clear(self):
        result,calls=self.evaluate(['loss','rl_gameplay_clear'],all_attempts=False)
        self.assertEqual(len(calls),2);self.assertTrue(result['stop_on_first_clear'])
        self.assertEqual(result['status'],'complete')

    def test_ten_early_clears_do_not_stop_twenty_attempts(self):
        result,calls=self.evaluate(['rl_gameplay_clear']*10+['loss']*10)
        self.assertEqual(calls,[f'l3-{i:03d}' for i in range(1,21)])
        self.assertFalse(result['stop_on_first_clear']);self.assertEqual(result['attempts_requested'],20)
        self.assertEqual(result['status'],'complete')
        self.assertEqual(sum(a['outcome']=='rl_gameplay_clear' for a in result['attempts']),10)
        self.assertEqual(sum(a['outcome']=='loss' for a in result['attempts']),10)

    def test_invalid_after_clear_is_retained_and_never_replaced(self):
        result,calls=self.evaluate(['rl_gameplay_clear',RuntimeError('native coin gate failed')])
        self.assertEqual(len(calls),2);self.assertEqual(result['status'],'invalid')
        self.assertEqual([a['outcome'] for a in result['attempts']],['rl_gameplay_clear','invalid'])
        self.assertIn('native coin gate failed',result['error'])

    def test_both_clis_forward_explicit_all_attempts(self):
        for module in (continuous,native_continuous):
            with self.subTest(module=module.__name__),patch('sys.argv',['cli','--model','x.zip','--output','out','--attempts','20','--all-attempts']),patch.object(module,'load_config',return_value={}),patch.object(module.signal,'signal'),patch.object(module,'evaluate',return_value={'status':'complete','attempts':[{'outcome':'loss'}]}) as evaluate:
                with self.assertRaises(SystemExit) as error:module.main()
                self.assertEqual(error.exception.code,1)
                self.assertIs(evaluate.call_args.kwargs['stop_on_first_clear'],False)
