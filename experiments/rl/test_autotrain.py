import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from .autotrain import parser, validate_args, effective_budgets, training_environment


class AutotrainTests(unittest.TestCase):
    def args(self, folder, *options):
        manifest=Path(folder)/'manifest.json';manifest.write_text('{}')
        return parser().parse_args(['--dataset',str(manifest),*options])

    def test_budget_required_and_worker_update_multiples(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError,'Provide'):validate_args(self.args(folder))
            for opts in (['--rounds','0'],['--rounds','1','--steps-per-round','0'],['--rounds','1','--checkpoint-every','0']):
                with self.assertRaises(ValueError):validate_args(self.args(folder,*opts))
            args=self.args(folder,'--hours','2','--rounds','5','--workers','16')
            validate_args(args);self.assertFalse(args.stop_on_clear);self.assertEqual(args.device,'cpu')

    def test_linux_training_selects_dummy_backend_without_global_mutation(self):
        import os
        with patch('sys.platform','linux'), patch.dict(os.environ,{'SDL_VIDEODRIVER':'wayland'}):
            env=training_environment('cuda')
            self.assertEqual(env['SDL_VIDEODRIVER'],'dummy')
            self.assertEqual(env['ASTRA_RL_DEVICE'],'cuda')
            self.assertEqual(env['ASTRA_RL_OPPONENT_SAMPLING'],'adaptive')
            self.assertEqual(training_environment('cpu','uniform')['ASTRA_RL_OPPONENT_SAMPLING'],'uniform')
            self.assertEqual(os.environ['SDL_VIDEODRIVER'],'wayland')
        with patch('sys.platform','darwin'), patch.dict(os.environ,{'SDL_VIDEODRIVER':'cocoa'}):
            self.assertEqual(training_environment('cpu')['SDL_VIDEODRIVER'],'dummy')
            self.assertEqual(os.environ['SDL_VIDEODRIVER'],'cocoa')
        with patch('sys.platform','win32'):
            self.assertNotIn('SDL_VIDEODRIVER',training_environment('cpu'))

    def test_custom_workers_align_budgets_without_changing_request(self):
        with tempfile.TemporaryDirectory() as folder:
            for workers in (1,3,6,12,20,24,32):
                args=self.args(folder,'--rounds','1','--workers',str(workers))
                validate_args(args)
                steps, checkpoint=effective_budgets(args)
                for actual,requested in ((steps,409600),(checkpoint,4096)):
                    self.assertEqual(actual%args.rollout_steps,0)
                    self.assertGreaterEqual(actual,requested)
                    self.assertLess(actual-requested,args.rollout_steps)
                self.assertEqual(args.steps_per_round,409600)
                self.assertEqual((steps,checkpoint),(409600,4096))
            for invalid in ('0','-1','1.5'):
                with self.assertRaises(SystemExit):self.args(folder,'--rounds','1','--workers',invalid)

    def test_resume_requires_new_output_and_model_is_exclusive(self):
        with tempfile.TemporaryDirectory() as folder:
            args=self.args(folder,'--rounds','1','--resume',folder,'--output',folder)
            with self.assertRaisesRegex(ValueError,'Output already exists'):validate_args(args)
            with self.assertRaises(SystemExit):self.args(folder,'--rounds','1','--init-model','a.zip','--resume',folder)


if __name__=='__main__':unittest.main()
