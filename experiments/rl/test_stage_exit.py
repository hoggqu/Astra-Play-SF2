"""Real harmless process exit races and cleanup. No MAME, PPO or game state."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from .reliability_pipeline import OwnedStage, live_group_members
from .stage_exit import finish_owned_stage, group_snapshot

@unittest.skipUnless(os.name=='posix','Owned process groups require POSIX')
class StageExitTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);(self.root/'src').mkdir();self.index=0
    def spawn(self, source):
        self.index+=1;name='child'+str(self.index);(self.root/(name+'.py')).write_text(source)
        stage=OwnedStage(self.root,name,[],self.root/(name+'.log'),10)
        self.addCleanup(lambda: stage.stop('test_finalizer',grace=.2))
        return stage
    def exited(self, stage):
        until=time.monotonic()+5
        while stage.poll() is None and time.monotonic()<until:time.sleep(.002)
        self.assertIsNotNone(stage.poll())
    def test_short_lived_owned_descendant_drains_without_signal(self):
        stage=self.spawn("import subprocess,sys\nsubprocess.Popen([sys.executable,'-c','import time;time.sleep(.3)'])\n")
        self.exited(stage);self.assertTrue(live_group_members(stage.pgid))
        self.assertEqual(finish_owned_stage(stage,grace=2),0)
        audit=stage.exit_audit;self.assertEqual(audit['status'],'drained');self.assertFalse(audit['signals_sent_by_finish'])
        self.assertTrue(any(r['members'] for r in audit['observed']));self.assertFalse(live_group_members(stage.pgid))
    def test_persistent_descendant_stays_invalid_and_unrelated_survives(self):
        unrelated=subprocess.Popen([sys.executable,'-c','import time;time.sleep(10)'],start_new_session=True)
        self.addCleanup(lambda:(unrelated.terminate(),unrelated.wait()))
        stage=self.spawn("import subprocess,sys\nsubprocess.Popen([sys.executable,'-c','import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(30)'])\n")
        self.exited(stage)
        with self.assertRaisesRegex(RuntimeError,'persistent owned descendants'):finish_owned_stage(stage,grace=.15)
        self.assertEqual(stage.exit_audit['status'],'invalid_survivors')
        self.assertTrue(stage.exit_audit['survivors_before_cleanup']);self.assertEqual(stage.exit_audit['final_live_pids'],[])
        self.assertIsNone(unrelated.poll())
        saved=json.loads(Path(str(stage.log.name)+'.exit-audit.json').read_text());self.assertIn('cleanup',saved)
    def test_real_resource_tracker_eof_exit_is_allowed_only_after_group_empty(self):
        stage=self.spawn("from multiprocessing import resource_tracker\nimport os\nresource_tracker.ensure_running()\nprint(resource_tracker._resource_tracker._pid,flush=True)\nos._exit(0)\n")
        self.exited(stage);self.assertEqual(finish_owned_stage(stage),0)
        self.assertFalse(live_group_members(stage.pgid));self.assertEqual(stage.exit_audit['status'],'drained')
        self.assertTrue(Path(stage.log.name).read_text().strip().isdigit())
    def test_nonzero_parent_exit_is_not_promoted_to_success(self):
        stage=self.spawn('raise SystemExit(7)\n');self.exited(stage)
        self.assertEqual(finish_owned_stage(stage),7)
    def test_running_parent_not_finished_or_signalled(self):
        stage=self.spawn('import time;time.sleep(10)\n')
        with self.assertRaisesRegex(RuntimeError,'running stage'):finish_owned_stage(stage)
        self.assertIsNone(stage.poll())
    def test_reject_orchestrator_group(self):
        with self.assertRaisesRegex(RuntimeError,'orchestrator'):group_snapshot(os.getpgrp())

if __name__=='__main__':unittest.main()
