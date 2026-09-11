"""Bounded post-exit drain, retaining owned descendant identities; no process allowlist."""
import json
import os
from pathlib import Path
import subprocess
import time


def group_snapshot(pgid):
    if pgid == os.getpgrp():
        raise RuntimeError('Refusing the orchestrator process group')
    output = subprocess.check_output(['ps', '-axo', 'pid=,ppid=,pgid=,stat=,args='], text=True)
    rows = []
    for line in output.splitlines():
        parts = line.strip().split(None, 4)
        if len(parts) == 5 and int(parts[2]) == pgid and not parts[3].startswith('Z'):
            rows.append({'pid': int(parts[0]), 'ppid': int(parts[1]), 'pgid': int(parts[2]),
                         'state': parts[3], 'command': parts[4]})
    return rows


def finish_owned_stage(stage, grace=2.0, interval=.02):
    """An exited parent may precede normal helper EOF exit by a short interval.

    Accept only an empty live process group; a survivor after the bounded drain
    is still invalid and receives the existing owned-group cleanup. Nonzero
    parent exit codes are returned unchanged for the caller's result checks.
    """
    if not 0 <= grace <= 5 or not 0 < interval <= .1:
        raise ValueError('Drain grace must be 0..5 seconds and polling 0..0.1')
    code = stage.process.poll()
    if code is None:
        raise RuntimeError('Cannot finish a running stage')
    start = time.monotonic()
    record = {'schema': 'astra.rl-owned-stage-exit.v1', 'status': 'draining',
              'pid': stage.pid, 'pgid': stage.pgid, 'parent_returncode': code,
              'grace_seconds': grace, 'allowlisted_processes': False, 'observed': []}
    path = Path(str(stage.log.name)+'.exit-audit.json')
    def save():
        stage.exit_audit = record
        temporary = path.with_suffix(path.suffix+'.tmp')
        temporary.write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
        temporary.replace(path)
    previous = None
    try:
        while True:
            rows = group_snapshot(stage.pgid)
            if rows != previous:
                record['observed'].append({'elapsed_seconds': time.monotonic()-start, 'members': rows})
                previous = rows
                save()
            if not rows:
                record.update(status='drained', waited_seconds=time.monotonic()-start,
                              final_live_pids=[], signals_sent_by_finish=False)
                save(); stage.log.close()
                return code
            if time.monotonic()-start >= grace:
                break
            time.sleep(interval)
        record.update(status='invalid_survivors', waited_seconds=time.monotonic()-start,
                      survivors_before_cleanup=rows, signals_sent_by_finish=True)
        save()
        try:
            record['cleanup'] = stage.stop('live_descendants_after_bounded_exit_drain')
        except BaseException as error:
            record['cleanup_error'] = f'{type(error).__name__}: {error}'
            save(); raise
        record['final_live_pids'] = [r['pid'] for r in group_snapshot(stage.pgid)]
        save()
        raise RuntimeError('Stage exited with persistent owned descendants; cleaned and retained as invalid; '+str(path))
    except BaseException:
        stage.exit_audit = record
        raise


def run_isolated(code, module, arguments, log_path, timeout):
    """Opt-in replacement for a new isolated serial flow; never patches old ones."""
    from .reliability_pipeline import OwnedStage, live_group_members
    stage = OwnedStage(code, module, arguments, log_path, timeout)
    try:
        while stage.poll() is None:
            time.sleep(.05)
        return finish_owned_stage(stage)
    finally:
        if stage.process.poll() is None or live_group_members(stage.pgid):
            stage.stop('owned_serial_stage_finally')
        else:
            stage.log.close()
