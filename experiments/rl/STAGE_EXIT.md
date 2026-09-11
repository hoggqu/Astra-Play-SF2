# Bounded owned-process exit drain

A completed training job can return zero before every helper in its process
group has finished reacting to closed pipes. The earlier
`OwnedStage.finish()` checked the group once, immediately cleaned any surviving
process, and marked the orchestration invalid. Its first cleanup return value
was discarded, so the subsequent exception-handler cleanup could only record
an already-empty group.

The observed affected stage completed its full 1,638,400-decision budget and
saved an independently verified model. The outer flow became invalid before
starting its first formal evaluation. Retain both records: completed training
is usable as a new candidate, while the old flow remains invalid. The old logs
lack the identities of the first observed descendants; they do not establish
that those processes were only Python's `resource_tracker`. A tracker exit race
is a plausible explanation, not a verified diagnosis of that particular run.

`stage_exit.finish_owned_stage` is an **opt-in helper for new isolated flows**.
It uses POSIX process groups and `ps` on Linux/macOS; it is not a Windows process supervisor.
It waits at most two seconds for an already-exited parent's owned group to drain
naturally, excluding zombies. It accepts completion only when no live members
remain, preserves the parent's actual exit code, and does not allowlist process
names. Persistent descendants still trigger the existing owned-group cleanup
and an invalid result. A sidecar `<stage-log>.exit-audit.json` records observed
PID, PPID, PGID, state and command, the wait, and any cleanup result before the
exception can discard that evidence.

For a future copied pipeline, replace only its `finish()` implementation with a
call to this helper and add `stage_exit.py` to the orchestrator source pins. For
a new serial flow, the helper also exposes the existing `run_isolated` signature
using an owned stage. Repin the new snapshot before launch. Existing frozen
flows are not modified. The current diverse-development flow uses the previous
serial `run_isolated` and does not call the immediate-failure `finish()` method;
this helper is not required to unblock that flow.

Six offline tests use real harmless Python subprocesses, including a real
multiprocessing resource tracker, a short-lived orphan, a persistent orphan
that ignores SIGTERM, and an unrelated process. They verify natural draining,
retained invalid cleanup, untouched unrelated processes, unchanged nonzero exit
codes, and refusal to finish a live parent or signal the orchestrator group.
No MAME process, model training or game state is used by these tests.
