"""Native-time learned-policy evaluation; legacy continuous entry is unchanged."""
import argparse
from pathlib import Path
import json
import signal
import math
from unittest.mock import patch

from astra_play_sf2.config import atomic_json, load_config
from astra_play_sf2.runner import seal_run, sha256
from astra_play_sf2.transport import read_json
from . import continuous

HERE = Path(__file__).resolve().parent
ORIGINAL_STAGE = continuous.stage_policy


def stage_native_policy(run, level, payload):
    ORIGINAL_STAGE(run, level, payload)
    runtime = run/'training/runtime'
    (runtime/'rl_settlement.lua').write_bytes((HERE/'settlement.lua').read_bytes())
    (runtime/'rl_continuous_core.lua').write_bytes((HERE/'native_continuous_core.lua').read_bytes())
    path = runtime/'play.lua'
    source = path.read_text()
    source = continuous.replace_once(source, "policy_kind='ppo',model_sha256=Model.model_sha256",
                                     "policy_kind='ppo',model_sha256=Model.model_sha256,native_timing=true,native_frame_period=r.native_frame_period,native_period_checks=r.native_period_checks")
    source = continuous.replace_once(source, 'local initial=r.core:rl_prime(opening)',
                                     'r.native_last_time=opening.emulated_seconds;r.native_frame_period=opening.native_frame_period;r.native_period_checks=0\n  assert(type(r.native_frame_period)=="number" and r.native_frame_period>0)\n  local initial=r.core:rl_prime(opening)')
    source = continuous.replace_once(source, 'Speed.check(m.video,r.speed);r.speed_checks=r.speed_checks+1',
                                     'local now=m.time:as_double()\n  if now==r.native_last_time then return end\n  assert(math.abs(now-r.native_last_time-r.native_frame_period)<1e-7,"Skipped native frame")\n  assert(m.screens[":screen"].frame_period==r.native_frame_period,"Native screen period changed")\n  r.native_period_checks=r.native_period_checks+1\n  r.native_last_time=now\n  Speed.check(m.video,r.speed);r.speed_checks=r.speed_checks+1')
    source = continuous.replace_once(source, 's.emulated_seconds=m.time:as_double()',
                                     's.emulated_seconds=m.time:as_double();s.native_frame_period=m.screens[":screen"].frame_period')
    source = continuous.replace_once(source, 'local effects=r.core:tick(r.latest)',
                                     'local effects=r.core:tick(r.latest)\n  r.rl_deferred_input=effects.rl_deferred_input')
    source += '''
-- The first macro input crosses the same input-update boundary as batch sampling.
emu.register_frame_done(function()
 if not active or active.rl_deferred_input==nil then return end
 local r=active;local text=r.rl_deferred_input;r.rl_deferred_input=nil
 local ok,err=pcall(function()
  r.release()
  for key in text:gmatch('%S+') do assert(key~='C' and key~='S' and r.keys[key]):set_value(1) end
  r.recorder.last_input=text
  if r.recorder.row then r.recorder.row[25]=text end
 end)
 if not ok then finish({valid=false,reason='deferred native input: '..tostring(err)}) end
end)
'''
    path.write_text(source)
    return {p.name: sha256(p) for p in sorted(runtime.iterdir()) if p.is_file()}


def audit_native_match(match):
    summary=match['summary']
    if summary.get('native_timing') is not True:
        raise RuntimeError('Match does not attest native timing')
    trace=match['trace'];columns=trace['columns'];rows=trace['rows']
    frame_index=columns.index('frame');time_index=columns.index('emulated_seconds')
    if len(rows)!=summary['frame'] or [row[frame_index] for row in rows]!=list(range(1,len(rows)+1)):
        raise RuntimeError('Incomplete native frame sequence')
    times=[row[time_index] for row in rows]
    if len(times)<2 or any(type(t) not in (int,float) or not math.isfinite(t) for t in times):
        raise RuntimeError('Missing finite native times')
    deltas=[b-a for a,b in zip(times,times[1:])]
    if any(dt<=0 for dt in deltas):
        raise RuntimeError('Native times are not strictly increasing')
    period=summary.get('native_frame_period')
    if type(period) not in (int,float) or not math.isfinite(period) or period<=0 or summary.get('native_period_checks')!=len(rows):
        raise RuntimeError('Missing independently observed screen frame period')
    if any(abs(dt-period)>1e-7 for dt in deltas):
        raise RuntimeError('Native frame gaps are inconsistent')
    starts=[event for event in match['events'] if event['kind']=='match_start']
    if len(starts)!=1:
        raise RuntimeError('Missing native match opening')
    if abs(times[0]-starts[0]['state']['emulated_seconds']-period)>1e-7:
        raise RuntimeError('First observed frame is not one native period after opening')
    frame_times={row[frame_index]:row[time_index] for row in rows}
    frame_times[0]=starts[0]['state']['emulated_seconds']
    decisions=trace['decisions']
    openings={1:0};stops={}
    for event in match['events']:
        if event['kind']=='round_start':
            if event['round'] in openings: raise RuntimeError('Duplicate round opening')
            openings[event['round']]=event['frame']
        elif event['kind']=='round_stop':
            if event['round'] in stops: raise RuntimeError('Duplicate round stop')
            stops[event['round']]=event['frame']
    if not decisions or set(openings)!=set(stops) or {d['round'] for d in decisions}!=set(openings):
        raise RuntimeError('Missing round/decision coverage')
    for round_number,start in openings.items():
        if not 0<=start<stops[round_number]<=summary['frame']:
            raise RuntimeError('Invalid native round boundaries')
        expected=list(range(start,stops[round_number],12))
        actual=[d['frame'] for d in decisions if d['round']==round_number]
        if not expected or actual!=expected:
            raise RuntimeError('Incomplete twelve-frame fighting decision coverage')
    for before,after in zip(decisions,decisions[1:]):
        if before['round']!=after['round']:
            continue
        if after['frame']-before['frame']!=12:
            raise RuntimeError('Fighting decisions are not twelve native frames apart')
        elapsed=frame_times[after['frame']]-frame_times[before['frame']]
        if abs(elapsed-12*period)>1e-7:
            raise RuntimeError('Fighting decision native-time interval is not twelve frames')
    return {'ok':True,'frames':len(rows),'decisions':len(decisions),'native_period_seconds':period}


def evaluate(*args, **kwargs):
    with patch.object(continuous, 'stage_policy', stage_native_policy):
        result = continuous.evaluate(*args, **kwargs)
    output = Path(kwargs.get('output', args[2] if len(args)>2 else '')).resolve()
    result['native_timing'] = True
    result['native_timing_audit'] = {'ok': False, 'reason': 'Run did not complete'}
    if result['status']=='complete':
        try:
            audited=[]
            for attempt in result['attempts']:
                for relative in attempt['matches']:
                    audited.append(audit_native_match(read_json(output/relative)))
            result['native_timing_audit']={'ok':True,'matches':len(audited),'frames':sum(row['frames'] for row in audited),'decisions':sum(row['decisions'] for row in audited)}
        except Exception as error:
            result.update(status='invalid',error=f'Native timing audit: {type(error).__name__}: {error}')
            result['native_timing_audit']={'ok':False,'reason':str(error)}
    result['native_timing_contract'] = '12 advancing emulated frames; action0 at frame_done; no combat pauses'
    atomic_json(output/'result.json', result)
    seal_run(output)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--difficulty', type=int, choices=range(3,8), default=3)
    parser.add_argument('--attempts', type=int, default=1)
    parser.add_argument('--speed', choices=('normal','2x','4x','fast'), default='fast')
    parser.add_argument('--show-window', action='store_true')
    parser.add_argument('--all-attempts', action='store_true', help='Complete all requested natural-coin attempts, including after a clear; stop on invalid execution')
    args = parser.parse_args()
    def stop(_signal, _frame):
        raise KeyboardInterrupt('SIGTERM')
    signal.signal(signal.SIGTERM, stop)
    result = evaluate(load_config(), args.model, args.output, args.difficulty, args.attempts, args.speed, args.show_window, stop_on_first_clear=not args.all_attempts)
    print(json.dumps(result, indent=2), flush=True)
    raise SystemExit(2 if result['status']!='complete' else 0 if any(a['outcome']=='rl_gameplay_clear' for a in result['attempts']) else 1)


if __name__ == '__main__':
    main()
