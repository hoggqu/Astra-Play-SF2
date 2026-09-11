"""Gymnasium adapter around one owned MAME process, training only."""
from collections import deque
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import shutil
import multiprocessing
from multiprocessing.util import Finalize
import signal
import time

import gymnasium as gym
import numpy as np

from astra_play_sf2.config import doctor
from astra_play_sf2.opening import make_opening_guard
from astra_play_sf2.runner import boot_config, stage_runtime, mame_command, sha256
from astra_play_sf2.transport import Bridge, read_json

ACTION_NAMES = ('neutral', 'forward', 'back', 'crouch_guard', 'jump_forward',
                'jump_back', 'jab', 'fierce', 'crouch_jab', 'sweep', 'medium_kick',
                'heavy_kick', 'fireball', 'uppercut', 'jump_heavy_kick')
TRAIN_LEADS = (0, 4, 8, 12)
EVAL_LEADS = (2, 6, 10)


def features(state):
    a, b = state['p1'], state['p2']
    values = [a['hp']/144, b['hp']/144, state['timer']/99,
              (b['x']-a['x'])/512, a['x']/1024, b['x']/1024,
              (a['y']-40)/256, (b['y']-40)/256, float(a['y']==40), float(b['y']==40)]
    values += [float(b['char']==i) for i in range(12)]
    for p in (a, b):
        values += [float(p['a']==i) for i in range(32)]
    return np.clip(np.asarray(values, dtype=np.float32), -1, 1)


def reward(before, after, outcome=None):
    # Round ending dominates; no reward for elapsed time or choosing a move.
    hp = lambda s, key: max(0, min(144, s[key]['hp']))
    damage = (hp(before, 'p2')-hp(after, 'p2')) - (hp(before, 'p1')-hp(after, 'p1'))
    return float(0.25*damage/144 + {'win': 1, 'loss': -1, 'draw': 0, None: 0}[outcome])


class MameEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, config, run, difficulty=7, show_window=False, checkpoints=None):
        super().__init__()
        self.run = Path(run).resolve()
        self.action_space = gym.spaces.Discrete(len(ACTION_NAMES))
        self.observation_space = gym.spaces.Box(-1, 1, shape=(4*86,), dtype=np.float32)
        self.history = deque(maxlen=4)
        self.process = None
        self.log = None
        self.index = 0
        self.closed = False
        self.must_reset = True
        self.rpc_seconds = 0.0
        self.total_frames = 0
        self.phase = 'train'
        self.baseline = False
        self.episodes = []
        # SB3 workers are daemon processes. Register cleanup before preflight or
        # MAME startup so a failed sibling cannot leave an orphan simulator.
        if multiprocessing.current_process().daemon:
            Finalize(None, self.close, exitpriority=10)
            def terminate_worker(_signum, _frame):
                try:
                    self.close()
                finally:
                    raise SystemExit(1)
            signal.signal(signal.SIGTERM, terminate_worker)
        self.difficulty = difficulty
        self.checkpoints = list(checkpoints or [])
        self.checkpoint_groups = {}
        for i, sample in enumerate(self.checkpoints):
            if sample['opponent'] not in (0,1,2,3,5,6,7,8,9,10,11):
                raise ValueError('Invalid checkpoint opponent')
            self.checkpoint_groups.setdefault(sample['opponent'], []).append(i)
        if not 3 <= difficulty <= 7:
            raise ValueError('Difficulty must be 3..7')
        preflight = doctor(config)
        if not preflight['ok']:
            raise RuntimeError(json.dumps(preflight))
        self.run.mkdir(parents=True, exist_ok=False)
        runtime = stage_runtime(self.run, difficulty)
        boot_config(self.run, difficulty)
        source = Path(__file__).parent
        for original, name in [('runtime.lua', 'rl.lua'), ('actions.lua', 'rl_actions.lua')]:
            (self.run/'training/runtime'/name).write_bytes((source/original).read_bytes())
        paths = []
        for index, sample in enumerate(self.checkpoints):
            original = Path(sample['path'])
            if sha256(original) != sample['sha256'] or sample['difficulty'] != difficulty:
                raise ValueError('Checkpoint hash/difficulty mismatch')
            target = self.run/f'training/rl-start-{index:03d}.sta'
            shutil.copyfile(original, target)
            paths.append(target)
        if not paths:
            paths = [self.run/'training/rl-start.sta']
        expression = '{'+','.join(json.dumps(path.as_posix(), ensure_ascii=False) for path in paths)+'}'
        (self.run/'training/runtime/rl_checkpoint.lua').write_text('return '+expression+'\n', encoding='utf-8')
        runtime.update({name: sha256(self.run/'training/runtime'/name) for name in ('rl.lua', 'rl_actions.lua', 'rl_checkpoint.lua')})
        self.log = (self.run/'mame.log').open('wb')
        try:
            command = mame_command(config) + ['-sound', 'none']
            if not show_window:
                command += ['-video', 'none']
            self.process = subprocess.Popen(command, cwd=self.run, stdout=self.log,
                                            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            bridge = Bridge(self.run, self.process, timeout=120)
            bridge.wait(lambda: read_json(self.run/'training/ready.json'), 60)
            # Detach only this training process's no-load session listeners.
            bridge.send('astra_load_sub:unsubscribe();astra_save_sub:unsubscribe();astra_reset_sub:unsubscribe();observe()')
            if self.checkpoints:
                opponent = self.checkpoints[0]['opponent']
                checkpoint = paths[0]
            else:
                bridge.send("speed('fast');wait_coin_ready(9000)")
                bridge.send("astra_entry.require_ready('coin');act({{3,'C'},{1,''}})")
                bridge.send('wait_start_ready(9000)')
                bridge.send("astra_entry.require_ready('start');act({{3,'S'},{120,''},{3,'D'},{12,''}})")
                bridge.send("act({{6,'LP'},{120,''}})")
                state = bridge.send('next_round(1200)')
                opponent = state['p2']['character']
                make_opening_guard(7-difficulty)[3](state, opponent, lambda _s, _o, n: bridge.send(f'next_round({n})'))
                checkpoint = self.run/'training/rl-start.sta'
                path_literal = json.dumps(checkpoint.as_posix(), ensure_ascii=False)
                bridge.send(f'checkpoint({path_literal})')
                if not checkpoint.is_file():
                    raise RuntimeError('Checkpoint was not saved')
            self.manifest = {'schema': 'astra.rl-pilot.v1', 'training_only': True,
                             'difficulty': difficulty, 'opponent': opponent, 'sound': 'none', 'show_window': show_window,
                             'checkpoint_sha256': sha256(checkpoint), 'runtime_sha256': runtime,
                             'opponents': sorted(self.checkpoint_groups) if self.checkpoints else [opponent],
                             'checkpoints': [{'id': sample.get('id', str(i)), 'sha256': sample['sha256'], 'opponent': sample['opponent']} for i, sample in enumerate(self.checkpoints)],
                             'experiment_sources': {p.name: sha256(p) for p in source.iterdir() if p.suffix in ('.py', '.lua')},
                             'actions': ACTION_NAMES, 'decision_frames': 12, 'observation_history': 4,
                             'train_leads': TRAIN_LEADS, 'eval_leads': EVAL_LEADS,
                             'sampling': 'uniform opponent, then uniform saved-state within opponent' if self.checkpoints else 'one fresh native opening; saved-state resets plus lead jitter, not independent starts',
                             'versions': {name: version(name) for name in ('stable-baselines3', 'torch', 'gymnasium', 'numpy')}}
            (self.run/'manifest.json').write_text(json.dumps(self.manifest, indent=2)+'\n', encoding='utf-8')
            bridge.send("assert(loadfile('training/runtime/rl.lua'))();observe()", snapshot=False)
        except BaseException as error:
            (self.run/'startup-error.json').write_text(json.dumps({'status': 'invalid', 'error': f'{type(error).__name__}: {error}'})+'\n', encoding='utf-8')
            self.close()
            raise

    def rpc(self, operation, arg=0, flag=0):
        if self.closed:
            raise RuntimeError('Environment closed')
        self.index += 1
        inbox = self.run/'training/rl-request.txt'
        if inbox.exists():
            raise RuntimeError('Pending request; never overwrite or replay')
        temporary = inbox.with_suffix('.tmp')
        temporary.write_text(f'{self.index} {operation} {int(arg)} {int(flag)}\n', encoding='ascii')
        temporary.replace(inbox)
        reply = self.run/f'training/rl-reply-{self.index:08d}.json'
        start = time.monotonic()
        try:
            while time.monotonic()-start < 45:
                result = read_json(reply)
                if result is not None:
                    if result.get('id') != self.index:
                        raise RuntimeError('Reply identity mismatch')
                    if 'error' in result:
                        raise RuntimeError(result['error'])
                    if not result.get('training_only') or result['state']['effective_difficulty'] != self.difficulty:
                        raise RuntimeError('Training/difficulty identity mismatch')
                    reply.unlink()
                    return result
                error = read_json(self.run/'training/rl-error.json')
                if error:
                    raise RuntimeError(str(error))
                if self.process.poll() is not None:
                    raise RuntimeError('MAME exited; inspect mame.log')
                time.sleep(0.002)
            raise TimeoutError('RL command timed out; no replay or recovery')
        except BaseException:
            self.close()
            raise
        finally:
            self.rpc_seconds += time.monotonic()-start

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        lead = options.get('lead', int(self.np_random.choice(TRAIN_LEADS)))
        if type(lead) is not int or not 0 <= lead <= 12:
            raise ValueError('Lead must be an integer in 0..12')
        if 'checkpoint' in options:
            index = options['checkpoint']
        elif self.checkpoints:
            opponent = int(self.np_random.choice(sorted(self.checkpoint_groups)))
            index = int(self.np_random.choice(self.checkpoint_groups[opponent]))
        else:
            index = 0
        if type(index) is not int or not 0 <= index < max(1, len(self.checkpoints)):
            raise ValueError('Invalid checkpoint index')
        self.record_partial('reset_before_round_end')
        result = self.rpc('reset', lead, int(self.baseline)+2*index)
        self.checkpoint_index = index
        if not result.get('reset_confirmed'):
            raise RuntimeError('Missing native post-load confirmation')
        state = result['state']
        opponent = self.checkpoints[index]['opponent'] if self.checkpoints else self.manifest['opponent']
        if state['p1']['char'] != 4 or state['p2']['char'] != opponent:
            raise RuntimeError('Restored actor mismatch')
        self.episode_opponent = opponent
        self.lead = lead
        self.episode_phase = self.phase
        self.previous = state
        self.history.clear()
        for _ in range(4):
            self.history.append(features(state))
        self.must_reset = False
        self.episode_return = 0.0
        self.episode_steps = 0
        self.last_frames = 0
        self.episode_started = time.monotonic()
        return np.concatenate(self.history), {'training_only': True, 'lead': lead}

    def step(self, action):
        if self.must_reset:
            raise RuntimeError('Reset before stepping')
        if not self.action_space.contains(action):
            raise ValueError('Invalid action')
        result = self.rpc('step', int(action))
        state = result['state']
        outcome = result.get('outcome')
        value = reward(self.previous, state, outcome)
        self.previous = state
        self.history.append(features(state))
        self.episode_return += value
        self.episode_steps += 1
        self.total_frames += result['frames']-self.last_frames
        self.last_frames = result['frames']
        done = bool(result.get('done'))
        info = {'training_only': True, 'outcome': outcome, 'frames': result['frames']}
        if done:
            row = {'phase': self.episode_phase, 'baseline': self.baseline, 'lead': self.lead,
                   'checkpoint': self.checkpoint_index, 'episode': result['episode'], 'opponent': self.episode_opponent,
                   'outcome': outcome, 'return': self.episode_return, 'steps': self.episode_steps,
                   'frames': result['frames'], 'wall_seconds': time.monotonic()-self.episode_started,
                   'final_state': state, 'native_round': result.get('native_round')}
            self.episodes.append(row)
            with (self.run/'episodes.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row)+'\n')
            self.must_reset = True
        return np.concatenate(self.history), value, done, False, info

    def record_partial(self, reason):
        if not self.must_reset:
            with (self.run/'partial-episodes.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps({'phase': self.episode_phase, 'lead': self.lead,
                                         'steps': self.episode_steps, 'frames': self.last_frames,
                                         'return': self.episode_return, 'checkpoint': self.checkpoint_index, 'reason': reason})+'\n')
            self.must_reset = True

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.record_partial('environment_closed')
        finally:
            try:
                if self.process is not None and self.process.poll() is None:
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        self.process.wait(timeout=10)
            finally:
                if self.log is not None:
                    self.log.close()
