"""Optional batch transport; original per-decision environment is unchanged."""
import json
from pathlib import Path
import time
from unittest.mock import patch

import gymnasium as gym
import numpy as np

from astra_play_sf2.config import atomic_json
from astra_play_sf2.runner import sha256
from astra_play_sf2.transport import Bridge, read_json
from . import env as env_module
from .env import MameEnv, TRAIN_LEADS
from .export import lua_literal

HERE = Path(__file__).resolve().parent
ATTACH = "assert(loadfile('training/runtime/rl.lua'))();observe()"


class BatchBridge(Bridge):
    """Swap only this fresh process's staged training module before first attach."""
    def send(self, command, snapshot=True):
        if command == ATTACH:
            runtime = self.base / 'runtime'
            (runtime / 'rl.lua').write_bytes((HERE / 'batch_runtime.lua').read_bytes())
            (runtime / 'rl_nn.lua').write_bytes((HERE / 'nn.lua').read_bytes())
            (runtime / 'rl_native_core.lua').write_bytes((HERE / 'native_continuous_core.lua').read_bytes())
            manifest = read_json(self.base.parent / 'manifest.json')
            expected = [sample['opponent'] for sample in manifest['checkpoints']] or [manifest['opponent']]
            (runtime / 'rl_batch_checkpoints.lua').write_text('return ' + lua_literal(expected) + '\n', encoding='utf-8')
        return super().send(command, snapshot=snapshot)


class BatchEnv(MameEnv):
    def __init__(self, *args, **kwargs):
        # Each sampling worker constructs one environment, without other threads.
        # The scoped class binding changes no source or pre-existing MAME process.
        with patch.object(env_module, 'Bridge', BatchBridge):
            super().__init__(*args, **kwargs)
        try:
            actual = sha256(self.run / 'training/runtime/rl.lua')
            if actual != sha256(HERE / 'batch_runtime.lua'):
                raise RuntimeError('Batch attach hook did not run; refusing protocol mismatch')
            self.manifest.update(schema='astra.rl-batch-env.v1', protocol='batch-v1')
            self.manifest['runtime_sha256'].update({'rl.lua': actual,
                'rl_nn.lua': sha256(self.run / 'training/runtime/rl_nn.lua'),
                'rl_native_core.lua': sha256(self.run / 'training/runtime/rl_native_core.lua'),
                'rl_batch_checkpoints.lua': sha256(self.run / 'training/runtime/rl_batch_checkpoints.lua')})
            atomic_json(self.run / 'manifest.json', self.manifest)
            self.batch_index = 0
            self.batch_calls = 0
            self.batch_steps = 0
            self.batch_partial = None
            self.batch_pending = False
        except BaseException:
            self.close()
            raise

    def batch_rpc(self, body):
        if self.closed:
            raise RuntimeError('Environment closed')
        self.batch_index += 1
        body = dict(body, id=self.batch_index)
        inbox = self.run / 'training/rl-batch-request.lua'
        if inbox.exists():
            raise RuntimeError('Pending batch request; never replay')
        tmp = inbox.with_suffix('.tmp')
        tmp.write_text('return ' + lua_literal(body) + '\n', encoding='utf-8')
        tmp.replace(inbox)
        reply = self.run / f'training/rl-batch-reply-{self.batch_index:08d}.json'
        started = time.monotonic()
        self.batch_pending = True
        try:
            while time.monotonic()-started < 180:
                result = read_json(reply)
                if result is not None:
                    if result.get('id') != self.batch_index or 'error' in result:
                        raise RuntimeError(str(result))
                    if result.get('training_only') is not True or result['state']['effective_difficulty'] != self.difficulty:
                        raise RuntimeError('Batch identity/difficulty mismatch')
                    reply.unlink()
                    self.batch_pending = False
                    self.batch_partial = result.get('partial_episode') or None
                    for key in ('transitions', 'episodes'):
                        if result.get(key) == {}:
                            result[key] = []
                    return result
                error = read_json(self.run / 'training/rl-batch-error.json')
                if error:
                    raise RuntimeError(str(error))
                if self.process.poll() is not None:
                    raise RuntimeError('MAME exited during batch')
                time.sleep(.002)
            raise TimeoutError('Batch timed out; no replay or recovery')
        except BaseException:
            self.close()
            raise
        finally:
            self.rpc_seconds += time.monotonic()-started

    def reset_choice(self):
        opponents = sorted(self.checkpoint_groups)
        opponent = int(self.np_random.choice(opponents))
        return {'checkpoint': int(self.np_random.choice(self.checkpoint_groups[opponent])),
                'lead': int(self.np_random.choice(TRAIN_LEADS))}

    def reset(self, *, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        options = options or self.reset_choice()
        result = self.batch_rpc({'op': 'reset', 'reset': options})
        self.must_reset = True  # Parent's partial-episode logger is not this protocol's owner.
        return np.asarray(result['observation'], dtype=np.float32), {'training_only': True}

    def step(self, action):
        raise RuntimeError('BatchEnv uses batch_rollout; use MameEnv for ordinary Gym stepping')

    def batch_rollout(self, count, model=None, actions=None, uniforms=None, resets=None, native_reference=False):
        if type(count) is not int or not 1 <= count <= 256:
            raise ValueError('Batch count must be 1..256')
        body = {'op': 'rollout', 'count': count,
                'resets': resets if resets is not None else [self.reset_choice() for _ in range(count)]}
        if native_reference:
            body['native_reference'] = 1
        if actions is not None:
            if len(actions) != count:
                raise ValueError('Action count mismatch')
            body['actions'] = [int(a) for a in actions]
        else:
            body['uniforms'] = list(uniforms) if uniforms is not None else self.np_random.random(count).tolist()
        if model is not None:
            body['model'] = model
        result = self.batch_rpc(body)
        if len(result['transitions']) != count:
            raise RuntimeError('Incomplete batch; never count partial rollout as complete')
        self.batch_calls += 1
        self.batch_steps += count
        for row in result['episodes']:
            row.update(phase=self.phase, baseline=False)
            self.episodes.append(row)
            with (self.run / 'episodes.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row) + '\n')
        return result

    def close(self):
        if self.closed:
            return
        try:
            partial = getattr(self, 'batch_partial', None)
            if partial:
                row = dict(partial, phase=self.phase, baseline=False,
                           reason='environment_closed', incomplete=True,
                           unobserved_tail=bool(getattr(self, 'batch_pending', False)))
                with (self.run / 'partial-episodes.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(row) + '\n')
        finally:
            super().close()
