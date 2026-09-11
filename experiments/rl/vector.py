"""Bounded shutdown for owned sampling workers, including partial startup.

Worker finalizers close MAME on normal multiprocessing exit/EOF/errors; SIGTERM
gets the same cleanup. An OS kill cannot run Python cleanup, so remote jobs also
use a dedicated systemd control group to contain and reap descendant processes.
"""
import time

from stable_baselines3.common.vec_env import SubprocVecEnv



class ManagedVec(SubprocVecEnv):
    """SubprocVecEnv that never blocks on a pending result during cleanup."""
    def __init__(self, env_fns, start_method='spawn', close_timeout=25):
        if close_timeout < 0:
            raise ValueError('close_timeout must be nonnegative')
        self.close_timeout = close_timeout
        self.close_errors = []
        self.processes = []
        self.remotes = ()
        self.work_remotes = ()
        self.closed = False
        self.waiting = False
        try:
            super().__init__(env_fns, start_method=start_method)
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.closed:
            return
        self.closed = True
        # A failed worker may leave waiting=True and an unreadable result pipe.
        # Sending close is ordered after its outstanding step, so it can finish
        # the command and close normally; never recv or replay that command.
        for remote in self.remotes:
            try:
                remote.send(('close', None))
            except (EOFError, OSError, ValueError) as error:
                self.close_errors.append(f'close request: {type(error).__name__}: {error}')
            finally:
                try:
                    remote.close()
                except OSError as error:
                    self.close_errors.append(f'pipe close: {error}')
        for remote in self.work_remotes:
            try:
                remote.close()
            except OSError as error:
                self.close_errors.append(f'worker pipe close: {error}')
        self._join_until(time.monotonic()+self.close_timeout)
        alive = [process for process in self.processes if process.is_alive()]
        for process in alive:
            try:
                process.terminate()  # Worker handler closes only its own MAME.
            except (OSError, ValueError) as error:
                self.close_errors.append(f'worker terminate: {error}')
        if alive:
            self._join_until(time.monotonic()+max(5, self.close_timeout))
        alive = [process for process in self.processes if process.is_alive()]
        for process in alive:
            try:
                process.kill()
            except (OSError, ValueError) as error:
                self.close_errors.append(f'worker kill: {error}')
        if alive:
            self._join_until(time.monotonic()+5)
        for process in self.processes:
            if process.is_alive():
                self.close_errors.append('Owned worker remained alive after bounded kill/join')
        self.waiting = False

    def _join_until(self, deadline):
        for process in self.processes:
            try:
                process.join(timeout=max(0., deadline-time.monotonic()))
            except (AssertionError, OSError, ValueError) as error:
                self.close_errors.append(f'worker join: {error}')
