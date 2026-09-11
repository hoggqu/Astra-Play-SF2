"""Real Lua 5.4 status I/O tests; optional locally, mandatory in Windows CI.

These exercise the packaged Lua implementation, not a Python reimplementation.
No emulator or ROM is needed. Install test-only dependency ``lupa`` to run them.
"""
from importlib.resources import files
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest

try:
    from lupa.lua54 import LuaError, LuaRuntime
except ImportError:
    LuaError = LuaRuntime = None


@unittest.skipUnless(LuaRuntime is not None, "Real Lua I/O test requires optional test dependency lupa")
class StatusIOTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "status test 中文"
        self.root.mkdir()
        # Like MAME, Lua gets relative ASCII filenames in a Unicode working dir.
        self.cwd = Path.cwd()
        os.chdir(self.root)
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(os.chdir, self.cwd)
        self.source = files("astra_play_sf2").joinpath("assets/status_io.lua").read_text(encoding="utf-8")

    def runtime(self):
        lua = LuaRuntime(unpack_returned_tuples=True)
        module = lua.execute(self.source)
        self.assertIsNotNone(module, "status_io.lua must return its API table")
        return lua, module

    def test_terminal_publish_is_complete_and_never_overwrites(self):
        lua, module = self.runtime()
        lua.execute("os.remove=function() error('Status I/O must never delete a file') end")
        body = json.dumps({"status": "complete", "payload": "terminal" * 10000}) + "\n"
        module.publish("match-status.json", body)
        self.assertEqual(Path("match-status.json").read_bytes(), body.encode())
        with Path("match-status.json").open("rb") as held:
            with self.assertRaises(LuaError):
                module.publish("match-status.json", '{"status":"invalid"}\n')
            self.assertEqual(held.read(), body.encode())
        self.assertEqual(Path("match-status.json").read_bytes(), body.encode())

    def test_terminal_stays_hidden_during_a_partial_native_write(self):
        lua, module = self.runtime()
        checkpoints = []

        def observe_partial_write():
            # Widen the partial-write window deterministically; this would fail
            # if publish opened the reader-visible terminal directly in wb mode.
            self.assertFalse(Path("match-status.json").exists())
            checkpoints.append("partial native file write observed")

        lua.globals().observe_partial_write = observe_partial_write
        lua.execute('''
            local real_open=io.open
            io.open=function(path,mode)
                local f,err=real_open(path,mode)
                if not f or mode~='wb' then return f,err end
                local proxy={}
                function proxy:write(body)
                    local middle=math.floor(#body/2)
                    assert(f:write(body:sub(1,middle)));assert(f:flush())
                    observe_partial_write()
                    assert(f:write(body:sub(middle+1)))
                    return self
                end
                function proxy:close() return f:close() end
                return proxy
            end
        ''')
        body = '{"status":"complete","payload":"native Lua file write"}\n'
        module.publish("match-status.json", body)
        self.assertEqual(len(checkpoints), 1)
        self.assertEqual(Path("match-status.json").read_bytes(), body.encode())

    def test_append_and_publish_with_concurrent_python_readers(self):
        progress = Path("match-progress.jsonl")
        terminal = Path("match-status.json")
        payload = {"status": "complete", "frame": 500, "payload": "ken" * 350000}
        body = json.dumps(payload, separators=(",", ":")) + "\n"
        created = threading.Event()
        reader_ready = threading.Event()
        halfway = threading.Event()
        resume = threading.Event()
        done = threading.Event()
        errors, observations = [], []

        def checkpoint():
            halfway.set()
            if not resume.wait(10):
                raise TimeoutError("Python reader did not observe the live progress stream")

        def writer():
            try:
                lua, module = self.runtime()
                lua.execute("os.remove=function() error('Status I/O must never delete a file') end")
                module.append(str(progress), '{"frame":0}\n')
                created.set()
                if not reader_ready.wait(10):
                    raise TimeoutError("Python reader did not open progress")
                lua.globals().status_io = module
                lua.globals().checkpoint = checkpoint
                lua.globals().terminal_body = body
                lua.execute('''
                    for frame=1,500 do
                        status_io.append('match-progress.jsonl', string.format('{"frame":%d}\\n',frame))
                        if frame==250 then checkpoint() end
                    end
                    status_io.publish('match-status.json',terminal_body)
                ''')
            except BaseException as exc:
                errors.append(exc)
            finally:
                created.set()
                done.set()

        thread = threading.Thread(target=writer, name="real-lua-status-writer", daemon=True)
        thread.start()
        try:
            self.assertTrue(created.wait(10), "Lua writer did not create progress")
            if errors:
                raise errors[0]
            # Keep this exact Windows handle open throughout append and publish.
            with progress.open("rb") as held:
                reader_ready.set()
                chunks = []
                deadline = time.monotonic() + 20
                missing_reads = 0
                while not done.is_set():
                    self.assertLess(time.monotonic(), deadline, "Lua writer stalled")
                    chunks.append(held.read())
                    try:
                        current = terminal.read_bytes()
                    except FileNotFoundError:
                        missing_reads += 1
                    else:
                        # Every visible terminal must already be the full JSON.
                        self.assertEqual(json.loads(current), payload)
                        self.assertEqual(current, body.encode())
                        observations.append(current)
                    if halfway.is_set() and not resume.is_set():
                        self.assertFalse(terminal.exists(), "Terminal appeared during progress")
                        chunks.append(held.read())
                        partial = b"".join(chunks)
                        self.assertIn(b'{"frame":250}\n', partial)
                        resume.set()
                    time.sleep(0.0005)
                chunks.append(held.read())
                self.assertGreater(missing_reads, 0, "Reader never polled before terminal publication")
                if errors:
                    raise errors[0]
                for _ in range(20):
                    current = terminal.read_bytes()
                    self.assertEqual(json.loads(current), payload)
                    self.assertEqual(current, body.encode())
                    observations.append(current)
                self.assertEqual([json.loads(line)["frame"] for line in b"".join(chunks).splitlines()],
                                 list(range(501)))
            self.assertTrue(observations)
            self.assertEqual(terminal.read_bytes(), body.encode())
        finally:
            reader_ready.set()
            resume.set()
            thread.join(10)
        self.assertFalse(thread.is_alive(), "Lua worker did not finish")

    @unittest.skipUnless(os.name == "nt", "Windows-only reproduction of delete sharing violation")
    def test_old_delete_then_rename_fails_with_held_python_reader(self):
        lua, _ = self.runtime()
        target = Path("old-status.json")
        target.write_bytes(b'{"frame":1}\n')
        old_publish = lua.eval('''function(path,body)
            local f=assert(io.open(path..'.tmp','wb'))
            assert(f:write(body));assert(f:close())
            assert(os.remove(path))
            assert(os.rename(path..'.tmp',path))
        end''')
        with target.open("rb") as held:
            with self.assertRaises(LuaError):
                old_publish(str(target), '{"frame":2}\n')
            self.assertEqual(held.read(), b'{"frame":1}\n')
            self.assertEqual(target.read_bytes(), b'{"frame":1}\n')
        # Prove the failure was the held reader, not the old Lua syntax/path.
        old_publish(str(target), '{"frame":2}\n')
        self.assertEqual(target.read_bytes(), b'{"frame":2}\n')


if __name__ == "__main__":
    unittest.main(verbosity=2)
