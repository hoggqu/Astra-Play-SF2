"""Single-owner file bridge. Never replay an uncertain command."""
import json
import os
from pathlib import Path
import time
import uuid


def read_json(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, ValueError, UnicodeError):
        return None


class Bridge:
    def __init__(self, run, process, timeout=720):
        self.base = Path(run) / "training"
        self.process = process
        self.timeout = timeout

    def alive(self):
        if self.process.poll() is not None:
            raise RuntimeError("MAME exited; inspect mame.log. The command will not be replayed.")
        error = self.base / "bootstrap-error.txt"
        if error.exists():
            raise RuntimeError(error.read_text(encoding="utf-8"))

    def wait(self, predicate, timeout=None):
        until = time.monotonic() + (self.timeout if timeout is None else timeout)
        while time.monotonic() < until:
            self.alive()
            result = predicate()
            if result:
                return result
            time.sleep(0.1)
        raise TimeoutError("MAME command timed out. Run is invalid; no automatic replay.")

    def event(self, identifier, event, command):
        with (self.base / "commands.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"id": identifier, "event": event, "command": command, "time": time.time()}) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def state(self):
        value = read_json(self.base / "status.json")
        if value is None:
            raise RuntimeError("No complete native observation")
        return value

    def send(self, command, snapshot=True):
        previous = self.state()["screenshot"]
        request = self.base / "request-v2.lua"
        status = self.base / "request-v2-status.txt"
        if request.exists():
            raise RuntimeError("A command is already pending; refusing to overwrite it")
        identifier = uuid.uuid4().hex
        self.event(identifier, "prepared", command)
        status.unlink(missing_ok=True)
        tmp = request.with_suffix(".tmp")
        tmp.write_text(command + "\n", encoding="utf-8")
        tmp.replace(request)
        def accepted():
            try:
                result = status.read_text(encoding="utf-8")
            except OSError:
                return False
            if not result:
                return False
            if result != "accepted":
                raise RuntimeError("Lua rejected command: " + result)
            return True
        self.wait(accepted, 30)
        self.event(identifier, "accepted", command)
        def completed():
            value = read_json(self.base / "status.json")
            if (value and value.get("screenshot") != previous and value.get("paused") is True
                    and value.get("controller_busy") is False
                    and (self.base / value["screenshot"]).is_file()):
                return value
            return False
        value = self.wait(completed) if snapshot else None
        self.event(identifier, "completed", command)
        return value
