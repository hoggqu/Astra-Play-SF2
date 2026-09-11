"""Bounded natural-coin verification; the frozen Lua policy owns each whole match."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from importlib.resources import files
import json
import os
from pathlib import Path
import platform
import subprocess
import uuid

from . import __version__
from .config import atomic_json, doctor
from .opening import make_opening_guard
from .transport import Bridge, read_json

NAMES = {0: "ryu", 1: "honda", 2: "blanka", 3: "guile", 5: "chunli", 6: "zangief", 7: "dhalsim", 8: "bison", 9: "sagat", 10: "balrog", 11: "vega"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def session_lock(data):
    data = Path(data)
    data.mkdir(parents=True, exist_ok=True)
    path = data / "verify.lock"
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise RuntimeError(f"Session lock exists: {path}. Inspect the recorded PID; do not start a second controller.") from None
    try:
        os.write(fd, json.dumps({"pid": os.getpid(), "host": platform.node()}).encode())
        os.close(fd)
        yield
    finally:
        path.unlink(missing_ok=True)


def stage_runtime(run, level):
    target = run / "training/runtime"
    target.mkdir(parents=True)
    assets = files("astra_play_sf2").joinpath("assets")
    for asset in assets.iterdir():
        if asset.name.endswith((".lua", ".json")):
            (target / asset.name).write_bytes(asset.read_bytes())
    (target / "settings.lua").write_text(f"astra_difficulty_bits={7-level}\nastra_difficulty_label='{level}'\n", encoding="utf-8")
    provenance = read_json(target / "provenance.json")
    for name, info in provenance["files"].items():
        if sha256(target / name) != info["sha256"]:
            raise RuntimeError("Packaged source checksum mismatch: " + name)
    return {path.name: sha256(path) for path in sorted(target.iterdir())}


def mame_command(config):
    # Default native video is portable; forcing SDL-only 'soft' breaks Windows builds.
    return [config["mame"], "sf2", "-noreadconfig", "-rompath", config["rom_dir"],
            "-homepath", ".", "-cfg_directory", "cfg", "-nvram_directory", "nvram",
            "-state_directory", "states", "-snapshot_directory", ".",
            "-window", "-skip_gameinfo", "-noconsole", "-noautosave", "-norewind",
            "-nocheat", "-noplugins", "-keyboardprovider", "none", "-joystickprovider", "none",
            "-mouseprovider", "none", "-lightgunprovider", "none",
            "-autoboot_script", "training/runtime/bootstrap.lua", "-autoboot_delay", "5"]


def seal_run(run):
    # Reports/reviews are derived or append-only. Everything else is sealed after process exit.
    excluded = {"evidence-sha256.json", "report.json", "report.html", "report.md", "reviews.jsonl"}
    items = {p.relative_to(run).as_posix(): sha256(p) for p in sorted(run.rglob("*"))
             if p.is_file() and p.name not in excluded}
    atomic_json(run / "evidence-sha256.json", {"schema": 1, "files": items})


def verify(config, levels, attempts=1, speed="normal", consecutive=None):
    if not levels or any(type(n) is not int or n not in range(3, 8) for n in levels):
        raise ValueError("Difficulty must be 3 through 7")
    if attempts < 1 or (consecutive is not None and not 1 <= consecutive <= attempts):
        raise ValueError("Attempts must be positive; consecutive must be between 1 and attempts")
    if speed not in ("normal", "fast"):
        raise ValueError("Speed must be normal or fast")
    preflight = doctor(config)
    if not preflight["ok"]:
        raise RuntimeError("Preflight failed: " + json.dumps(preflight))
    with session_lock(config["data_dir"]):
        run = Path(config["data_dir"]) / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8])
        run.mkdir(parents=True)
        hashes = stage_runtime(run, levels[0])
        command = mame_command(config)
        manifest = {"schema": "astra.run.v1", "status": "running", "version": __version__,
                    "created_utc": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
                    "python": platform.python_version(), "mame_version": "0.288", "rom": "sf2",
                    "difficulty": levels, "attempts_requested": attempts, "consecutive": consecutive,
                    "speed": speed, "policy_sha256": hashes["fighter.lua"], "runtime_sha256": hashes,
                    "command": command, "attempts": [], "preflight": preflight}
        atomic_json(run / "run.json", manifest)
        print(f"Run directory: {run}", flush=True)
        process = None
        failure = None
        log = (run / "mame.log").open("wb")
        try:
            process = subprocess.Popen(command, cwd=run, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            bridge = Bridge(run, process)
            bridge.wait(lambda: read_json(run / "training/ready.json"), 60)
            for level in levels:
                observed = bridge.send(f"session_difficulty({level})")
                if observed.get("difficulty_bits") != 7-level:
                    raise RuntimeError("Native DIP setting did not latch before the attempt")
                streak = 0
                for ordinal in range(1, attempts + 1):
                    attempt = {"id": f"l{level}-{ordinal:03d}", "difficulty": level, "outcome": "invalid", "matches": [], "images": {}}
                    manifest["attempts"].append(attempt)
                    atomic_json(run / "run.json", manifest)
                    _attempt(run, bridge, attempt, speed, lambda: atomic_json(run / "run.json", manifest))
                    streak = streak + 1 if attempt["outcome"] == "gameplay_clear" else 0
                    print(f"{attempt['id']}: {attempt['outcome']} (streak {streak})", flush=True)
                    if consecutive and streak >= consecutive:
                        break
            manifest["status"] = "complete"
        except (Exception, KeyboardInterrupt) as error:
            failure = f"{type(error).__name__}: {error}"
            manifest["status"] = "invalid"
            manifest["error"] = failure
            if manifest["attempts"] and manifest["attempts"][-1]["outcome"] == "invalid":
                manifest["attempts"][-1]["error"] = failure
        finally:
            # Only this Popen belongs to us. Interruptions never replay or resume a match.
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
            log.close()
            manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
            atomic_json(run / "run.json", manifest)
            seal_run(run)
        from .evidence import audit_run, write_report
        audit = audit_run(run)
        write_report(run)
        if failure or not audit["ok"]:
            print(f"Invalid run: {failure or audit['errors']}", flush=True)
            return run, 2
        passed = True
        for level in levels:
            outcomes = [a["outcome"] for a in manifest["attempts"] if a["difficulty"] == level]
            passed &= (all(x == "gameplay_clear" for x in outcomes) if consecutive is None
                       else len(outcomes) >= consecutive and all(x == "gameplay_clear" for x in outcomes[-consecutive:]))
        return run, 0 if passed else 1


def _attempt(run, bridge, attempt, speed, save):
    prefix = f"training/attempts/{attempt['id']}"
    (run / prefix).mkdir(parents=True)
    attempt["lifecycle"] = prefix + "/session-lifecycle.json"
    def action(command):
        return bridge.send(f"speed('{speed}');" + command)
    def image():
        return "training/" + bridge.state()["screenshot"]
    def advance(frames):
        return action(f"next_round({frames})")
    # One credit is consumed at initial start. Long neutral wait lets the native
    # continue countdown and ending finish, without any new Start input or reset.
    action("act({{9000,''}})")
    bridge.send(f"session_begin('{prefix}/session')")
    action("act({{3,'C'},{120,''},{3,'S'},{120,''},{3,'D'},{12,''}})")
    attempt["images"]["selection"] = image()
    save()
    action("act({{6,'LP'},{120,''}})")
    advance(1200)
    seen = []
    guard = make_opening_guard(7 - attempt["difficulty"])[3]
    for number in range(1, 12):
        state = bridge.state()
        opponent = state["p2"]["character"]
        guard(state, opponent, lambda _state, _opponent, frames: advance(frames))
        expected = (opponent in {0, 1, 2, 3, 5, 6, 7} if number <= 7 else opponent == [10, 11, 9, 8][number - 8])
        if not expected or opponent in seen:
            raise RuntimeError("Unexpected native opponent route; refusing to advance combat")
        seen.append(opponent)
        match = f"{prefix}/m{number:02d}-{NAMES[opponent]}"
        attempt["matches"].append(match + ".json")
        save()
        print(f"{attempt['id']} match {number}/11: {NAMES[opponent]}", flush=True)
        bridge.send(f"play_match('{match}',{opponent},{{training_validation=false,speed='{speed}'}})", snapshot=False)
        def finished():
            state = read_json(run / (match + "-status.json"))
            return state if state and state.get("status") in ("complete", "invalid") else False
        result = bridge.wait(finished)
        if result.get("valid_continuous") is not True or result.get("status") != "complete":
            raise RuntimeError("Invalid continuous match: " + str(result.get("reason")))
        if result.get("result") == "cpu_win":
            bridge.send("session_end()")
            attempt["outcome"] = "loss"
            save()
            return
        if result.get("result") != "ken_win":
            raise RuntimeError("Unexpected match result: " + str(result.get("result")))
        if opponent == 8:
            attempt["images"]["bison"] = image()
            attempt["images"]["ending"] = []
            for frames in (600, 900, 600):
                action(f"act({{{{{frames},''}}}})")
                attempt["images"]["ending"].append(image())
            bridge.send("session_end()")
            attempt["outcome"] = "gameplay_clear"
            save()
            return
        if number == 3:
            advance(1200)
            action("local s=repeatseq({{6,'D HK'},{6,''}},100);s[#s+1]={740,''};act(s)")
            advance(1800)
        elif number in (6, 9):
            advance(1200)
            action("act({{" + str(2000 if number == 6 else 1940) + ",''}})")
            advance(1800)
        else:
            advance(1800 if number == 7 else 1200)
