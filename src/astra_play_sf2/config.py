"""Configuration and read-only emulator preflight; no shell execution."""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

MAME_VERSION = "0.288"


def config_path():
    return Path(os.environ.get("ASTRA_SF2_CONFIG", Path.home() / ".astra-play-sf2/config.json")).expanduser().resolve()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def configure(mame, rom_dir, data_dir=None):
    executable = str(Path(shutil.which(mame) or mame).expanduser().resolve())
    value = {"schema": 1, "mame": executable,
             "rom_dir": str(Path(rom_dir).expanduser().resolve()),
             "data_dir": str(Path(data_dir or Path.home() / ".astra-play-sf2").expanduser().resolve())}
    if not Path(executable).is_file():
        raise ValueError("MAME executable does not exist: " + executable)
    if not Path(value["rom_dir"]).is_dir():
        raise ValueError("ROM directory does not exist: " + value["rom_dir"])
    atomic_json(config_path(), value)
    return value


def load_config():
    try:
        value = json.loads(config_path().read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("schema") != 1 or not all(isinstance(value.get(k), str) and value[k] for k in ("mame", "rom_dir", "data_dir")):
            raise ValueError("Unsupported configuration")
        return value
    except FileNotFoundError:
        raise ValueError("Run astra-sf2 configure --mame PATH --rom-dir PATH first") from None


def doctor(config):
    checks = []
    def check(name, okay, detail):
        checks.append({"name": name, "ok": bool(okay), "detail": detail})
    exe = Path(config["mame"])
    check("executable", exe.is_file(), str(exe))
    check("rom_directory", Path(config["rom_dir"]).is_dir(), config["rom_dir"])
    if not all(c["ok"] for c in checks):
        return {"ok": False, "checks": checks}
    with tempfile.TemporaryDirectory(prefix="astra-doctor-") as work:
        def run(*args):
            return subprocess.run([str(exe), "-noreadconfig", *args], cwd=work,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, encoding="utf-8", errors="replace", timeout=90)
        version = run("-version")
        match = re.search(r"\b(0\.\d{3})\b", version.stdout)
        check("mame_version", version.returncode == 0 and match and match[1] == MAME_VERSION, version.stdout.strip())
        roms = run("-rompath", config["rom_dir"], "-verifyroms", "sf2")
        check("sf2_roms", roms.returncode == 0, roms.stdout.strip())
        driver = run("-listfull", "sf2")
        check("driver", driver.returncode == 0 and "World 910522" in driver.stdout, driver.stdout.strip())
    data = Path(config["data_dir"])
    try:
        data.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=data):
            pass
        check("data_directory", True, str(data))
    except OSError as error:
        check("data_directory", False, str(error))
    return {"ok": all(c["ok"] for c in checks), "checks": checks,
            "note": "Runtime Lua/device/port checks run at verification startup. MAME 0.288 is required."}
