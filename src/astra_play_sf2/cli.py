import argparse
import json
from pathlib import Path
import subprocess

from . import __version__
from .config import config_path, configure, doctor, load_config


def main(argv=None):
    parser = argparse.ArgumentParser(prog="astra-sf2", description="Agent-independent MAME SF2 Ken verification")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    config = commands.add_parser("configure", help="Save local MAME, ROM and output paths")
    config.add_argument("--mame", required=True)
    config.add_argument("--rom-dir", required=True)
    config.add_argument("--data-dir")
    commands.add_parser("doctor", help="Check MAME 0.288, sf2 ROMs and output permissions")
    verify = commands.add_parser("verify", help="Run bounded natural-coin attempts")
    verify.add_argument("--difficulty", choices=["3", "4", "5", "6", "7", "all"], default="3")
    verify.add_argument("--attempts", type=int, default=1)
    verify.add_argument("--consecutive", type=int)
    verify.add_argument("--speed", choices=["normal", "2x", "4x", "fast"], default="normal")
    for name in ("audit", "report", "review"):
        command = commands.add_parser(name)
        command.add_argument("run_dir", type=Path)
        if name == "review":
            command.add_argument("--attempt", required=True)
            command.add_argument("--reviewer", required=True)
            command.add_argument("--decision", choices=["approve", "reject"], required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "configure":
            configure(args.mame, args.rom_dir, args.data_dir)
            print(f"Configuration: {config_path()}")
            return 0
        if args.command == "doctor":
            result = doctor(load_config())
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0 if result["ok"] else 2
        if args.command == "verify":
            from .runner import verify as run_verify
            levels = list(range(3, 8)) if args.difficulty == "all" else [int(args.difficulty)]
            path, code = run_verify(load_config(), levels, args.attempts, args.speed, args.consecutive)
            print(f"Report: {path / 'report.html'}")
            return code
        from .evidence import audit_run, write_report, record_review
        path = args.run_dir.expanduser().resolve()
        if args.command == "audit":
            result = audit_run(path)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0 if result["ok"] else 2
        if args.command == "review":
            result = record_review(path, args.attempt, args.reviewer, args.decision)
            print(json.dumps(result, indent=2, ensure_ascii=False))
        write_report(path)
        print(path / "report.html")
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"Error: {error}")
        return 2
