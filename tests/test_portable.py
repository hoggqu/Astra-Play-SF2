"""Portable orchestration tests. Every emulator/process interaction is mocked."""
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from astra_play_sf2 import cli, config, runner, transport
from astra_play_sf2.opening import make_opening_guard, require_difficulty
import xml.etree.ElementTree as ET


class TemporaryCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="astra test ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "空 格 workspace"
        self.root.mkdir()


class ConfigurationTests(TemporaryCase):
    def setUp(self):
        super().setUp()
        self.config_file = self.root / "settings 文件" / "config.json"
        self.env = patch.dict(os.environ, {"ASTRA_SF2_CONFIG": str(self.config_file)})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.exe = self.root / "MAME executable.exe"
        self.exe.write_bytes(b"not an emulator")
        self.roms = self.root / "ROM 空格"
        self.roms.mkdir()
        self.data = self.root / "run 数据"

    def configured(self):
        return config.configure(str(self.exe), str(self.roms), str(self.data))

    def test_unicode_space_paths_round_trip_without_shell(self):
        with patch.object(subprocess, "run", side_effect=AssertionError("must not execute")):
            result = self.configured()
        self.assertEqual(config.load_config(), result)
        self.assertEqual(config.config_path(), self.config_file.resolve())
        self.assertEqual(result["mame"], str(self.exe.resolve()))
        self.assertEqual(result["rom_dir"], str(self.roms.resolve()))
        self.assertEqual(result["data_dir"], str(self.data.resolve()))
        self.assertFalse(self.config_file.with_suffix(".json.tmp").exists())

    def test_missing_executable_does_not_replace_configuration(self):
        self.configured()
        before = self.config_file.read_bytes()
        with self.assertRaises(ValueError):
            config.configure(str(self.root / "missing"), str(self.roms))
        self.assertEqual(self.config_file.read_bytes(), before)

    def test_missing_rom_directory_is_rejected(self):
        with self.assertRaises(ValueError):
            config.configure(str(self.exe), str(self.root / "missing roms"))
        self.assertFalse(self.config_file.exists())

    def test_missing_config_gives_setup_guidance(self):
        with self.assertRaisesRegex(ValueError, "configure"):
            config.load_config()

    def test_invalid_config_schema_and_fields_are_rejected(self):
        for value in ({}, {"schema": 2}, {"schema": 1, "mame": 9},
                      {"schema": 1, "mame": "x", "rom_dir": "", "data_dir": "x"}):
            with self.subTest(value=value):
                config.atomic_json(self.config_file, value)
                with self.assertRaises(ValueError):
                    config.load_config()

    def test_config_non_object_is_a_validation_error(self):
        for value in ([], None, "bad configuration"):
            with self.subTest(value=value):
                config.atomic_json(self.config_file, value)
                with self.assertRaises(ValueError):
                    config.load_config()

    def test_doctor_uses_argument_lists_and_isolated_working_directory(self):
        value = self.configured()
        calls = []
        def fake_run(args, **kwargs):
            self.assertIsInstance(args, list)
            self.assertNotIn("shell", kwargs)
            self.assertEqual(args[:2], [str(self.exe.resolve()), "-noreadconfig"])
            self.assertTrue(Path(kwargs["cwd"]).is_dir())
            self.assertNotEqual(Path(kwargs["cwd"]), self.root)
            self.assertEqual(kwargs["timeout"], 90)
            calls.append((args, kwargs))
            output = "0.288 (mame0288)" if "-version" in args else (
                'sf2 "Street Fighter II: The World Warrior (World 910522)"' if "-listfull" in args else "romset sf2 is good")
            return subprocess.CompletedProcess(args, 0, output)
        with patch.object(config.subprocess, "run", side_effect=fake_run):
            result = config.doctor(value)
        self.assertTrue(result["ok"])
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[1][0][2:], ["-rompath", str(self.roms.resolve()), "-verifyroms", "sf2"])
        self.assertEqual(len({str(x[1]["cwd"]) for x in calls}), 1)
        self.assertFalse(Path(calls[0][1]["cwd"]).exists())

    def test_doctor_rejects_other_mame_version(self):
        value = self.configured()
        responses = [subprocess.CompletedProcess([], 0, "0.289"),
                     subprocess.CompletedProcess([], 0, "good"),
                     subprocess.CompletedProcess([], 0, "World 910522")]
        with patch.object(config.subprocess, "run", side_effect=responses):
            result = config.doctor(value)
        self.assertFalse(result["ok"])
        self.assertFalse(next(c["ok"] for c in result["checks"] if c["name"] == "mame_version"))

    def test_doctor_missing_paths_never_launches_process(self):
        value = self.configured()
        self.exe.unlink()
        with patch.object(config.subprocess, "run") as launch:
            self.assertFalse(config.doctor(value)["ok"])
        launch.assert_not_called()


class BridgeTests(TemporaryCase):
    def setUp(self):
        super().setUp()
        self.base = self.root / "training"
        self.base.mkdir()
        self.process = Mock()
        self.process.poll.return_value = None
        self.bridge = transport.Bridge(self.root, self.process, timeout=2)
        self.observation = {"screenshot": "old.png", "paused": True, "controller_busy": False}
        config.atomic_json(self.base / "status.json", self.observation)
        self.sleep = patch.object(transport.time, "sleep")
        self.sleep.start()
        self.addCleanup(self.sleep.stop)

    def events(self):
        p = self.base / "commands.jsonl"
        return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()] if p.exists() else []

    def acknowledge(self):
        (self.base / "request-v2.lua").unlink()
        (self.base / "request-v2-status.txt").write_text("accepted", encoding="utf-8")

    def test_pending_request_cannot_be_overwritten_or_logged_as_new(self):
        pending = self.base / "request-v2.lua"
        pending.write_text("original command\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "pending"):
            self.bridge.send("replacement")
        self.assertEqual(pending.read_text(), "original command\n")
        self.assertEqual(self.events(), [])

    def test_acceptance_timeout_never_replays_and_preserves_pending_request(self):
        # Stale acknowledgement must be removed before publishing the new request.
        (self.base / "request-v2-status.txt").write_text("accepted", encoding="utf-8")
        with patch.object(transport.time, "monotonic", side_effect=[0, 0, 31]):
            with self.assertRaises(TimeoutError):
                self.bridge.send("once_only()")
        self.assertEqual((self.base / "request-v2.lua").read_text(), "once_only()\n")
        self.assertEqual([x["event"] for x in self.events()], ["prepared"])
        with self.assertRaisesRegex(RuntimeError, "pending"):
            self.bridge.send("once_only()")
        self.assertEqual(len(self.events()), 1)

    def test_accepted_command_completion_timeout_is_not_replayed(self):
        seen = []
        def poll():
            request = self.base / "request-v2.lua"
            if request.exists():
                seen.append(request.read_text())
                self.acknowledge()
            return None
        self.process.poll.side_effect = poll
        with patch.object(transport.time, "monotonic", side_effect=[0, 0, 0, 0, 3]):
            with self.assertRaises(TimeoutError):
                self.bridge.send("long_action()")
        self.assertEqual(seen, ["long_action()\n"])
        self.assertEqual([x["event"] for x in self.events()], ["prepared", "accepted"])

    def test_process_death_stops_polling_before_predicate(self):
        self.process.poll.return_value = 1
        predicate = Mock()
        with self.assertRaisesRegex(RuntimeError, "exited"):
            self.bridge.wait(predicate)
        predicate.assert_not_called()
        self.assertEqual(self.events(), [])

    def test_bootstrap_error_stops_wait_without_replay(self):
        (self.base / "bootstrap-error.txt").write_text("wrong Lua port", encoding="utf-8")
        predicate = Mock()
        with self.assertRaisesRegex(RuntimeError, "wrong Lua port"):
            self.bridge.wait(predicate)
        predicate.assert_not_called()

    def test_rejection_is_not_marked_accepted_or_completed(self):
        def reject():
            (self.base / "request-v2-status.txt").write_text("error: denied", encoding="utf-8")
            return None
        self.process.poll.side_effect = reject
        with self.assertRaisesRegex(RuntimeError, "Lua rejected"):
            self.bridge.send("denied()")
        self.assertEqual([x["event"] for x in self.events()], ["prepared"])

    def test_new_snapshot_requires_file_paused_and_idle(self):
        observations = [
            {"screenshot": "new.png", "paused": True, "controller_busy": False},
            {"screenshot": "new.png", "paused": False, "controller_busy": False},
            {"screenshot": "new.png", "paused": True, "controller_busy": True},
            {"screenshot": "new.png", "paused": True, "controller_busy": False},
        ]
        calls = 0
        def poll():
            nonlocal calls
            if calls == 0:
                self.acknowledge()
            else:
                if calls >= 2:
                    (self.base / "new.png").write_bytes(b"snapshot fixture")
                config.atomic_json(self.base / "status.json", observations[calls-1])
            calls += 1
            return None
        self.process.poll.side_effect = poll
        result = self.bridge.send("observe()")
        self.assertEqual(calls, 5)
        self.assertEqual(result, observations[-1])
        events = self.events()
        self.assertEqual([x["event"] for x in events], ["prepared", "accepted", "completed"])
        self.assertEqual(len({x["id"] for x in events}), 1)

    def test_snapshot_false_only_waits_for_dispatch_acknowledgement(self):
        self.process.poll.side_effect = lambda: self.acknowledge()
        self.assertIsNone(self.bridge.send("play_match()", snapshot=False))
        self.assertEqual(self.process.poll.call_count, 1)
        self.assertEqual([x["event"] for x in self.events()], ["prepared", "accepted", "completed"])

    def test_partial_or_non_object_observation_is_not_native_state(self):
        for text in ("{", "[]", "null", "not json"):
            with self.subTest(text=text):
                (self.base / "status.json").write_text(text, encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "observation"):
                    self.bridge.state()


class RuntimeTests(TemporaryCase):
    FROZEN = {
        "fighter.lua": "80e3a83c3db48c14825ec39aea8248440fed9432cd5d476f7d4b9d23db4c6a99",
        "play_core.lua": "28153fcf787baba6fe3c7636739d4ffc608302ddba58ac1fc4ec08e9285909e7",
        "selection.json": "cd2d6c8462f27e8569afdafde94491ed33b1ea988b64aef05b313251ec6042e4",
    }

    def test_packaged_runtime_provenance_and_frozen_sources(self):
        assets = runner.files("astra_play_sf2").joinpath("assets")
        provenance = json.loads(assets.joinpath("provenance.json").read_text(encoding="utf-8"))
        self.assertEqual(provenance["original_snapshot_sha256"],
                         "e8061e601bc28ca17805521f584f0705c21e0045228ecc47daf87b19683493a5")
        self.assertEqual(provenance["mame_version"], "0.288")
        self.assertEqual(provenance["rom"], "sf2")
        for name, info in provenance["files"].items():
            self.assertEqual(hashlib.sha256(assets.joinpath(name).read_bytes()).hexdigest(), info["sha256"], name)
        for name, digest in self.FROZEN.items():
            self.assertTrue(provenance["files"][name]["unchanged"])
            self.assertEqual(provenance["files"][name]["sha256"], digest)
        selection = json.loads(assets.joinpath("selection.json").read_text(encoding="utf-8"))
        self.assertEqual(set(map(int, selection)), set(runner.NAMES))
        self.assertEqual(selection["7"], "c22v4_dhalsim_far_ground_start_guard")
        self.assertEqual(selection["2"], "c19_blanka_low_lead_hold")

    def test_stage_runtime_copies_assets_into_new_isolated_run(self):
        hashes = runner.stage_runtime(self.root, 3)
        folder = self.root / "training/runtime"
        for name in ("bootstrap.lua", "session.lua", "settings.lua", *self.FROZEN):
            self.assertIn(name, hashes)
            self.assertEqual(runner.sha256(folder / name), hashes[name])
        self.assertIn("astra_difficulty_bits=4", (folder / "settings.lua").read_text())
        with self.assertRaises(FileExistsError):
            runner.stage_runtime(self.root, 7)
        self.assertIn("astra_difficulty_bits=4", (folder / "settings.lua").read_text())
        other = self.root / "another run"
        runner.stage_runtime(other, 7)
        self.assertIn("astra_difficulty_bits=0", (other / "training/runtime/settings.lua").read_text())

    def test_stage_rejects_tampered_asset_before_gameplay(self):
        source = runner.files("astra_play_sf2").joinpath("assets")
        package = self.root / "mock-package"
        assets = package / "assets"
        assets.mkdir(parents=True)
        for item in source.iterdir():
            if item.is_file():
                (assets / item.name).write_bytes(item.read_bytes())
        with (assets / "fighter.lua").open("ab") as f:
            f.write(b"\n-- altered\n")
        with patch.object(runner, "files", return_value=package), patch.object(runner.subprocess, "Popen") as launch:
            with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
                runner.stage_runtime(self.root / "staged", 3)
        launch.assert_not_called()

    def test_mame_command_preserves_paths_and_disables_external_control(self):
        value = {"mame": str(self.root / "mame executable"), "rom_dir": str(self.root / "rom ; $literal")}
        command = runner.mame_command(value)
        self.assertIsInstance(command, list)
        self.assertEqual(command[:3], [value["mame"], "sf2", "-noreadconfig"])
        self.assertEqual(command[command.index("-rompath")+1], value["rom_dir"])
        for option in ("-noautosave", "-norewind", "-nocheat", "-noplugins", "-noconsole"):
            self.assertIn(option, command)
        for option in ("-homepath", "-snapshot_directory"):
            self.assertEqual(command[command.index(option)+1], ".")
        for option in ("-keyboardprovider", "-joystickprovider", "-mouseprovider", "-lightgunprovider"):
            self.assertEqual(command[command.index(option)+1], "none")
        self.assertNotIn("-state", command)
        self.assertNotIn("-playback", command)
        self.assertNotIn("-reset", command)
        self.assertNotIn("-video", command)

    def test_session_lock_conflict_preserves_original_owner(self):
        with runner.session_lock(self.root):
            path = self.root / "verify.lock"
            original = path.read_bytes()
            self.assertEqual(json.loads(original)["pid"], os.getpid())
            with self.assertRaisesRegex(RuntimeError, "Session lock"):
                with runner.session_lock(self.root):
                    self.fail("second controller entered")
            self.assertEqual(path.read_bytes(), original)
        self.assertFalse(path.exists())

    def test_session_lock_cleanup_after_exception_and_reacquisition(self):
        with self.assertRaisesRegex(RuntimeError, "fixture interruption"):
            with runner.session_lock(self.root):
                raise RuntimeError("fixture interruption")
        self.assertFalse((self.root / "verify.lock").exists())
        with runner.session_lock(self.root):
            self.assertTrue((self.root / "verify.lock").exists())

    def test_session_lock_metadata_failure_closes_handle_before_cleanup(self):
        with patch.object(runner.platform, "node", side_effect=RuntimeError("host lookup failed")), \
             patch.object(runner.os, "close", wraps=os.close) as close:
            with self.assertRaisesRegex(RuntimeError, "host lookup failed"):
                with runner.session_lock(self.root):
                    self.fail("lock initialization should fail")
            close.assert_called_once()
        self.assertFalse((self.root / "verify.lock").exists())
        with runner.session_lock(self.root):
            self.assertTrue((self.root / "verify.lock").exists())

    def test_startup_failure_seals_isolated_run_and_terminates_only_its_process(self):
        value = {"mame": str(self.root / "fake mame"), "rom_dir": str(self.root / "roms"),
                 "data_dir": str(self.root / "data")}
        process = Mock()
        process.poll.return_value = None
        bridge = Mock()
        bridge.wait.side_effect = RuntimeError("fixture bootstrap failure")
        launches = []
        def launch(command, **kwargs):
            self.assertEqual(command, runner.mame_command(value))
            self.assertNotIn("shell", kwargs)
            self.assertEqual(Path(kwargs["cwd"]).parent, Path(value["data_dir"]) / "runs")
            self.assertEqual(kwargs["stdin"], subprocess.DEVNULL)
            cfg = ET.parse(Path(kwargs["cwd"]) / "cfg/sf2.cfg").find("./system/input/port")
            self.assertEqual(cfg.attrib["value"], "4")
            launches.append(kwargs)
            return process
        with patch.object(runner, "doctor", return_value={"ok": True}), \
             patch.object(runner.platform, "platform", return_value="fixture-platform"), \
             patch.object(runner.platform, "node", return_value="fixture-host"), \
             patch.object(runner.subprocess, "Popen", side_effect=launch) as popen, \
             patch.object(runner, "Bridge", return_value=bridge), \
             patch("astra_play_sf2.evidence.audit_run", return_value={"ok": False, "errors": ["fixture failure"]}), \
             patch("astra_play_sf2.evidence.write_report"), redirect_stdout(io.StringIO()):
            run, code = runner.verify(value, [3])
        self.assertEqual(code, 2)
        self.assertEqual(popen.call_count, 1)
        bridge.send.assert_not_called()
        process.terminate.assert_called_once()
        process.kill.assert_not_called()
        self.assertTrue(launches[0]["stdout"].closed)
        self.assertFalse((Path(value["data_dir"]) / "verify.lock").exists())
        manifest = transport.read_json(run / "run.json")
        self.assertEqual(manifest["status"], "invalid")
        self.assertIn("fixture bootstrap failure", manifest["error"])
        self.assertTrue((run / "evidence-sha256.json").is_file())

    def test_consecutive_budget_retains_losses_and_restarts_streak_per_level(self):
        # Fake gameplay outcomes exercise scheduling only, not clear certification.
        value = {"mame": "fake", "rom_dir": "fake", "data_dir": str(self.root / "data")}
        process = Mock()
        process.poll.return_value = None
        bridge = Mock()
        bridge.send.side_effect = [{"difficulty_bits": 4, "difficulty_mirror": 3, "effective_difficulty": 3},
                                   {"difficulty_bits": 0, "difficulty_mirror": 7, "effective_difficulty": 7}]
        played = []
        def attempt(run, bridge, record, speed, save):
            played.append(record["id"])
            record["outcome"] = "loss" if record["id"] == "l3-001" else "gameplay_clear"
            save()
        with patch.object(runner, "doctor", return_value={"ok": True}), \
             patch.object(runner.platform, "platform", return_value="fixture-platform"), \
             patch.object(runner.platform, "node", return_value="fixture-host"), \
             patch.object(runner.subprocess, "Popen", return_value=process) as popen, \
             patch.object(runner, "Bridge", return_value=bridge), \
             patch.object(runner, "_attempt", side_effect=attempt), \
             patch("astra_play_sf2.evidence.audit_run", return_value={"ok": True}), \
             patch("astra_play_sf2.evidence.write_report"), redirect_stdout(io.StringIO()):
            run, code = runner.verify(value, [3, 7], attempts=4, consecutive=2)
        self.assertEqual(code, 0)
        self.assertEqual(played, ["l3-001", "l3-002", "l3-003", "l7-001", "l7-002"])
        records = transport.read_json(run / "run.json")["attempts"]
        self.assertEqual(records[0]["outcome"], "loss")
        self.assertEqual(len(records), 5)
        self.assertEqual([call.args[0] for call in bridge.send.call_args_list],
                         ["session_validate()", "session_validate()"])
        self.assertEqual(popen.call_count, 2)
        self.assertEqual(process.terminate.call_count, 2)
        self.assertEqual([Path(c.kwargs["cwd"]).relative_to(run).as_posix() for c in popen.call_args_list],
                         ["sessions/l3", "sessions/l7"])
        for level in (3, 7):
            folder = run / f"sessions/l{level}"
            node = ET.parse(folder / "boot-config.xml").find("./system/input/port")
            self.assertEqual(node.attrib["value"], str(7-level))
            self.assertEqual((folder / "boot-config.xml").read_bytes(), (folder / "cfg/sf2.cfg").read_bytes())

    def test_unlatched_native_difficulty_aborts_before_attempt_without_retry(self):
        value = {"mame": "fake", "rom_dir": "fake", "data_dir": str(self.root / "data")}
        process = Mock()
        process.poll.return_value = None
        bridge = Mock()
        # DIP says Hardest but the game still uses cached Normal, as reproduced live.
        bridge.send.return_value = {"difficulty_bits": 0, "difficulty_mirror": 7, "effective_difficulty": 3}
        with patch.object(runner, "doctor", return_value={"ok": True}), \
             patch.object(runner.platform, "platform", return_value="fixture-platform"), \
             patch.object(runner.platform, "node", return_value="fixture-host"), \
             patch.object(runner.subprocess, "Popen", return_value=process) as launch, \
             patch.object(runner, "Bridge", return_value=bridge), \
             patch.object(runner, "_attempt") as attempt, \
             patch("astra_play_sf2.evidence.audit_run", return_value={"ok": False, "errors": ["wrong difficulty"]}), \
             patch("astra_play_sf2.evidence.write_report"), redirect_stdout(io.StringIO()):
            run, code = runner.verify(value, [7], attempts=3)
        self.assertEqual(code, 2)
        attempt.assert_not_called()
        bridge.send.assert_called_once_with("session_validate()")
        launch.assert_called_once()
        process.terminate.assert_called_once()
        manifest = transport.read_json(run / "run.json")
        self.assertEqual(manifest["status"], "invalid")
        self.assertEqual(manifest["attempts"], [])
        self.assertIn("Difficulty mismatch", manifest["error"])
        self.assertFalse((Path(value["data_dir"]) / "verify.lock").exists())
        self.assertTrue((run / "evidence-sha256.json").exists())

    def test_coin_gate_failure_never_inserts_coin_or_starts_session(self):
        for number, result in enumerate((None, {'kind': 'coin', 'status': 'timeout'},
                       {'kind': 'coin', 'status': 'error'}, {'kind': 'start', 'status': 'ready'})):
            with self.subTest(result=result):
                bridge = Mock()
                bridge.send.return_value = {'session_gate': result}
                attempt = {'id': 'l3-001', 'difficulty': 3}
                with self.assertRaisesRegex(RuntimeError, 'Native coin readiness failed'):
                    runner._attempt(self.root / str(number), bridge, attempt, 'normal', Mock())
                bridge.send.assert_called_once_with("speed('normal');wait_coin_ready(9000)")
                self.assertEqual(attempt['readiness']['coin'], result)

    def test_start_gate_timeout_does_not_retry_coin_or_send_start(self):
        bridge = Mock()
        def send(command):
            if 'wait_coin_ready' in command:
                return {'session_gate': {'kind': 'coin', 'status': 'ready', 'frames': 583}}
            if 'wait_start_ready' in command:
                return {'session_gate': {'kind': 'start', 'status': 'timeout', 'frames': 9000}}
            return {}
        bridge.send.side_effect = send
        attempt = {'id': 'l3-001', 'difficulty': 3}
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, 'Native start readiness failed'):
            runner._attempt(self.root, bridge, attempt, 'normal', Mock())
        commands = [c.args[0] for c in bridge.send.call_args_list]
        self.assertEqual(commands, ["speed('normal');wait_coin_ready(9000)",
            "session_begin('training/attempts/l3-001/session')",
            "speed('normal');act({{3,'C'},{1,''}})", "speed('normal');wait_start_ready(9000)"])
        self.assertEqual(attempt['readiness']['start']['status'], 'timeout')

    def test_initialization_failure_is_preserved_without_launch(self):
        value = {"mame": "fake", "rom_dir": "fake", "data_dir": str(self.root / "data")}
        for stage in ("stage_runtime", "boot_config"):
            with self.subTest(stage=stage), \
                 patch.object(runner, "doctor", return_value={"ok": True}), \
                 patch.object(runner, stage, side_effect=OSError("fixture initialization failure")), \
                 patch.object(runner.platform, "platform", return_value="fixture-platform"), \
                 patch.object(runner.platform, "node", return_value="fixture-host"), \
                 patch.object(runner.subprocess, "Popen") as launch, \
                 redirect_stdout(io.StringIO()):
                run, code = runner.verify(value, [7])
            self.assertEqual(code, 2)
            launch.assert_not_called()
            manifest = transport.read_json(run / "run.json")
            self.assertEqual(manifest['status'], 'invalid')
            self.assertIn('fixture initialization failure', manifest['error'])
            self.assertEqual(manifest['attempts'], [])
            self.assertTrue((run / 'evidence-sha256.json').exists())
            self.assertTrue((run / 'report.json').exists())
            self.assertFalse((Path(value['data_dir']) / 'verify.lock').exists())

    def test_invalid_verify_budget_is_rejected_before_any_process(self):
        values = [([], 1, "normal", None), ([2], 1, "normal", None),
                  ([True], 1, "normal", None), ([3], 0, "normal", None),
                  ([3], 2, "normal", 3), ([3], 1, "turbo", None)]
        with patch.object(runner, "doctor") as doctor, patch.object(runner.subprocess, "Popen") as launch:
            for levels, attempts, speed, consecutive in values:
                with self.subTest(values=(levels, attempts, speed, consecutive)):
                    with self.assertRaises(ValueError):
                        runner.verify({}, levels, attempts, speed, consecutive)
        doctor.assert_not_called()
        launch.assert_not_called()


class DifficultyOpeningTests(unittest.TestCase):
    def test_wrong_or_missing_internal_value_cannot_start_or_advance_intro(self):
        state = {"paused": True, "controller_busy": False, "difficulty_bits": 0,
                 "difficulty_mirror": 7, "effective_difficulty": 7, "timer_seconds": 99,
                 "p1": {"character": 4, "hp": 144, "displayed_hp": 144, "round_wins": 0,
                        "y": 40, "x": 100, "animation": 123},
                 "p2": {"character": 0, "hp": 144, "displayed_hp": 144, "round_wins": 0,
                        "y": 40, "x": 200, "animation": 456}}
        guard = make_opening_guard(0)
        require_difficulty(state, 7)
        self.assertTrue(guard[1](state, 0))
        for value in (3, None, True, "7"):
            with self.subTest(value=value):
                state["effective_difficulty"] = value
                with self.assertRaises(ValueError): require_difficulty(state, 7)
                advance = Mock()
                with self.assertRaises(ValueError): guard[3](state, 0, advance)
                advance.assert_not_called()

    def test_all_five_native_encodings_and_input_mirror(self):
        for level in range(3, 8):
            state = dict(difficulty_bits=7-level, difficulty_mirror=level, effective_difficulty=level)
            require_difficulty(state, level)
            state['difficulty_mirror'] = 0
            with self.assertRaises(ValueError): require_difficulty(state, level)


class CliTests(unittest.TestCase):
    def test_bad_syntax_exits_two_without_loading_config(self):
        cases = [[], ["unknown"], ["verify", "--difficulty", "8"],
                 ["verify", "--speed", "turbo"], ["verify", "--attempts", "abc"],
                 ["review", ".", "--attempt", "l3-001", "--reviewer", "Tester", "--decision", "yes"]]
        with patch.object(cli, "load_config") as load:
            for argv in cases:
                with self.subTest(argv=argv), redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        cli.main(argv)
                    self.assertEqual(raised.exception.code, 2)
        load.assert_not_called()

    def test_bad_budget_returns_two_without_preflight_or_launch(self):
        with patch.object(cli, "load_config", return_value={}), patch.object(runner, "doctor") as doctor:
            for args in (["--attempts", "0"], ["--attempts", "2", "--consecutive", "3"]):
                with self.subTest(args=args), redirect_stdout(io.StringIO()):
                    self.assertEqual(cli.main(["verify", "--difficulty", "3", *args]), 2)
        doctor.assert_not_called()

    def test_verify_default_arguments_and_runner_exit_status_are_preserved(self):
        for code in (0, 1, 2):
            with self.subTest(code=code), patch.object(cli, "load_config", return_value={"fixture": True}), \
                 patch.object(runner, "verify", return_value=(Path("fixture-run"), code)) as verify, \
                 redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(["verify", "--difficulty", "3"]), code)
                verify.assert_called_once_with({"fixture": True}, [3], 1, "normal", None)

    def test_all_difficulties_and_explicit_budget_reach_runner_once(self):
        with patch.object(cli, "load_config", return_value={}), \
             patch.object(runner, "verify", return_value=(Path("fixture-run"), 1)) as verify, \
             redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(["verify", "--difficulty", "all", "--attempts", "20", "--consecutive", "5", "--speed", "fast"]), 1)
        verify.assert_called_once_with({}, [3, 4, 5, 6, 7], 20, "fast", 5)


if __name__ == "__main__":
    unittest.main()
