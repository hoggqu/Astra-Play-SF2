# Installation

Install Python and MAME yourself, then install this project into a virtual environment. The project does not download emulator binaries or game data.

## Prerequisites

- **Python 3.10 or newer**, with `venv` and pip. Official installers are available from [Python.org](https://www.python.org/downloads/). Python 3.10 and 3.12 are the CI test versions; this is not a promise that every newer interpreter has been tested.
- **MAME 0.288**, specifically. Use the [official 0.288 release](https://github.com/mamedev/mame/releases/tag/mame0288), not an unpinned latest-release link. The [MAME downloads page](https://www.mamedev.org/release.html) provides official Windows binaries and source information. For macOS or Linux builds, follow [MAME's build documentation](https://docs.mamedev.org/initialsetup/compilingmame.html) with the `mame0288` source tag; that documentation may describe a newer release, so keep the source version pinned.
- Your own ROM files compatible with MAME's **`sf2` / World 910522** driver. This project does not include or locate a ROM download. Other Street Fighter II editions or revisions are not interchangeable with the validated memory layout.

Do not substitute a newer package-manager MAME build and assume it has the same behavior. `doctor` checks the configured environment before a run. Configure the actual emulator executable, not an application folder or an unrelated launcher.

## macOS and Linux

From the project checkout:

```sh
bash scripts/bootstrap.sh
source .venv/bin/activate
astra-sf2 configure --mame "/absolute/path/to/mame" --rom-dir "/absolute/path/to/roms"
astra-sf2 doctor
```

To select a particular Python executable:

```sh
ASTRA_PYTHON=/absolute/path/to/python3.12 bash scripts/bootstrap.sh
```

Manual equivalent:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install .
.venv/bin/astra-sf2 --help
```

If your Linux Python installation lacks `venv` or pip, use your distribution's Python installation instructions to add them, then retry. The bootstrap does not run a system package manager or `sudo`.

## Windows PowerShell

```powershell
.\scripts\bootstrap.ps1
.\.venv\Scripts\astra-sf2.exe configure --mame "C:\MAME\mame.exe" --rom-dir "D:\Games\roms"
.\.venv\Scripts\astra-sf2.exe doctor
```

The script tries the Python launcher (`py -3`), then `python`. You can choose an executable explicitly:

```powershell
.\scripts\bootstrap.ps1 -Python "C:\Python312\python.exe"
```

If local PowerShell policy prevents executing the script, use the manual commands below. You do not need to change machine-wide execution policy or activate a PowerShell script:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install .
.\.venv\Scripts\astra-sf2.exe --help
```

## Configuration and local data

Configuration defaults to `~/.astra-play-sf2/config.json`, where `~` is your user home directory on each OS. Set the `ASTRA_SF2_CONFIG` environment variable to use a different configuration file (for example, an isolated test configuration); use the same override for `configure` and subsequent commands. Set a separate writable data directory if desired:

```sh
astra-sf2 configure --mame "/path/to/mame" --rom-dir "/path/to/roms" --data-dir "/path/to/astra-data"
astra-sf2 doctor
```

New runs go under the configured data directory's `runs` folder. Keep paths containing spaces quoted. Keep the entire run directory if you want to audit or review a result later.

Bootstrap installation can succeed while `doctor` reports that configuration or MAME is missing. The script prints that diagnostic and setup guidance; **it does not mean the emulator is ready**. Run `doctor` again after configuring and resolve its reported failures before `verify`.

Installing or upgrading the Python package does not authorize changing a running session. Finish or stop the active runner first, preserve its evidence, and start a new verification session after upgrading. Do not use a second controller to manipulate the emulator.
