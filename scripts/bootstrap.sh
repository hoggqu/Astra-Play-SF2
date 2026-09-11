#!/usr/bin/env bash
# Install this checkout locally. Does not install MAME/Python or start gameplay.
set -euo pipefail

astra_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
astra_repo_dir="$(cd -- "$astra_script_dir/.." && pwd)"
astra_python="${ASTRA_PYTHON:-python3}"

if ! command -v "$astra_python" >/dev/null 2>&1; then
  printf 'Python not found: %s\nInstall Python 3.10+ from https://www.python.org/downloads/\n' "$astra_python" >&2
  exit 1
fi
"$astra_python" -c 'import sys; sys.exit("Python 3.10+ is required") if sys.version_info < (3, 10) else None'
"$astra_python" -m venv "$astra_repo_dir/.venv"
astra_venv_python="$astra_repo_dir/.venv/bin/python"
astra_cli="$astra_repo_dir/.venv/bin/astra-sf2"
"$astra_venv_python" -m pip install "$astra_repo_dir"

printf '\nInstalled in %s\n' "$astra_repo_dir/.venv"
if ! "$astra_cli" doctor; then
  printf '\nDoctor did not pass. Resolve the diagnostics above, then configure and retry.\n'
fi
printf '\nNext steps (replace the example paths):\n'
printf '  source "%s/.venv/bin/activate"\n' "$astra_repo_dir"
printf '  astra-sf2 configure --mame "/path/to/mame" --rom-dir "/path/to/roms"\n'
printf '  astra-sf2 doctor\n'
printf '  astra-sf2 verify --difficulty 3\n'
