# Install this checkout locally. Does not install MAME/Python or start gameplay.
[CmdletBinding()]
param([string]$Python = "")

$ErrorActionPreference = "Stop"
$astraRepoDir = Split-Path -Parent $PSScriptRoot
$astraPythonArgs = @()

if ($Python) {
    $astraPythonCommand = $Python
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    $astraPythonCommand = "py"
    $astraPythonArgs = @("-3")
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $astraPythonCommand = "python"
} else {
    throw "Python not found. Install Python 3.10+ from https://www.python.org/downloads/"
}

& $astraPythonCommand @astraPythonArgs -c 'import sys; sys.exit(sys.version_info < (3, 10))'
if ($LASTEXITCODE -ne 0) { throw "Python 3.10+ is required." }

$astraVenv = Join-Path $astraRepoDir ".venv"
& $astraPythonCommand @astraPythonArgs -m venv $astraVenv
if ($LASTEXITCODE -ne 0) { throw "Could not create the local virtual environment." }

$astraVenvPython = Join-Path $astraVenv "Scripts\python.exe"
$astraCli = Join-Path $astraVenv "Scripts\astra-sf2.exe"
& $astraVenvPython -m pip install $astraRepoDir
if ($LASTEXITCODE -ne 0) { throw "Package installation failed." }

Write-Host "`nInstalled in $astraVenv"
& $astraCli doctor
if ($LASTEXITCODE -ne 0) {
    Write-Host "`nDoctor did not pass. Resolve the diagnostics above, then configure and retry."
}
Write-Host "`nNext steps (replace the example paths):"
Write-Host ('  & "{0}" configure --mame "C:\MAME\mame.exe" --rom-dir "C:\MAME\roms"' -f $astraCli)
Write-Host ('  & "{0}" doctor' -f $astraCli)
Write-Host ('  & "{0}" verify --difficulty 3' -f $astraCli)
exit 0
