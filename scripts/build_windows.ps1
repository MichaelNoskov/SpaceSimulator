# Build a single-file Windows executable (run from repo root in PowerShell):
#   .\scripts\build_windows.ps1
# Output: standalone\windows\TitanLandingSimulator.exe

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $root

$venvPy = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $venvPy)) {
    py -3 -m venv .venv
    if (-not (Test-Path $venvPy)) {
        python -m venv .venv
    }
}
& (Join-Path $root '.venv\Scripts\pip.exe') install --upgrade pip
& (Join-Path $root '.venv\Scripts\pip.exe') install -r requirements.txt pyinstaller
& (Join-Path $root '.venv\Scripts\pyinstaller.exe') --noconfirm --clean TitanLandingSimulator.spec

$exe = Join-Path $root 'dist\TitanLandingSimulator.exe'
if (-not (Test-Path $exe)) {
    Write-Error "Build failed: not found $exe"
}
$destDir = Join-Path $root 'standalone\windows'
New-Item -ItemType Directory -Force -Path $destDir | Out-Null
$dest = Join-Path $destDir 'TitanLandingSimulator.exe'
Copy-Item -Force $exe $dest
Write-Host "OK: $dest"
