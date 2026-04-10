# Build a single-file Windows executable (run from repo root in PowerShell):
#   .\scripts\build_windows.ps1
#
# Creates .venv if needed, installs requirements.txt + scripts/requirements-build.txt,
# runs PyInstaller on TitanLandingSimulator.spec, copies to standalone\windows\.
#
# Output: standalone\windows\TitanLandingSimulator.exe

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $root

$venvPy = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $venvPy)) {
    try {
        py -3 -m venv .venv
    } catch {
        python -m venv .venv
    }
    if (-not (Test-Path $venvPy)) {
        Write-Error "Could not create .venv (need py -3 or python on PATH)."
    }
}

# Активируем venv для корректной работы pip
$activatePs1 = Join-Path $root '.venv\Scripts\Activate.ps1'
& $activatePs1

# Теперь pip работает без проблем
pip install --upgrade pip
pip install -r requirements.txt -r scripts\requirements-build.txt

Remove-Item -Recurse -Force -ErrorAction SilentlyContinue (Join-Path $root 'build'), (Join-Path $root 'dist')
& $venvPy -m PyInstaller --noconfirm --clean (Join-Path $root 'TitanLandingSimulator.spec')

$exe = Join-Path $root 'dist\TitanLandingSimulator.exe'
if (-not (Test-Path $exe)) {
    Write-Error "Build failed: not found $exe"
}
$destDir = Join-Path $root 'standalone\windows'
New-Item -ItemType Directory -Force -Path $destDir | Out-Null
$dest = Join-Path $destDir 'TitanLandingSimulator.exe'
Copy-Item -Force $exe $dest
Write-Host "OK: $dest"

deactivate  # Деактивируем venv