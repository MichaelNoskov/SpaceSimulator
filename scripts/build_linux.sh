#!/usr/bin/env bash
# Rebuild the Linux one-file binary (from repo root):
#   ./scripts/build_linux.sh
#
# Creates .venv if missing, installs requirements.txt + scripts/requirements-build.txt,
# runs PyInstaller on TitanLandingSimulator.spec, copies to standalone/linux/.
#
# Override Python:   PYTHON=/usr/bin/python3.12 ./scripts/build_linux.sh
# Use existing PyInstaller only (no venv pip): PYINSTALLER=/path/to/pyinstaller ./scripts/build_linux.sh

set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

if [[ -n "${PYINSTALLER:-}" ]]; then
  rm -rf build dist
  "$PYINSTALLER" --noconfirm --clean "$root/TitanLandingSimulator.spec"
else
  PY="${PYTHON:-python3}"
  if [[ ! -x "$root/.venv/bin/python" ]]; then
    "$PY" -m venv "$root/.venv"
  fi
  # shellcheck disable=SC1091
  source "$root/.venv/bin/activate"
  pip install --upgrade pip
  pip install -r "$root/requirements.txt" -r "$root/scripts/requirements-build.txt"
  rm -rf build dist
  python -m PyInstaller --noconfirm --clean "$root/TitanLandingSimulator.spec"
fi

out="$root/dist/TitanLandingSimulator"
if [[ ! -f "$out" ]]; then
  echo "Build failed: missing $out" >&2
  exit 1
fi

install -d "$root/standalone/linux"
cp -f "$out" "$root/standalone/linux/TitanLandingSimulator"
chmod +x "$root/standalone/linux/TitanLandingSimulator"
echo "OK: $root/standalone/linux/TitanLandingSimulator"
