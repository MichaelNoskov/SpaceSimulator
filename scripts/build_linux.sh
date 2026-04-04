#!/usr/bin/env bash
# Rebuild the Linux one-file binary (run from repo root):
#   ./scripts/build_linux.sh
# Requires: .venv with pyinstaller, or set PYINSTALLER to pyinstaller binary.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

if [[ -x "$root/.venv/bin/pyinstaller" ]]; then
  PYI="$root/.venv/bin/pyinstaller"
else
  PYI="${PYINSTALLER:-pyinstaller}"
fi

rm -rf build dist
"$PYI" --noconfirm --clean TitanLandingSimulator.spec
cp -f "$root/dist/TitanLandingSimulator" "$root/standalone/linux/TitanLandingSimulator"
chmod +x "$root/standalone/linux/TitanLandingSimulator"
echo "OK: $root/standalone/linux/TitanLandingSimulator"
