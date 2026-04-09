# -*- mode: python ; coding: utf-8 -*-
# One-file bundle for main.py: pygame UI + render.py. data/ is bundled at data/.
# Optional render_gl.py (PyOpenGL) is not imported by main.py and is excluded here
# so the build works without PyOpenGL installed.

from PyInstaller.utils.hooks import collect_all

datas = [('data', 'data')]
binaries = []
hiddenimports = []

tmp_ret = collect_all('pygame')
datas += tmp_ret[0]
binaries += tmp_ret[1]
hiddenimports += tmp_ret[2]

# Explicit subpackages (helps one-file edge cases for digital_twin.models.*).
hiddenimports += [
    'digital_twin',
    'digital_twin.models.atmosphere',
    'digital_twin.models.wind',
    'control',
    'flight_program',
]

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='TitanLandingSimulator',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
