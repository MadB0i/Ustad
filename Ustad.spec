# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Ustad standalone executable.

Usage:
    pyinstaller Ustad.spec

The resulting executable will be in dist/Ustad.exe (Windows) or dist/Ustad (Linux/macOS).

Note: The executable is large (~2-4GB) due to PyTorch/transformers dependencies.
"""

block_cipher = None

a = Analysis(
    ['backend/server.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('frontend', 'frontend'),
        ('backend', 'backend'),
    ],
    hiddenimports=[
        'uvicorn.logging',
        'uvicorn.loops.auto',
        'uvicorn.protocols.http.auto',
        'uvicorn.protocols.http.h11_impl',
        'uvicorn.protocols.websockets.auto',
        'uvicorn.lifespan.on',
        'transformers',
        'peft',
        'bitsandbytes',
        'accelerate',
        'safetensors',
        'huggingface_hub',
        'httpx',
        'fastapi',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'scipy',
        'pandas',
        'jupyter',
        'IPython',
        'notebook',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='Ustad',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # Set to False for windowed app (no console)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
