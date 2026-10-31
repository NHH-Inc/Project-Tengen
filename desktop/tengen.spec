# PyInstaller spec for the Tengen desktop app. EXPERIMENTAL -- a starting point, not a finished
# build. See docs/DESKTOP.md "Packaging to a Windows app (experimental)" for the two known blockers
# (freeze-aware paths, and re-invoking the exe for the analysis/publish subprocess) that must be
# solved in app code before the produced exe actually runs a job.
#
#   .venv-desktop\Scripts\pyinstaller desktop\tengen.spec       (or: powershell build-windows.ps1)
#
# This captures the tedious part -- bundling PySide6, torch and ultralytics correctly, and shipping
# the season/contract data. The model weights are NOT bundled (hundreds of MB and they change):
# ship data/robot-v3.pt next to the exe, in data/ beside Tengen.exe.

from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules

ROOT = Path(SPECPATH).resolve().parent  # repo root (SPECPATH is the spec's dir, desktop/)

datas, binaries, hiddenimports = [], [], []
for package in ("ultralytics", "torch", "torchvision", "av", "lap", "scipy", "yt_dlp"):
    try:
        pkg_datas, pkg_binaries, pkg_hidden = collect_all(package)
    except Exception:
        continue  # torchvision is optional; skip anything not installed
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hidden
# tengen.py imports these lazily (inside functions), so collect them explicitly or the frozen
# build will not contain the worker, tracker, publisher or app.
for package in ("ingest", "desktop", "training", "tools"):
    hiddenimports += collect_submodules(package)

# Read-only data the app loads at runtime, shipped inside the bundle.
for rel in ("contracts", "analysis/config", "fixtures/2026casf_qm42", "brand/tengen-mark.svg"):
    source = ROOT / rel
    if source.exists():
        datas.append((str(source), rel))

ICON = ROOT / "brand" / "tengen.ico"  # optional; make one from brand/tengen-mark.svg if wanted

a = Analysis(
    [str(ROOT / "tengen.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz, a.scripts, [], exclude_binaries=True, name="Tengen",
    console=False, icon=str(ICON) if ICON.exists() else None,
)
coll = COLLECT(exe, a.binaries, a.datas, name="Tengen")  # one-folder build (faster start, big libs)
