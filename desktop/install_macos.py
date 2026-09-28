"""Install a small Finder launcher pointing to this checkout and its Python environment."""
import plistlib
from pathlib import Path
import shlex
import sys

from desktop.core import ROOT


def main():
    if sys.platform != "darwin":
        raise SystemExit("This installer is for macOS. Elsewhere, run python tengen.py.")
    app = Path.home() / "Applications/Tengen.app"
    marker = app / "Contents/tengen-project.txt"
    if app.exists() and (not marker.exists() or marker.read_text().strip() != str(ROOT)):
        raise SystemExit(f"Another application already exists at {app}; it was left unchanged.")
    executable = app / "Contents/MacOS/Tengen"
    executable.parent.mkdir(parents=True, exist_ok=True)
    (app / "Contents/Resources").mkdir(exist_ok=True)
    executable.write_text(
        "#!/bin/zsh\n"
        "export PATH=/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin\n"
        f"cd {shlex.quote(str(ROOT))} || exit 1\n"
        f"exec {shlex.quote(str(ROOT / '.venv-desktop/bin/python'))} tengen.py \"$@\"\n"
    )
    executable.chmod(0o755)
    marker.write_text(str(ROOT) + "\n")
    with (app / "Contents/Info.plist").open("wb") as file:
        plistlib.dump(dict(CFBundleName="Tengen", CFBundleDisplayName="Tengen",
                          CFBundleIdentifier="local.tengen.desktop", CFBundleExecutable="Tengen",
                          CFBundlePackageType="APPL", CFBundleVersion="1", CFBundleShortVersionString="1.0",
                          NSHighResolutionCapable=True), file)
    print(app)


if __name__ == "__main__":
    main()
