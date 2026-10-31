# Build the Tengen desktop app into a Windows executable (EXPERIMENTAL).
# See docs/DESKTOP.md "Packaging to a Windows app (experimental)".
#
#   powershell -ExecutionPolicy Bypass -File build-windows.ps1
#
# Output: dist\Tengen\Tengen.exe (a one-folder build). Ship data\robot-v3.pt next to the exe,
# under dist\Tengen\data\. The build is large (torch) and takes several minutes.

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$python = Join-Path $root ".venv-desktop\Scripts\python.exe"
if (-not (Test-Path $python)) { $python = "python" }  # fall back to whatever python is on PATH

Write-Host "Installing build dependencies..." -ForegroundColor Cyan
& $python -m pip install -r (Join-Path $root "requirements-desktop.txt")
& $python -m pip install -r (Join-Path $root "requirements-desktop-build.txt")

Write-Host "Running PyInstaller..." -ForegroundColor Cyan
& $python -m PyInstaller --noconfirm (Join-Path $root "desktop\tengen.spec")

$exe = Join-Path $root "dist\Tengen\Tengen.exe"
if (Test-Path $exe) {
  Write-Host "Built $exe" -ForegroundColor Green
  Write-Host "Now copy your model weights into dist\Tengen\data\ (e.g. data\robot-v3.pt)."
} else {
  Write-Host "Build finished but Tengen.exe was not found; check the PyInstaller output above." -ForegroundColor Yellow
}
