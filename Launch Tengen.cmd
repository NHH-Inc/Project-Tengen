@echo off
rem Start the Tengen desktop app on Windows. First-time setup is in docs\DESKTOP.md.
cd /d "%~dp0"
if not exist ".venv-desktop\Scripts\pythonw.exe" (
  echo Run the Windows setup in docs\DESKTOP.md first.
  pause
  exit /b 1
)
start "" ".venv-desktop\Scripts\pythonw.exe" tengen.py %*
