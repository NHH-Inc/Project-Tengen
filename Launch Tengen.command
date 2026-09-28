#!/bin/zsh
set -e
cd "$(dirname "$0")"
if [[ ! -x .venv-desktop/bin/python ]]; then
  echo 'Run ./setup-desktop.sh first.'
  read -r '?Press Return to close.'
  exit 1
fi
exec .venv-desktop/bin/python tengen.py "$@"
