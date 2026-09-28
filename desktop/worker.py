"""Analysis subprocess protocol: one JSON progress message per line."""
import argparse
import json
import os
import signal
from pathlib import Path
import sys
import traceback

from desktop.core import AnalysisOptions, analyze, read_json


def emit(**message):
    print(json.dumps(message), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("options", type=Path)
    args = parser.parse_args()
    # Give the UI one process group to cancel, including the tracker child.
    if os.name == "posix" and os.getpgrp() != os.getpid():
        os.setsid()
    def interrupted(signum, frame):
        raise InterruptedError("Analysis canceled")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        path = analyze(AnalysisOptions(**read_json(args.options)),
                       lambda value, stage: emit(progress=value, stage=stage))
        emit(complete=str(path))
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        emit(error=str(exc))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
