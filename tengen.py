"""Start the native application: python tengen.py [video.mp4 | --demo].

A frozen Windows build cannot run `Tengen.exe -m desktop.worker`, so it re-invokes the exe with a
flag instead, which this entry point dispatches to the right helper (see desktop/core.child_process):

    Tengen.exe --run-worker  <request.json>       # the analysis subprocess
    Tengen.exe --run-tracker <track_yolo args>    # the YOLO tracker the analysis spawns
    Tengen.exe --run-publish <run-dir> [args...]  # publishing to the hosted server

From source these paths are never taken (the app uses `python -m ...` directly); they exist only so
the one bundled exe can be both the GUI and its own background workers.
"""
import sys


def _dispatch() -> int:
    flag = sys.argv[1] if len(sys.argv) >= 2 else ""
    if flag == "--run-worker":
        from desktop.worker import main as worker_main
        sys.argv = [sys.argv[0], *sys.argv[2:]]
        return worker_main() or 0
    if flag == "--run-tracker":
        from training.track_yolo import main as tracker_main
        sys.argv = [sys.argv[0], *sys.argv[2:]]
        return tracker_main() or 0
    if flag == "--run-publish":
        from tools.publish import main as publish_main
        return publish_main(sys.argv[2:])
    from desktop.app import main
    return main()


if __name__ == "__main__":
    raise SystemExit(_dispatch())
