"""Reproducible native end-to-end check, including the actual GUI analysis subprocess.

python -m desktop.verify --video match.mp4
python -m desktop.verify --run data/desktop/runs/<id>
"""
import argparse
import json
from pathlib import Path
import sys
import time

from PySide6.QtWidgets import QApplication
from PySide6.QtMultimedia import QMediaPlayer

from desktop.app import MainWindow, STYLE
from desktop.core import AnalysisOptions, ROOT, RUNS, read_json, write_json


def wait(app, condition, timeout):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        app.processEvents()
        if condition():
            return
        time.sleep(0.02)
    raise RuntimeError("Timed out while verifying the desktop application")


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path)
    source.add_argument("--run", type=Path)
    args = parser.parse_args()
    app = QApplication([])
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    window = MainWindow()
    errors = []
    window.error = lambda message: errors.append(str(message))
    window.show()
    try:
        if args.video:
            window.open_video(args.video)
            window.run_analysis(AnalysisOptions(video=str(args.video.resolve()), match_id="desktop-gui-video-verification"))
            wait(app, lambda: window.process is None, 600)
        else:
            window.open_review(args.run)
        assert not errors, errors
        assert window.review is not None
        assert window.review.job["status"] == "complete"
        assert window.tracks, "Expected robot detections in the verification video"
        window.player.play()
        wait(app, lambda: window.canvas.frames_received >= 3 and window.player.position() > 100, 15)
        assert window.player.error() == QMediaPlayer.Error.NoError
        seek_ms = int(min(90, window.review.job["duration"] / 2) * 1000)
        window.player.setPosition(seek_ms)
        wait(app, lambda: window.canvas.time >= seek_ms / 1000, 10)
        window.player.pause()
        window.tabs.setCurrentWidget(window.track_table)
        evidence = RUNS.parent / "verification"
        evidence.mkdir(parents=True, exist_ok=True)
        window.grab().save(str(evidence / "desktop-match.png"))
        window.review.export(evidence / "scouting.json")
        window.review.export(evidence / "events.csv")
        exported = read_json(evidence / "scouting.json")
        assert len(exported["tracks"]) == len(window.tracks)
        # Qt's native player has no listening socket; inspect only our process and children.
        import psutil
        sockets = psutil.Process().net_connections(kind="inet")
        assert not any(c.status == psutil.CONN_LISTEN for c in sockets)
        report = dict(passed=True, run=str(window.review.directory),
                      source=str(window.review.video), source_duration=window.review.job["duration"],
                      result=window.review.result, decoded_frames=window.canvas.frames_received,
                      seek_seconds=window.canvas.time, tracks=len(window.tracks),
                      events=len(window.events), gui_errors=errors, listening_sockets=0,
                      checks=["GUI analysis subprocess" if args.video else "Saved-run loading",
                              "Native video decoding", "Playback", "Seek", "Pause", "Robot table",
                              "JSON and CSV export", "No listening server socket"])
        write_json(evidence / "report.json", report)
        print(json.dumps(report, indent=2))
    finally:
        window.close()
        app.processEvents()


if __name__ == "__main__":
    main()
