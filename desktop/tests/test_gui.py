"""Exercise real Qt playback and native review controls with the bundled MP4."""
import time

import pytest
from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtWidgets import QApplication, QDialog, QInputDialog

from desktop.app import MainWindow, STYLE
from desktop.core import FIXTURE


@pytest.fixture(scope="module")
def app():
    instance = QApplication.instance() or QApplication([])
    instance.setStyle("Fusion")
    instance.setStyleSheet(STYLE)
    return instance


def until(app, predicate, seconds=10):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return
        time.sleep(0.02)
    assert predicate(), "Qt condition did not become true before timeout"


def test_play_seek_pause_and_review(app, tmp_path, monkeypatch):
    window = MainWindow()
    window.show()
    window.open_review(FIXTURE, tmp_path / "corrections.json")
    try:
        assert len(window.tracks) == 7
        window.player.play()
        until(app, lambda: window.canvas.frames_received >= 5 and window.player.position() > 100)
        assert not window.canvas.image.isNull()
        assert window.player.error() == QMediaPlayer.Error.NoError
        window.player.setPosition(30000)
        until(app, lambda: window.canvas.time >= 30)
        window.player.pause()
        assert window.player.playbackState() == QMediaPlayer.PlaybackState.PausedState
        window.track_table.selectRow(0)
        monkeypatch.setattr(QInputDialog, "getInt", lambda *a, **k: (1234, True))
        window.assign_team()
        assert window.tracks[0]["team"] == 1234
        window.raw.setChecked(True)
        assert window.tracks[0]["team"] != 1234
        window.raw.setChecked(False)
        window.undo()
        assert window.tracks[0]["team"] != 1234
        # Exercise the actual editor constructor/schema and save path.
        monkeypatch.setattr(QDialog, "exec", lambda self: QDialog.DialogCode.Accepted)
        before = len(window.events)
        window.add_event()
        assert len(window.events) == before + 1
        window.undo()
        assert len(window.events) == before
        assert window.grab().save(str(tmp_path / "desktop-demo.png"))
    finally:
        window.close()
        app.processEvents()


def test_worker_failure_returns_control_to_gui(app, tmp_path):
    from desktop.core import AnalysisOptions
    window = MainWindow()
    errors = []
    window.error = lambda message: errors.append(str(message))
    window.open_video(FIXTURE / "segment.mp4")
    try:
        window.run_analysis(AnalysisOptions(video=str(FIXTURE / "segment.mp4"), model=str(tmp_path / "missing.pt")))
        until(app, lambda: window.process is None)
        assert errors and "Robot model not found" in errors[0]
        assert window.analyze_button.isEnabled()
        assert not window.cancel.isEnabled()
    finally:
        window.close()


def test_cancel_stops_worker_and_detector(app, tmp_path):
    import psutil
    from desktop.core import AnalysisOptions, RUNS, read_json
    before = set(RUNS.glob("*/job.json"))
    window = MainWindow()
    errors = []
    window.error = lambda message: errors.append(str(message))
    window.open_video(FIXTURE / "segment.mp4")
    try:
        window.run_analysis(AnalysisOptions(video=str(FIXTURE / "segment.mp4")))
        until(app, lambda: window.statusBar().currentMessage() == "Analyzing · decoding", seconds=15)
        pid = window.process.processId()
        children = psutil.Process(pid).children(recursive=True)
        window.cancel_analysis()
        until(app, lambda: window.process is None)
        assert not errors
        assert not psutil.pid_exists(pid)
        assert all(not p.is_running() or p.status() == psutil.STATUS_ZOMBIE for p in children)
        paths = set(RUNS.glob("*/job.json")) - before
        assert len(paths) == 1
        job_path = paths.pop()
        job = read_json(job_path)
        assert job["status"] == "failed"
        assert job["error"] == "Analysis canceled"
        assert window.analyze_button.isEnabled()
        # Keep test-only canceled jobs out of the user's run library.
        import shutil
        shutil.move(str(job_path.parent), str(tmp_path / "canceled-run"))
    finally:
        window.close()
