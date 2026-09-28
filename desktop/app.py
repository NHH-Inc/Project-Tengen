"""Native Qt interface for Tengen. All media and analysis stay on the filesystem."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import sys
import uuid

from PySide6.QtCore import Qt, QProcess, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer, QMediaMetaData
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QFileDialog, QFormLayout, QHBoxLayout, QHeaderView, QInputDialog, QLabel,
    QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QPlainTextEdit,
    QProgressBar, QPushButton, QSlider, QSplitter, QTabWidget, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from desktop.core import (AnalysisOptions, DEFAULT_MODEL, DEFAULT_BALL_CONFIG, FIXTURE, ROOT, RUNS, Review,
                          read_json, write_json)
from ingest.yolo_orchestrator import phase_at, MODEL_CROP
from desktop.rendering import VideoCanvas, HeatMap, FULL_FRAME

STYLE = """
QWidget { background: #111820; color: #e6edf3; font-size: 13px; }
QMainWindow { background: #111820; }
QLabel#brand { font-size: 27px; font-weight: 700; color: #5ee0c0; }
QLabel#title { font-size: 20px; font-weight: 600; }
QLabel#muted { color: #9caebd; }
QPushButton { background: #243443; border: 1px solid #344b5d; border-radius: 6px; padding: 8px 12px; }
QPushButton:hover { background: #335266; }
QPushButton:disabled { color: #677580; background: #19232c; }
QPushButton#primary { background: #167e6c; border-color: #30baa0; color: white; }
QLineEdit, QComboBox, QDoubleSpinBox { background: #1a2530; border: 1px solid #344b5d; padding: 7px; border-radius: 4px; }
QListWidget, QTableWidget, QPlainTextEdit { background: #151f29; border: 1px solid #293b49; border-radius: 5px; }
QListWidget::item { padding: 12px 6px; }
QListWidget::item:selected { background: #214b50; }
QHeaderView::section { background: #243443; padding: 6px; border: 0; }
QTabBar::tab { background: #1a2530; padding: 10px 18px; }
QTabBar::tab:selected { background: #28505a; }
QProgressBar { border: 1px solid #344b5d; border-radius: 4px; text-align: center; }
QProgressBar::chunk { background: #167e6c; }
QSlider::groove:horizontal { height: 5px; background: #344b5d; }
QSlider::handle:horizontal { background: #5ee0c0; width: 13px; margin: -5px 0; border-radius: 6px; }
"""


def button(text, callback, primary=False):
    item = QPushButton(text)
    item.clicked.connect(callback)
    if primary:
        item.setObjectName("primary")
    return item


def table(headers):
    widget = QTableWidget(0, len(headers))
    widget.setHorizontalHeaderLabels(headers)
    widget.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    widget.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
    widget.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
    widget.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    widget.verticalHeader().hide()
    return widget


def fill_table(widget, rows):
    widget.setRowCount(len(rows))
    for y, row in enumerate(rows):
        for x, value in enumerate(row):
            widget.setItem(y, x, QTableWidgetItem("—" if value is None else str(value)))


class AnalysisDialog(QDialog):
    def __init__(self, parent, source):
        super().__init__(parent)
        self.setWindowTitle("Analyze match video")
        self.resize(660, 440)
        layout = QFormLayout(self)
        self.model = QLineEdit(str(DEFAULT_MODEL))
        model_row = QHBoxLayout()
        model_row.addWidget(self.model)
        model_row.addWidget(button("Browse…", self.select_model))
        layout.addRow("Robot model", model_row)
        layout.addRow("Video", QLabel(Path(source).name))
        self.match = QLineEdit()
        self.match.setPlaceholderText("Optional match identifier")
        layout.addRow("Match", self.match)
        self.season = QComboBox()
        self.season.addItems(sorted(p.stem for p in (ROOT / "contracts/seasons").glob("*.json")))
        self.season.setCurrentText("2026")
        layout.addRow("Season", self.season)
        self.device = QComboBox()
        self.device.addItems(["cpu", "mps", "0"])
        self.device.setToolTip("cpu: portable; mps: Apple GPU; 0: NVIDIA GPU")
        layout.addRow("Compute device", self.device)
        self.confidence = QDoubleSpinBox()
        self.confidence.setRange(0.01, 1)
        self.confidence.setSingleStep(0.05)
        self.confidence.setValue(0.25)
        layout.addRow("Detection confidence", self.confidence)
        self.ball = QLineEdit(str(DEFAULT_BALL_CONFIG))
        ball_row = QHBoxLayout()
        ball_row.addWidget(self.ball)
        ball_row.addWidget(button("Browse…", lambda: self.pick_json(self.ball)))
        layout.addRow("Ball scouting config", ball_row)
        self.calibration = QLineEdit()
        cal_row = QHBoxLayout()
        cal_row.addWidget(self.calibration)
        cal_row.addWidget(button("Browse…", lambda: self.pick_json(self.calibration)))
        layout.addRow("Field calibration", cal_row)
        self.auto_calibrate = QCheckBox("Try automatic AprilTag field calibration")
        self.auto_calibrate.setChecked(True)
        layout.addRow(self.auto_calibrate)
        self.annotated = QCheckBox("Save a video with detection overlays")
        self.annotated.setChecked(True)
        layout.addRow(self.annotated)
        note = QLabel("Ball scouting needs a camera-specific config. Without it, analysis produces robot tracks and match boundaries. Team numbers can be assigned in review.")
        note.setWordWrap(True)
        note.setObjectName("muted")
        layout.addRow(note)
        controls = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        controls.accepted.connect(self.accept)
        controls.rejected.connect(self.reject)
        layout.addRow(controls)
        self.source = source

    def select_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Robot detector", str(ROOT), "YOLO models (*.pt)")
        if path:
            self.model.setText(path)

    def pick_json(self, target):
        path, _ = QFileDialog.getOpenFileName(self, "Configuration", str(ROOT / "analysis/config"), "JSON (*.json)")
        if path:
            target.setText(path)

    def options(self):
        return AnalysisOptions(video=self.source, model=self.model.text(), season=int(self.season.currentText()),
                               match_id=self.match.text(), device=self.device.currentText(),
                               confidence=self.confidence.value(), ball_config=self.ball.text(),
                               homography=self.calibration.text(), auto_homography=self.auto_calibrate.isChecked(),
                               annotated=self.annotated.isChecked())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tengen • FRC Auto-Scouting")
        self.resize(1350, 880)
        self.review = None
        self.source = None
        self.process = None
        self.buffer = ""
        self.cancelled = False
        self.worker_error = None
        self.events = []
        self.tracks = []
        self.request_path = None
        shell = QWidget()
        self.setCentralWidget(shell)
        root = QHBoxLayout(shell)
        sidebar = QWidget()
        sidebar.setMaximumWidth(265)
        side = QVBoxLayout(sidebar)
        brand = QLabel("TENGEN")
        brand.setObjectName("brand")
        side.addWidget(brand)
        subtitle = QLabel("FRC AUTO-SCOUTING\nPython desktop")
        subtitle.setObjectName("muted")
        side.addWidget(subtitle)
        side.addSpacing(18)
        side.addWidget(button("Open video…", self.choose_video, True))
        side.addWidget(button("Open saved run…", self.choose_run))
        side.addWidget(button("Load demo", self.load_demo))
        side.addSpacing(15)
        side.addWidget(QLabel("SAVED ANALYSES"))
        self.history = QListWidget()
        self.history.itemActivated.connect(lambda item: self.open_review(Path(item.data(Qt.ItemDataRole.UserRole))))
        self.history.itemClicked.connect(lambda item: self.open_review(Path(item.data(Qt.ItemDataRole.UserRole))))
        side.addWidget(self.history, 1)
        side.addWidget(button("Show data folder", lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(RUNS.parent)))))
        root.addWidget(sidebar)
        main = QVBoxLayout()
        root.addLayout(main, 1)
        self.title = QLabel("Your match review workspace")
        self.title.setObjectName("title")
        main.addWidget(self.title)
        self.summary = QLabel("Open a local video to begin, or try the included demo.")
        self.summary.setObjectName("muted")
        self.summary.setWordWrap(True)
        main.addWidget(self.summary)
        splitter = QSplitter(Qt.Orientation.Vertical)
        main.addWidget(splitter, 1)
        video_section = QWidget()
        video_layout = QVBoxLayout(video_section)
        video_layout.setContentsMargins(0, 0, 0, 0)
        self.canvas = VideoCanvas()
        video_layout.addWidget(self.canvas, 1)
        self.player = QMediaPlayer(self)
        self.audio = QAudioOutput(self)
        self.audio.setVolume(0.65)
        self.player.setAudioOutput(self.audio)
        self.sink = self.canvas.video_item.videoSink()
        self.player.setVideoOutput(self.canvas.video_item)
        self.canvas.frame_time_changed.connect(self.frame_presented)
        self.player.metaDataChanged.connect(self.update_media_info)
        self.player.positionChanged.connect(self.position_changed)
        self.player.durationChanged.connect(lambda duration: self.seek.setRange(0, duration))
        self.player.errorOccurred.connect(lambda error, message: self.statusBar().showMessage(f"Playback: {message}"))
        transport = QHBoxLayout()
        self.play = button("Play", self.toggle_play)
        self.player.playbackStateChanged.connect(lambda state: self.play.setText("Pause" if state == QMediaPlayer.PlaybackState.PlayingState else "Play"))
        transport.addWidget(self.play)
        self.seek = QSlider(Qt.Orientation.Horizontal)
        self.seek.sliderMoved.connect(self.player.setPosition)
        transport.addWidget(self.seek, 1)
        self.clock = QLabel("0:00 / 0:00")
        transport.addWidget(self.clock)
        self.speed = QComboBox()
        self.speed.addItems(["0.5×", "1×", "1.5×", "2×"])
        self.speed.setCurrentIndex(1)
        self.speed.currentTextChanged.connect(lambda text: self.player.setPlaybackRate(float(text[:-1])))
        transport.addWidget(self.speed)
        self.overlay = QCheckBox("Boxes")
        self.overlay.setChecked(True)
        self.overlay.toggled.connect(self.toggle_overlay)
        transport.addWidget(self.overlay)
        self.shot_overlay = QCheckBox("Shots")
        self.shot_overlay.setChecked(True)
        self.shot_overlay.toggled.connect(self.toggle_shots)
        transport.addWidget(self.shot_overlay)
        self.crop_video = QCheckBox("Crop broadcast")
        self.crop_video.setChecked(True)
        self.crop_video.toggled.connect(self.toggle_crop)
        transport.addWidget(self.crop_video)
        mute = QCheckBox("Mute")
        mute.toggled.connect(self.audio.setMuted)
        transport.addWidget(mute)
        video_layout.addLayout(transport)
        splitter.addWidget(video_section)
        self.tabs = QTabWidget()
        splitter.addWidget(self.tabs)
        splitter.setSizes([480, 250])
        self.event_table = table(["Time", "Phase", "Event", "Team", "Track", "Confidence"])
        self.event_table.cellDoubleClicked.connect(lambda row, col: self.player.setPosition(int(self.events[row]["t_seconds"] * 1000)))
        self.tabs.addTab(self.event_table, "Timeline")
        self.track_table = table(["Track", "Team", "Alliance", "Samples", "First seen", "Last seen"])
        self.track_table.cellDoubleClicked.connect(lambda row, col: self.assign_team())
        self.tabs.addTab(self.track_table, "Robots")
        self.stats_table = table(["Team", "Attempts", "Made", "Reloads", "Median cycle (s)"])
        self.tabs.addTab(self.stats_table, "Team stats")
        self.heatmap = HeatMap()
        heat_panel = QWidget()
        heat_layout = QVBoxLayout(heat_panel)
        self.heat_team = QComboBox()
        self.heat_team.addItem("All robots", None)
        self.heat_team.currentIndexChanged.connect(self.filter_heatmap)
        heat_layout.addWidget(self.heat_team)
        heat_layout.addWidget(self.heatmap, 1)
        self.tabs.addTab(heat_panel, "Heat map")
        self.shot_table = table(["Launch", "Robot", "Outcome", "Outcome time", "Goal", "Confidence"])
        self.shot_table.cellDoubleClicked.connect(lambda row, col: self.player.setPosition(int(self.review.shots[row]["launch_t_seconds"] * 1000)))
        self.tabs.addTab(self.shot_table, "Shots")
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.tabs.addTab(self.details, "Run details")
        self.tabs.currentChanged.connect(lambda: self.frame_presented(self.canvas.time))
        actions = QHBoxLayout()
        self.analyze_button = button("Analyze video…", self.start_analysis, True)
        self.analyze_button.setEnabled(False)
        actions.addWidget(self.analyze_button)
        actions.addWidget(button("Assign team", self.assign_team))
        actions.addWidget(button("Add event", self.add_event))
        actions.addWidget(button("Edit event", self.edit_event))
        actions.addWidget(button("Delete event", self.delete_event))
        actions.addWidget(button("Undo", self.undo))
        actions.addWidget(button("Export…", self.export))
        self.raw = QCheckBox("Raw model output")
        self.raw.toggled.connect(self.refresh_review)
        actions.addWidget(self.raw)
        main.addLayout(actions)
        progress_row = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setValue(0)
        progress_row.addWidget(self.progress, 1)
        self.cancel = button("Cancel analysis", self.cancel_analysis)
        self.cancel.setEnabled(False)
        progress_row.addWidget(self.cancel)
        main.addLayout(progress_row)
        self.statusBar().showMessage("Ready • local files • no server required")
        self.refresh_history()

    def error(self, message):
        QMessageBox.warning(self, "Tengen", str(message))

    def refresh_history(self):
        RUNS.mkdir(parents=True, exist_ok=True)
        self.history.clear()
        for path in sorted(RUNS.glob("*/job.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                job = read_json(path)
                item = QListWidgetItem(f"{job.get('match_id', path.parent.name)}\n{job.get('status', 'unknown')} · {Path(job.get('local_path', '')).name}")
                item.setData(Qt.ItemDataRole.UserRole, str(path.parent))
                self.history.addItem(item)
            except (OSError, ValueError):
                continue

    def choose_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open match video", str(ROOT), "Videos (*.mp4 *.mov *.mkv *.avi *.webm);;All files (*)")
        if path:
            self.open_video(Path(path))

    def open_video(self, path, clear=True):
        if not path.is_file():
            self.error(f"Video is missing: {path}")
            return
        self.player.stop()
        self.source = str(path.resolve())
        self.canvas.reset()
        self.canvas.time = 0
        if clear:
            self.canvas.model_crop = MODEL_CROP
            self.canvas.update_geometry()
            self.review = None
            self.clear_review()
            self.title.setText(path.name)
            self.summary.setText("Local video ready. Analyze to detect and track robots.")
        self.player.setSource(QUrl.fromLocalFile(self.source))
        self.analyze_button.setEnabled(self.process is None)
        self.player.pause()
        self.canvas.update()

    def clear_review(self):
        self.tracks, self.events = [], []
        self.canvas.tracks = []
        self.canvas.events, self.canvas.shots, self.canvas.goals, self.canvas.goal_entries, self.canvas.goal_gaps = [], [], [], [], []
        self.canvas.shots_configured = False
        for widget in (self.event_table, self.track_table, self.stats_table, self.shot_table):
            widget.setRowCount(0)
        self.heatmap.set_data([], {})
        self.details.clear()

    def choose_run(self):
        path = QFileDialog.getExistingDirectory(self, "Open analysis folder (containing job.json)", str(RUNS))
        if path:
            self.open_review(Path(path))

    def load_demo(self):
        self.open_review(FIXTURE, RUNS.parent / "demo-corrections.json")

    def open_review(self, directory, corrections_path=None):
        try:
            review = Review(directory, corrections_path)
            self.review = review
            self.title.setText(review.job.get("match_id") or directory.name)
            self.player.stop()
            self.player.setSource(QUrl())
            self.source = None
            self.canvas.reset()
            self.analyze_button.setEnabled(False)
            if review.video.is_file():
                self.open_video(review.video, clear=False)
            else:
                self.statusBar().showMessage(f"Video missing: {review.video}. Analysis remains available.")
            crop = review.result.get("model_crop")
            self.canvas.model_crop = tuple(crop[k] for k in ("left", "top", "right", "bottom")) if crop else (FULL_FRAME if review.directory == FIXTURE else MODEL_CROP)
            self.canvas.update_geometry()
            self.canvas.hold = 1 / (review.result.get("box_sample_rate") or review.job.get("fps") or 30)
            self.refresh_review()
        except Exception as exc:
            self.error(f"Cannot open analysis: {exc}")

    def refresh_review(self):
        if not self.review:
            return
        self.tracks, self.events = self.review.rows(self.raw.isChecked())
        self.canvas.tracks = self.tracks
        self.canvas.events = self.events
        self.canvas.shots = self.review.shots
        self.canvas.shots_configured = self.review.shots_configured
        self.canvas.goals = self.review.goals
        self.canvas.goal_entries = self.review.goal_entries
        self.canvas.goal_gaps = self.review.goal_calibration.get("camera_gaps", [])
        self.canvas.overlay_item.update()
        fill_table(self.shot_table, [[f"{s['launch_t_seconds']:.3f}s", s.get("robot_track_id"), s.get("outcome"),
                                    s.get("outcome_t_seconds"), s.get("goal"), f"{s.get('confidence', 0):.0%}"] for s in self.review.shots])
        fill_table(self.event_table, [[f"{e['t_seconds']:.2f}s", e.get("phase"), e["event_type"], e.get("team"), e.get("track_id"), f"{e.get('confidence', 0):.0%}"] for e in self.events])
        fill_table(self.track_table, [[t["track_id"], t.get("team"), t.get("alliance"), len(t["boxes"]), f"{t['boxes'][0]['t']:.2f}s" if t["boxes"] else "—", f"{t['boxes'][-1]['t']:.2f}s" if t["boxes"] else "—"] for t in self.tracks])
        from statistics import median
        teams = sorted({r["team"] for r in self.tracks + self.events if r.get("team") is not None})
        rows = []
        for team in teams:
            events = [e for e in self.events if e.get("team") == team]
            reloads = [e["t_seconds"] for e in events if e["event_type"] == "reload"]
            cycles = [b - a for a, b in zip(reloads, reloads[1:])]
            rows.append([team, sum(e["event_type"] == "shot_attempt" for e in events), sum(e["event_type"] == "shot_made" for e in events), len(reloads), round(median(cycles), 2) if cycles else None])
        fill_table(self.stats_table, rows)
        self.heatmap.set_data(self.tracks, self.review.season)
        selected = self.heat_team.currentData()
        self.heat_team.blockSignals(True)
        self.heat_team.clear()
        self.heat_team.addItem("All robots", None)
        for team in teams:
            self.heat_team.addItem(f"Team {team}", team)
        self.heat_team.setCurrentIndex(max(0, self.heat_team.findData(selected)))
        self.heat_team.blockSignals(False)
        self.filter_heatmap()
        self.update_media_info()
        from ingest.stats import scoring_is_meaningful, reconstruct_score
        accuracy = "Score comparison unavailable: season point values are placeholders."
        if scoring_is_meaningful(self.review.season):
            if self.review.job.get("tba_score") and self.review.job.get("alliances"):
                calculated = reconstruct_score(self.review.raw_events, self.review.job["alliances"], self.review.season)
                accuracy = f"Raw reconstructed score: {calculated}\nOfficial score: {self.review.job['tba_score']}"
            else:
                accuracy = "No official score/alliance data for this local video."
        self.details.setPlainText(f"{accuracy}\n\nSaved in: {self.review.directory}\nCorrections: {len(self.review.corrections)}\n\n" + json.dumps(self.review.result, indent=2))

    def position_changed(self, position):
        if not self.seek.isSliderDown():
            self.seek.setValue(position)
        def clock(ms):
            return f"{ms // 60000}:{ms // 1000 % 60:02}"
        self.clock.setText(f"{clock(position)} / {clock(self.player.duration())}")

    def toggle_play(self):
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def toggle_overlay(self, value):
        self.canvas.overlays = value
        self.canvas.overlay_item.update()

    def toggle_shots(self, value):
        self.canvas.show_shots = value
        self.canvas.overlay_item.update()

    def toggle_crop(self, value):
        self.canvas.crop_enabled = value
        self.canvas.update_geometry()

    def filter_heatmap(self):
        self.heatmap.selected_team = self.heat_team.currentData()
        self.heatmap.rebuild_samples()

    def frame_presented(self, seconds):
        # The heat layer is at most 4 Hz; live markers are lightweight and follow each frame.
        if self.heatmap.isVisible():
            self.heatmap.set_time(seconds)

    def update_media_info(self):
        fps = self.player.metaData().value(QMediaMetaData.Key.VideoFrameRate)
        fps = float(fps or (self.review.job.get("fps", 0) if self.review else 0))
        if self.review:
            text = (f"{len(self.tracks)} track fragments · {len(self.events)} events · {len(self.review.shots)} shot records · "
                    f"{self.review.job.get('duration', 0):.1f}s · {fps:.2f} fps source")
            if not self.review.shots_configured:
                text += " · re-analyze to add ball scouting"
        else:
            text = f"Local video ready · {fps:.2f} fps source · playback follows the original frame rate"
        if 0 < fps < 29:
            text += " · low-frame-rate file: open the original 30/60 fps recording for smooth motion"
        self.summary.setText(text)

    def can_correct(self):
        if not self.review:
            self.statusBar().showMessage("Load an analysis before reviewing events or teams.")
            return False
        if self.raw.isChecked():
            self.statusBar().showMessage("Uncheck Raw model output to make review corrections.")
            return False
        return True

    def assign_team(self):
        if not self.can_correct():
            return
        row = self.track_table.currentRow()
        if row < 0:
            self.tabs.setCurrentWidget(self.track_table)
            self.statusBar().showMessage("Select a robot row, then Assign team (or double-click it).")
            return
        track = self.tracks[row]
        value, ok = QInputDialog.getInt(self, "Assign robot team", "Team number (0 = unknown)", track.get("team") or 0, 0, 99999)
        if ok:
            self.review.correct("track", track["track_id"], {"team": value or None})
            self.refresh_review()

    def event_dialog(self, existing=None):
        dialog = QDialog(self)
        dialog.setWindowTitle("Edit event" if existing else "Add event at playhead")
        layout = QFormLayout(dialog)
        kind = QComboBox()
        schema = read_json(ROOT / "contracts/events.schema.json")
        kind.addItems(schema["properties"]["event_type"]["enum"])
        kind.setCurrentText((existing or {}).get("event_type", "shot_attempt"))
        layout.addRow("Event", kind)
        seconds = QDoubleSpinBox()
        seconds.setRange(0, self.review.job.get("duration") or 99999)
        seconds.setDecimals(3)
        seconds.setValue((existing or {}).get("t_seconds", self.player.position() / 1000))
        layout.addRow("Time (s)", seconds)
        robot = QComboBox()
        robot.addItem("Match / unknown", None)
        for track in self.tracks:
            robot.addItem(f"Track {track['track_id']} · team {track.get('team') or 'unknown'}", track["track_id"])
        if existing:
            robot.setCurrentIndex(max(0, robot.findData(existing.get("track_id"))))
        layout.addRow("Robot", robot)
        goal = QComboBox()
        goal.addItem("Unknown", None)
        for name in self.review.season.get("goals", []):
            goal.addItem(name, name)
        if existing:
            goal.setCurrentIndex(max(0, goal.findData(existing.get("goal"))))
        layout.addRow("Goal (shots)", goal)
        controls = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        controls.accepted.connect(dialog.accept)
        controls.rejected.connect(dialog.reject)
        layout.addRow(controls)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        track = next((t for t in self.tracks if t["track_id"] == robot.currentData()), {})
        return dict(event_type=kind.currentText(), t_seconds=seconds.value(),
                    phase=phase_at(seconds.value(), self.review.season), track_id=robot.currentData(),
                    team=track.get("team"), goal=goal.currentData() if kind.currentText().startswith("shot_") else None)

    def add_event(self):
        if not self.can_correct():
            return
        fields = self.event_dialog()
        if fields:
            event_id = str(uuid.uuid4())
            fields.update(event_id=event_id, schema_version=3, job_id=self.review.job["job_id"],
                          match_id=self.review.job.get("match_id"), confidence=1.0, source="manual", field_x=None, field_y=None)
            self.review.correct("event", event_id, fields, "create")
            self.refresh_review()

    def edit_event(self):
        if not self.can_correct():
            return
        row = self.event_table.currentRow()
        if row < 0:
            self.statusBar().showMessage("Select an event in the timeline first.")
            return
        event = self.events[row]
        fields = self.event_dialog(event)
        if fields:
            self.review.correct("event", event["event_id"], fields)
            self.refresh_review()

    def delete_event(self):
        if self.can_correct() and self.event_table.currentRow() >= 0:
            self.review.correct("event", self.events[self.event_table.currentRow()]["event_id"], {}, "delete")
            self.refresh_review()

    def undo(self):
        if self.can_correct():
            self.review.undo()
            self.refresh_review()

    def export(self):
        if not self.review:
            self.statusBar().showMessage("Load an analysis before exporting.")
            return
        path, selected_filter = QFileDialog.getSaveFileName(self, "Export scouting data", str(Path.home() / "Downloads/scouting.json"), "Full analysis (*.json);;Event table (*.csv)")
        if path:
            try:
                target = Path(path)
                if not target.suffix:
                    target = target.with_suffix(".csv" if "csv" in selected_filter else ".json")
                self.review.export(target, self.raw.isChecked())
                self.statusBar().showMessage(f"Exported {target}")
            except Exception as exc:
                self.error(exc)

    def start_analysis(self):
        if not self.source or self.process is not None:
            return
        dialog = AnalysisDialog(self, self.source)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.run_analysis(dialog.options())

    def run_analysis(self, options):
        self.cancelled = False
        self.worker_error = None
        self.completed_path = None
        self.buffer = ""
        self.stderr = ""
        self.request_path = RUNS.parent / f"request-{uuid.uuid4()}.json"
        write_json(self.request_path, vars(options))
        process = QProcess(self)
        self.process = process
        process.setWorkingDirectory(str(ROOT))
        process.setProgram(sys.executable)
        process.setArguments(["-m", "desktop.worker", str(self.request_path)])
        process.readyReadStandardOutput.connect(self.read_progress)
        process.readyReadStandardError.connect(self.read_stderr)
        process.finished.connect(self.analysis_finished)
        process.errorOccurred.connect(self.process_error)
        self.analyze_button.setEnabled(False)
        self.cancel.setEnabled(True)
        self.progress.setRange(0, 0)
        self.statusBar().showMessage("Starting robot analysis…")
        process.start()

    def read_stderr(self):
        self.stderr = (self.stderr + bytes(self.process.readAllStandardError()).decode(errors="replace"))[-16000:]

    def read_progress(self):
        self.buffer += bytes(self.process.readAllStandardOutput()).decode(errors="replace")
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            try:
                message = json.loads(line)
            except ValueError:
                continue
            if not isinstance(message, dict):
                continue
            if "stage" in message:
                progress = message.get("progress")
                if progress is not None:
                    self.progress.setRange(0, 100)
                    self.progress.setValue(round(progress * 100))
                self.statusBar().showMessage(f"Analyzing · {message['stage']}")
            if "complete" in message:
                self.completed_path = Path(message["complete"])
            if "error" in message:
                self.worker_error = message["error"]

    def process_error(self, error):
        if error == QProcess.ProcessError.FailedToStart:
            self.worker_error = self.process.errorString()
            self.analysis_finished(1)

    def analysis_finished(self, exit_code, exit_status=None):
        self.read_progress()
        self.read_stderr()
        self.process.deleteLater()
        self.process = None
        self.request_path.unlink(missing_ok=True)
        self.cancel.setEnabled(False)
        self.analyze_button.setEnabled(bool(self.source))
        self.progress.setRange(0, 100)
        self.progress.setValue(100 if self.completed_path else 0)
        self.refresh_history()
        if self.cancelled:
            self.statusBar().showMessage("Analysis canceled. Original video is unchanged.")
        elif exit_code == 0 and self.completed_path:
            self.open_review(self.completed_path)
            self.statusBar().showMessage("Analysis complete • double-click an event to seek or a robot to assign its team")
        else:
            self.error(self.worker_error or self.stderr or f"Analysis stopped (exit {exit_code}).")

    def cancel_analysis(self):
        if self.process is None:
            return
        self.cancelled = True
        pid = self.process.processId()
        if pid:
            try:
                if os.name == "posix":
                    os.killpg(pid, signal.SIGTERM)
                else:
                    import subprocess
                    subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
            except ProcessLookupError:
                self.process.terminate()
        self.cancel.setEnabled(False)

    def closeEvent(self, event):
        if self.process is not None:
            self.cancel_analysis()
            # Process-group termination also stops the model child before the GUI exits.
            self.process.waitForFinished(3000)
        self.player.stop()
        event.accept()


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Tengen native Python desktop")
    parser.add_argument("video", nargs="?", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--seek", type=float, default=0, help="Initial playback position in seconds")
    parser.add_argument("--tab", choices=["Timeline", "Robots", "Team stats", "Heat map", "Shots", "Run details"])
    args = parser.parse_args(argv)
    app = QApplication(sys.argv[:1])
    app.setApplicationName("Tengen")
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    window = MainWindow()
    window.show()
    if args.demo:
        window.load_demo()
    elif args.run:
        window.open_review(args.run)
    elif args.video:
        window.open_video(args.video)
    if args.tab:
        for index in range(window.tabs.count()):
            if window.tabs.tabText(index) == args.tab:
                window.tabs.setCurrentIndex(index)
                break
    if args.seek > 0:
        def seek_when_loaded(status):
            if status in (QMediaPlayer.MediaStatus.LoadedMedia, QMediaPlayer.MediaStatus.BufferedMedia):
                window.player.mediaStatusChanged.disconnect(seek_when_loaded)
                window.player.setPosition(round(args.seek * 1000))
        window.player.mediaStatusChanged.connect(seek_when_loaded)
        seek_when_loaded(window.player.mediaStatus())
    return app.exec()
