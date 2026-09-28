"""Native video presentation and scouting overlays, sharing the web view's coordinates.

Qt presents video directly. Python receives timestamps only; no per-frame RGB conversion
or image copy. Heat density is cached at quarter-second intervals, not rebuilt at video FPS.
"""
from __future__ import annotations

from bisect import bisect_right
import math
import time

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSizeF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtMultimedia import QVideoFrame
from PySide6.QtMultimediaWidgets import QGraphicsVideoItem
from PySide6.QtWidgets import QGraphicsItem, QGraphicsRectItem, QGraphicsScene, QGraphicsView, QWidget

from desktop.core import ROOT, box_at
from ingest.yolo_orchestrator import MODEL_CROP

FULL_FRAME = (0.0, 0.0, 1.0, 1.0)


def scoring_counts(shots, entries, t):
    visible = [e for e in entries if e["t_seconds"] <= t]
    linked = {e.get("shot_id") for e in visible}
    legacy = [s for s in shots if s.get("outcome") == "made"
              and s.get("outcome_t_seconds") is not None and s["outcome_t_seconds"] <= t
              and s["shot_id"] not in linked]
    return len(visible) + len(legacy), sum(e.get("robot_track_id") is None for e in visible + legacy)


def goal_counting_paused(gaps, t):
    return any(t >= start and (end is None or t < end) for start, end in gaps)


def shot_path_at(shot, t):
    points = shot.get("ball_track", [])
    end = points[-1]["t_seconds"] if points else shot["launch_t_seconds"]
    if t < shot["launch_t_seconds"] - .2 or t > end + .2:
        return []
    return [p for p in points if t - .35 <= p["t_seconds"] <= t + .001]


class ScoutingOverlay(QGraphicsItem):
    def __init__(self, canvas, parent):
        super().__init__(parent)
        self.canvas = canvas
        self.setZValue(2)
        self.bounds = QRectF(0, 0, 1, 1)

    def boundingRect(self):
        return self.bounds

    def resize(self, rect):
        self.prepareGeometryChange()
        self.bounds = rect

    def paint(self, painter, option, widget=None):
        c = self.canvas
        # Cosmetic pens/text stay readable when the scene is fitted into the window.
        scale = max(c.transform().m11(), .01)
        unit = 1 / scale
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setFont(QFont("Arial", 11))
        font = painter.font()
        font.setPixelSize(max(1, round(12 * unit)))
        painter.setFont(font)

        def pen(color, width=2):
            result = QPen(QColor(color), width * unit)
            return result

        def text(point, label, color="#f2f3f5"):
            painter.setPen(QColor(color))
            painter.drawText(point, label)

        def path(points, color, width=2, close=False):
            if not points:
                return
            painter.setPen(pen(color, width))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            p = QPainterPath()
            for i, point in enumerate(points):
                target = c.point(point["x"], point["y"])
                if i == 0 or (point.get("t_seconds", 0) - points[i-1].get("t_seconds", 0) > .085):
                    p.moveTo(target)
                else:
                    p.lineTo(target)
            if close:
                p.closeSubpath()
            painter.drawPath(p)

        if c.show_shots:
            paused = goal_counting_paused(c.goal_gaps, c.time)
            for index, goal in enumerate([] if paused else c.goals):
                region = goal.get("region_id", f"{goal.get('id', 'goal')}_{index+1}")
                polygon = goal.get("polygon") or []
                confirmation = goal.get("confirmation_polygon") or []
                if confirmation:
                    painter.setBrush(QColor(80, 220, 114, 26))
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.drawPolygon(QPolygonF([c.point(*p) for p in confirmation]))
                if polygon:
                    path([dict(x=x, y=y) for x, y in polygon], "#50dc72", close=True)
                boundaries = [(goal.get("made_boundary"), "#50dc72", 3)]
                boundaries += [(b, "#ef5964", 2) for b in goal.get("miss_boundaries", [])]
                for boundary, color, width in boundaries:
                    if boundary:
                        path([dict(x=x, y=y) for x, y in boundary["line"]], color, width)
                label_point = (goal.get("made_boundary") or {}).get("line", polygon)
                if label_point:
                    count = sum(e.get("region_id") == region and e["t_seconds"] <= c.time for e in c.goal_entries)
                    text(c.point(*label_point[0]) + QPointF(0, -8 * unit), f"{region.replace('_', ' ')}: {count} in", "#50dc72")
            for shot in c.shots:
                points = shot_path_at(shot, c.time)
                if not points:
                    continue
                known = shot.get("outcome_t_seconds") is not None and c.time >= shot["outcome_t_seconds"]
                outcome = shot.get("outcome") if known else "unknown"
                color = {"made": "#50dc72", "missed": "#ef5964"}.get(outcome, "#4de4ee")
                path(points, color)
                point = points[-1]
                pos = c.point(point["x"], point["y"])
                radius = max(3 * unit, point.get("radius", 0) * c.model_diagonal())
                painter.setPen(pen(color))
                painter.drawEllipse(pos, radius, radius)
                if outcome in {"made", "missed"}:
                    text(pos + QPointF(radius + 3 * unit, -radius), "IN" if outcome == "made" else "MISS", color)
            for entry in c.goal_entries:
                if entry["t_seconds"] <= c.time <= entry["t_seconds"] + .25:
                    points = [p for p in entry.get("ball_track", []) if p["t_seconds"] <= c.time + .001]
                    if points:
                        path(points, "#50dc72", 3)
                        text(c.point(points[-1]["x"], points[-1]["y"]) + QPointF(8 * unit, 0), "IN", "#50dc72")
            attempted = sum(s["launch_t_seconds"] <= c.time + .001 for s in c.shots)
            made, unassigned = scoring_counts(c.shots, c.goal_entries, c.time)
            rows = [f"Shots detected: {attempted}" if c.shots_configured else "Shots: re-analyze to enable ball scouting",
                    f"Balls in: {made} · source unknown: {unassigned}" if c.goals else "Goal regions not calibrated"]
            panel_width = max(painter.fontMetrics().horizontalAdvance(row) for row in rows) + 18 * unit
            panel = QRectF(self.bounds.width() - panel_width - 8 * unit, 8 * unit, panel_width, 49 * unit)
            painter.fillRect(panel, QColor(10, 12, 16, 215))
            for i, row in enumerate(rows):
                text(QPointF(panel.x() + 9 * unit, panel.y() + (19 + i * 20) * unit), row)
            if paused:
                text(QPointF(12 * unit, 48 * unit), "Goal counting paused: AprilTags not verified", "#ffd078")

        if c.overlays:
            for track in c.tracks:
                box = box_at(track, c.time, c.hold)
                if box is None:
                    continue
                color = {"red": "#ff6b7a", "blue": "#60b6ff"}.get(track.get("alliance"), "#f9d66b")
                rect = QRectF(c.point(box["x"], box["y"]), c.box_size(box["w"], box["h"]))
                outline = pen(color)
                if track.get("team") is None:
                    outline.setStyle(Qt.PenStyle.DashLine)
                painter.setPen(outline)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRect(rect)
                label = str(track.get("team") or track.get("robot_name") or f"Track {track['track_id']}")
                width = painter.fontMetrics().horizontalAdvance(label) + 10 * unit
                label_rect = QRectF(rect.x(), max(0, rect.y() - 20 * unit), width, 20 * unit)
                painter.fillRect(label_rect, QColor("#101c29"))
                painter.setPen(QColor(color))
                painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, label)
                near = [e for e in c.events if e.get("track_id") == track["track_id"] and abs(e["t_seconds"] - c.time) < .6]
                if near:
                    event = near[0]
                    text(QPointF(rect.x(), rect.bottom() + 14 * unit), event["event_type"].replace("_", " "),
                         "#e8b93b" if event.get("confidence", 1) < .5 else "#ffffff")


class VideoCanvas(QGraphicsView):
    frame_time_changed = Signal(float)

    def __init__(self):
        super().__init__()
        self.tracks, self.events, self.shots, self.goals, self.goal_entries, self.goal_gaps = [], [], [], [], [], []
        self.shots_configured = False
        self.time = 0.0
        self.hold = 1 / 30
        self.overlays = True
        self.show_shots = True
        self.crop_enabled = True
        self.model_crop = MODEL_CROP
        self.source_size = QSizeF(1920, 1080)
        self.frames_received = 0
        self.painted_frames = 0
        self._last_painted_time = None
        self.frame_times = []
        self._frame = QVideoFrame()
        scene = QGraphicsScene(self)
        self.setScene(scene)
        self.clip = QGraphicsRectItem()
        self.clip.setPen(QPen(Qt.PenStyle.NoPen))
        self.clip.setFlag(QGraphicsItem.GraphicsItemFlag.ItemClipsChildrenToShape)
        scene.addItem(self.clip)
        self.video_item = QGraphicsVideoItem(self.clip)
        self.video_item.setAspectRatioMode(Qt.AspectRatioMode.IgnoreAspectRatio)
        self.video_item.nativeSizeChanged.connect(self.set_source_size)
        self.overlay_item = ScoutingOverlay(self, self.clip)
        self.video_item.videoSink().videoFrameChanged.connect(self.receive)
        self.setBackgroundBrush(QColor("#080e14"))
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameShape(QGraphicsView.Shape.NoFrame)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setMinimumSize(520, 220)
        self.update_geometry()

    @property
    def image(self):
        """On-demand diagnostic snapshot only; normal playback never calls toImage()."""
        return self._frame.toImage() if self._frame.isValid() else QImage()

    def reset(self):
        self._frame = QVideoFrame()
        self.time = 0
        self.frames_received = self.painted_frames = 0
        self._last_painted_time = None
        self.frame_times.clear()

    def set_source_size(self, size):
        if size.width() > 0 and size.height() > 0:
            self.source_size = QSizeF(size)
            self.update_geometry()

    @property
    def display_crop(self):
        return self.model_crop if self.crop_enabled else FULL_FRAME

    def point(self, x, y):
        ml, mt, mr, mb = self.model_crop
        left, top, _, _ = self.display_crop
        return QPointF((ml + x * (mr - ml) - left) * self.source_size.width(),
                       (mt + y * (mb - mt) - top) * self.source_size.height())

    def box_size(self, w, h):
        l, t, r, b = self.model_crop
        return QSizeF(w * (r - l) * self.source_size.width(), h * (b - t) * self.source_size.height())

    def model_diagonal(self):
        size = self.box_size(1, 1)
        return math.hypot(size.width(), size.height())

    def update_geometry(self):
        left, top, right, bottom = self.display_crop
        w, h = self.source_size.width(), self.source_size.height()
        rect = QRectF(0, 0, w * (right - left), h * (bottom - top))
        self.clip.setRect(rect)
        self.video_item.setSize(self.source_size)
        self.video_item.setPos(-left * w, -top * h)
        self.overlay_item.resize(rect)
        self.setSceneRect(rect)
        self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
        self.viewport().update()

    def receive(self, frame):
        if frame.isValid():
            self._frame = QVideoFrame(frame)
            if frame.startTime() >= 0:
                self.time = frame.startTime() / 1_000_000
            self.frames_received += 1
            self.frame_times.append((time.monotonic(), self.time))
            if len(self.frame_times) > 600:
                del self.frame_times[:300]
            self.overlay_item.update()
            self.frame_time_changed.emit(self.time)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_geometry()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._frame.isValid() and self.time != self._last_painted_time:
            self.painted_frames += 1
            self._last_painted_time = self.time


class HeatMap(QWidget):
    """Dwell density, trails and live markers over the same field image as the web UI."""
    def __init__(self):
        super().__init__()
        self.tracks = []
        self.season = {}
        self.time = 0.0
        self.selected_team = None
        self.samples = []
        self.times = []
        self.last_bucket = None
        self.grid_image = QImage()
        self.background = QImage(str(ROOT / "web/src/assets/field-heatmap-background.png"))
        self.setMinimumHeight(180)
        self.visible_samples = []
        self.missing = 0
        self.valid_count = 0
        self.path_cache = []

    def set_data(self, tracks, season):
        self.tracks, self.season = tracks, season
        self.rebuild_samples()

    def rebuild_samples(self):
        self.samples = []
        self.all_observation_times, self.missing_times = [], []
        length, width = self.extents()
        for track in self.tracks:
            if self.selected_team is not None and track.get("team") != self.selected_team:
                continue
            last = -math.inf
            for box in track.get("boxes", []):
                self.all_observation_times.append(box["t"])
                x, y = box.get("field_x"), box.get("field_y")
                if x is None or y is None or not (0 <= x <= length and 0 <= y <= width):
                    self.missing_times.append(box["t"])
                    continue
                if box["t"] - last < .25 - 1e-6:
                    continue
                last = box["t"]
                self.samples.append((box["t"], x / length, 1 - y / width, track))
        self.samples.sort(key=lambda p: p[0])
        self.times = [p[0] for p in self.samples]
        self.all_observation_times.sort()
        self.missing_times.sort()
        self.last_bucket = None
        self.set_time(self.time)

    def extents(self):
        return self.season.get("field_length_ft", 54), self.season.get("field_width_ft", 26.6)

    def set_time(self, t):
        self.time = t
        bucket = math.floor(t * 4)
        if bucket != self.last_bucket:
            self.last_bucket = bucket
            # Floor the cache time; never reveal positions from a later frame.
            cutoff = bucket / 4
            count = bisect_right(self.times, cutoff)
            self.visible_samples = self.samples[:count]
            paths, previous = {}, {}
            for stamp, x, y, track in self.visible_samples:
                track_id = track["track_id"]
                if track_id not in paths:
                    paths[track_id] = (QPainterPath(), track.get("alliance"))
                path = paths[track_id][0]
                prior = previous.get(track_id)
                continuous = prior is not None and stamp - prior <= .75 and not any(
                    g["start"] < stamp and g["end"] > prior for g in track.get("gaps", []))
                if continuous:
                    path.lineTo(x, y)
                else:
                    path.moveTo(x, y)
                previous[track_id] = stamp
            self.path_cache = list(paths.values())
            self.missing = bisect_right(self.missing_times, cutoff)
            self.valid_count = bisect_right(self.all_observation_times, cutoff) - self.missing
            grid = np.zeros((36, 72), dtype=np.float32)
            for _, x, y, _ in self.visible_samples:
                grid[min(35, round(y * 35)), min(71, round(x * 71))] += 1
            if grid.max() > 0:
                from scipy.ndimage import gaussian_filter
                grid = gaussian_filter(grid, 1.6)
                grid /= grid.max()
                stops = [0, .35, .65, 1]
                colors = [(30, 60, 130), (40, 160, 180), (232, 185, 59), (255, 245, 235)]
                rgba = np.zeros((36, 72, 4), dtype=np.uint8)
                for channel in range(3):
                    rgba[:, :, channel] = np.interp(grid, stops, [c[channel] for c in colors])
                rgba[:, :, 3] = np.where(grid < .02, 0, 255 * (.25 + .65 * grid)).astype(np.uint8)
                self.grid_image = QImage(rgba.data, 72, 36, 72 * 4, QImage.Format.Format_RGBA8888).copy()
            else:
                self.grid_image = QImage()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        length, width = self.extents()
        available = QRectF(12, 8, self.width() - 24, self.height() - 40)
        scale = min(available.width() / length, available.height() / width)
        area = QRectF((self.width() - length * scale) / 2, 8, length * scale, width * scale)
        painter.fillRect(self.rect(), QColor("#101827"))
        if not self.background.isNull():
            painter.drawImage(area, self.background)
        if not self.grid_image.isNull():
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            painter.drawImage(area, self.grid_image)
        def point(x, y):
            return QPointF(area.x() + x * area.width(), area.y() + y * area.height())
        painter.save()
        painter.translate(area.x(), area.y())
        painter.scale(area.width(), area.height())
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for path, alliance in self.path_cache:
            color = QColor({"red": "#ff6973", "blue": "#69a5ff"}.get(alliance, "#f0f3f8"))
            color.setAlpha(110)
            pen = QPen(color, 1.4)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.drawPath(path)
        painter.restore()
        for track in self.tracks:
            if self.selected_team is not None and track.get("team") != self.selected_team:
                continue
            box = box_at(track, self.time, .25)
            if not box or box.get("field_x") is None or box.get("field_y") is None:
                continue
            x, y = box["field_x"], box["field_y"]
            if not (0 <= x <= length and 0 <= y <= width):
                continue
            pos = point(x / length, 1 - y / width)
            color = QColor({"red": "#ff6973", "blue": "#69a5ff"}.get(track.get("alliance"), "#f0f3f8"))
            painter.setPen(QPen(QColor("#0d0f14"), 2))
            painter.setBrush(color)
            painter.drawEllipse(pos, 5, 5)
            painter.setPen(QColor("#f4f6fb"))
            painter.drawText(pos + QPointF(7, 3), str(track.get("team") or track.get("robot_name") or track["track_id"]))
        painter.setPen(QColor("#b5c7d6"))
        label = f"Through {self.time:.1f}s · {self.valid_count} positioned samples · {self.missing} without field position"
        painter.drawText(QPointF(12, self.height() - 10), label)
        if not self.samples:
            painter.fillRect(area, QColor(10, 18, 28, 170))
            painter.drawText(area, Qt.AlignmentFlag.AlignCenter, "No calibrated positions. Re-analyze with automatic field calibration enabled.")
