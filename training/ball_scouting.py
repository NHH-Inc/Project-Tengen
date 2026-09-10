"""Colour-based game-piece tracking and conservative shot event extraction.

This module deliberately knows nothing about YOLO.  Callers provide stable robot observations;
the detector can therefore be replaced by a learned ball detector later without changing the
tracking, launch, goal, or persistence layers.

Coordinates in configuration and persisted shot paths are normalized to the image passed to
``process_frame``.  In ``training.track_yolo`` that image is the configured model crop.
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence


Point = tuple[float, float]
Box = tuple[float, float, float, float]


def _number(value: object, name: str, *, minimum: float | None = None) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        suffix = f" >= {minimum}" if minimum is not None else ""
        raise ValueError(f"{name} must be finite{suffix}")
    return result


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _point(value: object, name: str) -> Point:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must be [x, y]")
    point = (_number(value[0], f"{name}[0]"), _number(value[1], f"{name}[1]"))
    if not all(0.0 <= component <= 1.0 for component in point):
        raise ValueError(f"{name} coordinates must be normalized between zero and one")
    return point


def _direction(value: object, name: str) -> Point:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must be [dx, dy]")
    direction = (_number(value[0], f"{name}[0]"), _number(value[1], f"{name}[1]"))
    length = math.hypot(*direction)
    if length <= 1e-9:
        raise ValueError(f"{name} cannot be the zero vector")
    return direction[0] / length, direction[1] / length


@dataclass(frozen=True)
class BallDetectorConfig:
    hsv_lower: tuple[int, int, int] = (20, 90, 110)
    hsv_upper: tuple[int, int, int] = (38, 255, 255)
    blur_kernel: int = 3
    morph_open_iterations: int = 1
    morph_close_iterations: int = 1
    min_area_px: float = 18.0
    max_area_px: float = 2200.0
    min_radius_px: float = 2.0
    max_radius_px: float = 38.0
    min_circularity: float = 0.28
    min_fill_ratio: float = 0.35
    min_aspect_ratio: float = 0.30
    max_aspect_ratio: float = 3.30


@dataclass(frozen=True)
class BlobTrackingConfig:
    minimum_confirmed_hits: int = 3
    maximum_missed_frames: int = 5
    base_link_distance_ratio: float = 0.004
    acceleration_allowance_ratio_per_second: float = 0.50
    maximum_speed_ratio_per_second: float = 1.8
    maximum_radius_ratio: float = 3.0
    maximum_history_points: int = 180


@dataclass(frozen=True)
class ShotDetectionConfig:
    carry_confirmation_frames: int = 4
    source_memory_frames: int = 8
    robot_padding_ratio: float = 0.08
    maximum_carried_relative_speed_ratio_per_second: float = 0.16
    minimum_launch_relative_speed_ratio_per_second: float = 0.32
    minimum_radial_speed_ratio_per_second: float = 0.12
    minimum_departure_distance_ratio: float = 0.012
    confirmation_frames: int = 3
    maximum_candidate_frames: int = 14
    minimum_direction_consistency: float = 0.62
    edge_launch_max_distance_ratio: float = 0.025
    edge_launch_max_track_points: int = 5
    maximum_launch_distance_ratio: float = 0.045
    maximum_launch_origin_y_ratio: float = 0.80
    maximum_shot_track_seconds: float = 3.0
    minimum_shot_track_speed_ratio_per_second: float = 0.035
    maximum_shot_track_reversal_cosine: float = -0.15


@dataclass(frozen=True)
class DebugConfig:
    show_mask: bool = False
    trail_points: int = 24


@dataclass(frozen=True)
class DirectedBoundary:
    line: tuple[Point, Point]
    direction: Point


@dataclass(frozen=True)
class GoalGeometry:
    goal_id: str
    polygon: tuple[Point, ...] | None
    entry_direction: Point
    made_line: DirectedBoundary | None = None
    miss_boundaries: tuple[DirectedBoundary, ...] = ()

    @property
    def center(self) -> Point:
        if self.polygon:
            return (
                sum(point[0] for point in self.polygon) / len(self.polygon),
                sum(point[1] for point in self.polygon) / len(self.polygon),
            )
        assert self.made_line is not None
        first, second = self.made_line.line
        return ((first[0] + second[0]) / 2.0, (first[1] + second[1]) / 2.0)


@dataclass(frozen=True)
class BallScoutingConfig:
    enabled: bool = True
    detector: BallDetectorConfig = field(default_factory=BallDetectorConfig)
    tracking: BlobTrackingConfig = field(default_factory=BlobTrackingConfig)
    shot: ShotDetectionConfig = field(default_factory=ShotDetectionConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    goals: tuple[GoalGeometry, ...] = ()


def _hsv_triplet(value: object, name: str) -> tuple[int, int, int]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{name} must be [h, s, v]")
    limits = (179, 255, 255)
    result: list[int] = []
    for index, (component, limit) in enumerate(zip(value, limits)):
        parsed = _integer(component, f"{name}[{index}]")
        if parsed > limit:
            raise ValueError(f"{name}[{index}] must be <= {limit}")
        result.append(parsed)
    return tuple(result)  # type: ignore[return-value]


def _directed_boundary(value: object, name: str) -> DirectedBoundary:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    raw_line = value.get("line")
    if not isinstance(raw_line, list) or len(raw_line) != 2:
        raise ValueError(f"{name}.line must contain two normalized points")
    return DirectedBoundary(
        line=(_point(raw_line[0], f"{name}.line[0]"), _point(raw_line[1], f"{name}.line[1]")),
        direction=_direction(value.get("direction"), f"{name}.direction"),
    )


def _goal(value: object, index: int) -> GoalGeometry:
    name = f"goals[{index}]"
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    goal_id = value.get("id")
    if not isinstance(goal_id, str) or not goal_id.strip():
        raise ValueError(f"{name}.id must be a non-empty string")
    raw_polygon = value.get("polygon")
    polygon = None
    if raw_polygon is not None:
        if not isinstance(raw_polygon, list) or len(raw_polygon) < 3:
            raise ValueError(f"{name}.polygon must contain at least three normalized points")
        polygon = tuple(_point(point, f"{name}.polygon") for point in raw_polygon)
    made_line = (
        _directed_boundary(value["made_boundary"], f"{name}.made_boundary")
        if value.get("made_boundary") is not None
        else None
    )
    if polygon is None and made_line is None:
        raise ValueError(f"{name} needs a polygon or made_boundary")
    entry_direction = _direction(value.get("entry_direction"), f"{name}.entry_direction")
    raw_misses = value.get("miss_boundaries", [])
    if not isinstance(raw_misses, list):
        raise ValueError(f"{name}.miss_boundaries must be an array")
    return GoalGeometry(
        goal_id=goal_id.strip(),
        polygon=polygon,
        entry_direction=entry_direction,
        made_line=made_line,
        miss_boundaries=tuple(
            _directed_boundary(boundary, f"{name}.miss_boundaries[{miss_index}]")
            for miss_index, boundary in enumerate(raw_misses)
        ),
    )


def load_ball_scouting_config(path: str | Path) -> BallScoutingConfig:
    """Load and strictly validate the version-one JSON configuration."""

    config_path = Path(path)
    try:
        document = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read ball scouting config {config_path}: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError("ball scouting config must be a JSON object")
    if document.get("schema_version", 1) != 1:
        raise ValueError("ball scouting config schema_version must be 1")

    detector_raw = document.get("detector") or {}
    tracking_raw = document.get("tracking") or {}
    shot_raw = document.get("shot") or {}
    debug_raw = document.get("debug") or {}
    for raw, name in (
        (detector_raw, "detector"), (tracking_raw, "tracking"),
        (shot_raw, "shot"), (debug_raw, "debug"),
    ):
        if not isinstance(raw, dict):
            raise ValueError(f"{name} must be an object")

    detector_defaults = BallDetectorConfig()
    lower = _hsv_triplet(
        detector_raw.get("hsv_lower", list(detector_defaults.hsv_lower)), "detector.hsv_lower"
    )
    upper = _hsv_triplet(
        detector_raw.get("hsv_upper", list(detector_defaults.hsv_upper)), "detector.hsv_upper"
    )
    if any(low > high for low, high in zip(lower, upper)):
        raise ValueError("detector.hsv_lower cannot exceed detector.hsv_upper")
    blur_kernel = _integer(
        detector_raw.get("blur_kernel", detector_defaults.blur_kernel),
        "detector.blur_kernel",
        minimum=0,
    )
    if blur_kernel not in {0, 1} and blur_kernel % 2 == 0:
        raise ValueError("detector.blur_kernel must be zero, one, or an odd integer")
    detector = BallDetectorConfig(
        hsv_lower=lower,
        hsv_upper=upper,
        blur_kernel=blur_kernel,
        morph_open_iterations=_integer(
            detector_raw.get("morph_open_iterations", detector_defaults.morph_open_iterations),
            "detector.morph_open_iterations",
        ),
        morph_close_iterations=_integer(
            detector_raw.get("morph_close_iterations", detector_defaults.morph_close_iterations),
            "detector.morph_close_iterations",
        ),
        min_area_px=_number(
            detector_raw.get("min_area_px", detector_defaults.min_area_px),
            "detector.min_area_px", minimum=0.0,
        ),
        max_area_px=_number(
            detector_raw.get("max_area_px", detector_defaults.max_area_px),
            "detector.max_area_px", minimum=0.0,
        ),
        min_radius_px=_number(
            detector_raw.get("min_radius_px", detector_defaults.min_radius_px),
            "detector.min_radius_px", minimum=0.0,
        ),
        max_radius_px=_number(
            detector_raw.get("max_radius_px", detector_defaults.max_radius_px),
            "detector.max_radius_px", minimum=0.0,
        ),
        min_circularity=_number(
            detector_raw.get("min_circularity", detector_defaults.min_circularity),
            "detector.min_circularity", minimum=0.0,
        ),
        min_fill_ratio=_number(
            detector_raw.get("min_fill_ratio", detector_defaults.min_fill_ratio),
            "detector.min_fill_ratio", minimum=0.0,
        ),
        min_aspect_ratio=_number(
            detector_raw.get("min_aspect_ratio", detector_defaults.min_aspect_ratio),
            "detector.min_aspect_ratio", minimum=0.0,
        ),
        max_aspect_ratio=_number(
            detector_raw.get("max_aspect_ratio", detector_defaults.max_aspect_ratio),
            "detector.max_aspect_ratio", minimum=0.0,
        ),
    )
    if detector.min_area_px > detector.max_area_px:
        raise ValueError("detector.min_area_px cannot exceed detector.max_area_px")
    if detector.min_radius_px > detector.max_radius_px:
        raise ValueError("detector.min_radius_px cannot exceed detector.max_radius_px")
    if detector.min_aspect_ratio > detector.max_aspect_ratio:
        raise ValueError("detector.min_aspect_ratio cannot exceed detector.max_aspect_ratio")
    if detector.min_circularity > 1 or detector.min_fill_ratio > 1:
        raise ValueError("detector circularity and fill thresholds cannot exceed one")

    tracking_defaults = BlobTrackingConfig()
    tracking = BlobTrackingConfig(
        minimum_confirmed_hits=_integer(
            tracking_raw.get("minimum_confirmed_hits", tracking_defaults.minimum_confirmed_hits),
            "tracking.minimum_confirmed_hits", minimum=1,
        ),
        maximum_missed_frames=_integer(
            tracking_raw.get("maximum_missed_frames", tracking_defaults.maximum_missed_frames),
            "tracking.maximum_missed_frames",
        ),
        base_link_distance_ratio=_number(
            tracking_raw.get("base_link_distance_ratio", tracking_defaults.base_link_distance_ratio),
            "tracking.base_link_distance_ratio", minimum=0.0,
        ),
        acceleration_allowance_ratio_per_second=_number(
            tracking_raw.get(
                "acceleration_allowance_ratio_per_second",
                tracking_defaults.acceleration_allowance_ratio_per_second,
            ),
            "tracking.acceleration_allowance_ratio_per_second", minimum=0.0,
        ),
        maximum_speed_ratio_per_second=_number(
            tracking_raw.get(
                "maximum_speed_ratio_per_second", tracking_defaults.maximum_speed_ratio_per_second
            ),
            "tracking.maximum_speed_ratio_per_second", minimum=0.0,
        ),
        maximum_radius_ratio=_number(
            tracking_raw.get("maximum_radius_ratio", tracking_defaults.maximum_radius_ratio),
            "tracking.maximum_radius_ratio", minimum=1.0,
        ),
        maximum_history_points=_integer(
            tracking_raw.get("maximum_history_points", tracking_defaults.maximum_history_points),
            "tracking.maximum_history_points", minimum=2,
        ),
    )

    shot_defaults = ShotDetectionConfig()
    shot = ShotDetectionConfig(
        carry_confirmation_frames=_integer(
            shot_raw.get("carry_confirmation_frames", shot_defaults.carry_confirmation_frames),
            "shot.carry_confirmation_frames", minimum=2,
        ),
        source_memory_frames=_integer(
            shot_raw.get("source_memory_frames", shot_defaults.source_memory_frames),
            "shot.source_memory_frames", minimum=1,
        ),
        robot_padding_ratio=_number(
            shot_raw.get("robot_padding_ratio", shot_defaults.robot_padding_ratio),
            "shot.robot_padding_ratio", minimum=0.0,
        ),
        maximum_carried_relative_speed_ratio_per_second=_number(
            shot_raw.get(
                "maximum_carried_relative_speed_ratio_per_second",
                shot_defaults.maximum_carried_relative_speed_ratio_per_second,
            ),
            "shot.maximum_carried_relative_speed_ratio_per_second", minimum=0.0,
        ),
        minimum_launch_relative_speed_ratio_per_second=_number(
            shot_raw.get(
                "minimum_launch_relative_speed_ratio_per_second",
                shot_defaults.minimum_launch_relative_speed_ratio_per_second,
            ),
            "shot.minimum_launch_relative_speed_ratio_per_second", minimum=0.0,
        ),
        minimum_radial_speed_ratio_per_second=_number(
            shot_raw.get(
                "minimum_radial_speed_ratio_per_second",
                shot_defaults.minimum_radial_speed_ratio_per_second,
            ),
            "shot.minimum_radial_speed_ratio_per_second", minimum=0.0,
        ),
        minimum_departure_distance_ratio=_number(
            shot_raw.get(
                "minimum_departure_distance_ratio", shot_defaults.minimum_departure_distance_ratio
            ),
            "shot.minimum_departure_distance_ratio", minimum=0.0,
        ),
        confirmation_frames=_integer(
            shot_raw.get("confirmation_frames", shot_defaults.confirmation_frames),
            "shot.confirmation_frames", minimum=2,
        ),
        maximum_candidate_frames=_integer(
            shot_raw.get("maximum_candidate_frames", shot_defaults.maximum_candidate_frames),
            "shot.maximum_candidate_frames", minimum=2,
        ),
        minimum_direction_consistency=_number(
            shot_raw.get(
                "minimum_direction_consistency", shot_defaults.minimum_direction_consistency
            ),
            "shot.minimum_direction_consistency", minimum=0.0,
        ),
        edge_launch_max_distance_ratio=_number(
            shot_raw.get(
                "edge_launch_max_distance_ratio",
                shot_defaults.edge_launch_max_distance_ratio,
            ),
            "shot.edge_launch_max_distance_ratio", minimum=0.0,
        ),
        edge_launch_max_track_points=_integer(
            shot_raw.get(
                "edge_launch_max_track_points",
                shot_defaults.edge_launch_max_track_points,
            ),
            "shot.edge_launch_max_track_points", minimum=2,
        ),
        maximum_launch_distance_ratio=_number(
            shot_raw.get(
                "maximum_launch_distance_ratio",
                shot_defaults.maximum_launch_distance_ratio,
            ),
            "shot.maximum_launch_distance_ratio", minimum=0.0,
        ),
        maximum_launch_origin_y_ratio=_number(
            shot_raw.get(
                "maximum_launch_origin_y_ratio",
                shot_defaults.maximum_launch_origin_y_ratio,
            ),
            "shot.maximum_launch_origin_y_ratio", minimum=0.0,
        ),
        maximum_shot_track_seconds=_number(
            shot_raw.get(
                "maximum_shot_track_seconds",
                shot_defaults.maximum_shot_track_seconds,
            ),
            "shot.maximum_shot_track_seconds", minimum=0.1,
        ),
        minimum_shot_track_speed_ratio_per_second=_number(
            shot_raw.get(
                "minimum_shot_track_speed_ratio_per_second",
                shot_defaults.minimum_shot_track_speed_ratio_per_second,
            ),
            "shot.minimum_shot_track_speed_ratio_per_second", minimum=0.0,
        ),
        maximum_shot_track_reversal_cosine=_number(
            shot_raw.get(
                "maximum_shot_track_reversal_cosine",
                shot_defaults.maximum_shot_track_reversal_cosine,
            ),
            "shot.maximum_shot_track_reversal_cosine",
        ),
    )
    if shot.minimum_direction_consistency > 1:
        raise ValueError("shot.minimum_direction_consistency cannot exceed one")
    if shot.maximum_launch_origin_y_ratio > 1:
        raise ValueError("shot.maximum_launch_origin_y_ratio cannot exceed one")
    if not -1 <= shot.maximum_shot_track_reversal_cosine <= 1:
        raise ValueError("shot.maximum_shot_track_reversal_cosine must be between -1 and one")

    debug_defaults = DebugConfig()
    show_mask = debug_raw.get("show_mask", debug_defaults.show_mask)
    if not isinstance(show_mask, bool):
        raise ValueError("debug.show_mask must be a boolean")
    debug = DebugConfig(
        show_mask=show_mask,
        trail_points=_integer(
            debug_raw.get("trail_points", debug_defaults.trail_points),
            "debug.trail_points", minimum=2,
        ),
    )
    raw_goals = document.get("goals", [])
    if not isinstance(raw_goals, list):
        raise ValueError("goals must be an array")
    goals = tuple(_goal(value, index) for index, value in enumerate(raw_goals))
    enabled = document.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError("enabled must be a boolean")
    return BallScoutingConfig(enabled, detector, tracking, shot, debug, goals)


@dataclass(frozen=True)
class BallDetection:
    center: Point
    radius: float
    area: float
    circularity: float
    bbox: Box


class YellowBallDetector:
    """HSV thresholding with morphology plus size and shape rejection."""

    def __init__(self, config: BallDetectorConfig):
        self.config = config
        self.last_mask = None
        self.last_detections: list[BallDetection] = []

    def detect(self, frame) -> list[BallDetection]:
        import cv2
        import numpy as np

        source = frame
        if self.config.blur_kernel > 1:
            source = cv2.GaussianBlur(
                frame, (self.config.blur_kernel, self.config.blur_kernel), 0
            )
        hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            np.array(self.config.hsv_lower, dtype=np.uint8),
            np.array(self.config.hsv_upper, dtype=np.uint8),
        )
        kernel = np.ones((3, 3), dtype=np.uint8)
        if self.config.morph_open_iterations:
            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_OPEN,
                kernel,
                iterations=self.config.morph_open_iterations,
            )
        if self.config.morph_close_iterations:
            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_CLOSE,
                kernel,
                iterations=self.config.morph_close_iterations,
            )

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: list[BallDetection] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if not self.config.min_area_px <= area <= self.config.max_area_px:
                continue
            perimeter = float(cv2.arcLength(contour, True))
            if perimeter <= 1e-6:
                continue
            circularity = 4.0 * math.pi * area / (perimeter * perimeter)
            if circularity < self.config.min_circularity:
                continue
            left, top, width, height = cv2.boundingRect(contour)
            aspect = width / max(1.0, float(height))
            if not self.config.min_aspect_ratio <= aspect <= self.config.max_aspect_ratio:
                continue
            fill_ratio = area / max(1.0, float(width * height))
            if fill_ratio < self.config.min_fill_ratio:
                continue
            (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
            if not self.config.min_radius_px <= radius <= self.config.max_radius_px:
                continue
            detections.append(BallDetection(
                center=(float(center_x), float(center_y)),
                radius=float(radius),
                area=area,
                circularity=min(1.0, circularity),
                bbox=(float(left), float(top), float(left + width), float(top + height)),
            ))
        detections.sort(key=lambda detection: (detection.center[0], detection.center[1]))
        self.last_mask = mask
        self.last_detections = detections
        return detections


@dataclass(frozen=True)
class BallTrackPoint:
    frame_index: int
    t_seconds: float
    center: Point
    radius: float
    area: float
    circularity: float


@dataclass
class BallTrack:
    track_id: int
    points: list[BallTrackPoint] = field(default_factory=list)
    hits: int = 0
    missed_frames: int = 0
    matched_this_frame: bool = False

    @property
    def last(self) -> BallTrackPoint:
        return self.points[-1]

    @property
    def velocity(self) -> Point:
        if len(self.points) < 2:
            return (0.0, 0.0)
        # A short baseline damps one-pixel centroid noise without hiding a launch.
        first = self.points[max(0, len(self.points) - 3)]
        last = self.points[-1]
        elapsed = last.t_seconds - first.t_seconds
        if elapsed <= 1e-9:
            return (0.0, 0.0)
        return (
            (last.center[0] - first.center[0]) / elapsed,
            (last.center[1] - first.center[1]) / elapsed,
        )

    def predicted_center(self, t_seconds: float) -> Point:
        elapsed = max(0.0, t_seconds - self.last.t_seconds)
        velocity = self.velocity
        return self.last.center[0] + velocity[0] * elapsed, self.last.center[1] + velocity[1] * elapsed


class BallBlobTracker:
    """Small constant-velocity nearest-neighbour tracker with short occlusion coast."""

    def __init__(self, config: BlobTrackingConfig):
        self.config = config
        self.active: dict[int, BallTrack] = {}
        self.retired: dict[int, BallTrack] = {}
        self.next_track_id = 1

    def update(
        self,
        detections: Sequence[BallDetection],
        frame_index: int,
        t_seconds: float,
        frame_size: tuple[int, int],
        wide_gate_track_ids: set[int] | None = None,
    ) -> tuple[list[BallTrack], list[BallTrack]]:
        width, height = frame_size
        diagonal = math.hypot(width, height)
        candidates: list[tuple[float, int, int]] = []
        for track_id, track in self.active.items():
            predicted = track.predicted_center(t_seconds)
            elapsed = max(0.0, t_seconds - track.last.t_seconds)
            observed_speed_ratio = math.hypot(*track.velocity) / max(diagonal, 1.0)
            if wide_gate_track_ids and track_id in wide_gate_track_ids:
                # A confirmed carried ball is the one place a sudden acceleration is expected.
                # Keeping this wider gate away from ordinary floor/decor tracks prevents dense
                # piles from being stitched into implausible zig-zags.
                gate_speed_ratio = self.config.maximum_speed_ratio_per_second
            else:
                gate_speed_ratio = min(
                    self.config.maximum_speed_ratio_per_second,
                    max(
                        self.config.acceleration_allowance_ratio_per_second,
                        observed_speed_ratio * 1.75,
                    ),
                )
            allowed = diagonal * (
                self.config.base_link_distance_ratio
                + gate_speed_ratio * elapsed
            )
            for detection_index, detection in enumerate(detections):
                radius_ratio = max(
                    detection.radius / max(track.last.radius, 1e-6),
                    track.last.radius / max(detection.radius, 1e-6),
                )
                if radius_ratio > self.config.maximum_radius_ratio:
                    continue
                distance = math.hypot(
                    detection.center[0] - predicted[0], detection.center[1] - predicted[1]
                )
                if distance > allowed:
                    continue
                cost = distance / max(allowed, 1e-6) + 0.18 * abs(math.log(radius_ratio))
                candidates.append((cost, track_id, detection_index))
        candidates.sort()

        for track in self.active.values():
            track.matched_this_frame = False
        assignments: dict[int, int] = {}
        used_tracks: set[int] = set()
        used_detections: set[int] = set()
        for _cost, track_id, detection_index in candidates:
            if track_id in used_tracks or detection_index in used_detections:
                continue
            assignments[track_id] = detection_index
            used_tracks.add(track_id)
            used_detections.add(detection_index)

        retired_now: list[BallTrack] = []
        for track_id, track in list(self.active.items()):
            detection_index = assignments.get(track_id)
            if detection_index is None:
                track.missed_frames += 1
                if track.missed_frames > self.config.maximum_missed_frames:
                    retired = self.active.pop(track_id)
                    self.retired[track_id] = retired
                    if len(self.retired) > 256:
                        self.retired.pop(next(iter(self.retired)))
                    retired_now.append(retired)
                continue
            detection = detections[detection_index]
            track.points.append(BallTrackPoint(
                frame_index=frame_index,
                t_seconds=t_seconds,
                center=detection.center,
                radius=detection.radius,
                area=detection.area,
                circularity=detection.circularity,
            ))
            track.points[:] = track.points[-self.config.maximum_history_points:]
            track.hits += 1
            track.missed_frames = 0
            track.matched_this_frame = True

        for detection_index, detection in enumerate(detections):
            if detection_index in used_detections:
                continue
            track = BallTrack(track_id=self.next_track_id)
            self.next_track_id += 1
            track.points.append(BallTrackPoint(
                frame_index=frame_index,
                t_seconds=t_seconds,
                center=detection.center,
                radius=detection.radius,
                area=detection.area,
                circularity=detection.circularity,
            ))
            track.hits = 1
            track.matched_this_frame = True
            self.active[track.track_id] = track

        matched = [
            track for track in self.active.values()
            if track.matched_this_frame and track.hits >= self.config.minimum_confirmed_hits
        ]
        return matched, retired_now


@dataclass(frozen=True)
class RobotObservation:
    track_id: int
    bbox: Box

    @property
    def center(self) -> Point:
        left, top, right, bottom = self.bbox
        return (left + right) / 2.0, (top + bottom) / 2.0


@dataclass
class _RobotMotion:
    observation: RobotObservation
    t_seconds: float
    velocity: Point = (0.0, 0.0)


@dataclass
class _LaunchCandidate:
    source_robot_ids: tuple[int, ...]
    robot_track_id: int | None
    launch_frame: int
    launch_t_seconds: float
    launch_center: Point
    trajectory: list[Point] = field(default_factory=list)
    candidate_frames: int = 1
    away_frames: int = 1


@dataclass
class _TrackState:
    owner_robot_id: int | None = None
    owner_streak: int = 0
    ambiguous_robot_ids: tuple[int, ...] = ()
    ambiguous_streak: int = 0
    source_robot_ids: tuple[int, ...] = ()
    last_carried_step: int = -1
    candidate: _LaunchCandidate | None = None
    shot_id: str | None = None


@dataclass
class ShotRecord:
    shot_id: str
    launch_frame: int
    launch_t_seconds: float
    robot_track_id: int | None
    ball_track_id: int
    confidence: float
    attribution_confidence: float
    outcome: str = "unknown"
    goal: str | None = None
    outcome_frame: int | None = None
    outcome_t_seconds: float | None = None
    ball_track: list[dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "shot_id": self.shot_id,
            "launch_frame": self.launch_frame,
            "launch_t_seconds": round(self.launch_t_seconds, 6),
            "robot_track_id": self.robot_track_id,
            "ball_track_id": self.ball_track_id,
            "confidence": round(self.confidence, 6),
            "attribution_confidence": round(self.attribution_confidence, 6),
            "outcome": self.outcome,
            "goal": self.goal,
            "outcome_frame": self.outcome_frame,
            "outcome_t_seconds": (
                round(self.outcome_t_seconds, 6)
                if self.outcome_t_seconds is not None else None
            ),
            "coordinate_space": "model_crop_normalized",
            "ball_track": self.ball_track,
        }


def _inside_box(point: Point, box: Box, padding_ratio: float) -> bool:
    left, top, right, bottom = box
    pad_x = (right - left) * padding_ratio
    pad_y = (bottom - top) * padding_ratio
    return (
        left - pad_x <= point[0] <= right + pad_x
        and top - pad_y <= point[1] <= bottom + pad_y
    )


def _distance_to_box(point: Point, box: Box) -> float:
    """Shortest image-space distance from a point to a closed rectangle."""

    left, top, right, bottom = box
    dx = max(left - point[0], 0.0, point[0] - right)
    dy = max(top - point[1], 0.0, point[1] - bottom)
    return math.hypot(dx, dy)


def _point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    inside = False
    x, y = point
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = previous
        x2, y2 = current
        if (y1 > y) != (y2 > y):
            crossing_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < crossing_x:
                inside = not inside
        previous = current
    return inside


def _orientation(first: Point, second: Point, third: Point) -> float:
    return (second[0] - first[0]) * (third[1] - first[1]) - (
        second[1] - first[1]
    ) * (third[0] - first[0])


def segments_intersect(first: Point, second: Point, third: Point, fourth: Point) -> bool:
    """Whether two closed line segments intersect, including endpoint contact."""

    o1 = _orientation(first, second, third)
    o2 = _orientation(first, second, fourth)
    o3 = _orientation(third, fourth, first)
    o4 = _orientation(third, fourth, second)
    epsilon = 1e-9
    if (o1 > epsilon and o2 < -epsilon or o1 < -epsilon and o2 > epsilon) and (
        o3 > epsilon and o4 < -epsilon or o3 < -epsilon and o4 > epsilon
    ):
        return True

    def on_segment(a: Point, b: Point, point: Point) -> bool:
        return (
            min(a[0], b[0]) - epsilon <= point[0] <= max(a[0], b[0]) + epsilon
            and min(a[1], b[1]) - epsilon <= point[1] <= max(a[1], b[1]) + epsilon
        )

    return (
        abs(o1) <= epsilon and on_segment(first, second, third)
        or abs(o2) <= epsilon and on_segment(first, second, fourth)
        or abs(o3) <= epsilon and on_segment(third, fourth, first)
        or abs(o4) <= epsilon and on_segment(third, fourth, second)
    )


def crosses_directed_boundary(previous: Point, current: Point, boundary: DirectedBoundary) -> bool:
    movement = current[0] - previous[0], current[1] - previous[1]
    if movement[0] * boundary.direction[0] + movement[1] * boundary.direction[1] <= 0.0:
        return False
    return segments_intersect(previous, current, boundary.line[0], boundary.line[1])


class BallShotAnalyzer:
    """End-to-end detector/tracker/shot state machine for one video stream."""

    def __init__(self, config: BallScoutingConfig):
        self.config = config
        self.detector = YellowBallDetector(config.detector)
        self.tracker = BallBlobTracker(config.tracking)
        self.track_states: dict[int, _TrackState] = {}
        self.robot_motion: dict[int, _RobotMotion] = {}
        self.shots: list[ShotRecord] = []
        self._shots_by_ball_track: dict[int, ShotRecord] = {}
        self._closed_shot_track_ids: set[int] = set()
        self.frame_size: tuple[int, int] = (1, 1)
        self._analysis_step = -1

    def _update_robots(
        self, observations: Sequence[RobotObservation], t_seconds: float
    ) -> dict[int, _RobotMotion]:
        current: dict[int, _RobotMotion] = {}
        for observation in observations:
            previous = self.robot_motion.get(observation.track_id)
            velocity = (0.0, 0.0)
            if previous is not None and t_seconds > previous.t_seconds:
                elapsed = t_seconds - previous.t_seconds
                velocity = (
                    (observation.center[0] - previous.observation.center[0]) / elapsed,
                    (observation.center[1] - previous.observation.center[1]) / elapsed,
                )
            motion = _RobotMotion(observation, t_seconds, velocity)
            current[observation.track_id] = motion
            self.robot_motion[observation.track_id] = motion
        return current

    def process_frame(
        self,
        frame,
        frame_index: int,
        t_seconds: float,
        robots: Sequence[RobotObservation],
    ) -> list[BallDetection]:
        height, width = frame.shape[:2]
        self.frame_size = (width, height)
        self._analysis_step += 1
        current_robots = self._update_robots(robots, t_seconds)
        detections = self.detector.detect(frame)
        carried_track_ids = {
            track_id for track_id, state in self.track_states.items()
            if state.source_robot_ids
            and self._analysis_step - state.last_carried_step
            <= self.config.shot.source_memory_frames
        }
        matched, retired = self.tracker.update(
            detections,
            frame_index,
            t_seconds,
            self.frame_size,
            wide_gate_track_ids=carried_track_ids,
        )
        for track in matched:
            self._update_track(track, current_robots, frame_index, self._analysis_step)
        for track in retired:
            if track.track_id not in self._shots_by_ball_track:
                self.track_states.pop(track.track_id, None)
        return detections

    def _relative_motion(
        self,
        track: BallTrack,
        robot_ids: Sequence[int],
        robots: dict[int, _RobotMotion],
    ) -> tuple[float, float]:
        diagonal = math.hypot(*self.frame_size)
        ball_velocity = track.velocity
        relative_speeds: list[float] = []
        radial_speeds: list[float] = []
        for robot_id in robot_ids:
            robot = robots.get(robot_id) or self.robot_motion.get(robot_id)
            if robot is None:
                continue
            relative_velocity = (
                ball_velocity[0] - robot.velocity[0],
                ball_velocity[1] - robot.velocity[1],
            )
            relative_speeds.append(math.hypot(*relative_velocity) / max(diagonal, 1.0))
            radial = (
                track.last.center[0] - robot.observation.center[0],
                track.last.center[1] - robot.observation.center[1],
            )
            radial_length = math.hypot(*radial)
            if radial_length <= 1e-6:
                radial_speeds.append(0.0)
            else:
                radial_speeds.append(
                    (relative_velocity[0] * radial[0] + relative_velocity[1] * radial[1])
                    / radial_length
                    / max(diagonal, 1.0)
                )
        if not relative_speeds:
            return (0.0, 0.0)
        # For an ambiguous source, the ball must separate from every plausible robot.  Using the
        # minimum keeps a close pair from manufacturing a confident attribution or launch.
        return min(relative_speeds), min(radial_speeds)

    def _update_track(
        self,
        track: BallTrack,
        robots: dict[int, _RobotMotion],
        frame_index: int,
        analysis_step: int,
    ) -> None:
        existing_shot = self._shots_by_ball_track.get(track.track_id)
        if existing_shot is not None:
            if track.track_id in self._closed_shot_track_ids:
                return
            if (
                track.last.t_seconds - existing_shot.launch_t_seconds
                > self.config.shot.maximum_shot_track_seconds
            ):
                self._closed_shot_track_ids.add(track.track_id)
                return
            point = self._serialized_point(track.last)
            if not existing_shot.ball_track or existing_shot.ball_track[-1]["frame_index"] != frame_index:
                existing_shot.ball_track.append(point)
                if len(existing_shot.ball_track) >= 2 and existing_shot.outcome == "unknown":
                    self._evaluate_outcome(
                        existing_shot,
                        existing_shot.ball_track[-2],
                        existing_shot.ball_track[-1],
                    )
                if (
                    existing_shot.outcome != "unknown"
                    or self._shot_track_has_ended(existing_shot)
                ):
                    self._closed_shot_track_ids.add(track.track_id)
            return

        state = self.track_states.setdefault(track.track_id, _TrackState())
        point = track.last.center
        containing = tuple(sorted(
            robot_id for robot_id, robot in robots.items()
            if _inside_box(point, robot.observation.bbox, self.config.shot.robot_padding_ratio)
        ))
        relative_ids = containing or state.source_robot_ids
        relative_speed, radial_speed = self._relative_motion(track, relative_ids, robots)

        carried_motion = (
            len(track.points) < 2
            or relative_speed <= self.config.shot.maximum_carried_relative_speed_ratio_per_second
        )
        if len(containing) == 1 and carried_motion:
            robot_id = containing[0]
            if state.owner_robot_id == robot_id:
                state.owner_streak += 1
            else:
                state.owner_robot_id = robot_id
                state.owner_streak = 1
            state.ambiguous_robot_ids = ()
            state.ambiguous_streak = 0
            if state.owner_streak >= self.config.shot.carry_confirmation_frames:
                state.source_robot_ids = (robot_id,)
                state.last_carried_step = analysis_step
        elif len(containing) > 1 and carried_motion:
            if state.ambiguous_robot_ids == containing:
                state.ambiguous_streak += 1
            else:
                state.ambiguous_robot_ids = containing
                state.ambiguous_streak = 1
            state.owner_robot_id = None
            state.owner_streak = 0
            if state.ambiguous_streak >= self.config.shot.carry_confirmation_frames:
                state.source_robot_ids = containing
                state.last_carried_step = analysis_step

        # Rapid-fire shooters often hide the ball until it has already cleared the mechanism.
        # A newly confirmed track may therefore have no observations inside the robot at all.
        # Recover those launches only when the first point is very close to a robot edge and the
        # short observed path is already fast, outward, and geometrically consistent. Slow floor
        # balls and old tracks cannot enter through this path.
        if (
            not state.source_robot_ids
            and len(track.points) <= self.config.shot.edge_launch_max_track_points
        ):
            diagonal = math.hypot(*self.frame_size)
            maximum_distance = (
                self.config.shot.edge_launch_max_distance_ratio * max(diagonal, 1.0)
            )
            first_point = track.points[0]
            nearby_ids = tuple(sorted(
                robot_id for robot_id, robot in robots.items()
                if _distance_to_box(first_point.center, robot.observation.bbox)
                <= maximum_distance
                and self._launch_height_is_plausible(
                    first_point.center, robot.observation.bbox
                )
            ))
            if nearby_ids:
                inferred_speed, inferred_radial_speed = self._relative_motion(
                    track, nearby_ids, robots
                )
                outside_nearby = all(
                    not _inside_box(
                        point,
                        robots[robot_id].observation.bbox,
                        self.config.shot.robot_padding_ratio,
                    )
                    for robot_id in nearby_ids
                    if robot_id in robots
                )
                if (
                    outside_nearby
                    and inferred_speed
                    >= self.config.shot.minimum_launch_relative_speed_ratio_per_second
                    and inferred_radial_speed
                    >= self.config.shot.minimum_radial_speed_ratio_per_second
                ):
                    state.source_robot_ids = nearby_ids
                    state.last_carried_step = analysis_step
                    state.candidate = _LaunchCandidate(
                        source_robot_ids=nearby_ids,
                        robot_track_id=nearby_ids[0] if len(nearby_ids) == 1 else None,
                        launch_frame=first_point.frame_index,
                        launch_t_seconds=first_point.t_seconds,
                        launch_center=first_point.center,
                        trajectory=[item.center for item in track.points[:-1]],
                        candidate_frames=max(0, len(track.points) - 1),
                        away_frames=max(0, len(track.points) - 1),
                    )
                    relative_speed = inferred_speed
                    radial_speed = inferred_radial_speed

        if (
            state.source_robot_ids
            and analysis_step - state.last_carried_step > self.config.shot.source_memory_frames
        ):
            state.source_robot_ids = ()
            state.candidate = None
        if not state.source_robot_ids:
            return

        outside_source = all(
            robot_id not in robots
            or not _inside_box(
                point,
                robots[robot_id].observation.bbox,
                self.config.shot.robot_padding_ratio,
            )
            for robot_id in state.source_robot_ids
        )
        fast = relative_speed >= self.config.shot.minimum_launch_relative_speed_ratio_per_second
        moving_away = radial_speed >= self.config.shot.minimum_radial_speed_ratio_per_second

        if state.candidate is None:
            if not (
                outside_source
                and fast
                and moving_away
                and self._has_nearby_launch_robot(point, state.source_robot_ids, robots)
            ):
                return
            robot_track_id = (
                state.source_robot_ids[0] if len(state.source_robot_ids) == 1 else None
            )
            state.candidate = _LaunchCandidate(
                source_robot_ids=state.source_robot_ids,
                robot_track_id=robot_track_id,
                launch_frame=frame_index,
                launch_t_seconds=track.last.t_seconds,
                launch_center=track.last.center,
                trajectory=[track.last.center],
            )
            return

        candidate = state.candidate
        candidate.candidate_frames += 1
        candidate.trajectory.append(point)
        if outside_source and fast and moving_away:
            candidate.away_frames += 1
        else:
            candidate.away_frames = max(0, candidate.away_frames - 1)
        if candidate.candidate_frames > self.config.shot.maximum_candidate_frames:
            state.candidate = None
            return

        launch_point = candidate.launch_center
        displacement = math.hypot(point[0] - launch_point[0], point[1] - launch_point[1])
        diagonal = math.hypot(*self.frame_size)
        path_length = sum(
            math.hypot(
                candidate.trajectory[index][0] - candidate.trajectory[index - 1][0],
                candidate.trajectory[index][1] - candidate.trajectory[index - 1][1],
            )
            for index in range(1, len(candidate.trajectory))
        )
        consistency = displacement / path_length if path_length > 1e-6 else 0.0
        if (
            candidate.away_frames < self.config.shot.confirmation_frames
            or displacement / max(diagonal, 1.0)
            < self.config.shot.minimum_departure_distance_ratio
            or consistency < self.config.shot.minimum_direction_consistency
        ):
            return

        speed_score = min(
            1.0,
            relative_speed
            / max(self.config.shot.minimum_launch_relative_speed_ratio_per_second, 1e-6),
        )
        carry_score = min(
            1.0,
            max(state.owner_streak, state.ambiguous_streak)
            / self.config.shot.carry_confirmation_frames,
        )
        attributed = candidate.robot_track_id is not None
        confidence = min(
            0.99,
            0.35 + 0.20 * speed_score + 0.15 * carry_score + 0.20 * consistency
            + (0.10 if attributed else 0.0),
        )
        attribution_confidence = min(0.99, 0.55 + 0.10 * carry_score) if attributed else 0.0
        shot = ShotRecord(
            shot_id=str(uuid.uuid4()),
            launch_frame=candidate.launch_frame,
            launch_t_seconds=candidate.launch_t_seconds,
            robot_track_id=candidate.robot_track_id,
            ball_track_id=track.track_id,
            confidence=confidence,
            attribution_confidence=attribution_confidence,
            ball_track=[
                self._serialized_point(item)
                for item in track.points
                if item.frame_index >= candidate.launch_frame
            ],
        )
        self.shots.append(shot)
        self._shots_by_ball_track[track.track_id] = shot
        state.shot_id = shot.shot_id
        for index in range(1, len(shot.ball_track)):
            self._evaluate_outcome(shot, shot.ball_track[index - 1], shot.ball_track[index])
            if shot.outcome != "unknown":
                break

    def _launch_height_is_plausible(self, point: Point, box: Box) -> bool:
        """Reject floor-level contacts while retaining top and side-mounted shooters."""

        _left, top, _right, bottom = box
        height = max(1.0, bottom - top)
        return (point[1] - top) / height <= self.config.shot.maximum_launch_origin_y_ratio

    def _has_nearby_launch_robot(
        self,
        point: Point,
        source_robot_ids: Sequence[int],
        robots: dict[int, _RobotMotion],
    ) -> bool:
        """A launch must originate beside a currently visible plausible source robot."""

        maximum_distance = (
            self.config.shot.maximum_launch_distance_ratio
            * max(math.hypot(*self.frame_size), 1.0)
        )
        return any(
            robot_id in robots
            and _distance_to_box(point, robots[robot_id].observation.bbox) <= maximum_distance
            and self._launch_height_is_plausible(point, robots[robot_id].observation.bbox)
            for robot_id in source_robot_ids
        )

    def _shot_track_has_ended(self, shot: ShotRecord) -> bool:
        """Stop an overlay trail when a flight slows substantially or reverses direction."""

        points = shot.ball_track
        if len(points) < 4:
            return False
        width, height = self.frame_size

        def movement(first: dict[str, object], second: dict[str, object]) -> Point:
            return (
                (float(second["x"]) - float(first["x"])) * width,
                (float(second["y"]) - float(first["y"])) * height,
            )

        launch_vector = movement(points[0], points[min(2, len(points) - 1)])
        recent_vector = movement(points[-3], points[-1])
        launch_length = math.hypot(*launch_vector)
        recent_length = math.hypot(*recent_vector)
        recent_elapsed = float(points[-1]["t_seconds"]) - float(points[-3]["t_seconds"])
        if recent_elapsed <= 1e-9:
            return False
        recent_speed_ratio = (
            recent_length / recent_elapsed / max(math.hypot(width, height), 1.0)
        )
        if (
            float(points[-1]["t_seconds"]) - shot.launch_t_seconds >= 0.25
            and recent_speed_ratio
            < self.config.shot.minimum_shot_track_speed_ratio_per_second
        ):
            return True
        if launch_length <= 1e-6 or recent_length <= 1e-6:
            return False
        direction_cosine = (
            launch_vector[0] * recent_vector[0]
            + launch_vector[1] * recent_vector[1]
        ) / (launch_length * recent_length)
        return direction_cosine < self.config.shot.maximum_shot_track_reversal_cosine

    def _serialized_point(self, point: BallTrackPoint) -> dict[str, object]:
        width, height = self.frame_size
        diagonal = math.hypot(width, height)
        return {
            "frame_index": point.frame_index,
            "t_seconds": round(point.t_seconds, 6),
            "x": round(point.center[0] / max(width, 1), 6),
            "y": round(point.center[1] / max(height, 1), 6),
            "radius": round(point.radius / max(diagonal, 1.0), 6),
            "observed": True,
        }

    def _evaluate_outcome(
        self, shot: ShotRecord, previous_point: dict[str, object], current_point: dict[str, object]
    ) -> None:
        previous = (float(previous_point["x"]), float(previous_point["y"]))
        current = (float(current_point["x"]), float(current_point["y"]))
        movement = current[0] - previous[0], current[1] - previous[1]
        for goal in self.config.goals:
            correct_direction = (
                movement[0] * goal.entry_direction[0]
                + movement[1] * goal.entry_direction[1]
                > 0.0
            )
            made = False
            if goal.made_line is not None:
                made = crosses_directed_boundary(previous, current, goal.made_line)
            elif goal.polygon is not None:
                made = (
                    not _point_in_polygon(previous, goal.polygon)
                    and _point_in_polygon(current, goal.polygon)
                    and correct_direction
                )
            if made:
                shot.outcome = "made"
                shot.goal = goal.goal_id
                shot.outcome_frame = int(current_point["frame_index"])
                shot.outcome_t_seconds = float(current_point["t_seconds"])
                return
        # Misses are never inferred from disappearance or proximity.  They require a separate,
        # explicitly configured directed boundary and a visible crossing.
        for goal in self.config.goals:
            for boundary in goal.miss_boundaries:
                if crosses_directed_boundary(previous, current, boundary):
                    shot.outcome = "missed"
                    shot.goal = goal.goal_id
                    shot.outcome_frame = int(current_point["frame_index"])
                    shot.outcome_t_seconds = float(current_point["t_seconds"])
                    return

    def draw_debug_overlay(self, frame) -> None:
        """Draw candidates, blob tracks, goals, shot trails, and per-robot counts in place."""

        import cv2
        import numpy as np

        height, width = frame.shape[:2]
        for goal in self.config.goals:
            if goal.polygon:
                polygon = [
                    (int(round(x * width)), int(round(y * height)))
                    for x, y in goal.polygon
                ]
                cv2.polylines(frame, [np.array(polygon)], True, (40, 220, 40), 2)
            if goal.made_line:
                first, second = goal.made_line.line
                cv2.line(
                    frame,
                    (int(first[0] * width), int(first[1] * height)),
                    (int(second[0] * width), int(second[1] * height)),
                    (40, 230, 40),
                    3,
                )
            for boundary in goal.miss_boundaries:
                first, second = boundary.line
                cv2.line(
                    frame,
                    (int(first[0] * width), int(first[1] * height)),
                    (int(second[0] * width), int(second[1] * height)),
                    (40, 80, 230),
                    2,
                )
            label_point = goal.center
            cv2.putText(
                frame,
                goal.goal_id,
                (int(label_point[0] * width), int(label_point[1] * height) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        for detection in self.detector.last_detections:
            cv2.circle(
                frame,
                (int(round(detection.center[0])), int(round(detection.center[1]))),
                max(2, int(round(detection.radius))),
                (0, 255, 255),
                1,
            )
        for track in self.tracker.active.values():
            if track.hits < self.config.tracking.minimum_confirmed_hits:
                continue
            trail = track.points[-self.config.debug.trail_points:]
            points = [(int(point.center[0]), int(point.center[1])) for point in trail]
            shot = self._shots_by_ball_track.get(track.track_id)
            colour = (0, 170, 255)
            if shot is not None:
                colour = (80, 230, 80) if shot.outcome == "made" else (
                    (60, 80, 240) if shot.outcome == "missed" else (255, 220, 40)
                )
            if len(points) >= 2:
                cv2.polylines(frame, [np.array(points)], False, colour, 2)
            cv2.putText(
                frame,
                f"b{track.track_id}",
                (points[-1][0] + 3, points[-1][1] - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                colour,
                1,
                cv2.LINE_AA,
            )

        statistics = shot_statistics(self.shots)
        rows = [
            f"shots {statistics['attempted']}  made {statistics['made']}  "
            f"miss {statistics['missed']}  ? {statistics['unknown']}"
        ]
        for robot_id, values in sorted(statistics["per_robot"].items()):
            rows.append(
                f"R{robot_id}: {values['attempted']} / {values['made']} / "
                f"{values['missed']} / ?{values['unknown']}"
            )
        panel_width = max(290, max((len(row) for row in rows), default=0) * 8)
        cv2.rectangle(frame, (8, 8), (8 + panel_width, 17 + 20 * len(rows)), (12, 12, 12), -1)
        for row_index, row in enumerate(rows):
            cv2.putText(
                frame,
                row,
                (15, 25 + row_index * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (240, 240, 240),
                1,
                cv2.LINE_AA,
            )

        if self.config.debug.show_mask and self.detector.last_mask is not None:
            inset_width = min(320, width // 4)
            inset_height = max(1, int(inset_width * height / width))
            mask = cv2.resize(self.detector.last_mask, (inset_width, inset_height))
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            frame[height - inset_height:height, width - inset_width:width] = mask_bgr


def shot_statistics(shots: Iterable[ShotRecord | dict[str, object]]) -> dict[str, object]:
    """Derive counts while keeping unknown outcomes distinct from confirmed misses."""

    totals = {"attempted": 0, "made": 0, "missed": 0, "unknown": 0}
    per_robot: dict[str, dict[str, int]] = {}
    for raw in shots:
        if isinstance(raw, ShotRecord):
            robot_id, outcome = raw.robot_track_id, raw.outcome
        else:
            robot_id, outcome = raw.get("robot_track_id"), raw.get("outcome", "unknown")
        outcome = str(outcome) if outcome in {"made", "missed", "unknown"} else "unknown"
        totals["attempted"] += 1
        totals[outcome] += 1
        key = str(robot_id) if isinstance(robot_id, int) else "unassigned"
        values = per_robot.setdefault(
            key, {"attempted": 0, "made": 0, "missed": 0, "unknown": 0}
        )
        values["attempted"] += 1
        values[outcome] += 1
    return {**totals, "per_robot": per_robot}


def write_shot_records(path: str | Path, shots: Iterable[ShotRecord]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for shot in shots:
            handle.write(json.dumps(shot.to_dict(), sort_keys=True) + "\n")
    temporary.replace(output)
