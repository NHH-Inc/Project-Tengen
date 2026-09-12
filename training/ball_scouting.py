"""Full-rate game-piece tracking and evidence-based rapid-fire launch extraction.

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
from collections import deque
from pathlib import Path
from typing import Iterable, Sequence

from training.launch_signals import LaunchSignalConfig, LaunchSignalBank
from training.goal_scoring import GoalEntryCounter, goal_statistics, trajectory_link_cost


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
    morph_open_iterations: int = 0
    morph_close_iterations: int = 0
    min_area_px: float = 18.0
    max_area_px: float = 2200.0
    min_radius_px: float = 2.0
    max_radius_px: float = 38.0
    min_circularity: float = 0.28
    min_fill_ratio: float = 0.35
    min_aspect_ratio: float = 0.30
    max_aspect_ratio: float = 3.30
    split_touching: bool = True
    motion_blur_max_aspect: float = 9.0
    motion_min_fraction: float = 0.25
    motion_difference_threshold: float = 18.0


@dataclass(frozen=True)
class BlobTrackingConfig:
    minimum_confirmed_hits: int = 2
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
    region_id: str = ""
    approach_line: DirectedBoundary | None = None
    confirmation_polygon: tuple[Point, ...] | None = None
    confirmation_frames: int = 1
    minimum_depth_radii: float = 0.0
    maximum_gap_seconds: float = 0.085
    maximum_confirmation_seconds: float = 0.25
    allow_partial_approach: bool = False
    expected_ball_radius: float = 0.0

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
    launch_signals: LaunchSignalConfig = field(default_factory=LaunchSignalConfig)
    automatic_goals: bool = False
    calibration_anchors: tuple = ()


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
    boundary = DirectedBoundary(
        line=(_point(raw_line[0], f"{name}.line[0]"), _point(raw_line[1], f"{name}.line[1]")),
        direction=_direction(value.get("direction"), f"{name}.direction"),
    )
    a, b = boundary.line
    if abs((b[0] - a[0]) * boundary.direction[1]
           - (b[1] - a[1]) * boundary.direction[0]) < 1e-9:
        raise ValueError(f"{name} needs a nonzero line and a direction crossing it")
    return boundary


def _goal(value: object, index: int) -> GoalGeometry:
    name = f"goals[{index}]"
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    allowed = {"id", "region_id", "polygon", "entry_direction", "made_boundary", "miss_boundaries",
               "approach_boundary", "confirmation_polygon", "confirmation_frames", "minimum_depth_radii",
               "maximum_gap_seconds", "maximum_confirmation_seconds", "allow_partial_approach",
               "expected_ball_radius", "_comment"}
    if set(value) - allowed:
        raise ValueError(f"{name} has unknown settings: {sorted(set(value) - allowed)}")
    goal_id = value.get("id")
    if not isinstance(value.get("allow_partial_approach", False), bool):
        raise ValueError(f"{name}.allow_partial_approach must be a boolean")
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
    confirmation_polygon = value.get("confirmation_polygon")
    if confirmation_polygon is not None:
        if not isinstance(confirmation_polygon, list) or len(confirmation_polygon) < 3:
            raise ValueError(f"{name}.confirmation_polygon needs at least three points")
        confirmation_polygon = tuple(_point(p, f"{name}.confirmation_polygon") for p in confirmation_polygon)
    region_id = value.get("region_id", f"{goal_id.strip()}_{index + 1}")
    if not isinstance(region_id, str) or not region_id.strip():
        raise ValueError(f"{name}.region_id must be a non-empty string")
    return GoalGeometry(
        goal_id=goal_id.strip(),
        polygon=polygon,
        entry_direction=entry_direction,
        made_line=made_line,
        miss_boundaries=tuple(
            _directed_boundary(boundary, f"{name}.miss_boundaries[{miss_index}]")
            for miss_index, boundary in enumerate(raw_misses)
        ),
        region_id=region_id.strip(),
        allow_partial_approach=value.get("allow_partial_approach", False) is True,
        expected_ball_radius=_number(value.get("expected_ball_radius", 0),
                                     f"{name}.expected_ball_radius", minimum=0),
        approach_line=(_directed_boundary(value["approach_boundary"], f"{name}.approach_boundary")
                       if value.get("approach_boundary") is not None else None),
        confirmation_polygon=confirmation_polygon,
        confirmation_frames=_integer(value.get("confirmation_frames", 1), f"{name}.confirmation_frames", minimum=1),
        minimum_depth_radii=_number(value.get("minimum_depth_radii", 0), f"{name}.minimum_depth_radii", minimum=0),
        maximum_gap_seconds=_number(value.get("maximum_gap_seconds", .085), f"{name}.maximum_gap_seconds", minimum=.001),
        maximum_confirmation_seconds=_number(value.get("maximum_confirmation_seconds", .25),
                                            f"{name}.maximum_confirmation_seconds", minimum=.001),
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
    signal_raw = document.get("launch_signals") or {}
    for raw, name in (
        (detector_raw, "detector"), (tracking_raw, "tracking"),
        (shot_raw, "shot"), (debug_raw, "debug"), (signal_raw, "launch_signals"),
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
    from dataclasses import replace

    split_touching = detector_raw.get("split_touching", True)
    if not isinstance(split_touching, bool):
        raise ValueError("detector.split_touching must be a boolean")
    detector = replace(
        detector, split_touching=split_touching,
        motion_blur_max_aspect=_number(detector_raw.get("motion_blur_max_aspect", 9.0),
                                      "detector.motion_blur_max_aspect", minimum=1),
        motion_min_fraction=_number(detector_raw.get("motion_min_fraction", 0.25),
                                    "detector.motion_min_fraction", minimum=0),
        motion_difference_threshold=_number(
            detector_raw.get("motion_difference_threshold", 18.0),
            "detector.motion_difference_threshold", minimum=1),
    )
    if detector.motion_min_fraction > 1 or detector.motion_difference_threshold > 255:
        raise ValueError("detector motion thresholds exceed their valid range")

    signal_defaults = LaunchSignalConfig()
    signal_values = {}
    unknown_signal_keys = set(signal_raw) - set(signal_defaults.__dataclass_fields__)
    if unknown_signal_keys:
        raise ValueError(f"unknown launch_signals settings: {sorted(unknown_signal_keys)}")
    for name in signal_defaults.__dataclass_fields__:
        default = getattr(signal_defaults, name)
        value = signal_raw.get(name, default)
        if isinstance(default, bool):
            if not isinstance(value, bool):
                raise ValueError(f"launch_signals.{name} must be a boolean")
        elif isinstance(default, int):
            value = _integer(value, f"launch_signals.{name}", minimum=2)
        else:
            value = _number(value, f"launch_signals.{name}", minimum=0.001)
        signal_values[name] = value
    signals = LaunchSignalConfig(**signal_values)
    if signals.pulse_relative_prominence > 1:
        raise ValueError("launch_signals.pulse_relative_prominence cannot exceed one")

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
    if len({goal.region_id for goal in goals}) != len(goals):
        raise ValueError("goals.region_id must be unique for each physical goal")
    enabled = document.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError("enabled must be a boolean")
    return BallScoutingConfig(enabled, detector, tracking, shot, debug, goals, signals,
        (document.get("goal_calibration") or {}).get("source") == "apriltag_pose",
        tuple((document.get("goal_calibration") or {}).get("anchors", [])))


@dataclass(frozen=True)
class BallDetection:
    center: Point
    radius: float
    area: float
    circularity: float
    bbox: Box
    novelty: float = 1.0


class YellowBallDetector:
    """Colour + temporal foreground, with distance-peak separation of touching balls."""

    def __init__(self, config: BallDetectorConfig):
        self.config = config
        self.last_mask = None
        self.last_detections: list[BallDetection] = []
        self.previous_gray = None
        self.scene_changed = False
        self.previous_colour = None

    def _split_contour(self, contour):
        import cv2
        import numpy as np

        if not self.config.split_touching:
            return [contour]
        left, top, width, height = cv2.boundingRect(contour)
        if width * height > self.config.max_area_px * 16:
            return [contour]
        local = np.zeros((height + 2, width + 2), dtype=np.uint8)
        offset = np.array([[[left - 1, top - 1]]], dtype=np.int32)
        cv2.drawContours(local, [contour - offset], -1, 255, -1)
        distance = cv2.distanceTransform(local, cv2.DIST_L2, 5)
        peak = float(distance.max())
        size = max(3, int(self.config.min_radius_px * 2) + 1)
        maxima = ((distance >= cv2.dilate(distance, np.ones((size, size), np.uint8)) - 1e-5)
                  & (distance >= max(self.config.min_radius_px, peak * 0.45))).astype(np.uint8)
        count, labels, stats, centers = cv2.connectedComponentsWithStats(maxima)
        seeds = []
        for label in range(1, count):
            x, y = centers[label]
            r = float(distance[int(round(y)), int(round(x))])
            # A long flat ridge is a smear/rectangle, not many individual balls.
            if stats[label, cv2.CC_STAT_AREA] > max(4, r * r):
                continue
            seeds.append((r, float(x), float(y)))
        selected = []
        for r, x, y in sorted(seeds, reverse=True):
            if r > self.config.max_radius_px:
                continue
            if any(math.hypot(x - a, y - b) < 1.5 * max(r, other)
                   for other, a, b in selected):
                continue
            selected.append((r, x, y))
            if len(selected) >= 16:
                break
        if len(selected) < 2:
            return [contour]
        yy, xx = np.indices(local.shape)
        partition = np.argmin(np.stack([
            ((xx - x) ** 2 + (yy - y) ** 2) / max(r * r, 1)
            for r, x, y in selected
        ]), axis=0)
        pieces = []
        for index in range(len(selected)):
            part = ((partition == index) & (local > 0)).astype(np.uint8) * 255
            contours, _ = cv2.findContours(part, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                pieces.append(max(contours, key=cv2.contourArea) + offset)
        return pieces or [contour]

    def detect(self, frame) -> list[BallDetection]:
        import cv2
        import numpy as np

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        moving = np.zeros_like(gray)
        self.scene_changed = False
        if self.previous_gray is not None and self.previous_gray.shape == gray.shape:
            difference = cv2.absdiff(gray, self.previous_gray)
            moving = (difference >= self.config.motion_difference_threshold).astype(np.uint8) * 255
            self.scene_changed = float(difference.mean()) > 55 and float((moving > 0).mean()) > 0.65
        self.previous_gray = gray
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
        # Preserve native-resolution colour cores that Gaussian blur/morphology can erase.
        raw_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        raw_mask = cv2.inRange(raw_hsv, np.array(self.config.hsv_lower, np.uint8),
                              np.array(self.config.hsv_upper, np.uint8))
        previous_colour = self.previous_colour
        self.previous_colour = raw_mask
        lower = self.config.hsv_lower
        weak = cv2.inRange(raw_hsv, np.array((max(0, lower[0] - 2), int(lower[1] * 0.55),
                                            int(lower[2] * 0.65)), np.uint8),
                           np.array((min(179, self.config.hsv_upper[0] + 2),
                                     self.config.hsv_upper[1], self.config.hsv_upper[2]), np.uint8))
        mask |= raw_mask | (weak & moving)
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
        pieces = [piece for contour in contours for piece in self._split_contour(contour)]
        for contour in pieces:
            area = float(cv2.contourArea(contour))
            if area <= 0 or not self.config.min_area_px <= area <= self.config.max_area_px:
                continue
            perimeter = float(cv2.arcLength(contour, True))
            if perimeter <= 1e-6:
                continue
            circularity = 4.0 * math.pi * area / (perimeter * perimeter)
            left, top, width, height = cv2.boundingRect(contour)
            aspect = width / max(1.0, float(height))
            fill_ratio = area / max(1.0, float(width * height))
            if fill_ratio < self.config.min_fill_ratio:
                continue
            circular = (circularity >= self.config.min_circularity
                        and self.config.min_aspect_ratio <= aspect <= self.config.max_aspect_ratio)
            foreground = moving[top:top + height, left:left + width]
            coloured = mask[top:top + height, left:left + width] > 0
            motion_fraction = float(((foreground > 0) & coloured).sum()) / max(1, int(coloured.sum()))
            novelty = 0.0
            if previous_colour is not None and previous_colour.shape == mask.shape:
                old_colour = previous_colour[top:top + height, left:left + width] > 0
                novelty = float((coloured & ~old_colour).sum()) / max(1, int(coloured.sum()))
            elongated = max(aspect, 1 / max(aspect, 1e-6))
            if not circular and not (motion_fraction >= self.config.motion_min_fraction
                                     and elongated <= self.config.motion_blur_max_aspect):
                continue
            moments = cv2.moments(contour)
            center_x, center_y = moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]
            # Equivalent-area radius remains stable when the contour is a motion smear.
            radius = math.sqrt(area / math.pi)
            if not self.config.min_radius_px <= radius <= self.config.max_radius_px:
                continue
            detections.append(BallDetection(
                center=(float(center_x), float(center_y)),
                radius=float(radius),
                area=area,
                circularity=min(1.0, circularity),
                bbox=(float(left), float(top), float(left + width), float(top + height)),
                novelty=novelty,
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
    novelty: float = 1.0


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
        if len(self.points) >= 4:
            import numpy as np

            recent = self.points[-5:]
            times = np.array([p.t_seconds - recent[-1].t_seconds for p in recent])
            scale = max(1e-6, -times[0])
            matrix = np.stack([np.ones_like(times), times / scale, (times / scale) ** 2], axis=1)
            positions = np.array([p.center for p in recent])
            coefficients, *_ = np.linalg.lstsq(matrix, positions, rcond=None)
            error = float(np.sqrt(np.mean((matrix @ coefficients - positions) ** 2)))
            if error <= max(1.5, self.last.radius * 0.4) and elapsed <= scale:
                # Local ballistic prediction; a poor fit falls back to the short velocity
                # model, e.g. at the exact transition from carried to launched.
                result = np.array([1, elapsed / scale, (elapsed / scale) ** 2]) @ coefficients
                return float(result[0]), float(result[1])
        velocity = self.velocity
        return self.last.center[0] + velocity[0] * elapsed, self.last.center[1] + velocity[1] * elapsed


class BallBlobTracker:
    """Global one-to-one assignment with ballistic prediction and occlusion coasting."""

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
            wide_gate = bool(wide_gate_track_ids and track_id in wide_gate_track_ids)
            if wide_gate:
                # A confirmed carried ball is the one place a sudden acceleration is expected.
                # Keeping this wider gate away from ordinary floor/decor tracks prevents dense
                # piles from being stitched into implausible zig-zags.
                gate_speed_ratio = self.config.maximum_speed_ratio_per_second
            else:
                gate_speed_ratio = min(
                    self.config.maximum_speed_ratio_per_second,
                    self.config.acceleration_allowance_ratio_per_second,
                )
            allowed = diagonal * (
                self.config.base_link_distance_ratio
                + gate_speed_ratio * elapsed
            )
            allowed += track.last.radius * 0.6
            if not wide_gate and track.hits >= 2 and observed_speed_ratio > .25:
                # A coasting projectile must not latch onto the next ball at its old
                # muzzle position just because elapsed time widened the uncertainty gate.
                expected_travel = math.dist(predicted, track.last.center)
                allowed = min(allowed, max(track.last.radius * 1.2,
                                          diagonal * self.config.base_link_distance_ratio
                                          + expected_travel * .6))
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
                step = (detection.center[0] - track.last.center[0],
                        detection.center[1] - track.last.center[1])
                if math.hypot(*step) > diagonal * (
                        self.config.maximum_speed_ratio_per_second * elapsed
                        + self.config.base_link_distance_ratio):
                    continue
                velocity = track.velocity
                step_length = math.hypot(*step)
                cosine = sum(a * b for a, b in zip(step, velocity)) / max(
                    step_length * math.hypot(*velocity), 1e-6)
                if (not wide_gate and track.hits >= 3 and observed_speed_ratio > 0.25
                        and step_length > track.last.radius and cosine < -0.2):
                    continue
                cost = (distance / max(allowed, 1e-6) + 0.18 * abs(math.log(radius_ratio))
                        + 0.04 * track.missed_frames)
                candidates.append((cost, track_id, detection_index))

        for track in self.active.values():
            track.matched_this_frame = False
        assignments: dict[int, int] = {}
        used_detections: set[int] = set()
        if candidates:
            import numpy as np
            from scipy.optimize import linear_sum_assignment

            ids = sorted({track_id for _, track_id, _ in candidates})
            rows = {track_id: row for row, track_id in enumerate(ids)}
            costs = np.full((len(ids), len(detections) + len(ids)), 1e6)
            # Each track has its own explicit 'missed' option; an invalid match is never
            # forced just to fill the assignment matrix.
            for row in range(len(ids)):
                costs[row, len(detections) + row] = 1.35
            for cost, track_id, detection_index in candidates:
                costs[rows[track_id], detection_index] = cost
            for row, column in zip(*linear_sum_assignment(costs)):
                if column < len(detections) and costs[row, column] < 1.35:
                    assignments[ids[row]] = int(column)
                    used_detections.add(int(column))

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
                novelty=detection.novelty,
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
                novelty=detection.novelty,
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
        self.launch_signals = LaunchSignalBank(config.launch_signals)
        self._robot_history: deque = deque(maxlen=120)
        self._gray_history: deque = deque(maxlen=8)
        self._previous_time: float | None = None
        self._previous_frame: int | None = None
        self.shot_methods: dict[str, str] = {}
        self.goal_counter = GoalEntryCounter(config.goals)
        self.goals_valid = True
        self._last_camera_check = -math.inf
        self._last_camera_verified = -math.inf
        self._camera_tags_visible = False
        self.goal_camera_gaps: list[list[float | None]] = []

    @property
    def goal_entries(self):
        return self.goal_counter.entries

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
        if (not math.isfinite(t_seconds) or t_seconds < 0 or frame_index < 0
                or (self._previous_time is not None and t_seconds <= self._previous_time)
                or (self._previous_frame is not None and frame_index <= self._previous_frame)):
            raise ValueError("ball analysis requires increasing source frame indices and timestamps")
        height, width = frame.shape[:2]
        interval = t_seconds - self._previous_time if self._previous_time is not None else 0.0
        discontinuity = (self._previous_time is not None and (
            interval > 0.25 or self.frame_size != (width, height)))
        self._previous_time, self._previous_frame = t_seconds, frame_index
        self.frame_size = (width, height)
        self._analysis_step += 1
        detections = self.detector.detect(frame)
        if self.config.automatic_goals and (self.detector.scene_changed or discontinuity
                or t_seconds - self._last_camera_check >= (1.0 if self._camera_tags_visible else .1)):
            from training.auto_goals import camera_matches
            was_valid = self.goals_valid
            self._camera_tags_visible = camera_matches(frame, self.config.calibration_anchors)
            if self.detector.scene_changed or discontinuity:
                self._last_camera_verified = -math.inf
            if self._camera_tags_visible:
                self._last_camera_verified = t_seconds
            # A failed decode is not evidence of a moved camera. Keep a recently
            # verified pose through short occlusions, retrying ten times a second.
            # Cuts/gaps invalidate immediately; sustained failure also expires it.
            self.goals_valid = t_seconds - self._last_camera_verified <= 3.0
            self._last_camera_check = t_seconds
            if self.goals_valid != was_valid:
                self.goal_counter.reset_temporal(t_seconds)
                if not self.goals_valid:
                    self.goal_camera_gaps.append([t_seconds, None])
                elif self.goal_camera_gaps:
                    self.goal_camera_gaps[-1][1] = t_seconds
        if discontinuity or self.detector.scene_changed:
            self._closed_shot_track_ids.update(self._shots_by_ball_track)
            self.tracker.active.clear()
            self.track_states.clear()
            self.robot_motion.clear()
            self._robot_history.clear()
            self._gray_history.clear()
            self.launch_signals = LaunchSignalBank(self.config.launch_signals)
            self.goal_counter.reset_temporal(t_seconds)
        current_robots = self._update_robots(robots, t_seconds)
        self._robot_history.append((t_seconds, current_robots))
        self._gray_history.append((frame_index, self.detector.previous_gray))
        self.robot_motion = {key: value for key, value in self.robot_motion.items()
                             if t_seconds - value.t_seconds <= 0.25}
        carried_track_ids = {
            track_id for track_id, state in self.track_states.items()
            if state.source_robot_ids
            and track_id not in self._shots_by_ball_track
            and self._analysis_step - state.last_carried_step
            <= self.config.shot.source_memory_frames
        }
        # The first visible ball of a burst has no velocity yet. It needs a full speed
        # association gate at the shooter, otherwise it becomes a new ID every frame.
        diagonal = math.hypot(width, height)
        carried_track_ids.update(
            track.track_id for track in self.tracker.active.values()
            if track.hits < 2 and any(
                _distance_to_box(track.last.center, robot.observation.bbox)
                <= self.config.shot.edge_launch_max_distance_ratio * diagonal
                and self._launch_height_is_plausible(track.last.center, robot.observation.bbox)
                for robot in current_robots.values()
            )
        )
        # A fast inbound ball first seen at the basket also has no velocity yet.
        # The shooter-only birth gate used to fragment those balls before crossing.
        for track in self.tracker.active.values():
            if track.hits != 1:
                continue
            position = (track.last.center[0] / width, track.last.center[1] / height)
            for goal in self.config.goals:
                if goal.confirmation_polygon:
                    xs, ys = zip(*goal.confirmation_polygon)
                    pad = max(track.last.radius * 3 / width, .015)
                    if (min(xs) - pad <= position[0] <= max(xs) + pad
                            and min(ys) - pad * width / height <= position[1] <= max(ys)):
                        carried_track_ids.add(track.track_id)
                        break
        matched, retired = self.tracker.update(
            detections,
            frame_index,
            t_seconds,
            self.frame_size,
            wide_gate_track_ids=carried_track_ids,
        )
        for track in list(self.tracker.active.values()):
            if not track.matched_this_frame:
                continue
            if self.config.launch_signals.enabled:
                self._short_launch(track, current_robots, interval)
            if track.hits >= self.config.tracking.minimum_confirmed_hits:
                self._update_track(track, current_robots, frame_index, self._analysis_step)
        if self.config.launch_signals.enabled:
            pulses = self.launch_signals.update(
                self.detector.last_mask, current_robots, frame_index, t_seconds,
                self.config.shot.minimum_launch_relative_speed_ratio_per_second * diagonal,
            )
            for pulse in pulses:
                self._record_gate_pulse(pulse, current_robots, interval)
        entries = self.goal_counter.observe(
            list(self.tracker.active.values()), frame_index, t_seconds, self.frame_size,
        ) if self.goals_valid else []
        self._associate_goal_entries(entries)
        for track in retired:
            if track.track_id not in self._shots_by_ball_track:
                self.track_states.pop(track.track_id, None)
        return detections

    def _associate_goal_entries(self, entries):
        """Credit observed entries once; ambiguous or long missing flights stay unassigned."""
        unmatched = []
        for entry in entries:
            shot = self._shots_by_ball_track.get(entry.ball_track_id)
            if (shot is not None and shot.outcome == "unknown"
                    and 0 <= entry.t_seconds - shot.launch_t_seconds
                    <= self.config.shot.maximum_shot_track_seconds):
                self._credit_goal_entry(entry, shot, "same_track", 1.0)
            else:
                unmatched.append(entry)
        candidates = [shot for shot in self.shots if shot.outcome == "unknown"
                      and shot.launch_t_seconds >= self.goal_counter.scene_start]
        # Mutual uniqueness rejects intersecting lookalike paths. Select all links before
        # mutating shots so iteration order cannot turn an ambiguous pair into a sure one.
        links = []
        for entry in unmatched:
            for shot in candidates:
                if not 0 < entry.t_seconds - shot.launch_t_seconds <= self.config.shot.maximum_shot_track_seconds:
                    continue
                cost = trajectory_link_cost(shot, entry, self.frame_size)
                if cost is not None:
                    links.append((cost, entry, shot))
        accepted = []
        for cost, entry, shot in links:
            alternatives = [other_cost for other_cost, other_entry, other_shot in links
                            if (other_entry is entry and other_shot is not shot)
                            or (other_shot is shot and other_entry is not entry)]
            if not alternatives or min(alternatives) - cost >= .25:
                accepted.append((cost, entry, shot))
        for cost, entry, shot in accepted:
            self._credit_goal_entry(entry, shot, "trajectory_relink", max(.5, .9 - cost * .3))

    def _credit_goal_entry(self, entry, shot, association, confidence):
        entry.shot_id = shot.shot_id
        entry.robot_track_id = shot.robot_track_id
        entry.association = association
        entry.association_confidence = min(confidence, shot.attribution_confidence)
        shot.outcome, shot.goal = "made", entry.goal
        shot.outcome_frame, shot.outcome_t_seconds = entry.frame_index, entry.t_seconds
        evidence = {p["frame_index"]: p for p in shot.ball_track}
        evidence.update({p["frame_index"]: p for p in entry.ball_track
                         if p["frame_index"] >= shot.launch_frame})
        shot.ball_track = [evidence[index] for index in sorted(evidence)]
        self._shots_by_ball_track[entry.ball_track_id] = shot
        self._closed_shot_track_ids.update((shot.ball_track_id, entry.ball_track_id))

    def _flow_agrees(self, first: BallTrackPoint, last: BallTrackPoint) -> bool | None:
        """Check the actual local pixels with bidirectional pyramidal optical flow.

        Matching yellow colour alone cannot distinguish neighbouring hopper balls. Flow
        tests the short correspondence independently. An untrackable patch returns unknown,
        not a fabricated displacement. The crop bounds the work even on 1080p video.
        """
        import cv2
        import numpy as np

        before = next((gray for index, gray in self._gray_history if index == first.frame_index), None)
        after = next((gray for index, gray in self._gray_history if index == last.frame_index), None)
        if before is None or after is None:
            return None
        radius = min(first.radius, last.radius)
        pad = max(16, math.ceil(radius * 3))
        x0 = max(0, math.floor(min(first.center[0], last.center[0]) - pad))
        y0 = max(0, math.floor(min(first.center[1], last.center[1]) - pad))
        x1 = min(before.shape[1], math.ceil(max(first.center[0], last.center[0]) + pad))
        y1 = min(before.shape[0], math.ceil(max(first.center[1], last.center[1]) + pad))
        old = before[y0:y1, x0:x1]
        new = after[y0:y1, x0:x1]
        origin = np.array(first.center, dtype=np.float32) - (x0, y0)
        support = np.zeros(old.shape, np.uint8)
        cv2.circle(support, tuple(np.round(origin).astype(int)), max(3, round(radius * 1.4)), 255, -1)
        features = cv2.goodFeaturesToTrack(old, maxCorners=12, qualityLevel=.02,
                                          minDistance=2, mask=support, blockSize=3)
        if features is None or len(features) < 2:
            return None
        window = max(9, min(25, round(radius * 2) | 1))
        moved, status, error = cv2.calcOpticalFlowPyrLK(
            old, new, features, None, winSize=(window, window), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 25, .01),
        )
        if moved is None:
            return None
        backward, back_status, _ = cv2.calcOpticalFlowPyrLK(
            new, old, moved, None, winSize=(window, window), maxLevel=3,
        )
        if backward is None:
            return None
        round_trip = np.linalg.norm(backward[:, 0] - features[:, 0], axis=1)
        good = (status[:, 0] > 0) & (back_status[:, 0] > 0) & (error[:, 0] < 30)
        good &= round_trip < max(1.5, radius * .5)
        if int(good.sum()) < 2:
            return None
        displacement = np.median(moved[good, 0] - features[good, 0], axis=0)
        expected = np.array(last.center) - np.array(first.center)
        return bool(np.linalg.norm(displacement - expected) <= max(2.5, radius * .9))

    def _short_launch(self, track, robots, interval):
        """Count a measured local departure without waiting for a long flight track."""
        if track.track_id in self._shots_by_ball_track:
            return
        settings = self.config.launch_signals
        points = track.points
        if not settings.short_track_minimum_hits <= len(points) <= self.config.shot.edge_launch_max_track_points:
            return
        if any(b.t_seconds - a.t_seconds > settings.maximum_observation_gap_seconds
               for a, b in zip(points, points[1:])):
            return
        first, last = points[0], points[-1]
        # A two-point match between pre-existing yellow patches is not motion evidence.
        # Require newly occupied yellow pixels at the destination, or a longer measured
        # path; this prevents a hidden ball from jumping to a nearby hopper/floor ball.
        if len(points) == 2 and last.novelty < self.config.detector.motion_min_fraction:
            return
        elapsed = last.t_seconds - first.t_seconds
        origin_robots = next((value for timestamp, value in self._robot_history
                              if abs(timestamp - first.t_seconds) < 1e-6), {})
        diagonal = max(1.0, math.hypot(*self.frame_size))
        radius = min(first.radius, last.radius)
        possible = []
        for robot_id, initial in origin_robots.items():
            if robot_id not in robots:
                continue
            start_box = initial.observation.bbox
            current_box = robots[robot_id].observation.bbox
            if (_distance_to_box(first.center, start_box)
                    > min(self.config.shot.edge_launch_max_distance_ratio * diagonal,
                          max(3.0, radius * 2.0))
                    or not self._launch_height_is_plausible(first.center, start_box)):
                continue
            if _inside_box(first.center, start_box, 0):
                edge_distance = min(first.center[0] - start_box[0], start_box[2] - first.center[0],
                                    first.center[1] - start_box[1])
                if edge_distance > max(first.radius * 1.5, 2.0):
                    continue
            if _inside_box(last.center, current_box, self.config.shot.robot_padding_ratio):
                continue
            if any(other != robot_id and _inside_box(last.center, value.observation.bbox, 0)
                   for other, value in robots.items()):
                continue
            left, top, right, bottom = start_box
            x0, y0, x1, y1 = current_box
            sx = (x1 - x0) / max(1.0, right - left)
            sy = (y1 - y0) / max(1.0, bottom - top)
            if not (0.75 <= sx <= 1.33 and 0.75 <= sy <= 1.33):
                continue
            # Compare in robot coordinates to reject carried yellow during pan/zoom and
            # robot motion. Use the box at the ball's birth, not today's box at that point.
            translated_first = (x0 + (first.center[0] - left) * sx,
                                y0 + (first.center[1] - top) * sy)
            movement = (last.center[0] - translated_first[0],
                        last.center[1] - translated_first[1])
            travel = math.hypot(*movement)
            speed = travel / max(elapsed, 1e-6)
            radial = (last.center[0] - (x0 + x1) / 2, last.center[1] - (y0 + y1) / 2)
            outward = sum(a * b for a, b in zip(movement, radial)) / max(
                math.hypot(*radial) * elapsed, 1e-6)
            if (speed / diagonal < self.config.shot.minimum_launch_relative_speed_ratio_per_second
                    or outward / diagonal < self.config.shot.minimum_radial_speed_ratio_per_second
                    or travel < max(radius * settings.minimum_travel_radii, diagonal * 0.003)):
                continue
            path = sum(math.dist(a.center, b.center) for a, b in zip(points, points[1:]))
            straightness = math.dist(first.center, last.center) / max(path, 1e-6)
            if straightness < self.config.shot.minimum_direction_consistency:
                continue
            possible.append((robot_id, translated_first, movement, speed))
        if not possible:
            return
        flow = self._flow_agrees(points[-2], last)
        if flow is False:
            return
        if last.novelty < self.config.detector.motion_min_fraction and flow is not True:
            return
        if flow is None and len(points) == 2 and last.novelty < 0.65:
            return
        robot_id = possible[0][0] if len(possible) == 1 else None
        # Timestamp the first observed point outside the source; hidden births at the edge
        # retain their first observation, rather than backdating an unseen launch.
        launch = first
        if robot_id is not None:
            for point in points:
                frame_robots = next((value for timestamp, value in self._robot_history
                                     if abs(timestamp - point.t_seconds) < 1e-6), {})
                if robot_id in frame_robots and not _inside_box(
                        point.center, frame_robots[robot_id].observation.bbox,
                        self.config.shot.robot_padding_ratio):
                    launch = point
                    break
        shot = ShotRecord(
            str(uuid.uuid4()), launch.frame_index, launch.t_seconds, robot_id, track.track_id,
            0.86 if len(points) == 2 else 0.92, 0.72 if robot_id is not None else 0.0,
            ball_track=[self._serialized_point(p) for p in points],
        )
        existing = self._find_existing_launch(shot)
        if existing is not None:
            self._shots_by_ball_track[track.track_id] = existing
            return
        if robot_id is not None:
            _, origin, movement, speed = possible[0]
            port = self.launch_signals.seed(robot_id, robots[robot_id].observation.bbox,
                origin, (origin[0] + movement[0], origin[1] + movement[1]),
                radius, speed, last.t_seconds)
            if port is not None:
                existing = self.launch_signals.register_track(
                    port, robots[robot_id].observation.bbox, points, shot, interval)
                if existing is not None:
                    self._shots_by_ball_track[track.track_id] = existing
                    return
        self.shots.append(shot)
        self.shot_methods[shot.shot_id] = "short_track"
        self._shots_by_ball_track[track.track_id] = shot
        self.track_states.setdefault(track.track_id, _TrackState()).shot_id = shot.shot_id
        for a, b in zip(shot.ball_track, shot.ball_track[1:]):
            self._evaluate_outcome(shot, a, b)
            if shot.outcome != "unknown":
                break

    def _record_gate_pulse(self, pulse, robots, interval):
        robot = robots.get(pulse.robot_id)
        if robot is None:
            return
        # Do not attribute occupancy shared by another robot's box.
        if any(key != pulse.robot_id and _inside_box(pulse.inner.center, value.observation.bbox, 0)
               for key, value in robots.items()):
            return
        evidence = [BallTrackPoint(s.frame_index, s.t_seconds, s.center, s.radius,
                                   math.pi * s.radius ** 2, 1.0)
                    for s in (pulse.inner, pulse.outer)]
        shot = ShotRecord(
            str(uuid.uuid4()), pulse.inner.frame_index, pulse.inner.t_seconds, pulse.robot_id,
            self.tracker.next_track_id, 0.78, 0.65,
            ball_track=[self._serialized_point(p) for p in evidence],
        )
        existing = self._find_existing_launch(shot)
        registered = self.launch_signals.register_pulse(
            pulse, robot.observation.bbox, existing or shot, interval)
        existing = existing or registered
        if existing is not None:
            return
        # Reserve a real, unique evidence ID even if segmentation never produced a blob.
        self.tracker.next_track_id += 1
        self.shots.append(shot)
        self.shot_methods[shot.shot_id] = "paired_gate"
        # Attach only to a geometrically matching unclaimed track. No guessed ball flight
        # is generated when a gate pulse is the only available evidence.
        matches = []
        for track in self.tracker.active.values():
            if track.track_id in self._shots_by_ball_track:
                continue
            point = next((p for p in track.points if p.frame_index == pulse.outer.frame_index), None)
            if point is not None and math.dist(point.center, pulse.outer.center) <= pulse.outer.radius:
                matches.append(track)
        if len(matches) == 1:
            shot.ball_track_id = matches[0].track_id
            self._shots_by_ball_track[matches[0].track_id] = shot

    def _find_existing_launch(self, proposed: ShotRecord) -> ShotRecord | None:
        """Fuse overlapping observations across ports/track fragments, without a cooldown.

        Two observations must agree in both time and position. Adjacent balls in the same
        burst can therefore be counted even when their event timestamps are very close.
        Single-frame intersections are insufficient to collapse two crossing trajectories.
        """
        width, height = self.frame_size
        diagonal = math.hypot(width, height)
        for existing in reversed(self.shots):
            if (existing.robot_track_id != proposed.robot_track_id
                    or abs(existing.launch_t_seconds - proposed.launch_t_seconds) > .35):
                continue
            old = [p for p in existing.ball_track if p["frame_index"] >= existing.launch_frame]
            comparisons = 0
            agreements = 0
            for point in proposed.ball_track[:8]:
                if point["frame_index"] < proposed.launch_frame:
                    continue
                match = next((p for p in old if p["frame_index"] == point["frame_index"]), None)
                if match is None:
                    for a, b in zip(old, old[1:]):
                        dt = b["t_seconds"] - a["t_seconds"]
                        if (a["t_seconds"] < point["t_seconds"] < b["t_seconds"]
                                and 0 < dt <= self.config.launch_signals.maximum_observation_gap_seconds):
                            fraction = (point["t_seconds"] - a["t_seconds"]) / dt
                            match = {key: a[key] + fraction * (b[key] - a[key])
                                     for key in ("x", "y", "radius")}
                            break
                if match is None:
                    continue
                comparisons += 1
                distance = math.hypot((point["x"] - match["x"]) * width,
                                      (point["y"] - match["y"]) * height)
                tolerance = max(3.0, diagonal * min(point["radius"], match["radius"]) * 1.1)
                agreements += int(distance <= tolerance)
            if agreements >= 2 and agreements / comparisons >= .75:
                return existing
        return None

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
            if track.track_id in self._closed_shot_track_ids or existing_shot.outcome != "unknown":
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
                    or (not self.config.goals and self._shot_track_has_ended(existing_shot))
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
        else:
            state.owner_streak = 0
            state.ambiguous_streak = 0

        # Rapid-fire shooters often hide the ball until it has already cleared the mechanism.
        # A newly confirmed track may therefore have no observations inside the robot at all.
        # Recover those launches only when the first point is very close to a robot edge and the
        # short observed path is already fast, outward, and geometrically consistent. Slow floor
        # balls and old tracks cannot enter through this path.
        if (
            not self.config.launch_signals.enabled
            and not state.source_robot_ids
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
        if (self.config.launch_signals.enabled
                and self._flow_agrees(track.points[-2], track.points[-1]) is False):
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
        existing = self._find_existing_launch(shot)
        if existing is not None:
            self._shots_by_ball_track[track.track_id] = existing
            state.shot_id = existing.shot_id
            return
        if self.config.launch_signals.enabled and candidate.robot_track_id in robots:
            source = robots[candidate.robot_track_id]
            flight = [p for p in track.points if p.frame_index >= candidate.launch_frame]
            if len(flight) >= 2:
                a, b = flight[0], flight[-1]
                duration = max(b.t_seconds - a.t_seconds, 1e-6)
                port = self.launch_signals.seed(
                    candidate.robot_track_id, source.observation.bbox, a.center, b.center,
                    min(p.radius for p in flight), math.dist(a.center, b.center) / duration,
                    b.t_seconds,
                )
                if port is not None:
                    existing = self.launch_signals.register_track(
                        port, source.observation.bbox, flight, shot,
                        flight[-1].t_seconds - flight[-2].t_seconds,
                    )
                    if existing is not None:
                        self._shots_by_ball_track[track.track_id] = existing
                        state.shot_id = existing.shot_id
                        return
        self.shots.append(shot)
        self.shot_methods[shot.shot_id] = "trajectory"
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
        if float(current_point["t_seconds"]) - float(previous_point["t_seconds"]) > .085:
            return
        previous = (float(previous_point["x"]), float(previous_point["y"]))
        current = (float(current_point["x"]), float(current_point["y"]))
        # Makes are evaluated independently for every ball by GoalEntryCounter, including
        # tracks whose source robot was never visible. A line hit alone cannot bypass its
        # approach/interior/depth checks here.
        # Misses are never inferred from disappearance or proximity.  They require a separate,
        # explicitly configured directed boundary and a visible crossing.
        for goal in self.config.goals if self.goals_valid else ():
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
        for goal in self.config.goals if self.goals_valid else ():
            if goal.confirmation_polygon:
                interior = np.array([(round(x * width), round(y * height))
                                     for x, y in goal.confirmation_polygon])
                cv2.polylines(frame, [interior], True, (90, 180, 90), 1)
            if goal.approach_line:
                first, second = goal.approach_line.line
                cv2.line(frame, (round(first[0] * width), round(first[1] * height)),
                         (round(second[0] * width), round(second[1] * height)), (220, 180, 40), 1)
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
                f"{goal.region_id or goal.goal_id}: "
                f"{sum(e.region_id == (goal.region_id or goal.goal_id) for e in self.goal_entries)} made",
                (int(label_point[0] * width), int(label_point[1] * height) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        for sample in self.launch_signals.last_samples:
            cx, cy = sample["center"]
            dx, dy = sample["direction"]
            radius = sample["radius"]
            color = (255, 80, 220) if sample["gate"] == 0 else (220, 220, 80)
            cv2.line(frame, (round(cx - dy * radius * 2), round(cy + dx * radius * 2)),
                     (round(cx + dy * radius * 2), round(cy - dx * radius * 2)), color, 2)
            cv2.putText(frame, f"{sample['value']:.2f}", (round(cx + 3), round(cy - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, .3, color, 1)

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
        goals = goal_statistics(self.goal_entries)
        rows = [
            f"shots {statistics['attempted']}  made {statistics['made']}  "
            f"miss {statistics['missed']}  ? {statistics['unknown']}"
        ]
        if self.config.goals:
            rows.append(f"Goal entries: {goals['made']} made / {goals['unassigned']} source unknown")
            if not self.goals_valid:
                rows.append("Auto goals paused: camera changed; recalibration required")
        else:
            rows.append("Goals not calibrated: made count unavailable")
        for entry in self.goal_entries:
            if self._previous_time is None or self._previous_time - entry.t_seconds > .75:
                continue
            points = [(round(p['x'] * width), round(p['y'] * height)) for p in entry.ball_track]
            if points:
                cv2.polylines(frame, [np.array(points)], False, (60, 255, 100), 3)
                cv2.putText(frame, "IN", points[-1], cv2.FONT_HERSHEY_SIMPLEX, .6,
                            (60, 255, 100), 2, cv2.LINE_AA)
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
