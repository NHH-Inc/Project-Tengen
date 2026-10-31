"""Validation harness (T0): does the motion model reproduce a real auto?

A sim is only worth trusting if it can retrace an auto we already have on video. So: take a real
match's auto, feed its start pose and waypoints to the model, and measure how far the simulated
trajectory strays from the actual track. A model that can't retrace a known auto within a few feet
has no business predicting an unknown one. The tracks that drive the heatmaps are the ground truth,
so there is nothing to label.

This is the yardstick every higher tier is measured against (docs/SIM-DESIGN.md section 5).
"""

from __future__ import annotations

import math

from ..strategy.timeline import samples_as_rows
from .model import RobotParams, pose_at, simulate_path

#: Downsample the actual path to waypoints at least this far apart -- the "intended" path, as a
#: driver would think of it, rather than every jittery 10 Hz sample.
WAYPOINT_SPACING_FT = 3.0
#: Below this many positioned auto samples there is nothing to validate.
MIN_AUTO_SAMPLES = 4


def _downsample(points: list[tuple[float, float]], spacing_ft: float) -> list[tuple[float, float]]:
    kept = [points[0]]
    for point in points[1:]:
        if math.hypot(point[0] - kept[-1][0], point[1] - kept[-1][1]) >= spacing_ft:
            kept.append(point)
    if kept[-1] != points[-1]:
        kept.append(points[-1])
    return kept[1:]  # drop the start pose; the waypoints are the targets after it


def auto_waypoints(timeline: dict, *, spacing_ft: float = WAYPOINT_SPACING_FT):
    """A robot's actual auto path from its timeline: the start pose, the downsampled waypoints to
    drive, and the full actual `(elapsed_s, x, y)` points to score against. None if too little."""
    rows = samples_as_rows(timeline)
    auto = [(r["t"], r["x"], r["y"]) for r in rows
            if r.get("phase") == "auto" and r.get("x") is not None]
    if len(auto) < MIN_AUTO_SAMPLES:
        return None
    t0 = auto[0][0]
    actual = [(t - t0, x, y) for (t, x, y) in auto]
    start = (actual[0][1], actual[0][2])
    waypoints = _downsample([(x, y) for (_, x, y) in actual], spacing_ft)
    return start, waypoints, actual


def trajectory_error(sim, actual: list[tuple[float, float, float]]) -> dict:
    """Position error (ft) of the simulated trajectory against the actual path, sampled at the
    actual track's timestamps."""
    errors = []
    for t, ax, ay in actual:
        point = pose_at(sim, t)
        if point is None:
            continue
        errors.append(math.hypot(point[0] - ax, point[1] - ay))
    if not errors:
        return {"n": 0, "mean_ft": None, "max_ft": None}
    return {"n": len(errors), "mean_ft": round(sum(errors) / len(errors), 2),
            "max_ft": round(max(errors), 2)}


def validate_auto(timeline: dict, params: RobotParams | None = None) -> dict | None:
    """Simulate a robot's own auto and score the result. None when the robot has no usable auto."""
    found = auto_waypoints(timeline)
    if found is None:
        return None
    start, waypoints, actual = found
    if not waypoints:
        return None
    duration = actual[-1][0]
    params = params or RobotParams()
    sim = simulate_path(start, waypoints, params, max_time_s=duration + 2.0)
    return {
        "team": timeline.get("team"),
        "match_id": timeline.get("match_id"),
        "robot_key": timeline.get("robot_key"),
        "n_waypoints": len(waypoints),
        "actual_duration_s": round(duration, 2),
        "sim_duration_s": round(sim[-1].t, 2) if sim else 0.0,
        "error": trajectory_error(sim, actual),
    }
