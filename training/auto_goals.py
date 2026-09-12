"""Project the 2026 hub funnels from a surveyed AprilTag camera pose.

The carpet homography alone cannot locate a six-foot-high opening. We retain the
3-D camera pose, intersect the funnel at two heights, and normalize only after
projecting into the original image. No camera-specific image coordinates live here.

Geometry: FIRST 2026 manual section 5.4 (41.7 inch flat-to-flat rim, 72 inch height),
GE-26329 funnel side drawing (17.90 inch height, 24.09/16.30 inch top/bottom edges).
https://firstfrc.blob.core.windows.net/frc2026/Manual/HTML/2026GameManual.htm
https://firstfrc.blob.core.windows.net/frc2026/FieldAssets/TE-26300-build-instructions.pdf
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

HUB_TAGS = {"blue_hub": (18, 19, 20, 21, 24, 25, 26, 27),
            "red_hub": (2, 3, 4, 5, 8, 9, 10, 11)}


def project_goals(calibration, image_size, crop, layout_path=None):
    from ingest.collection.apriltag_layout import load_layout

    trusted = calibration.get("trustworthy", (calibration.get("solution") or {}).get("trustworthy"))
    pose = calibration.get("pose")
    if not trusted or not isinstance(pose, dict):
        return []
    layout_path = layout_path or Path(__file__).resolve().parents[1] / "contracts/fields/2026-apriltags.json"
    layout = load_layout(layout_path)
    if layout.season != 2026:
        return []
    width, height = image_size
    left, top, right, bottom = crop
    origin = np.array([round(left * width), round(top * height)])
    scale = np.array([round(right * width), round(bottom * height)]) - origin
    if np.any(scale <= 0):
        raise ValueError("Goal projection requires a valid model crop")
    camera = np.array(pose["camera_matrix"], dtype=float)
    # Calibration is stored in original source pixels, never in model-crop pixels.
    original_size = calibration.get("image_size") or image_size
    camera = np.diag([width / original_size[0], height / original_size[1], 1]) @ camera
    rotation = np.array(pose["rvec"], dtype=float)
    translation = np.array(pose["tvec"], dtype=float)
    matrix, _ = cv2.Rodrigues(rotation)
    goals = []
    for region, ids in HUB_TAGS.items():
        tags = [layout.tags[i] for i in ids]
        center = np.array([(min(t.x_ft for t in tags) + max(t.x_ft for t in tags)) / 2,
                           (min(t.y_ft for t in tags) + max(t.y_ft for t in tags)) / 2])

        def section(depth):
            # Regular hexagon with flats parallel to field X; shrink along the funnel.
            diameter = (41.7 - (41.7 - 16.30 * math.sqrt(3)) * depth / (17.90 / 12)) / 12
            radius = diameter / math.sqrt(3)
            angles = np.arange(6) * math.pi / 3 + math.pi / 6
            points = np.c_[center + radius * np.c_[np.cos(angles), np.sin(angles)],
                           np.full(6, 6.0 - depth)]
            if np.any((points @ matrix.T + translation)[:, 2] <= 0):
                return None
            pixels, _ = cv2.projectPoints(points, rotation, translation, camera, None)
            return (pixels.reshape(-1, 2) - origin) / scale

        rim, scoring, deep = section(0), section(.65), section(1.0)
        if any(p is None for p in (rim, scoring, deep)):
            continue
        # A finite line across each projected cross-section, perpendicular to gravity.
        centers = [p.mean(axis=0) for p in (rim, scoring, deep)]
        down_pixels = (centers[2] - centers[0]) * scale
        length = np.linalg.norm(down_pixels)
        if length < 5:
            continue
        down = down_pixels / length
        across = np.array([down[1], -down[0]])

        def gate(points, center):
            offsets = ((points - center) * scale) @ across
            return np.array([center + across * offsets.min() / scale,
                             center + across * offsets.max() / scale])

        approach, made, end = [gate(p, c) for p, c in zip((rim, scoring, deep), centers)]
        # A ball centre must clear the rim, not just overlap its outer edge.
        # FIRST section 5.10.1: FUEL diameter is 5.91 inches. Project a sphere
        # at the scoring section and include half the measured calibration residual.
        depth = (matrix @ np.r_[center, 6.0 - .65] + translation)[2]
        radius_ft = 5.91 / 24
        ball_radius_px = math.sqrt(camera[0, 0] * camera[1, 1]) * radius_ft / depth
        across_radius = np.linalg.norm(across * [camera[0, 0], camera[1, 1]]) * radius_ft / depth
        inset = across * (across_radius + min(3, pose.get("reprojection_px", 0) / 2)) / scale
        made[0] += inset
        made[1] -= inset
        end[0] += inset
        end[1] -= inset
        polygon = np.array([approach[0], approach[1], end[1], end[0]])
        if not np.isfinite(polygon).all() or np.any(polygon < 0) or np.any(polygon > 1):
            continue
        direction = down / scale
        direction /= np.linalg.norm(direction)
        goals.append(dict(id="high", region_id=region, entry_direction=direction.tolist(),
                          approach_boundary=dict(line=approach.tolist(), direction=direction.tolist()),
                          made_boundary=dict(line=made.tolist(), direction=direction.tolist()),
                          confirmation_polygon=polygon.tolist(), confirmation_frames=2,
                          minimum_depth_radii=.25, maximum_gap_seconds=.085,
                          maximum_confirmation_seconds=.3, allow_partial_approach=True,
                          expected_ball_radius=ball_radius_px / math.hypot(*scale),
                          miss_boundaries=[], polygon=rim.tolist()))
    return goals


def configure_auto_goals(config_path, calibration_path, image_size, crop, output_path):
    """Keep explicit human regions; fill an empty goal list from a trusted camera pose."""
    document = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if document.get("goals") or document.get("auto_goals", True) is False:
        return Path(config_path)
    calibration = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
    goals = project_goals(calibration, image_size, crop)
    document["goals"] = goals
    document["goal_calibration"] = dict(source="apriltag_pose" if goals else "unavailable",
                                         tags_used=calibration.get("tags_used", []),
                                         image_size=list(image_size), crop=list(crop),
                                         reprojection_px=(calibration.get("pose") or {}).get("reprojection_px"))
    from ingest.collection.apriltag_layout import load_layout
    layout = load_layout(Path(__file__).resolve().parents[1] / "contracts/fields/2026-apriltags.json")
    anchors = []
    if goals:
        pose = calibration["pose"]
        original_size = calibration.get("image_size") or image_size
        camera = np.diag([image_size[0] / original_size[0], image_size[1] / original_size[1], 1]) @ np.array(pose["camera_matrix"])
        origin = np.round(np.array(crop[:2]) * image_size)
        scale = np.round(np.array(crop[2:]) * image_size) - origin
        for tag_id in calibration.get("tags_used", []):
            if tag_id not in layout.tags:
                continue
            pixels, _ = cv2.projectPoints(layout.tags[tag_id].corners_ft(), np.array(pose["rvec"], float),
                np.array(pose["tvec"], float), camera, None)
            corners = (pixels.reshape(4, 2) - origin) / scale
            anchors.append(dict(tag_id=tag_id, corners=corners.tolist()))
    document["goal_calibration"]["anchors"] = anchors
    Path(output_path).write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return Path(output_path)


def camera_matches(frame, anchors):
    """Decode tags close to their calibrated locations, including after a lighting flash.

    Two known markers must agree. False also includes unreadable/occluded tags;
    it is not proof of camera motion. The analyzer supplies bounded occlusion
    tolerance and invalidates immediately on detected cuts or timestamp gaps.
    """
    from ingest.collection.calibrate import build_detector

    height, width = frame.shape[:2]
    detector = build_detector()
    found = 0
    for anchor in anchors:
        corners = np.array(anchor["corners"]) * [width, height]
        center = corners.mean(axis=0)
        margin = max(10, np.ptp(corners, axis=0).max() * .5)
        x0, y0 = np.maximum(0, np.floor(corners.min(axis=0) - margin)).astype(int)
        x1, y1 = np.minimum([width, height], np.ceil(corners.max(axis=0) + margin)).astype(int)
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue
        gray = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        detected, ids, _ = detector.detectMarkers(cv2.resize(gray, None, fx=2, fy=2,
                                                            interpolation=cv2.INTER_CUBIC))
        for pixels, tag_id in zip(detected, ids.ravel() if ids is not None else []):
            observed = pixels.reshape(4, 2).mean(axis=0) / 2 + [x0, y0]
            if int(tag_id) == anchor["tag_id"] and np.linalg.norm(observed - center) <= 8:
                found += 1
                break
        if found >= 2:
            return True
    return False


def save_camera_gaps(config_path, gaps):
    """Keep camera validity with the effective regions for synchronized playback."""
    path = Path(config_path)
    if not path.exists():
        return
    document = json.loads(path.read_text(encoding="utf-8"))
    if (document.get("goal_calibration") or {}).get("source") == "apriltag_pose":
        document["goal_calibration"]["camera_gaps"] = gaps
        path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
