"""FRC-aware spatial filtering and conservative temporal consistency for robot proposals."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import FRC_FILTER_VERSION


def box_iou(a: dict[str, Any], b: dict[str, Any]) -> float:
    left, top = max(float(a["x"]), float(b["x"])), max(float(a["y"]), float(b["y"]))
    right = min(float(a["x"]) + float(a["w"]), float(b["x"]) + float(b["w"]))
    bottom = min(float(a["y"]) + float(a["h"]), float(b["y"]) + float(b["h"]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = float(a["w"]) * float(a["h"]) + float(b["w"]) * float(b["h"]) - intersection
    return intersection / union if union > 0 else 0.0


def _center_distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    ax, ay = float(a["x"]) + float(a["w"]) / 2, float(a["y"]) + float(a["h"]) / 2
    bx, by = float(b["x"]) + float(b["w"]) / 2, float(b["y"]) + float(b["h"]) / 2
    return math.hypot(ax - bx, ay - by)


@dataclass(frozen=True)
class FilterSettings:
    min_confidence: float = 0.25
    high_confidence: float = 0.60
    min_area: float = 0.00015
    max_area: float = 0.30
    min_aspect_ratio: float = 0.20
    max_aspect_ratio: float = 5.0
    nms_iou: float = 0.75
    roi: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
    bumper_color_enabled: bool = True
    bumper_min_fraction: float = 0.02
    bumper_confidence_bonus: float = 0.08

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "FilterSettings":
        bumper = raw.get("bumper_color", {})
        roi = tuple(float(value) for value in raw.get("roi", (0, 0, 1, 1)))
        if len(roi) != 4:
            raise ValueError("frc_filter.roi must contain [left, top, right, bottom]")
        settings = cls(
            min_confidence=float(raw.get("min_confidence", cls.min_confidence)),
            high_confidence=float(raw.get("high_confidence", cls.high_confidence)),
            min_area=float(raw.get("min_area", cls.min_area)),
            max_area=float(raw.get("max_area", cls.max_area)),
            min_aspect_ratio=float(raw.get("min_aspect_ratio", cls.min_aspect_ratio)),
            max_aspect_ratio=float(raw.get("max_aspect_ratio", cls.max_aspect_ratio)),
            nms_iou=float(raw.get("nms_iou", cls.nms_iou)),
            roi=roi,
            bumper_color_enabled=bool(bumper.get("enabled", cls.bumper_color_enabled)),
            bumper_min_fraction=float(bumper.get("min_fraction", cls.bumper_min_fraction)),
            bumper_confidence_bonus=float(bumper.get("confidence_bonus", cls.bumper_confidence_bonus)),
        )
        if not 0 <= settings.min_confidence <= settings.high_confidence <= 1:
            raise ValueError("FRC confidence thresholds must satisfy 0 <= min <= high <= 1")
        if not 0 < settings.min_area < settings.max_area <= 1:
            raise ValueError("FRC area thresholds must satisfy 0 < min < max <= 1")
        if not 0 < settings.min_aspect_ratio < settings.max_aspect_ratio:
            raise ValueError("FRC aspect-ratio thresholds are invalid")
        if not 0 < settings.nms_iou <= 1:
            raise ValueError("frc_filter.nms_iou must be in (0, 1]")
        left, top, right, bottom = settings.roi
        if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
            raise ValueError("frc_filter.roi must be normalized and non-empty")
        return settings


@dataclass(frozen=True)
class TemporalSettings:
    enabled: bool = True
    association_iou: float = 0.15
    max_center_distance: float = 0.10
    max_gap_frames: int = 2
    min_track_hits: int = 2

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "TemporalSettings":
        settings = cls(
            enabled=bool(raw.get("enabled", cls.enabled)),
            association_iou=float(raw.get("association_iou", cls.association_iou)),
            max_center_distance=float(raw.get("max_center_distance", cls.max_center_distance)),
            max_gap_frames=int(raw.get("max_gap_frames", cls.max_gap_frames)),
            min_track_hits=int(raw.get("min_track_hits", cls.min_track_hits)),
        )
        if not 0 <= settings.association_iou <= 1 or settings.max_center_distance < 0:
            raise ValueError("Temporal association thresholds are invalid")
        if settings.max_gap_frames < 0 or settings.min_track_hits < 1:
            raise ValueError("Temporal gap/hit settings are invalid")
        return settings


def _validate_box(raw: dict[str, Any]) -> dict[str, Any] | None:
    try:
        x, y, width, height = (float(raw[name]) for name in ("x", "y", "w", "h"))
        confidence = float(raw["confidence"])
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (x, y, width, height, confidence)):
        return None
    if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > 1.000001 or y + height > 1.000001:
        return None
    result = dict(raw)
    result.update(x=x, y=y, w=width, h=height, confidence=max(0.0, min(1.0, confidence)))
    result.setdefault("class_name", "robot")
    return result


def geometry_rejection_reason(box: dict[str, Any], settings: FilterSettings) -> str | None:
    area = float(box["w"]) * float(box["h"])
    aspect = float(box["w"]) / float(box["h"])
    center_x = float(box["x"]) + float(box["w"]) / 2
    center_y = float(box["y"]) + float(box["h"]) / 2
    left, top, right, bottom = settings.roi
    if float(box["confidence"]) < settings.min_confidence:
        return "below_min_confidence"
    if area < settings.min_area:
        return "box_too_small"
    if area > settings.max_area:
        return "box_too_large"
    if aspect < settings.min_aspect_ratio or aspect > settings.max_aspect_ratio:
        return "implausible_aspect_ratio"
    if not (left <= center_x <= right and top <= center_y <= bottom):
        return "center_outside_field_roi"
    return None


def bumper_color_fraction(image: Any, box: dict[str, Any]) -> dict[str, float]:
    """Measure red/blue HSV pixels inside a box. OpenCV is imported only for real image runs."""
    import cv2
    import numpy as np

    height, width = image.shape[:2]
    left = max(0, min(width, int(float(box["x"]) * width)))
    top = max(0, min(height, int(float(box["y"]) * height)))
    right = max(left + 1, min(width, int((float(box["x"]) + float(box["w"])) * width)))
    bottom = max(top + 1, min(height, int((float(box["y"]) + float(box["h"])) * height)))
    crop = image[top:bottom, left:right]
    if crop.size == 0:
        return {"red": 0.0, "blue": 0.0, "max": 0.0}
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    red = cv2.inRange(hsv, np.array([0, 80, 55]), np.array([12, 255, 255]))
    red |= cv2.inRange(hsv, np.array([168, 80, 55]), np.array([179, 255, 255]))
    blue = cv2.inRange(hsv, np.array([92, 70, 45]), np.array([132, 255, 255]))
    pixels = float(crop.shape[0] * crop.shape[1])
    red_fraction = float((red > 0).sum()) / pixels
    blue_fraction = float((blue > 0).sum()) / pixels
    return {"red": round(red_fraction, 6), "blue": round(blue_fraction, 6), "max": round(max(red_fraction, blue_fraction), 6)}


def filter_frame_boxes(
    boxes: list[dict[str, Any]], settings: FilterSettings, image: Any | None = None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for raw in boxes:
        box = _validate_box(raw)
        if box is None:
            rejected.append({**raw, "filter_reasons": ["invalid_box"]})
            continue
        reason = geometry_rejection_reason(box, settings)
        if reason:
            rejected.append({**box, "filter_reasons": [reason]})
            continue
        reasons = ["geometry_pass", "confidence_pass"]
        evidence = {"red": 0.0, "blue": 0.0, "max": 0.0}
        if settings.bumper_color_enabled and image is not None:
            evidence = bumper_color_fraction(image, box)
            if evidence["max"] >= settings.bumper_min_fraction:
                reasons.append("bumper_color_support")
        adjusted = min(1.0, float(box["confidence"]) + (
            settings.bumper_confidence_bonus if "bumper_color_support" in reasons else 0.0
        ))
        accepted.append({
            **box,
            "raw_confidence": float(box["confidence"]),
            "confidence": round(adjusted, 6),
            "bumper_color_fraction": evidence,
            "filter_reasons": reasons,
        })

    kept: list[dict[str, Any]] = []
    for candidate in sorted(accepted, key=lambda value: float(value["confidence"]), reverse=True):
        if any(box_iou(candidate, prior) >= settings.nms_iou for prior in kept):
            rejected.append({**candidate, "filter_reasons": candidate["filter_reasons"] + ["duplicate_nms"]})
        else:
            kept.append(candidate)
    return kept, rejected


def apply_temporal_consistency(
    rows: list[dict[str, Any]], temporal: TemporalSettings, high_confidence: float
) -> list[dict[str, Any]]:
    if not temporal.enabled:
        return rows
    active: list[dict[str, Any]] = []
    tracks: dict[int, list[tuple[int, int]]] = {}
    next_track_id = 1

    for frame_index, row in enumerate(rows):
        boxes = row.get("boxes", [])
        candidates: list[tuple[float, int, int]] = []
        for box_index, box in enumerate(boxes):
            for track_index, track in enumerate(active):
                gap = frame_index - int(track["last_frame"]) - 1
                if gap > temporal.max_gap_frames:
                    continue
                iou = box_iou(box, track["last_box"])
                distance = _center_distance(box, track["last_box"])
                if iou >= temporal.association_iou or distance <= temporal.max_center_distance:
                    candidates.append((iou - distance, box_index, track_index))
        used_boxes: set[int] = set()
        used_tracks: set[int] = set()
        for _, box_index, track_index in sorted(candidates, reverse=True):
            if box_index in used_boxes or track_index in used_tracks:
                continue
            track = active[track_index]
            track_id = int(track["track_id"])
            boxes[box_index]["track_id"] = track_id
            track.update(last_frame=frame_index, last_box=boxes[box_index])
            tracks[track_id].append((frame_index, box_index))
            used_boxes.add(box_index)
            used_tracks.add(track_index)
        for box_index, box in enumerate(boxes):
            if box_index in used_boxes:
                continue
            box["track_id"] = next_track_id
            active.append({"track_id": next_track_id, "last_frame": frame_index, "last_box": box})
            tracks[next_track_id] = [(frame_index, box_index)]
            next_track_id += 1
        active = [
            track for track in active
            if frame_index - int(track["last_frame"]) <= temporal.max_gap_frames
        ]

    track_hits = {track_id: len(positions) for track_id, positions in tracks.items()}
    for row in rows:
        accepted, temporal_rejected = [], []
        for box in row.get("boxes", []):
            hits = track_hits[int(box["track_id"])]
            box["temporal_hits"] = hits
            if hits >= temporal.min_track_hits:
                box["filter_reasons"].append("temporal_support")
                accepted.append(box)
            elif float(box["confidence"]) >= high_confidence:
                box["filter_reasons"].append("high_confidence_single_frame")
                accepted.append(box)
            else:
                temporal_rejected.append({
                    **box,
                    "filter_reasons": box["filter_reasons"] + ["insufficient_temporal_support"],
                })
        row["boxes"] = accepted
        row.setdefault("rejected_boxes", []).extend(temporal_rejected)
    return rows


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(path)


def filter_collection(
    *, collection: str | Path, input_name: str, output_name: str,
    filter_settings: dict[str, Any], temporal_settings: dict[str, Any],
) -> dict[str, Any]:
    collection_path = Path(collection)
    frames = _read_jsonl(collection_path / "frames.jsonl")
    proposals = _read_jsonl(collection_path / input_name)
    proposal_by_id = {str(row["frame_id"]): row for row in proposals if row.get("status") != "failed"}
    settings = FilterSettings.from_mapping(filter_settings)
    temporal = TemporalSettings.from_mapping(temporal_settings)
    use_images = settings.bumper_color_enabled
    cv2 = None
    if use_images:
        try:
            import cv2 as loaded_cv2
        except ImportError as exc:
            raise RuntimeError(
                "OpenCV is required when frc_filter.bumper_color.enabled is true; install opencv-python"
            ) from exc
        cv2 = loaded_cv2

    output_rows = []
    for frame in sorted(frames, key=lambda row: (float(row.get("source_video_time_seconds", 0)), str(row["frame_id"]))):
        frame_id = str(frame["frame_id"])
        proposal = proposal_by_id.get(frame_id, {"boxes": []})
        image = None
        if use_images and proposal.get("boxes"):
            image_path = collection_path / str(frame["image_path"])
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"OpenCV could not read frame image: {image_path}")
        accepted, rejected = filter_frame_boxes(list(proposal.get("boxes", [])), settings, image)
        output_rows.append({
            "frame_id": frame_id,
            "annotator_version": FRC_FILTER_VERSION,
            "input_file": input_name,
            "input_source": proposal.get("model", proposal.get("annotator_version", "unknown")),
            "status": "proposed",
            "review_status": "unreviewed",
            "human_review_required": True,
            "boxes": accepted,
            "rejected_boxes": rejected,
        })
    apply_temporal_consistency(output_rows, temporal, settings.high_confidence)
    output_path = collection_path / output_name
    _write_jsonl(output_path, output_rows)
    accepted_count = sum(len(row["boxes"]) for row in output_rows)
    rejected_count = sum(len(row["rejected_boxes"]) for row in output_rows)
    report = {
        "output": str(output_path),
        "frames": len(output_rows),
        "accepted_boxes": accepted_count,
        "rejected_boxes": rejected_count,
        "human_review_required": True,
        "warning": "Automatic filters rank and reject proposals; they do not create ground truth.",
    }
    (collection_path / "filter-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
