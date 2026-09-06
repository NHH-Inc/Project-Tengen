"""Use a trained Ultralytics YOLO detector to pre-label additional FRC collections."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import YOLO_ANNOTATOR_VERSION


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(path)


def xyxy_pixels_to_normalized(
    raw_box: list[float] | tuple[float, ...], confidence: float, width: int, height: int
) -> dict[str, Any] | None:
    if len(raw_box) != 4 or width <= 0 or height <= 0:
        return None
    left, top, right, bottom = (float(value) for value in raw_box)
    left, right = max(0.0, min(left, width)), max(0.0, min(right, width))
    top, bottom = max(0.0, min(top, height)), max(0.0, min(bottom, height))
    if right <= left or bottom <= top:
        return None
    return {
        "class_name": "robot",
        "team": None,
        "x": round(left / width, 6),
        "y": round(top / height, 6),
        "w": round((right - left) / width, 6),
        "h": round((bottom - top) / height, 6),
        "confidence": round(max(0.0, min(1.0, float(confidence))), 6),
        "source": "yolo_bootstrap",
    }


def annotate_collection_yolo(
    *, collection: str | Path, model_path: str | Path, confidence: float = 0.25,
    image_size: int = 960, device: str = "0", limit: int | None = None,
    force: bool = False, preview_dir: str | Path | None = None,
) -> Path:
    if not 0 <= confidence <= 1:
        raise ValueError("confidence must be between 0 and 1")
    if image_size <= 0 or image_size % 32:
        raise ValueError("image_size must be a positive multiple of 32")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics is not installed in this Python environment") from exc
    model_file = Path(model_path)
    if not model_file.is_file():
        raise FileNotFoundError(f"YOLO model is missing: {model_file}")
    model = YOLO(str(model_file))
    collection_path = Path(collection)
    frames = _read_jsonl(collection_path / "frames.jsonl")
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        frames = frames[:limit]
    output = collection_path / "yolo-proposals.jsonl"
    rows = _read_jsonl(output)
    requested_ids = {str(frame["frame_id"]) for frame in frames}
    if force:
        rows = [row for row in rows if str(row.get("frame_id")) not in requested_ids]
    existing = {str(row.get("frame_id")) for row in rows}
    previews = Path(preview_dir) if preview_dir else None
    if previews:
        previews.mkdir(parents=True, exist_ok=True)

    for frame in frames:
        frame_id = str(frame["frame_id"])
        if frame_id in existing:
            continue
        image_path = collection_path / str(frame["image_path"])
        if not image_path.is_file():
            raise FileNotFoundError(f"Frame image is missing: {image_path}")
        results = model.predict(
            source=str(image_path), conf=confidence, imgsz=image_size, device=device,
            verbose=False, save=False,
        )
        result = results[0]
        raw_boxes = result.boxes.xyxy.detach().cpu().tolist() if result.boxes is not None else []
        raw_confidences = result.boxes.conf.detach().cpu().tolist() if result.boxes is not None else []
        boxes = [
            box for raw_box, score in zip(raw_boxes, raw_confidences)
            if (box := xyxy_pixels_to_normalized(
                raw_box, score, int(frame["width"]), int(frame["height"])
            )) is not None
        ]
        rows.append({
            "frame_id": frame_id,
            "model": str(model_file),
            "annotator_version": YOLO_ANNOTATOR_VERSION,
            "generated_at": _utc_now(),
            "status": "proposed",
            "review_status": "unreviewed",
            "human_review_required": True,
            "confidence_threshold": confidence,
            "image_size": image_size,
            "boxes": boxes,
        })
        if previews:
            result.save(filename=str(previews / f"{frame_id}.jpg"))
        rows.sort(key=lambda value: str(value["frame_id"]))
        _write_jsonl(output, rows)
        print(f"{frame_id}: YOLO proposed {len(boxes)} boxes", flush=True)
    return output
