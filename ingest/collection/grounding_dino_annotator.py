"""Grounding DINO Swin-T bootstrap proposals for extracted full-field FRC frames."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import GROUNDING_DINO_ANNOTATOR_VERSION


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


def cxcywh_to_xywh(box: list[float] | tuple[float, ...], confidence: float, phrase: str) -> dict[str, Any] | None:
    if len(box) != 4:
        return None
    center_x, center_y, width, height = (float(value) for value in box)
    left, top = center_x - width / 2, center_y - height / 2
    right, bottom = center_x + width / 2, center_y + height / 2
    left, top, right, bottom = max(0.0, left), max(0.0, top), min(1.0, right), min(1.0, bottom)
    if right <= left or bottom <= top:
        return None
    return {
        "class_name": "robot",
        "team": None,
        "x": round(left, 6),
        "y": round(top, 6),
        "w": round(right - left, 6),
        "h": round(bottom - top, 6),
        "confidence": round(max(0.0, min(1.0, float(confidence))), 6),
        "phrase": str(phrase),
        "source": "grounding_dino_swint",
    }


def _load_grounding_dino(model_config: Path, checkpoint: Path, device: str):
    try:
        import cv2
        import torch
        from groundingdino.util.inference import annotate, load_image, load_model, predict
    except ImportError as exc:
        raise RuntimeError(
            "Grounding DINO is not installed. Install the official IDEA-Research/GroundingDINO "
            "repository into the dedicated vision environment."
        ) from exc
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Grounding DINO requested CUDA, but torch.cuda.is_available() is false")
    if not model_config.is_file():
        raise FileNotFoundError(f"Grounding DINO model config is missing: {model_config}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Grounding DINO checkpoint is missing: {checkpoint}")
    model = load_model(str(model_config), str(checkpoint), device=device)
    return cv2, model, load_image, predict, annotate


def annotate_collection_grounding_dino(
    *, collection: str | Path, model_config: str | Path, checkpoint: str | Path,
    settings: dict[str, Any], device: str = "cuda", limit: int | None = None,
    force: bool = False, preview_dir: str | Path | None = None,
) -> Path:
    collection_path = Path(collection)
    frames = _read_jsonl(collection_path / "frames.jsonl")
    if not frames:
        raise ValueError(f"No frames.jsonl records found in {collection_path}")
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        frames = frames[:limit]
    prompt = str(settings.get("prompt", "robot . competition robot . wheeled robot ."))
    box_threshold = float(settings.get("box_threshold", 0.25))
    text_threshold = float(settings.get("text_threshold", 0.20))
    if not 0 <= box_threshold <= 1 or not 0 <= text_threshold <= 1:
        raise ValueError("Grounding DINO thresholds must be between 0 and 1")
    cv2, model, load_image, predict, annotate = _load_grounding_dino(
        Path(model_config), Path(checkpoint), device
    )
    previews = Path(preview_dir) if preview_dir else None
    if previews:
        previews.mkdir(parents=True, exist_ok=True)
    output = collection_path / "grounding-dino-proposals.jsonl"
    rows = _read_jsonl(output)
    requested_ids = {str(frame["frame_id"]) for frame in frames}
    if force:
        rows = [row for row in rows if str(row.get("frame_id")) not in requested_ids]
    existing = {str(row.get("frame_id")) for row in rows}

    for frame in frames:
        frame_id = str(frame["frame_id"])
        if frame_id in existing:
            continue
        image_path = collection_path / str(frame["image_path"])
        if not image_path.is_file():
            raise FileNotFoundError(f"Frame image is missing: {image_path}")
        try:
            image_source, transformed = load_image(str(image_path))
            boxes, logits, phrases = predict(
                model=model, image=transformed, caption=prompt,
                box_threshold=box_threshold, text_threshold=text_threshold, device=device,
            )
            raw_boxes = boxes.detach().cpu().tolist() if hasattr(boxes, "detach") else list(boxes)
            raw_logits = logits.detach().cpu().tolist() if hasattr(logits, "detach") else list(logits)
            converted = [
                result for raw_box, confidence, phrase in zip(raw_boxes, raw_logits, phrases)
                if (result := cxcywh_to_xywh(raw_box, float(confidence), str(phrase))) is not None
            ]
            row = {
                "frame_id": frame_id,
                "model": "GroundingDINO_SwinT_OGC",
                "annotator_version": GROUNDING_DINO_ANNOTATOR_VERSION,
                "generated_at": _utc_now(),
                "status": "proposed",
                "review_status": "unreviewed",
                "human_review_required": True,
                "prompt": prompt,
                "box_threshold": box_threshold,
                "text_threshold": text_threshold,
                "boxes": converted,
            }
            if previews:
                preview = annotate(
                    image_source=image_source, boxes=boxes, logits=logits, phrases=phrases
                )
                cv2.imwrite(str(previews / f"{frame_id}.jpg"), preview)
        except Exception as exc:
            row = {
                "frame_id": frame_id,
                "model": "GroundingDINO_SwinT_OGC",
                "annotator_version": GROUNDING_DINO_ANNOTATOR_VERSION,
                "generated_at": _utc_now(),
                "status": "failed",
                "human_review_required": True,
                "error": str(exc)[:1000],
                "boxes": [],
            }
        rows.append(row)
        rows.sort(key=lambda value: str(value["frame_id"]))
        _write_jsonl(output, rows)
        print(f"{frame_id}: Grounding DINO {row['status']} with {len(row['boxes'])} boxes", flush=True)
    return output
