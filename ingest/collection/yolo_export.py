"""Materialise collection manifests as a leakage-safe Ultralytics YOLO dataset."""

from __future__ import annotations

import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from .config import CollectionConfig


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _copy_or_link(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def export_yolo_dataset(
    *, collection: str | Path | list[str | Path], config: CollectionConfig,
    output: str | Path, allow_unreviewed: bool, labels_file: str = "filtered-proposals.jsonl",
) -> dict[str, Any]:
    collection_paths = [Path(item) for item in collection] if isinstance(collection, list) else [Path(collection)]
    if not collection_paths:
        raise ValueError("At least one collection is required")
    output_path = Path(output).resolve()
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty dataset directory: {output_path}")

    classes = [str(item["name"]) for item in config.raw["classes"]]
    class_ids = {name: index for index, name in enumerate(classes)}
    frames: dict[str, tuple[dict[str, Any], Path]] = {}
    labels: dict[str, tuple[dict[str, Any], Path]] = {}
    group_splits: dict[str, str] = {}
    for collection_path in collection_paths:
        for frame in _read_jsonl(collection_path / "frames.jsonl"):
            frame_id = str(frame["frame_id"])
            if frame_id in frames:
                raise ValueError(f"Duplicate frame_id across collections: {frame_id}")
            split = str(frame.get("split", ""))
            if split not in {"train", "val", "test"}:
                raise ValueError(f"Frame has an invalid split: {frame_id} -> {split}")
            group = str(frame.get("split_group") or frame.get("event_id") or frame.get("match_id") or frame_id)
            previous = group_splits.setdefault(group, split)
            if previous != split:
                raise ValueError(f"Split leakage: group {group} appears in both {previous} and {split}")
            frames[frame_id] = (frame, collection_path)
        label_path = collection_path / labels_file
        if not label_path.is_file():
            raise FileNotFoundError(f"Collection has no {labels_file}: {collection_path}")
        for row in _read_jsonl(label_path):
            frame_id = str(row["frame_id"])
            if frame_id in labels:
                raise ValueError(f"Duplicate labels across collections: {frame_id}")
            labels[frame_id] = (row, collection_path)
    if not frames:
        raise ValueError("Collections contain no frames")

    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"images": 0, "annotations": 0, "negative_images": 0})
    output_path.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        (output_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_path / "labels" / split).mkdir(parents=True, exist_ok=True)

    for frame_id, (frame, collection_path) in sorted(frames.items()):
        label_entry = labels.get(frame_id)
        if label_entry is None:
            raise ValueError(f"No label record for frame: {frame_id}")
        label_row, label_collection = label_entry
        if label_collection != collection_path:
            raise ValueError(f"Label belongs to the wrong collection: {frame_id}")
        status = str(label_row.get("review_status", label_row.get("status", "unreviewed")))
        if status not in {"reviewed", "accepted", "approved"} and not allow_unreviewed:
            raise ValueError(
                "Refusing unreviewed proposals. Human-review them first, or pass --allow-unreviewed "
                "only for an explicitly temporary bootstrap dataset."
            )
        split = str(frame["split"])
        source = collection_path / str(frame["image_path"])
        if not source.is_file():
            raise FileNotFoundError(f"Frame image is missing: {source}")
        suffix = source.suffix.lower() or ".jpg"
        _copy_or_link(source, output_path / "images" / split / f"{frame_id}{suffix}")
        lines = []
        for box in label_row.get("boxes", []):
            class_name = str(box.get("class_name", "robot"))
            if class_name not in class_ids:
                raise ValueError(f"Unknown class in {frame_id}: {class_name}")
            x, y, width, height = (float(box[name]) for name in ("x", "y", "w", "h"))
            left, top, right, bottom = max(0.0, x), max(0.0, y), min(1.0, x + width), min(1.0, y + height)
            if right <= left or bottom <= top:
                continue
            center_x, center_y = (left + right) / 2, (top + bottom) / 2
            lines.append(
                f"{class_ids[class_name]} {center_x:.6f} {center_y:.6f} "
                f"{right - left:.6f} {bottom - top:.6f}"
            )
        (output_path / "labels" / split / f"{frame_id}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
        counts[split]["images"] += 1
        counts[split]["annotations"] += len(lines)
        counts[split]["negative_images"] += int(not lines)

    dataset_yaml = {
        "path": output_path.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {index: name for index, name in enumerate(classes)},
    }
    (output_path / "dataset.yaml").write_text(
        yaml.safe_dump(dataset_yaml, sort_keys=False), encoding="utf-8"
    )
    summary = {
        "format": "ultralytics-yolo",
        "collections": [str(path) for path in collection_paths],
        "labels_file": labels_file,
        "allow_unreviewed": allow_unreviewed,
        "split_group_by": config.split_group_by,
        "split_groups": len(group_splits),
        "classes": classes,
        "splits": dict(counts),
        "warning": (
            "Automatically generated labels remain proposals until human-reviewed. "
            "The collection JSONL files are the provenance record."
        ),
    }
    (output_path / "dataset.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
