"""Convert reviewed competition-image exports into a leakage-safe YOLO dataset.

The source format is one annotations.json per video folder. The exporter keeps each
video folder in exactly one split, preserves empty frames, and applies offline
horizontal-flip and motion-blur augmentations to training images only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--val-folders", type=int, default=4)
    parser.add_argument("--test-folders", type=int, default=4)
    return parser.parse_args()


def yolo_rows(boxes: list[dict], width: int, height: int, *, flip: bool = False) -> list[str]:
    rows: list[str] = []
    for box in boxes:
        if box.get("label", "robot") != "robot":
            continue
        raw = box.get("bbox_xyxy")
        if not raw or len(raw) != 4:
            continue
        x1, y1, x2, y2 = (float(value) for value in raw)
        x1, x2 = sorted((max(0.0, min(width, x1)), max(0.0, min(width, x2))))
        y1, y2 = sorted((max(0.0, min(height, y1)), max(0.0, min(height, y2))))
        if x2 <= x1 or y2 <= y1:
            continue
        if flip:
            x1, x2 = width - x2, width - x1
        xc = ((x1 + x2) / 2.0) / width
        yc = ((y1 + y2) / 2.0) / height
        bw = (x2 - x1) / width
        bh = (y2 - y1) / height
        rows.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    return rows


def motion_blur(image: np.ndarray, length: int, angle_degrees: float) -> np.ndarray:
    """Apply a mild directional blur without changing image dimensions."""
    length = max(3, int(length) | 1)
    kernel = np.zeros((length, length), dtype=np.float32)
    center = length // 2
    cv2.line(kernel, (0, center), (length - 1, center), 1.0, 1)
    rotation = cv2.getRotationMatrix2D((center, center), angle_degrees, 1.0)
    kernel = cv2.warpAffine(kernel, rotation, (length, length))
    total = float(kernel.sum())
    if total <= 0:
        kernel[center, :] = 1.0
        total = float(kernel.sum())
    kernel /= total
    return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REPLICATE)


def write_image(source: Path, destination: Path, transform: str, rng: random.Random) -> tuple[int, int]:
    with Image.open(source) as pil_image:
        image = np.asarray(pil_image.convert("RGB"))
    height, width = image.shape[:2]
    if transform == "flip":
        image = np.ascontiguousarray(image[:, ::-1, :])
    elif transform == "motion_blur":
        length = rng.choice((5, 7, 9, 11))
        angle = rng.choice((-30.0, -15.0, 0.0, 15.0, 30.0))
        image = motion_blur(image, length, angle)
    elif transform != "original":
        raise ValueError(f"unknown transform: {transform}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(destination), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise OSError(f"could not write {destination}")
    return width, height


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if not source.is_dir():
        raise SystemExit(f"source directory is missing: {source}")
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty dataset directory: {output}")
    if args.val_folders < 1 or args.test_folders < 1:
        raise SystemExit("val-folders and test-folders must both be positive")

    folders = sorted(path for path in source.iterdir() if path.is_dir() and (path / "annotations.json").is_file())
    if len(folders) <= args.val_folders + args.test_folders:
        raise SystemExit("not enough video folders for the requested split")
    shuffled = folders[:]
    random.Random(args.seed).shuffle(shuffled)
    test_folders = shuffled[: args.test_folders]
    val_folders = shuffled[args.test_folders : args.test_folders + args.val_folders]
    train_folders = shuffled[args.test_folders + args.val_folders :]
    split_for_folder = {
        **{folder.name: "train" for folder in train_folders},
        **{folder.name: "val" for folder in val_folders},
        **{folder.name: "test" for folder in test_folders},
    }

    counts = Counter()
    missing_images: list[str] = []
    invalid_boxes: list[str] = []
    source_frames = 0
    source_boxes = 0
    rng = random.Random(args.seed)

    for folder in folders:
        split = split_for_folder[folder.name]
        data = json.loads((folder / "annotations.json").read_text(encoding="utf-8"))
        rows = data.get("images", [])
        for index, row in enumerate(rows):
            source_frames += 1
            relative_name = str(row.get("image", ""))
            source_image = (folder / relative_name).resolve()
            if not source_image.is_file() or source_image.suffix.lower() not in IMAGE_SUFFIXES:
                missing_images.append(str(source_image))
                continue
            boxes = row.get("boxes", []) or []
            source_boxes += len(boxes)
            with Image.open(source_image) as image:
                width, height = image.size
            original_rows = yolo_rows(boxes, width, height)
            if len(original_rows) != len(boxes):
                invalid_boxes.append(str(source_image))

            stem = f"{folder.name.replace(' ', '_').replace('-', '_')}_{Path(relative_name).stem}_{index:04d}"
            transforms = ["original"]
            if split == "train":
                transforms += ["flip", "motion_blur"]
            for transform in transforms:
                suffix = "" if transform == "original" else f"_{transform}"
                name = f"{stem}{suffix}.jpg"
                image_path = output / split / "images" / name
                label_path = output / split / "labels" / f"{Path(name).stem}.txt"
                out_width, out_height = write_image(source_image, image_path, transform, rng)
                if (out_width, out_height) != (width, height):
                    raise RuntimeError(f"image dimensions changed for {source_image}")
                label_rows = original_rows if transform in {"original", "motion_blur"} else yolo_rows(boxes, width, height, flip=True)
                label_path.parent.mkdir(parents=True, exist_ok=True)
                label_path.write_text("\n".join(label_rows) + ("\n" if label_rows else ""), encoding="utf-8")
                counts[(split, transform)] += 1

    dataset_yaml = output / "dataset.yaml"
    dataset_yaml.write_text(
        "path: " + str(output).replace("\\", "/") + "\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n\n"
        "nc: 1\n"
        "names: ['robot']\n",
        encoding="utf-8",
    )
    folder_hash = hashlib.sha256("\n".join(sorted(split_for_folder)).encode()).hexdigest()[:16]
    manifest = {
        "created_at": "2026-09-05",
        "source": str(source),
        "output": str(output),
        "seed": args.seed,
        "class_names": ["robot"],
        "source_frames_in_annotations": source_frames,
        "source_images_written": sum(counts.values()) - counts[("train", "flip")] - counts[("train", "motion_blur")],
        "source_boxes_in_present_images": source_boxes,
        "missing_image_references": len(missing_images),
        "invalid_box_rows": len(invalid_boxes),
        "counts_by_split_and_transform": {f"{split}/{transform}": count for (split, transform), count in sorted(counts.items())},
        "folder_counts": {"train": len(train_folders), "val": len(val_folders), "test": len(test_folders)},
        "folders": {"train": sorted(folder.name for folder in train_folders), "val": sorted(folder.name for folder in val_folders), "test": sorted(folder.name for folder in test_folders)},
        "folder_assignment_hash": folder_hash,
        "missing_images": missing_images,
        "invalid_box_images": invalid_boxes,
        "augmentation": {
            "horizontal_flip": "one offline copy per training image; x coordinates mirrored",
            "motion_blur": "one offline copy per training image; directional kernel length 5/7/9/11 and angle -30/-15/0/15/30 degrees",
        },
    }
    (output / "dataset-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("source_frames_in_annotations", "source_images_written", "source_boxes_in_present_images", "missing_image_references", "counts_by_split_and_transform", "folder_counts")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
