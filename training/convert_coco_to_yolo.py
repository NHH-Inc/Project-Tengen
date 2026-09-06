"""Convert a split COCO object-detection dataset into an Ultralytics YOLO dataset."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any


SPLITS = ("train", "valid", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="COCO root with train/valid/test subdirectories")
    parser.add_argument("--output", required=True, help="new YOLO dataset directory; never overwritten")
    return parser.parse_args()


def link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def read_coco(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid COCO JSON: {path}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"COCO document must be an object: {path}")
    return value


def main() -> int:
    args = parse_args()
    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    if not source.is_dir():
        raise SystemExit(f"COCO source directory is missing: {source}")
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty dataset directory: {output}")

    first = read_coco(source / "train" / "_annotations.coco.json")
    categories = sorted(first.get("categories", []), key=lambda item: int(item["id"]))
    if not categories:
        raise SystemExit("COCO dataset has no categories")
    names = [str(item["name"]) for item in categories]
    category_ids = {int(item["id"]): index for index, item in enumerate(categories)}

    counts: dict[str, dict[str, int]] = {}
    output.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        coco_path = source / split / "_annotations.coco.json"
        if not coco_path.is_file():
            raise SystemExit(f"Missing {split} COCO annotations: {coco_path}")
        coco = read_coco(coco_path)
        split_categories = sorted(coco.get("categories", []), key=lambda item: int(item["id"]))
        if [str(item["name"]) for item in split_categories] != names:
            raise SystemExit(f"Category mapping differs between splits: {coco_path}")

        images_by_id = {int(item["id"]): item for item in coco.get("images", [])}
        annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for annotation in coco.get("annotations", []):
            image_id = int(annotation["image_id"])
            if image_id not in images_by_id:
                raise SystemExit(f"Annotation references missing image {image_id}: {coco_path}")
            annotations_by_image[image_id].append(annotation)

        images_out = output / "images" / ("val" if split == "valid" else split)
        labels_out = output / "labels" / ("val" if split == "valid" else split)
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)
        image_count = annotation_count = negative_count = 0

        for image in images_by_id.values():
            image_id = int(image["id"])
            width, height = float(image["width"]), float(image["height"])
            if width <= 0 or height <= 0:
                raise SystemExit(f"Image has invalid dimensions: {image}")
            source_image = (source / split / str(image["file_name"])).resolve()
            if not source_image.is_file() or source_image.parent != (source / split).resolve():
                raise SystemExit(f"Image is missing or outside its split directory: {source_image}")
            destination_image = images_out / source_image.name
            link_or_copy(source_image, destination_image)

            lines: list[str] = []
            for annotation in annotations_by_image[image_id]:
                category_id = int(annotation["category_id"])
                if category_id not in category_ids:
                    raise SystemExit(f"Unknown category {category_id}: {coco_path}")
                x, y, box_width, box_height = (float(value) for value in annotation["bbox"])
                left, top = max(0.0, x), max(0.0, y)
                right, bottom = min(width, x + box_width), min(height, y + box_height)
                if right <= left or bottom <= top:
                    continue
                center_x = ((left + right) / 2.0) / width
                center_y = ((top + bottom) / 2.0) / height
                normalized_width = (right - left) / width
                normalized_height = (bottom - top) / height
                lines.append(
                    f"{category_ids[category_id]} {center_x:.6f} {center_y:.6f} "
                    f"{normalized_width:.6f} {normalized_height:.6f}"
                )

            (labels_out / f"{source_image.stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )
            image_count += 1
            annotation_count += len(lines)
            negative_count += int(not lines)

        counts[split] = {
            "images": image_count,
            "annotations": annotation_count,
            "negative_images": negative_count,
        }

    (output / "dataset.yaml").write_text(
        "path: " + output.as_posix() + "\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names: " + json.dumps(names) + "\n",
        encoding="utf-8",
    )
    summary = {
        "format": "ultralytics-yolo",
        "source": str(source),
        "allow_unreviewed": True,
        "classes": names,
        "splits": counts,
        "warning": "Converted from the unreviewed COCO bootstrap dataset; labels remain proposals until reviewed.",
    }
    (output / "dataset.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
