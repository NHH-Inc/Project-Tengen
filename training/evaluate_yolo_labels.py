"""Compare automatic YOLO labels with a small human-reviewed YOLO benchmark set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _labels(path: Path) -> list[tuple[int, float, float, float, float]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        class_id, center_x, center_y, width, height = line.split()[:5]
        rows.append((int(class_id), float(center_x), float(center_y), float(width), float(height)))
    return rows


def _iou(a: tuple[int, float, float, float, float], b: tuple[int, float, float, float, float]) -> float:
    if a[0] != b[0]:
        return 0.0
    _, ax, ay, aw, ah = a
    _, bx, by, bw, bh = b
    left, top = max(ax - aw / 2, bx - bw / 2), max(ay - ah / 2, by - bh / 2)
    right, bottom = min(ax + aw / 2, bx + bw / 2), min(ay + ah / 2, by + bh / 2)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union else 0.0


def evaluate(predictions: Path, ground_truth: Path, threshold: float) -> dict[str, float | int]:
    stems = {path.stem for path in predictions.glob("*.txt")} | {path.stem for path in ground_truth.glob("*.txt")}
    true_positive = false_positive = false_negative = 0
    for stem in stems:
        predicted, expected = _labels(predictions / f"{stem}.txt"), _labels(ground_truth / f"{stem}.txt")
        candidates = sorted(
            ((_iou(prediction, truth), p, t) for p, prediction in enumerate(predicted) for t, truth in enumerate(expected)),
            reverse=True,
        )
        used_predicted: set[int] = set()
        used_expected: set[int] = set()
        for iou, predicted_index, expected_index in candidates:
            if iou < threshold:
                break
            if predicted_index in used_predicted or expected_index in used_expected:
                continue
            used_predicted.add(predicted_index)
            used_expected.add(expected_index)
        true_positive += len(used_predicted)
        false_positive += len(predicted) - len(used_predicted)
        false_negative += len(expected) - len(used_expected)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "frames": len(stems), "iou_threshold": threshold,
        "true_positive": true_positive, "false_positive": false_positive, "false_negative": false_negative,
        "precision": round(precision, 6), "recall": round(recall, 6), "f1": round(f1, 6),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()
    if not 0 < args.iou <= 1:
        raise SystemExit("--iou must be in (0, 1]")
    print(json.dumps(evaluate(Path(args.predictions), Path(args.ground_truth), args.iou), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
