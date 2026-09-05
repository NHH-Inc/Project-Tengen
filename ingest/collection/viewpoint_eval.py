"""Score a detector on held-out 2026 footage, where the corpus is thin.

A validation split drawn from the training corpus measures accuracy on images that look like the
training images. That is not what broke: on real 2026 footage the detector recognised one to five
robots in a six-robot match, and nothing at all in the lower of two stacked camera views -- not at
a lower threshold, not under tiled inference, not at 3x magnification. A model can post an
excellent mAP against its own val split and still be blind here, because the failure is
distribution, not accuracy.

So this counts detections per frame on a pack sampled across venues, and needs no ground truth to
be useful: a six-robot match contains six robots, and a mean far below that is the gap, whatever
the val split reports. Labels, when a pack comes back from a human, sharpen it into real recall.

The per-segment breakdown is the part worth reading. A mean of three hides the difference between
a model that finds three robots everywhere and one that finds six at half the venues and none at
the other half -- and only the first is fixed by more labels of the same kind.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean

#: Frames come out of label_pack as <videoid>_<start>_<length>_<milliseconds>, so the segment --
#: one camera at one venue -- is everything before the timestamp.
def segment_of(stem: str) -> str:
    return "_".join(stem.split("_")[:-1])


def reviewed_stems(pack: Path) -> set[str] | None:
    """Which frames a human has actually looked at, according to the manifest.

    This matters more than it sounds. A pack ships with the detector's own proposals already in
    `labels/`, marked status "proposed", so scoring a model against them measures how well it
    agrees with itself -- v2 scores 0.986 against boxes v2 drew. Returning None when nothing has
    been reviewed makes the caller say "no ground truth" instead of printing a number that looks
    authoritative and means nothing.

    Labellers edit label files and send them back; they do not edit the manifest. So a returned
    pack is pointed at with --truth rather than detected, and this covers only the case where the
    manifest itself has been updated.
    """
    path = pack / "manifest.json"
    if not path.is_file():
        return None
    items = json.loads(path.read_text(encoding="utf-8")).get("items", [])
    done = {Path(i["image"]).stem for i in items
            if i.get("status") != "proposed" and not i.get("human_review_required")}
    return done or None


def truth_counts(labels_dir: Path, keep: set[str] | None = None) -> dict[str, int]:
    """How many boxes a human left per frame.

    An empty label file is ambiguous -- it means either "no robots here" or "nobody looked" -- and
    dataset_merge drops those for the same reason. Counting them as zero would invent perfect
    recall on frames nobody opened.
    """
    counts = {}
    for path in sorted(labels_dir.glob("*.txt")):
        if keep is not None and path.stem not in keep:
            continue
        lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        if lines:
            counts[path.stem] = len(lines)
    return counts


def evaluate(detector, pack: Path, expected: int = 6, progress=None,
             truth_dir: Path | None = None) -> dict:
    import cv2

    # Only labels a human has actually produced count. Without --truth, that is whatever the
    # manifest marks reviewed -- usually nothing, and nothing is the honest answer.
    if truth_dir is not None:
        truth = truth_counts(truth_dir)
    else:
        reviewed = reviewed_stems(pack)
        truth = truth_counts(pack / "labels", reviewed) if reviewed else {}
    counts, per_segment, matched, found_of_truth = [], {}, [], []
    images = sorted((pack / "images").glob("*.jpg"))
    for i, path in enumerate(images, 1):
        image = cv2.imread(str(path))
        if image is None:
            continue
        n = len(detector.detect(image))
        counts.append(n)
        per_segment.setdefault(segment_of(path.stem), []).append(n)
        if path.stem in truth:
            matched.append(truth[path.stem])
            found_of_truth.append(n)
        if progress and i % 50 == 0:
            progress(i, len(images), mean(counts))

    if not counts:
        return {"frames": 0}

    result = {
        "frames": len(counts),
        "mean": mean(counts),
        "expected": expected,
        "histogram": Counter(counts),
        "blind_frames": sum(1 for n in counts if n == 0),
        "full_frames": sum(1 for n in counts if n >= expected),
        "segments": {seg: mean(ns) for seg, ns in per_segment.items()},
    }
    # Only meaningful once a pack comes back. Detections are not matched to boxes by IoU here --
    # this is a count, so it cannot tell a right box from a wrong one in the right quantity. It
    # is an upper bound on recall, and a low upper bound is still decisive.
    if matched:
        result["labelled_frames"] = len(matched)
        result["ceiling_recall"] = sum(min(f, t) for f, t in zip(found_of_truth, matched)) / sum(matched)
    return result


def report(result: dict, weakest: int = 8) -> str:
    if not result.get("frames"):
        return "no readable frames in the pack"

    hist = result["histogram"]
    frames, expected = result["frames"], result["expected"]
    lines = [
        f"frames            {frames}",
        f"mean detections   {result['mean']:.2f}   (a {expected}-robot match has {expected})",
        f"frames with 0     {result['blind_frames']}"
        f"  ({100 * result['blind_frames'] / frames:.1f}%)",
        f"frames with >={expected}   {result['full_frames']}"
        f"  ({100 * result['full_frames'] / frames:.1f}%)",
    ]
    if "ceiling_recall" in result:
        lines.append(f"recall ceiling    {result['ceiling_recall']:.3f}   against "
                     f"{result['labelled_frames']} human-labelled frames")
    else:
        lines.append(
            "recall            unknown -- no human-labelled frames in this pack. The labels"
            "\n                  it ships are the detector's own proposals, so scoring"
            "\n                  against them measures self-agreement, not recall.")
    lines.append("\ndetections per frame:")
    for k in sorted(hist):
        lines.append(f"  {k:>2}: {'#' * min(60, hist[k])} {hist[k]}")

    order = sorted(result["segments"].items(), key=lambda kv: kv[1])
    lines.append(f"\nweakest of {len(order)} segments:")
    for seg, avg in order[:weakest]:
        lines.append(f"  {avg:>5.2f}  {seg}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="path to a YOLO-family .onnx export")
    parser.add_argument("--pack", type=Path, default=Path("data/label-packs/v3-viewpoint"))
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--expected", type=int, default=6,
                        help="robots on the field; 6 for a standard FRC match")
    parser.add_argument("--truth", type=Path,
                        help="a labels/ directory returned by a human. Without it the pack's own "
                             "labels are treated as proposals, because that is what they are")
    args = parser.parse_args(argv)

    if not (args.pack / "images").is_dir():
        print(f"no images under {args.pack / 'images'}")
        return 1

    from ingest.collection.detect_runner import OnnxDetector

    detector = OnnxDetector(model_path=args.model, confidence_threshold=args.threshold)
    print(f"model      {args.model}")
    print(f"input size {detector.input_size}   threshold {args.threshold}\n")

    result = evaluate(
        detector, args.pack, expected=args.expected, truth_dir=args.truth,
        progress=lambda i, n, m: print(f"  {i}/{n} frames, running mean {m:.2f}"))
    print("\n" + report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
