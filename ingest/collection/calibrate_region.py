"""Work out each video source's region of interest, and write it where the pipeline will read it.

A broadcast that composites a second camera under the main one shows every robot twice. The
tracker then makes two tracks of one robot, and since team attribution is per track, a human is
asked for the same team number twice while every shot on that robot is counted twice.

The seam cannot be read off a single frame -- measured, see view_region -- so this samples frames,
runs the detector, and looks for two separated bands of robots. That makes it a calibration pass
per source, run once, the same shape as `calibrate.py` for homography.

    python -m ingest.collection.calibrate_region --model data\\robot-v2.onnx \\
        --segments data\\segments --out analysis\\config\\regions.json

It refuses rather than guesses. A source whose frames produce too few detections is reported as
unknown and left at the full frame, because a bad crop silently discards a third of the picture
and nothing downstream can tell.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .view_region import CROP, FILTER, Region, calibrate

#: Frames to sample per source. Enough for a majority to mean something, few enough that sixty
#: sources are minutes rather than an evening.
SAMPLE_FRAMES = 12


def sample_frames(video: Path, count: int = SAMPLE_FRAMES) -> list:
    """Frames spread across a clip, skipping the very start and end.

    Evenly spread rather than consecutive: consecutive frames show one moment of one match, and a
    moment where everyone happens to be at one end looks exactly like a single camera view.
    """
    import cv2

    capture = cv2.VideoCapture(str(video))
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            return []
        # The first and last tenth are titles, replays and crowd shots often enough to skip.
        span = range(count)
        frames = []
        for i in span:
            at = int(total * (0.1 + 0.8 * (i + 0.5) / count))
            capture.set(cv2.CAP_PROP_POS_FRAMES, at)
            ok, frame = capture.read()
            if ok:
                frames.append(frame)
        return frames
    finally:
        capture.release()


def calibrate_source(detector, video: Path, count: int = SAMPLE_FRAMES) -> dict:
    frames = sample_frames(video, count)
    if not frames:
        return {"region": Region(), "stacked": False, "frames": 0, "frames_usable": 0,
                "frames_stacked": 0, "reason": "no readable frames"}
    return calibrate([detector.detect(f) for f in frames])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="a YOLO-family .onnx export")
    parser.add_argument("--segments", type=Path, default=Path("data/segments"))
    parser.add_argument("--out", type=Path, help="where to write the regions; prints if omitted")
    parser.add_argument("--frames", type=int, default=SAMPLE_FRAMES)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--mode", choices=(FILTER, CROP), default=FILTER,
                        help=f"how to apply the region; {FILTER} measured better, see view_region")
    args = parser.parse_args(argv)

    videos = sorted(args.segments.glob("*.mp4"))
    if not videos:
        print(f"no .mp4 under {args.segments}")
        return 1

    from .detect_runner import OnnxDetector
    detector = OnnxDetector(model_path=args.model, confidence_threshold=args.threshold)

    regions, stacked_n, unknown = {}, 0, 0
    print(f"{len(videos)} sources, {args.frames} frames each\n")
    print(f"{'source':<34}{'verdict':<12}{'keep':>7}  evidence")
    for video in videos:
        result = calibrate_source(detector, video, args.frames)
        region = result["region"]
        if result["stacked"]:
            region = Region(region.x, region.y, region.w, region.h, args.mode)
            regions[video.stem] = region.to_dict()
            stacked_n += 1
            verdict = "two views"
        elif not result["frames_usable"]:
            unknown += 1
            verdict = "unknown"
        else:
            verdict = "one view"
        print(f"{video.stem:<34}{verdict:<12}{region.h:>7.2f}  {result['reason']}")

    print(f"\n{stacked_n} of {len(videos)} sources composite a second view"
          f"{f', {unknown} could not be judged' if unknown else ''}.")
    if unknown:
        print("  Sources that could not be judged keep the whole frame, which is the safe "
              "default but\n  means their robots are still counted twice if they do have two "
              "views. Check the model\n  works on them before trusting the verdict.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"regions": regions}, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {len(regions)} regions to {args.out}")
    elif regions:
        print("\n" + json.dumps({"regions": regions}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
