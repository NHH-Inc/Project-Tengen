"""Turn every analysed match into labelling work: the frames the detector got wrong.

`label_pack.py` samples frames evenly, which was right when the detector was blind everywhere.
v3 is not blind any more (0 of 400 held-out frames empty), so an even sample is now mostly frames
it already gets right, and a person's labelling time is spent confirming. The frames that still
teach are the ones where tracking shows the detector failed, and an analysed match already
records where those are:

  * **vanished**: a track ends, or begins, away from the edges of the view and away from the clip
    boundaries. A robot does not teleport, so the detector lost it (or invented it).
  * **hole**: a track goes unobserved for a stretch inside its own lifetime with no camera cut
    listed. The robot was there and was missed.
  * **count_drop**: fewer robots are being tracked than usual for this match. Several were
    missed at once, usually a crowd or an occlusion the model has not learned.
  * **flicker**: a track shorter than a second. Almost always a false positive on a field
    element, a person, or a bumper reflection.

Moments are ranked, spread out in time (consecutive frames teach one thing many times), and
written as an ordinary label pack. The same rule applies as for every pack: every robot boxed, or
the frame skipped.

    python -m ingest.collection.hard_examples data/desktop/runs --out data/label-packs/hard-1 \\
        --per-match 25 --model data/robot-v3.onnx
"""

from __future__ import annotations

import argparse
import json
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from statistics import median

from .label_pack import PROPOSAL_CONFIDENCE, PackStats, build

#: A box centre this close to the edge of the analysed view may simply have left it.
EDGE_MARGIN = 0.08
#: Ignore track starts and ends this close to the clip boundaries: that is the clip, not the model.
CLIP_MARGIN_S = 2.0
#: An unobserved stretch inside a track at least this long is a hole worth a frame.
HOLE_S = 0.8
#: A track shorter than this is a flicker.
FLICKER_S = 1.0
#: Robot count this far below the match's median count is a drop.
COUNT_DROP = 2
#: Frames chosen from one match are at least this far apart.
MIN_GAP_S = 2.0

#: Higher first: a missed robot that later reappears is the clearest lesson.
PRIORITY = {"hole": 0, "vanished": 1, "count_drop": 2, "flicker": 3}


@dataclass(frozen=True)
class Moment:
    t: float
    reason: str
    track_id: int | None
    severity: float


def _near_edge(box: dict) -> bool:
    cx, cy = box["x"] + box["w"] / 2, box["y"] + box["h"] / 2
    return min(cx, cy, 1 - cx, 1 - cy) < EDGE_MARGIN


def _in_gap(a: float, b: float, gaps) -> bool:
    return any(g.get("start", 0) < b and g.get("end", 0) > a for g in gaps)


def find_moments(tracks: list[dict], duration: float, sample_rate: float | None = None) -> list[Moment]:
    """Every moment worth a human look in one match, unranked."""
    moments: list[Moment] = []
    sightings = []
    for track in tracks:
        boxes = sorted(track.get("boxes", []), key=lambda b: b["t"])
        if not boxes:
            continue
        gaps = track.get("gaps", [])
        first, last = boxes[0], boxes[-1]
        sightings.append([b["t"] for b in boxes])
        length = last["t"] - first["t"]
        if length < FLICKER_S:
            moments.append(Moment(first["t"], "flicker", track.get("track_id"), FLICKER_S - length))
            continue
        if first["t"] > CLIP_MARGIN_S and not _near_edge(first):
            moments.append(Moment(first["t"], "vanished", track.get("track_id"), 1.0))
        if last["t"] < duration - CLIP_MARGIN_S and not _near_edge(last):
            moments.append(Moment(last["t"] + 0.2, "vanished", track.get("track_id"), 1.0))
        for a, b in zip(boxes, boxes[1:]):
            hole = b["t"] - a["t"]
            if hole >= HOLE_S and not _in_gap(a["t"], b["t"], gaps):
                # Just after the last sighting: the robot is still where it was, unboxed.
                moments.append(Moment(round(a["t"] + min(0.3, hole / 2), 3), "hole",
                                      track.get("track_id"), hole))

    if sightings and duration > 0:
        # Count robots actually seen near each instant, not robots whose track merely spans it:
        # a track with a hole is exactly the robot the detector failed to see.
        def seen(times, t):
            i = bisect_left(times, t - 0.5)
            return i < len(times) and times[i] <= t + 0.5

        counts = []
        t = CLIP_MARGIN_S
        while t < duration - CLIP_MARGIN_S:
            counts.append((t, sum(1 for times in sightings if seen(times, t))))
            t += 1.0
        typical = median(c for _, c in counts) if counts else 0
        for t, count in counts:
            if typical - count >= COUNT_DROP:
                moments.append(Moment(t, "count_drop", None, float(typical - count)))
    return moments


def select(moments: list[Moment], limit: int, gap: float = MIN_GAP_S) -> list[Moment]:
    """Up to `limit` moments, never two within `gap` seconds, taking the reasons in turn.

    Strict priority filled whole packs with one kind of mistake (on a real match, 25 of 25 were
    holes). Taking the best remaining moment of each reason in turn keeps every failure mode in
    front of a labeller, which is what the next model needs to learn from.
    """
    queues = {reason: sorted((m for m in moments if m.reason == reason), key=lambda m: (-m.severity, m.t))
              for reason in sorted(PRIORITY, key=PRIORITY.get)}
    chosen: list[Moment] = []
    while len(chosen) < limit and any(queues.values()):
        for reason, queue in queues.items():
            while queue:
                moment = queue.pop(0)
                if all(abs(moment.t - other.t) >= gap for other in chosen):
                    chosen.append(moment)
                    break
            if len(chosen) >= limit:
                break
    return sorted(chosen, key=lambda m: m.t)


def _video_for(run: Path, job: dict) -> Path | None:
    for candidate in (run / "segment.mp4", Path(job.get("local_path") or "")):
        if candidate and candidate.is_file():
            return candidate
    return None


def frames_at(video: Path, name: str, times: list[float]):
    """Decode the requested instants by reading forward, never seeking (see label_pack)."""
    import cv2

    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    targets = {int(round(t * fps)): t for t in times}
    index, last = 0, max(targets) if targets else -1
    while index <= last:
        ok, frame = capture.read()
        if not ok:
            break
        if index in targets:
            yield name, targets[index], frame
        index += 1
    capture.release()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path,
                        help="analysed match folders, or folders containing them")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--per-match", type=int, default=25)
    parser.add_argument("--model", type=Path, help="ONNX detector for pre-filled proposals")
    args = parser.parse_args(argv)

    runs = []
    for path in args.runs:
        runs.extend([path] if (path / "tracks.jsonl").exists()
                    else sorted(p.parent for p in path.glob("*/tracks.jsonl")))

    detector = None
    if args.model:
        from .detect_runner import OnnxDetector
        detector = OnnxDetector(model_path=str(args.model), confidence_threshold=PROPOSAL_CONFIDENCE)
    from .frame_quality import measure_image

    stats = PackStats(segments=len(runs))
    reasons: dict[str, dict] = {}
    frames = []
    for run in runs:
        job = json.loads((run / "job.json").read_text(encoding="utf-8")) if (run / "job.json").exists() else {}
        video = _video_for(run, job)
        if video is None:
            print(f"  {run.name}: video not found, skipped")
            continue
        tracks = [json.loads(line) for line in (run / "tracks.jsonl").read_text(encoding="utf-8").splitlines()
                  if line.strip()]
        duration = float(job.get("duration") or max((b["t"] for t in tracks for b in t["boxes"]), default=0))
        chosen = select(find_moments(tracks, duration), args.per_match)
        name = f"{job.get('match_id') or run.name}".replace("/", "_")
        by_reason: dict[str, int] = {}
        for moment in chosen:
            by_reason[moment.reason] = by_reason.get(moment.reason, 0) + 1
            reasons[f"{name}_{int(round(moment.t * 1000)):08d}"] = {
                "reason": moment.reason, "track_id": moment.track_id, "t_seconds": moment.t,
                "run": str(run)}
        print(f"  {name}: {len(chosen)} frames " + ", ".join(f"{k} {v}" for k, v in sorted(by_reason.items())))
        frames.append(frames_at(video, name, [m.t for m in chosen]))

    def chained():
        for source in frames:
            yield from source

    build(chained(), args.out, detector=detector, stats=stats, quality_check=measure_image)
    (args.out / "hard_examples.json").write_text(json.dumps(reasons, indent=1), encoding="utf-8")
    print(f"\nwrote {stats.frames_written} frames to {args.out}; reasons in hard_examples.json")
    if stats.rejected_reasons:
        print("  rejected: " + ", ".join(f"{k} {v}" for k, v in sorted(stats.rejected_reasons.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
