"""Replay every video frame with cached robot boxes, without rerunning YOLO.

Run as ``python -m tools.replay_ball_scouting --help``. Reference shots are a JSON
array of human-labelled {launch_t_seconds, robot_track_id} records in source time.
Counts alone are not an accuracy measurement; precision/recall require that reference.
"""

from __future__ import annotations

import argparse
import bisect
import json
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from training.ball_scouting import (
    BallShotAnalyzer, RobotObservation, load_ball_scouting_config, shot_statistics, write_shot_records,
)
from training.goal_scoring import goal_statistics, write_goal_entries


def match_shots(predictions, reference, tolerance=0.075):
    """One-to-one matching, with incorrect robot attribution counted as an error."""
    costs = np.full((len(reference), len(predictions) + len(reference)), 1e6)
    for row, expected in enumerate(reference):
        costs[row, len(predictions) + row] = 2.0
        for col, actual in enumerate(predictions):
            delta = abs(actual["launch_t_seconds"] - expected["launch_t_seconds"])
            if delta <= tolerance and actual["robot_track_id"] == expected["robot_track_id"]:
                costs[row, col] = delta / max(tolerance, 1e-9)
    pairs = [(int(row), int(col)) for row, col in zip(*linear_sum_assignment(costs))
             if col < len(predictions) and costs[row, col] <= 1]
    true_positive = len(pairs)
    return dict(true_positive=true_positive, false_positive=len(predictions) - true_positive,
                false_negative=len(reference) - true_positive,
                precision=true_positive / len(predictions) if predictions else None,
                recall=true_positive / len(reference) if reference else None,
                matches=pairs)


class CachedRobots:
    def __init__(self, path):
        self.tracks = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            track = json.loads(line)
            boxes = sorted(track["boxes"], key=lambda box: box["t"])
            self.tracks.append((track["track_id"], [box["t"] for box in boxes], boxes))

    def at(self, timestamp, width, height):
        observations = []
        for track_id, times, boxes in self.tracks:
            index = bisect.bisect_left(times, timestamp)
            if index < len(times) and abs(times[index] - timestamp) < 1e-5:
                box = boxes[index]
            elif 0 < index < len(times) and times[index] - times[index - 1] <= 0.10:
                a, b = boxes[index - 1], boxes[index]
                fraction = (timestamp - a["t"]) / (b["t"] - a["t"])
                box = {key: a[key] + fraction * (b[key] - a[key]) for key in ("x", "y", "w", "h")}
            else:
                continue
            observations.append(RobotObservation(track_id, (
                box["x"] * width, box["y"] * height,
                (box["x"] + box["w"]) * width, (box["y"] + box["h"]) * height,
            )))
        return observations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--robot-tracks", required=True)
    parser.add_argument("--config", default="analysis/config/ball_scouting.example.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--crop", nargs=4, type=float, default=(0.02, 0.035, 0.98, 0.66))
    parser.add_argument("--reference-shots")
    parser.add_argument("--tolerance-seconds", type=float, default=0.075)
    parser.add_argument("--annotated", action="store_true")
    parser.add_argument("--homography", help="camera calibration used to fill an empty goals list")
    parser.add_argument("--auto-homography", action="store_true")
    args = parser.parse_args()
    left, top, right, bottom = args.crop
    if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
        parser.error("crop must be normalized left, top, right, bottom")
    if args.tolerance_seconds <= 0:
        parser.error("tolerance must be positive")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    robots = CachedRobots(args.robot_tracks)
    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {args.video}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Video has no valid source frame rate")
    calibration_path = args.homography
    if args.auto_homography and not calibration_path:
        from ingest.collection.calibrate import calibrate
        calibration = calibrate(args.video, region=(top, bottom), samples=24, optimize_hfov=True)
        calibration_path = output / "homography.json"
        calibration_path.write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")
    config_path = Path(args.config)
    if calibration_path:
        from training.auto_goals import configure_auto_goals
        config_path = configure_auto_goals(config_path, calibration_path,
            (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
            args.crop, output / "ball_scouting.config.json")
    if config_path.resolve() != (output / "ball_scouting.config.json").resolve():
        (output / "ball_scouting.config.json").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    analyzer = BallShotAnalyzer(load_ball_scouting_config(config_path))
    writer = None
    index = 0
    started = time.perf_counter()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            frame = frame[round(top * h):round(bottom * h), round(left * w):round(right * w)]
            h, w = frame.shape[:2]
            observations = robots.at(index / fps, w, h)
            analyzer.process_frame(frame, index, index / fps, observations)
            if args.annotated:
                if writer is None:
                    writer = cv2.VideoWriter(str(output / "annotated.mp4"),
                                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    if not writer.isOpened():
                        raise RuntimeError("Cannot open annotated video writer")
                plotted = frame.copy()
                for robot in observations:
                    x0, y0, x1, y1 = map(round, robot.bbox)
                    cv2.rectangle(plotted, (x0, y0), (x1, y1), (220, 140, 30), 1)
                    cv2.putText(plotted, f"R{robot.track_id}", (x0, y0 - 5), 0, .5, (255, 255, 255), 1)
                analyzer.draw_debug_overlay(plotted)
                cv2.putText(plotted, f"frame {index}   {index / fps:.3f}s", (w - 300, 25),
                            0, .5, (255, 255, 255), 1)
                writer.write(plotted)
            index += 1
            if index % 300 == 0:
                print(json.dumps(dict(frames=index, shots=len(analyzer.shots))), flush=True)
    finally:
        capture.release()
        if writer:
            writer.release()
    elapsed = time.perf_counter() - started
    write_shot_records(output / "shots.jsonl", analyzer.shots)
    write_goal_entries(output / "goal_entries.jsonl", analyzer.goal_entries)
    from training.auto_goals import save_camera_gaps
    save_camera_gaps(output / "ball_scouting.config.json", analyzer.goal_camera_gaps)
    report = dict(video=args.video, robot_tracks=args.robot_tracks, source_fps=fps,
                  frames_analyzed=index, frame_stride=1, wall_seconds=elapsed,
                  processing_fps=index / max(elapsed, 1e-9), counts=shot_statistics(analyzer.shots),
                  methods=dict(Counter(analyzer.shot_methods.values())),
                  goal_counts=goal_statistics(analyzer.goal_entries),
                  goal_camera_gaps=analyzer.goal_camera_gaps,
                  accuracy="unmeasured: no human-labelled reference supplied")
    (output / "shot_methods.json").write_text(json.dumps(analyzer.shot_methods, indent=2) + "\n",
                                             encoding="utf-8")
    if args.reference_shots:
        reference = json.loads(Path(args.reference_shots).read_text(encoding="utf-8"))
        report["accuracy"] = match_shots([s.to_dict() for s in analyzer.shots], reference,
                                         args.tolerance_seconds)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
