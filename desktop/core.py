"""Local run storage, analysis and review. No GUI or server dependencies."""
from __future__ import annotations

import copy
import csv
import json
import math
import sys
import uuid
from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "data" / "desktop" / "runs"
DEFAULT_MODEL = ROOT / "data/models/yolo-v3-960-20260906/weights/best.pt"
DEFAULT_BALL_CONFIG = ROOT / "analysis/config/ball_scouting.example.json"
FIXTURE = ROOT / "fixtures/2026casf_qm42"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def probe_video(path: Path) -> dict:
    import cv2
    capture = cv2.VideoCapture(str(path))
    try:
        ok, _ = capture.read()
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not ok or not math.isfinite(fps) or fps <= 0 or frames <= 0:
            raise ValueError(f"Cannot decode video: {path}")
        return dict(fps=fps, duration=frames / fps, frames=frames,
                    width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    finally:
        capture.release()


@dataclass
class AnalysisOptions:
    video: str
    model: str = str(DEFAULT_MODEL)
    season: int = 2026
    match_id: str = ""
    device: str = "cpu"
    confidence: float = 0.25
    image_size: int = 960
    ball_config: str = str(DEFAULT_BALL_CONFIG)
    homography: str = ""
    auto_homography: bool = True
    annotated: bool = True


def analyze(options: AnalysisOptions, progress=lambda value, stage: None, runs: Path = RUNS) -> Path:
    from ingest.yolo_orchestrator import YoloAnalysisOrchestrator
    video = Path(options.video).expanduser().resolve()
    model = Path(options.model).expanduser().resolve()
    if not model.is_file():
        raise ValueError(f"Robot model not found: {model}. Choose a trained .pt model.")
    season_path = ROOT / f"contracts/seasons/{options.season}.json"
    if not season_path.is_file():
        raise ValueError(f"No season configuration for {options.season}")
    metadata = probe_video(video)
    job_id = str(uuid.uuid4())
    directory = runs / job_id
    directory.mkdir(parents=True)
    job = dict(job_id=job_id, local_path=str(video), video_id=video.stem,
               match_id=options.match_id.strip() or f"local_{job_id[:8]}", season=options.season,
               start_offset=0, status="analyzing", created_at=datetime.now(timezone.utc).isoformat(),
               **metadata)
    write_json(directory / "job.json", job)
    write_json(directory / "options.json", vars(options))
    engine = YoloAnalysisOrchestrator(
        repo_root=ROOT, python_path=sys.executable, model_path=model, output_base_dir=runs,
        device=options.device, confidence=options.confidence, image_size=options.image_size,
        frame_stride=1, save_annotated=options.annotated,
        ball_config_path=options.ball_config or None,
        homography_path=options.homography or None, auto_homography=options.auto_homography,
        # Arbitrary local clips can begin midway through a match. Do not assign alliances by
        # starting field position unless we know that this is the opening of the match.
        startup_position_seconds=0,
    )
    try:
        engine.run_job(job, str(season_path), on_progress=progress)
        job["status"] = "complete"
        write_json(directory / "job.json", job)
        return directory
    except BaseException as exc:
        job.update(status="failed", error=str(exc))
        write_json(directory / "job.json", job)
        raise


def box_at(track: dict, t: float, hold: float = 0.0) -> dict | None:
    boxes = track.get("boxes", [])
    gaps = track.get("gaps", [])
    if not boxes or t < boxes[0]["t"] or t > boxes[-1]["t"] + hold:
        return None
    if any(g["start"] <= t <= g["end"] for g in gaps):
        return None
    index = bisect_right(boxes, t, key=lambda b: b["t"]) - 1
    a = boxes[index]
    if index == len(boxes) - 1:
        return a
    b = boxes[index + 1]
    if any(g["start"] < b["t"] and g["end"] > a["t"] for g in gaps):
        return a if t <= a["t"] + hold else None
    span = b["t"] - a["t"]
    u = (t - a["t"]) / span if span > 0 else 0
    result = {**a, **{k: a[k] + (b[k] - a[k]) * u for k in ("x", "y", "w", "h")}, "t": t}
    for key in ("field_x", "field_y", "speed_ftps"):
        av, bv = a.get(key), b.get(key)
        result[key] = av + (bv - av) * u if av is not None and bv is not None else None
    return result


class Review:
    def __init__(self, directory: Path, corrections_path: Path | None = None):
        self.directory = directory.resolve()
        self.job = read_json(directory / "job.json")
        self.raw_tracks = read_jsonl(directory / "tracks.jsonl")
        self.raw_events = read_jsonl(directory / "events.jsonl")
        self.shots = read_jsonl(directory / "shots.jsonl")
        self.shots_configured = (directory / "shots.jsonl").exists()
        self.goal_entries = read_jsonl(directory / "goal_entries.jsonl")
        config_path = directory / "ball_scouting.config.json"
        self.ball_config = read_json(config_path) if config_path.exists() else {}
        self.goals = self.ball_config.get("goals", [])
        self.goal_calibration = self.ball_config.get("goal_calibration") or {}
        self.result = read_json(directory / "result.json") if (directory / "result.json").exists() else {}
        self.corrections_path = corrections_path or directory / "desktop-corrections.json"
        self.corrections = read_json(self.corrections_path) if self.corrections_path.exists() else []
        # An optional synchronized original can be played at 60 fps while the analysis
        # source remains explicitly recorded at 30 fps. Track times stay in source seconds.
        self.video = directory / "segment.mp4" if (directory / "segment.mp4").exists() else Path(self.job.get("playback_path") or self.job.get("local_path") or "")
        self.season = read_json(ROOT / f"contracts/seasons/{self.job['season']}.json")

    def rows(self, raw=False):
        tracks, events = copy.deepcopy(self.raw_tracks), copy.deepcopy(self.raw_events)
        if raw:
            return tracks, events
        for c in self.corrections:
            if c["scope"] == "track":
                for track in tracks:
                    if track["track_id"] == c["target"]:
                        track["team"] = c["fields"]["team"]
                for event in events:
                    if event.get("track_id") == c["target"]:
                        event["team"] = c["fields"]["team"]
            elif c["action"] == "create":
                events.append(copy.deepcopy(c["fields"]))
            elif c["action"] == "delete":
                events = [e for e in events if e["event_id"] != c["target"]]
            else:
                for event in events:
                    if event["event_id"] == c["target"]:
                        event.update(c["fields"])
        return tracks, sorted(events, key=lambda e: e["t_seconds"])

    def correct(self, scope, target, fields, action="edit"):
        self.corrections.append(dict(scope=scope, target=target, fields=fields, action=action))
        write_json(self.corrections_path, self.corrections)

    def undo(self):
        if self.corrections:
            self.corrections.pop()
            write_json(self.corrections_path, self.corrections)

    def export(self, path: Path, raw=False):
        tracks, events = self.rows(raw)
        if path.suffix.lower() == ".csv":
            columns = ["event_id", "t_seconds", "phase", "event_type", "team", "track_id", "confidence", "goal", "source"]
            with path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(events)
        else:
            write_json(path, dict(job=self.job, result=self.result, tracks=tracks, events=events,
                                 shots=self.shots, goals=self.goals, goal_entries=self.goal_entries,
                                 goal_calibration=self.goal_calibration,
                                 corrections=[] if raw else self.corrections, raw=raw))
