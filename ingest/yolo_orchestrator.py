"""Run the trained YOLO detector through the ingest service.

The native analysis executable is an RF-DETR/ONNX backend.  The project also has a trained
Ultralytics model on the Windows vision machine, so this adapter lets the localhost API use that
model without importing CUDA/Ultralytics into the lighter ingest virtual environment.
"""

from __future__ import annotations

import json
import os
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable


ProgressCallback = Callable[[float | None, str], None]
MODEL_CROP = (0.02, 0.035, 0.98, 0.66)
AUTO_HOMOGRAPHY_REGION = (0.0, 0.68)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def phase_at(t_seconds: float, season: dict) -> str:
    """Derive a season phase from match-relative time, matching the native backend."""
    auto_seconds = float(season.get("auto_seconds", 0))
    teleop_seconds = float(season.get("teleop_seconds", 0))
    if t_seconds <= auto_seconds:
        return "auto"
    if t_seconds <= auto_seconds + teleop_seconds:
        return "teleop"
    return "endgame"


def match_events(job: dict, duration: float, season: dict) -> list[dict]:
    """Emit only safe match-boundary events; YOLO is a detector, not a scorer/OCR model."""
    match_id = job.get("match_id")
    if not match_id:
        return []
    common = {
        "schema_version": 3,
        "job_id": job["job_id"],
        "match_id": match_id,
        "team": None,
        "track_id": None,
        "confidence": 1.0,
        "field_x": None,
        "field_y": None,
        "source": "model",
    }
    return [
        {
            **common,
            "event_id": str(uuid.uuid4()),
            "t_seconds": 0.0,
            "phase": phase_at(0.0, season),
            "event_type": "match_start",
        },
        {
            **common,
            "event_id": str(uuid.uuid4()),
            "t_seconds": max(0.0, duration),
            "phase": phase_at(max(0.0, duration), season),
            "event_type": "match_end",
        },
    ]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


class YoloAnalysisOrchestrator:
    """Invoke ``training.track_yolo`` in the dedicated CUDA/Ultralytics environment."""

    def __init__(
        self,
        *,
        repo_root: str | Path,
        python_path: str | Path | None,
        model_path: str | Path | None,
        output_base_dir: str | Path,
        tracker: str = "bytetrack",
        confidence: float = 0.25,
        image_size: int = 960,
        device: str = "0",
        save_annotated: bool = False,
        snapshot_interval: float = 5.0,
        homography_path: str | Path | None = None,
        reid_memory_seconds: float = 5.0,
        reid_appearance_threshold: float = 0.60,
        reid_max_distance: float = 0.60,
        reid_score_margin: float = 0.08,
        reid_max_speed: float = 0.75,
        reid_edge_threshold: float = 0.10,
        reid_template_gallery_size: int = 5,
        reid_template_confirmation_frames: int = 3,
        reid_template_min_confidence: float = 0.50,
        auto_homography: bool = True,
        homography_hfov_deg: float = 70.0,
    ):
        self.repo_root = Path(repo_root).resolve()
        self.python_path = self._resolve_optional(python_path)
        self.model_path = self._resolve_optional(model_path)
        self.output_base_dir = Path(output_base_dir)
        self.tracker = tracker
        self.confidence = confidence
        self.image_size = image_size
        self.device = device
        self.save_annotated = save_annotated
        self.snapshot_interval = snapshot_interval
        self.reid_memory_seconds = reid_memory_seconds
        self.reid_appearance_threshold = reid_appearance_threshold
        self.reid_max_distance = reid_max_distance
        self.reid_score_margin = reid_score_margin
        self.reid_max_speed = reid_max_speed
        self.reid_edge_threshold = reid_edge_threshold
        self.reid_template_gallery_size = reid_template_gallery_size
        self.reid_template_confirmation_frames = reid_template_confirmation_frames
        self.reid_template_min_confidence = reid_template_min_confidence
        self.auto_homography = auto_homography
        self.homography_hfov_deg = homography_hfov_deg
        self.homography_path = self._resolve_optional(
            homography_path if homography_path is not None else os.environ.get("FRC_HOMOGRAPHY_CONFIG")
        )

    def _resolve_optional(self, value: str | Path | None) -> Path | None:
        if value is None or str(value).strip() == "":
            return None
        path = Path(value)
        return path if path.is_absolute() else (self.repo_root / path).resolve()

    @property
    def available(self) -> bool:
        return bool(
            self.python_path
            and self.python_path.is_file()
            and self.model_path
            and self.model_path.is_file()
        )

    def health(self) -> dict[str, object]:
        return {
            "backend": "yolo",
            "available": self.available,
            "python": str(self.python_path) if self.python_path else None,
            "model": str(self.model_path) if self.model_path else None,
            "tracker": self.tracker,
            "confidence": self.confidence,
            "image_size": self.image_size,
            "device": self.device,
            "snapshot_interval": self.snapshot_interval,
            "reid_memory_seconds": self.reid_memory_seconds,
            "reid_appearance_threshold": self.reid_appearance_threshold,
            "reid_max_distance": self.reid_max_distance,
            "reid_score_margin": self.reid_score_margin,
            "reid_max_speed": self.reid_max_speed,
            "reid_edge_threshold": self.reid_edge_threshold,
            "reid_template_gallery_size": self.reid_template_gallery_size,
            "reid_template_confirmation_frames": self.reid_template_confirmation_frames,
            "reid_template_min_confidence": self.reid_template_min_confidence,
            "homography": str(self.homography_path) if self.homography_path else None,
            "homography_available": bool(self.homography_path and self.homography_path.is_file()),
            "auto_homography": self.auto_homography,
            "homography_hfov_deg": self.homography_hfov_deg,
            "model_crop": {
                "left": MODEL_CROP[0],
                "top": MODEL_CROP[1],
                "right": MODEL_CROP[2],
                "bottom": MODEL_CROP[3],
            },
        }

    @property
    def model_version(self) -> str:
        if self.model_path and self.model_path.parent.name == "weights":
            label = self.model_path.parent.parent.name
        else:
            label = self.model_path.stem if self.model_path else "unconfigured"
        return f"{label}+{self.tracker}"

    def _probe_video(self, path: Path, fallback_fps: float, fallback_duration: float) -> tuple[int, float, float]:
        """Read the completed MP4 metadata in the ingest process for accurate result counts."""
        try:
            import cv2

            capture = cv2.VideoCapture(str(path))
            frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            capture.release()
        except ImportError:
            frames, fps = 0, 0.0
        fps = fps if fps > 0 else fallback_fps
        if frames <= 0 and fps > 0 and fallback_duration > 0:
            frames = max(1, int(round(fallback_duration * fps)))
        duration = (frames - 1) / fps if frames > 1 and fps > 0 else fallback_duration
        return frames, fps, max(0.0, duration)

    def _job_homography(self, video_path: Path | None, job_dir: Path) -> Path | None:
        """Use an explicit calibration, or derive one from steady AprilTags in a local clip."""

        if self.homography_path and self.homography_path.is_file():
            return self.homography_path
        if not self.auto_homography or video_path is None:
            # A stream has no seekable local frames for the legacy calibration helper. An
            # explicit FRC_HOMOGRAPHY_CONFIG still works, while the detector continues without
            # fabricated field coordinates when no trusted mapping is available.
            return None

        from .collection.apriltag_layout import load_layout
        from .collection.calibrate import DEFAULT_LAYOUT, calibrate

        layout_path = DEFAULT_LAYOUT if DEFAULT_LAYOUT.is_absolute() else self.repo_root / DEFAULT_LAYOUT

        try:
            result = calibrate(
                video_path,
                layout_path,
                region=AUTO_HOMOGRAPHY_REGION,
                method="pose",
                hfov_deg=self.homography_hfov_deg,
                optimize_hfov=True,
            )
        except Exception as exc:
            (job_dir / "homography.diagnostics.json").write_text(
                json.dumps({"error": str(exc)}, indent=2) + "\n", encoding="utf-8"
            )
            return None

        diagnostics_path = job_dir / "homography.diagnostics.json"
        diagnostics_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        solution = result.get("solution")
        if not isinstance(solution, dict) or not solution.get("trustworthy") or not result.get("matrix"):
            return None

        layout = load_layout(layout_path)
        calibration = {
            "_comment": (
                f"Auto-calibrated from {video_path.name} using AprilTags {result.get('tags_used', [])}. "
                "Valid for this fixed camera pose only."
            ),
            "mapping_source": "carpet_pose",
            "field_length_ft": layout.length_ft,
            "field_width_ft": layout.width_ft,
            "plane_height_ft": 0.0,
            "point_count": int(result.get("point_count", 0)),
            "has_redundancy": bool(solution.get("has_redundancy")),
            "trustworthy": True,
            "points": result.get("points", []),
            "matrix": result["matrix"],
            "pose": result.get("pose"),
        }
        path = job_dir / "homography.json"
        path.write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")
        return path

    def run_job(self, job_data: dict, season_path: str, on_progress=None) -> dict:
        if not self.available:
            raise RuntimeError(
                "YOLO backend is not configured. Set YOLO_PYTHON to the dedicated vision "
                "Python and YOLO_MODEL_PATH to a trained .pt file."
            )
        stream_url = str(job_data.get("stream_url") or "").strip()
        video_path = None if stream_url else Path(str(job_data.get("local_path", ""))).resolve()
        if not stream_url and (video_path is None or not video_path.is_file()):
            raise RuntimeError(f"Stream URL is missing and local video is unavailable: {video_path}")
        if self.tracker not in {"bytetrack", "botsort"}:
            raise RuntimeError(f"Unsupported YOLO tracker: {self.tracker}")
        if not 0.0 <= self.confidence <= 1.0:
            raise RuntimeError("FRC_YOLO_CONFIDENCE must be between 0 and 1")
        if self.image_size <= 0 or self.image_size % 32:
            raise RuntimeError("FRC_YOLO_IMAGE_SIZE must be a positive multiple of 32")
        if self.reid_memory_seconds <= 0:
            raise RuntimeError("FRC_YOLO_REID_MEMORY_SECONDS must be greater than zero")
        if not 0.0 <= self.reid_appearance_threshold <= 1.0:
            raise RuntimeError("FRC_YOLO_REID_APPEARANCE_THRESHOLD must be between zero and one")
        if self.reid_max_distance <= 0:
            raise RuntimeError("FRC_YOLO_REID_MAX_DISTANCE must be greater than zero")
        if self.reid_score_margin < 0:
            raise RuntimeError("FRC_YOLO_REID_SCORE_MARGIN cannot be negative")
        if self.reid_max_speed <= 0:
            raise RuntimeError("FRC_YOLO_REID_MAX_SPEED must be greater than zero")
        if not 0.0 < self.reid_edge_threshold < 0.5:
            raise RuntimeError("FRC_YOLO_REID_EDGE_THRESHOLD must be between zero and 0.5")
        if self.reid_template_gallery_size <= 0:
            raise RuntimeError("FRC_YOLO_REID_TEMPLATE_GALLERY_SIZE must be greater than zero")
        if self.reid_template_confirmation_frames <= 0:
            raise RuntimeError("FRC_YOLO_REID_TEMPLATE_CONFIRMATION_FRAMES must be greater than zero")
        if not 0.0 <= self.reid_template_min_confidence <= 1.0:
            raise RuntimeError("FRC_YOLO_REID_TEMPLATE_MIN_CONFIDENCE must be between zero and one")

        job_id = str(job_data["job_id"])
        job_dir = self.output_base_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        tracks_path = job_dir / "tracks.jsonl"
        raw_tracks_path = job_dir / "tracks.raw.jsonl"
        partial_tracks_path = job_dir / "tracks.partial.jsonl"
        events_path = job_dir / "events.jsonl"
        result_path = job_dir / "result.json"
        if tracks_path.exists() or events_path.exists() or result_path.exists():
            raise RuntimeError(f"Refusing to overwrite existing YOLO output: {job_dir}")
        job_homography = self._job_homography(video_path, job_dir)

        annotated_path = job_dir / "annotated.mp4" if self.save_annotated else None
        video_label = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in str(job_data.get("video_id") or "video")
        )
        snapshot_dir = self.output_base_dir.parent / "robot_image_exports" / f"{video_label}_{job_id}"
        command = [
            str(self.python_path),
            "-m",
            "training.track_yolo",
            "--model",
            str(self.model_path),
            *( ["--stream-url", stream_url] if stream_url else ["--video", str(video_path)] ),
            *( ["--auto-homography"] if stream_url and self.auto_homography and not job_homography else [] ),
            "--output",
            str(tracks_path),
            "--tracker",
            self.tracker,
            "--confidence",
            str(self.confidence),
            "--image-size",
            str(self.image_size),
            "--device",
            self.device,
            "--reid-memory-seconds",
            str(self.reid_memory_seconds),
            "--reid-appearance-threshold",
            str(self.reid_appearance_threshold),
            "--reid-max-distance",
            str(self.reid_max_distance),
            "--reid-score-margin",
            str(self.reid_score_margin),
            "--reid-max-speed",
            str(self.reid_max_speed),
            "--reid-edge-threshold",
            str(self.reid_edge_threshold),
            "--reid-template-gallery-size",
            str(self.reid_template_gallery_size),
            "--reid-template-confirmation-frames",
            str(self.reid_template_confirmation_frames),
            "--reid-template-min-confidence",
            str(self.reid_template_min_confidence),
            "--raw-output",
            str(raw_tracks_path),
            *(["--homography", str(job_homography)] if job_homography else []),
            "--snapshot-dir",
            str(snapshot_dir),
            "--snapshot-interval",
            str(self.snapshot_interval),
            "--partial-output",
            str(partial_tracks_path),
            "--crop",
            *(str(value) for value in MODEL_CROP),
        ]
        if annotated_path:
            command.extend(["--annotated-output", str(annotated_path)])

        started_at = utc_now()
        if on_progress:
            on_progress(0.05, "decoding")
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            item for item in (str(self.repo_root), env.get("PYTHONPATH", "")) if item
        )
        process = subprocess.Popen(
            command,
            cwd=str(self.repo_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            if on_progress and isinstance(message, dict) and message.get("stage"):
                on_progress(message.get("progress"), message["stage"])
        process.stdout.close()
        returncode = process.wait()
        stderr = process.stderr.read() if process.stderr else ""
        if process.stderr:
            process.stderr.close()
        if returncode != 0:
            reason = stderr.strip() or f"YOLO tracker exited {returncode}"
            error = RuntimeError(reason)
            error.error_code = "analysis_failed"
            raise error
        if not tracks_path.is_file():
            raise RuntimeError(f"YOLO tracker completed without writing: {tracks_path}")
        partial_tracks_path.unlink(missing_ok=True)

        try:
            track_rows = [
                json.loads(line)
                for line in tracks_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Could not read YOLO tracks: {exc}") from exc

        # Stream calibration is performed inside the dedicated vision process so it can share
        # the yt-dlp/PyAV environment. Surface that generated calibration in the run result just
        # like a local-file calibration instead of reporting a false homography failure.
        if job_homography is None:
            generated_homography = job_dir / "homography.json"
            if generated_homography.is_file():
                job_homography = generated_homography

        try:
            season = json.loads(Path(season_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Could not read season config: {exc}") from exc

        fallback_fps = float(job_data.get("fps") or 30.0)
        fallback_duration = float(job_data.get("duration") or 0.0)
        if video_path is not None:
            frames_total, fps, duration = self._probe_video(
                video_path, fallback_fps, fallback_duration
            )
        else:
            fps = fallback_fps
            duration = fallback_duration
            frames_total = max(0, int(round(duration * fps))) if duration > 0 else 0
            if duration <= 0 and track_rows:
                last_timestamp = max(
                    float(box.get("t", 0.0))
                    for track in track_rows
                    for box in track.get("boxes", [])
                )
                duration = max(0.0, last_timestamp + 1.0 / fps)
                frames_total = max(1, int(round(duration * fps)))
        events = match_events(job_data, duration, season)
        _write_jsonl(events_path, events)
        mapping_source = None
        if job_homography and job_homography.is_file():
            try:
                document = json.loads(job_homography.read_text(encoding="utf-8"))
                mapping_source = document.get("mapping_source", "tag_plane")
            except (OSError, json.JSONDecodeError):
                mapping_source = None
        result = {
            "schema_version": 3,
            "job_id": job_id,
            "model_version": self.model_version,
            "box_sample_rate": fps,
            "homography_ok": bool(mapping_source),
            "homography_source": mapping_source,
            "homography_path": str(job_homography) if job_homography else None,
            "homography_diagnostics": str(job_dir / "homography.diagnostics.json")
            if (job_dir / "homography.diagnostics.json").is_file() else None,
            "frames_total": frames_total,
            "frames_analyzed": frames_total,
            "duration": duration,
            "frames_skipped_shot_change": 0,
            "tracks_emitted": len(track_rows),
            "events_emitted": len(events),
            "reconstructed_score": None,
            "started_at": started_at,
            "finished_at": utc_now(),
            "snapshots_dir": str(snapshot_dir),
            "snapshots_annotations": str(snapshot_dir / "annotations.json"),
            "snapshots_interval_seconds": self.snapshot_interval,
            "snapshots_count": len(list(snapshot_dir.glob("snapshot_*.jpg"))),
            "raw_tracks_path": str(raw_tracks_path) if raw_tracks_path.is_file() else None,
            "model_crop": {
                "left": MODEL_CROP[0],
                "top": MODEL_CROP[1],
                "right": MODEL_CROP[2],
                "bottom": MODEL_CROP[3],
            },
        }
        result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        if on_progress:
            on_progress(0.95, "events")
            on_progress(1.0, "events")
        return {
            "events_path": str(events_path),
            "tracks_path": str(tracks_path),
            "raw_tracks_path": str(raw_tracks_path) if raw_tracks_path.is_file() else None,
            "result_path": str(result_path),
            "result": result,
        }
