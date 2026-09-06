"""Run a trained YOLO detector with ByteTrack or BoT-SORT and emit persistent robot tracks."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path


# Normalized coordinates of the yellow-marked upper broadcast panel in the supplied reference.
# The small margins keep the yellow markup itself out of the model input.
DEFAULT_CROP = (0.02, 0.035, 0.98, 0.66)  # left, top, right, bottom
# A sudden identity swap or bad calibration can manufacture an impossible velocity. Keep that
# out of scouting metrics instead of presenting it as a very fast robot.
MAX_PLAUSIBLE_FTPS = 20.0


def contract_track_records(
    tracks: dict[int, list[dict[str, object]]],
    fps: float,
    alliances: dict[int, str | None] | None = None,
    position_source: str | None = None,
) -> list[dict[str, object]]:
    if fps <= 0:
        raise ValueError("fps must be positive")
    frame_interval = 1.0 / fps
    records = []
    for track_id, boxes in sorted(tracks.items()):
        gaps = []
        for previous, current in zip(boxes, boxes[1:]):
            if current["t"] - previous["t"] > frame_interval * 1.5:
                gaps.append({
                    "start": round(previous["t"] + frame_interval, 6),
                    "end": round(current["t"] - frame_interval, 6),
                    "reason": "detection_lost",
                })
        record: dict[str, object] = {
            "schema_version": 3,
            "track_id": track_id,
            "team": None,
            "alliance": (alliances or {}).get(track_id),
            "team_confidence": None,
            "boxes": boxes,
            "gaps": gaps,
        }
        if position_source:
            record["position_source"] = position_source
        records.append(record)
    return records


def write_track_records(
    path: Path,
    tracks: dict[int, list[dict[str, object]]],
    fps: float,
    alliances: dict[int, str | None] | None = None,
    position_source: str | None = None,
) -> None:
    """Atomically publish the tracks accumulated so far for the live UI overlay."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in contract_track_records(tracks, fps, alliances, position_source):
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    # The browser polls the partial file while the worker is writing it. Windows can briefly
    # keep that read handle open, so retry the atomic swap instead of aborting the whole run.
    for attempt in range(30):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 29:
                raise
            time.sleep(0.1)


def bumper_color(image, raw_box: list[float]) -> str | None:
    """Infer red/blue from the lower part of a robot box; return None when ambiguous."""
    import cv2

    height, width = image.shape[:2]
    left, top, right, bottom = (int(round(value)) for value in raw_box)
    left = max(0, min(width - 1, left))
    right = max(left + 1, min(width, right))
    top = max(0, min(height - 1, top + int((bottom - top) * 0.55)))
    bottom = max(top + 1, min(height, bottom))
    region = image[top:bottom, left:right]
    if region.size == 0:
        return None

    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    vivid = (saturation > 75) & (value > 55)
    hue = hsv[:, :, 0]
    red = vivid & ((hue <= 10) | (hue >= 170))
    blue = vivid & (hue >= 95) & (hue <= 135)
    red_count = int(red.sum())
    blue_count = int(blue.sum())
    minimum = max(8, int(region.shape[0] * region.shape[1] * 0.015))
    if red_count >= minimum and red_count >= blue_count * 1.25:
        return "red"
    if blue_count >= minimum and blue_count >= red_count * 1.25:
        return "blue"
    return None


def resolved_alliances(votes: dict[int, dict[str, int]]) -> dict[int, str | None]:
    """Keep a bumper color only when repeated evidence gives it a clear majority."""
    resolved: dict[int, str | None] = {}
    for track_id, counts in votes.items():
        total = counts.get("red", 0) + counts.get("blue", 0)
        if total < 3:
            resolved[track_id] = None
            continue
        colour, count = max(counts.items(), key=lambda item: item[1])
        resolved[track_id] = colour if count / total >= 0.60 else None
    return resolved


def crop_bounds(width: int, height: int, crop: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    left, top, right, bottom = crop
    if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
        raise ValueError("crop must be normalized as left top right bottom within 0..1")
    x0 = max(0, min(width - 1, int(round(width * left))))
    y0 = max(0, min(height - 1, int(round(height * top))))
    x1 = max(x0 + 1, min(width, int(round(width * right))))
    y1 = max(y0 + 1, min(height, int(round(height * bottom))))
    # Keep the encoded model input compatible with common YUV 4:2:0 MP4 codecs.
    if (x1 - x0) % 2:
        x1 = x1 + 1 if x1 < width else x1 - 1
    if (y1 - y0) % 2:
        y1 = y1 + 1 if y1 < height else y1 - 1
    return x0, y0, x1, y1


def crop_box_to_source(
    box: tuple[float, float, float, float],
    crop: tuple[float, float, float, float],
    cropped_width: int,
    cropped_height: int,
) -> tuple[float, float, float, float]:
    """Convert a normalized box in the model crop back to full-source normalization."""

    left, top, right, bottom = crop
    crop_width = right - left
    crop_height = bottom - top
    x, y, width, height = box
    return (
        left + x * crop_width,
        top + y * crop_height,
        width * crop_width,
        height * crop_height,
    )


def add_field_motion(
    sample: dict[str, object],
    mapper,
    crop: tuple[float, float, float, float],
    cropped_width: int,
    cropped_height: int,
    history: dict[int, list[tuple[float, float, float]]],
    max_gap_seconds: float,
) -> None:
    """Attach carpet position and finite-difference motion to one tracked sample."""

    track_id = sample.get("track_id")
    if mapper is None or not isinstance(track_id, int):
        return
    bbox = sample.get("bbox_normalized")
    if not isinstance(bbox, dict):
        return
    try:
        source_box = crop_box_to_source(
            (float(bbox["x"]), float(bbox["y"]), float(bbox["width"]), float(bbox["height"])),
            crop,
            cropped_width,
            cropped_height,
        )
        image_width = cropped_width / (crop[2] - crop[0])
        image_height = cropped_height / (crop[3] - crop[1])
        field_x, field_y = mapper.box_to_field(*source_box, image_width, image_height)
    except (KeyError, TypeError, ValueError):
        return
    if not (math.isfinite(field_x) and math.isfinite(field_y)):
        return
    pixel_x = (source_box[0] + source_box[2] / 2.0) * image_width
    pixel_y = (source_box[1] + source_box[3]) * image_height
    if not mapper.on_field(pixel_x, pixel_y):
        return

    timestamp = float(sample["t"])
    sample["field_x"] = round(field_x, 4)
    sample["field_y"] = round(field_y, 4)
    observations = history.setdefault(track_id, [])
    previous = observations[-1] if observations else None
    if previous is not None:
        previous_time, previous_x, previous_y = previous
        dt = timestamp - previous_time
        if 0.0 < dt <= max_gap_seconds:
            velocity_x = (field_x - previous_x) / dt
            velocity_y = (field_y - previous_y) / dt
            speed = math.hypot(velocity_x, velocity_y)
            if speed <= MAX_PLAUSIBLE_FTPS:
                sample["velocity_x_ftps"] = round(velocity_x, 4)
                sample["velocity_y_ftps"] = round(velocity_y, 4)
                sample["speed_ftps"] = round(speed, 4)
                sample["motion_heading_rad"] = round(math.atan2(velocity_y, velocity_x), 6)
            else:
                # Preserve the valid current position, but do not let a bad jump contaminate the
                # next finite difference either.
                observations.clear()
    observations.append((timestamp, field_x, field_y))
    if len(observations) > 4:
        del observations[:-4]


def build_cropped_video(source: Path, destination: Path, crop: tuple[float, float, float, float]) -> None:
    """Materialize the marked panel so Ultralytics never receives the lower broadcast panel."""
    import cv2

    capture = cv2.VideoCapture(str(source))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(f"Could not read video dimensions: {source}")
    x0, y0, x1, y1 = crop_bounds(width, height, crop)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    writer = cv2.VideoWriter(
        str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (x1 - x0, y1 - y0)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create cropped model input: {destination}")
    frames = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(frame[y0:y1, x0:x1])
            frames += 1
    finally:
        capture.release()
        writer.release()
    if frames == 0:
        raise RuntimeError(f"Cropped model input contains no frames: {destination}")


STREAM_SELECTOR = "bv[height<=720][ext=mp4][vcodec^=avc1]/bv[height<=720][ext=mp4]/bv[height<=720]"


def stream_source_fps(url: str) -> float:
    """Read stream metadata without fetching media."""
    import yt_dlp

    with yt_dlp.YoutubeDL({
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "skip_download": True,
        "format": STREAM_SELECTOR,
    }) as ydl:
        info = ydl.extract_info(url, download=False)
    return float(info.get("fps") or 30.0)


def stream_cropped_frames(url: str, crop: tuple[float, float, float, float]):
    """Yield cropped BGR frames from yt-dlp/FFmpeg pipes without saving the source video."""
    try:
        import av
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError("The stream mode requires PyAV and yt-dlp in the vision environment") from exc

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg must be on PATH for YouTube stream mode")

    # A metadata-only yt-dlp call gives the source frame rate for snapshot timestamps. The
    # media itself is fetched by the separate stdout pipe below.
    # DASH MP4 is not seekable when sent to stdout. FFmpeg reads that pipe and emits a
    # streamable Matroska pipe; MJPEG keeps the decoder path broadly compatible with the
    # installed OpenCV/Ultralytics stack while still never touching disk.
    yt_process = subprocess.Popen(
        [
            sys.executable, "-m", "yt_dlp", "--quiet", "--no-warnings", "--no-playlist",
            "--format", STREAM_SELECTOR, "--output", "-", url,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert yt_process.stdout is not None
    ffmpeg_process = subprocess.Popen(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
            "-map", "0:v:0", "-an", "-c:v", "mjpeg", "-f", "matroska", "pipe:1",
        ],
        stdin=yt_process.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    yt_process.stdout.close()
    assert ffmpeg_process.stdout is not None
    try:
        with av.open(ffmpeg_process.stdout, mode="r", format="matroska") as container:
            for frame in container.decode(video=0):
                image = frame.to_ndarray(format="bgr24")
                height, width = image.shape[:2]
                x0, y0, x1, y1 = crop_bounds(width, height, crop)
                yield image[y0:y1, x0:x1]
    finally:
        if ffmpeg_process.poll() is None:
            ffmpeg_process.terminate()
        if yt_process.poll() is None:
            yt_process.terminate()
        ffmpeg_stderr = ffmpeg_process.stderr.read().decode("utf-8", errors="replace") if ffmpeg_process.stderr else ""
        yt_stderr = yt_process.stderr.read().decode("utf-8", errors="replace") if yt_process.stderr else ""
        ffmpeg_process.wait(timeout=30)
        yt_process.wait(timeout=30)
        if ffmpeg_process.returncode not in (0, -15, 15) and ffmpeg_stderr.strip():
            raise RuntimeError(f"ffmpeg stream failed: {ffmpeg_stderr.strip()[-1000:]}")
        if yt_process.returncode not in (0, -15, 15) and yt_stderr.strip():
            raise RuntimeError(f"yt-dlp stream failed: {yt_stderr.strip()[-1000:]}")


def colored_detections(result):
    """Draw every YOLO robot detection using its inferred bumper color or neutral gray."""
    import cv2

    source_image = result.orig_img
    image = source_image.copy()
    if result.boxes is None:
        return image
    boxes = result.boxes.xyxy.detach().cpu().tolist()
    confidences = result.boxes.conf.detach().cpu().tolist()
    identifiers = (
        result.boxes.id.detach().cpu().tolist()
        if result.boxes.id is not None else [None] * len(boxes)
    )
    palette = {
        "red": (55, 75, 225),
        "blue": (225, 120, 55),
        None: (155, 155, 155),
    }
    for raw_box, confidence, identifier in zip(boxes, confidences, identifiers):
        left, top, right, bottom = (int(round(value)) for value in raw_box)
        alliance = bumper_color(source_image, raw_box)
        colour = palette[alliance]
        cv2.rectangle(image, (left, top), (right, bottom), colour, 3)
        label = "robot"
        if identifier is not None:
            label += f" id:{int(identifier)}"
        label += f" {float(confidence):.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
        )
        label_top = max(0, top - text_height - baseline - 5)
        cv2.rectangle(
            image,
            (left, label_top),
            (min(image.shape[1], left + text_width + 10), top),
            colour,
            -1,
        )
        cv2.putText(
            image,
            label,
            (left + 5, max(text_height, top - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
    return image


def detection_records(result) -> list[dict[str, object]]:
    """Return box metadata for one clean source frame without drawing on the frame."""
    if result.boxes is None:
        return []
    image = result.orig_img
    height, width = image.shape[:2]
    boxes = result.boxes.xyxy.detach().cpu().tolist()
    confidences = result.boxes.conf.detach().cpu().tolist()
    identifiers = (
        result.boxes.id.detach().cpu().tolist()
        if result.boxes.id is not None else [None] * len(boxes)
    )
    records: list[dict[str, object]] = []
    for raw_box, confidence, identifier in zip(boxes, confidences, identifiers):
        left, top, right, bottom = (float(value) for value in raw_box)
        alliance = bumper_color(image, raw_box)
        records.append({
            "label": "robot",
            "track_id": int(identifier) if identifier is not None else None,
            "confidence": round(float(confidence), 6),
            "alliance": alliance,
            "bbox_xyxy": [
                int(round(left)), int(round(top)), int(round(right)), int(round(bottom)),
            ],
            "bbox_normalized": {
                "x": round(left / width, 6),
                "y": round(top / height, 6),
                "width": round((right - left) / width, 6),
                "height": round((bottom - top) / height, 6),
            },
        })
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--homography", help="calibration JSON for carpet positions and speed")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", help="local video file")
    source.add_argument("--stream-url", help="YouTube URL; media is consumed as a pipe")
    parser.add_argument("--output", required=True, help="JSONL track samples")
    parser.add_argument("--tracker", choices=("bytetrack", "botsort"), default="bytetrack")
    parser.add_argument("--confidence", type=float, default=0.20)
    parser.add_argument("--image-size", type=int, default=960)
    parser.add_argument("--device", default="0")
    parser.add_argument("--annotated-output", help="optional MP4 preview")
    parser.add_argument(
        "--snapshot-dir",
        help="directory for clean JPEG snapshots and one annotations.json file",
    )
    parser.add_argument(
        "--snapshot-interval",
        type=float,
        default=5.0,
        help="seconds between clean snapshots (default: 5)",
    )
    parser.add_argument(
        "--partial-output",
        help="optional JSONL file for live, partial track output",
    )
    parser.add_argument(
        "--crop",
        nargs=4,
        type=float,
        default=DEFAULT_CROP,
        metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
        help="normalized crop rectangle; defaults to the marked upper broadcast panel",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path, output = Path(args.model), Path(args.output)
    video_path = Path(args.video) if args.video else None
    if not model_path.is_file():
        raise SystemExit("--model must name an existing file")
    if video_path is not None and not video_path.is_file():
        raise SystemExit("--video must name an existing file")
    if output.exists():
        raise SystemExit(f"Refusing to overwrite track output: {output}")
    try:
        import cv2
        from ultralytics import YOLO
        from ultralytics.data.build import SourceTypes
        from ultralytics.data.loaders import LoadPilAndNumpy
    except ImportError as exc:
        raise SystemExit("Install training/requirements-yolo.txt in the dedicated vision venv") from exc
    if args.snapshot_interval <= 0:
        raise SystemExit("--snapshot-interval must be greater than zero")
    output.parent.mkdir(parents=True, exist_ok=True)
    mapper = None
    position_source = None
    if args.homography:
        try:
            from ingest.collection.homography import load_calibration
            mapper = load_calibration(args.homography)
        except (ImportError, OSError, ValueError) as exc:
            raise SystemExit(f"Could not load homography {args.homography}: {exc}") from exc
        if mapper is None:
            raise SystemExit(f"Homography is missing, malformed, or untrustworthy: {args.homography}")
        position_source = mapper.source

    if args.stream_url:
        fps = stream_source_fps(args.stream_url)
        total_frames = 0
        source_name = args.stream_url

        class YtFrameLoader(LoadPilAndNumpy):
            """Ultralytics-compatible lazy loader around the yt-dlp/FFmpeg frame pipe."""

            def __init__(self, frames):
                self.frames = iter(frames)
                self.mode = "stream"
                self.bs = 1
                self.count = 0
                self.source_type = SourceTypes(stream=True, screenshot=False, from_img=False, tensor=False)

            def __len__(self):
                return 0

            def __iter__(self):
                return self

            def __next__(self):
                frame = next(self.frames)
                self.count += 1
                return [f"youtube_stream_{self.count:08d}.jpg"], [frame], [""]

        source_frames = YtFrameLoader(stream_cropped_frames(args.stream_url, tuple(args.crop)))
    else:
        assert video_path is not None
        source_name = str(video_path)

        class LocalFrameLoader(LoadPilAndNumpy):
            """Lazy OpenCV source that crops each frame as YOLO requests it."""

            def __init__(self, path: Path, crop):
                self.capture = cv2.VideoCapture(str(path))
                if not self.capture.isOpened():
                    raise SystemExit(f"Could not open video: {path}")
                self.fps = float(self.capture.get(cv2.CAP_PROP_FPS) or 30.0)
                self.total_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                self.crop = tuple(crop)
                self.mode = "stream"
                self.bs = 1
                self.count = 0
                self.source_type = SourceTypes(stream=True, screenshot=False, from_img=False, tensor=False)

            def __len__(self):
                return 0

            def __iter__(self):
                return self

            def __next__(self):
                ok, frame = self.capture.read()
                if not ok:
                    self.capture.release()
                    raise StopIteration
                height, width = frame.shape[:2]
                x0, y0, x1, y1 = crop_bounds(width, height, self.crop)
                self.count += 1
                return [f"video_stream_{self.count:08d}.jpg"], [frame[y0:y1, x0:x1]], [""]

            def close(self):
                self.capture.release()

        source_frames = LocalFrameLoader(video_path, tuple(args.crop))
        fps = source_frames.fps
        total_frames = source_frames.total_frames
    model = YOLO(str(model_path))
    results = model.track(
        source=source_frames, stream=True, persist=True, tracker=f"{args.tracker}.yaml",
        conf=args.confidence, imgsz=args.image_size, device=args.device, verbose=False,
    )
    tracks: dict[int, list[dict[str, object]]] = {}
    alliance_votes: dict[int, dict[str, int]] = {}
    motion_history: dict[int, list[tuple[float, float, float]]] = {}
    writer = None
    annotated_path = Path(args.annotated_output) if args.annotated_output else None
    if annotated_path:
        annotated_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_dir = Path(args.snapshot_dir) if args.snapshot_dir else None
    if snapshot_dir:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
    next_snapshot = 0.0
    snapshot_annotations: list[dict[str, object]] = []
    partial_output = Path(args.partial_output) if args.partial_output else None
    # Publish immediately and then about once per source second. This keeps the UI useful for
    # both finite files and never-ending streams without rewriting the complete output every
    # frame.
    publish_interval = max(1, int(round(fps)))
    for frame_index, result in enumerate(results):
        timestamp = round(frame_index / fps, 6)
        if result.boxes is not None:
            xyxy = result.boxes.xyxy.detach().cpu().tolist()
            identifiers = (
                result.boxes.id.detach().cpu().tolist()
                if result.boxes.id is not None else [None] * len(xyxy)
            )
            height, width = result.orig_img.shape[:2]
            for raw_box, identifier in zip(xyxy, identifiers):
                left, top, right, bottom = (float(value) for value in raw_box)
                if identifier is None:
                    continue
                track_id = int(identifier)
                alliance = bumper_color(result.orig_img, raw_box)
                if alliance:
                    counts = alliance_votes.setdefault(track_id, {"red": 0, "blue": 0})
                    counts[alliance] += 1
                sample: dict[str, object] = {
                    "t": timestamp,
                    "x": round(left / width, 6), "y": round(top / height, 6),
                    "w": round((right - left) / width, 6),
                    "h": round((bottom - top) / height, 6),
                    "track_id": track_id,
                    "bbox_normalized": {
                        "x": round(left / width, 6), "y": round(top / height, 6),
                        "width": round((right - left) / width, 6),
                        "height": round((bottom - top) / height, 6),
                    },
                }
                add_field_motion(
                    sample,
                    mapper,
                    tuple(args.crop),
                    int(width),
                    int(height),
                    motion_history,
                    max(1.5 / fps, 1.0),
                )
                sample.pop("track_id", None)
                sample.pop("bbox_normalized", None)
                tracks.setdefault(track_id, []).append(sample)
        if snapshot_dir and timestamp + (0.5 / fps) >= next_snapshot:
            clean_frame = result.orig_img.copy()
            snapshot_name = f"snapshot_{int(round(next_snapshot)):06d}s.jpg"
            snapshot_path = snapshot_dir / snapshot_name
            if not cv2.imwrite(str(snapshot_path), clean_frame, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                raise RuntimeError(f"Could not write YOLO snapshot: {snapshot_path}")
            snapshot_annotations.append({
                "image": snapshot_name,
                "timestamp_seconds": round(next_snapshot, 3),
                "frame_index": frame_index,
                "width": int(clean_frame.shape[1]),
                "height": int(clean_frame.shape[0]),
                "boxes": detection_records(result),
            })
            next_snapshot += args.snapshot_interval
        if annotated_path:
            plotted = colored_detections(result)
            if writer is None:
                height, width = plotted.shape[:2]
                writer = cv2.VideoWriter(
                    str(annotated_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
                )
            writer.write(plotted)
        if frame_index == 0 or (frame_index + 1) % publish_interval == 0:
            fraction = min(1.0, (frame_index + 1) / total_frames) if total_frames else None
            if partial_output:
                write_track_records(
                    partial_output, tracks, fps, resolved_alliances(alliance_votes), position_source
                )
            print(json.dumps({
                "progress": round(0.10 + fraction * 0.80, 6) if fraction is not None else None,
                "stage": "tracking",
            }), flush=True)
    if writer is not None:
        writer.release()
    alliances = resolved_alliances(alliance_votes)
    write_track_records(output, tracks, fps, alliances, position_source)
    if snapshot_dir:
        (snapshot_dir / "annotations.json").write_text(
            json.dumps({
                "schema_version": 1,
                "source_video": source_name,
                "crop": {
                    "left": args.crop[0],
                    "top": args.crop[1],
                    "right": args.crop[2],
                    "bottom": args.crop[3],
                },
                "interval_seconds": args.snapshot_interval,
                "images": snapshot_annotations,
            }, indent=2) + "\n",
            encoding="utf-8",
        )
    if hasattr(source_frames, "close"):
        source_frames.close()
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
