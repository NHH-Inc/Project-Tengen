"""Component 2's HTTP API -- Contract E.

Component 3 talks to this and nothing else, so every endpoint doc 0 lists exists here even
when the implementation behind it is thin. A missing endpoint is indistinguishable from a
broken one at the browser.
"""

import json
import os
import shutil
import uuid
from pathlib import Path

import httpx
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool

# Must run before database/TBA/Sheets read os.environ. See ingest/settings.py.
from . import settings  # noqa: F401
from . import (
    apps_script_sheets,
    database,
    downloader,
    models,
    orchestrator,
    sheets,
    stats,
    tba,
    yolo_orchestrator,
)
from .corrections import apply_corrections, apply_track_corrections
from .serializers import (
    normalize_public_track_labels,
    JOB_STATUSES,
    SCHEMA_VERSION,
    correction_to_dict,
    event_to_dict,
    job_to_dict,
    track_to_dict,
    validate_event_fields,
)

app = FastAPI(title="FRC Auto-Scouting Ingest Service")


@app.exception_handler(HTTPException)
async def contract_http_error(_request: Request, exc: HTTPException):
    """Contract E errors are ``{"error_code": "...", "error": "message"}``.

    FastAPI's default ``detail`` shape is not what component 3 parses.
    """
    code = getattr(exc, "error_code", None) or (
        "internal" if exc.status_code >= 500 else None
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"error_code": code, "error": str(exc.detail)},
    )


def _classify(exc: Exception) -> str:
    """Map a failure onto the closed error_code set so the UI knows whether to offer retry."""
    text = str(exc).lower()
    # YouTube's bot check is gating, not a broken download. It used to land in download_failed
    # because the message happens to contain the string "yt-dlp" (in a help URL), and the UI then
    # offered an immediate retry that could not possibly work. Backing off, or supplying cookies,
    # is what fixes it.
    if "not a bot" in text or "sign in to confirm" in text:
        return "rate_limited"
    if "429" in text or "rate" in text and "limit" in text:
        return "rate_limited"
    if "unavailable" in text or "private" in text or "removed" in text or "404" in text:
        return "video_unavailable"
    if "timed out" in text or "timeout" in text:
        return "timeout"
    if (
        "analysis exited" in text
        or "did not write" in text
        or "yolo tracker exited" in text
        or "yolo backend" in text
    ):
        return "analysis_failed"
    if "yt-dlp" in text or "download" in text or "403" in text:
        return "download_failed"
    return "internal"

# The web dev server runs on 5173 (doc 0 default). Vite proxies /api in dev, but a build
# served from anywhere else talks to this directly.
# Same-origin needs no CORS: `run.ps1 serve` has this service hand out the built UI, which is
# the recommended setup. These origins only matter when someone runs a separate Vite dev server,
# including from another machine -- add theirs to FRC_CORS_ORIGINS, comma separated.
_cors_origins = [
    o.strip()
    for o in os.environ.get(
        "FRC_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEFAULT_SEASON = int(os.environ.get("FRC_DEFAULT_SEASON", "2026"))
SEASONS_DIR = Path(__file__).resolve().parent.parent / "contracts" / "seasons"

tba_client = tba.TBAClient()
sheets_exporter = sheets.SheetsExporter()
#: Alternative transport for accounts where an administrator has disabled Google Cloud, which is
#: normal for school Workspace accounts. Same row semantics, no service account required.
apps_script_exporter = apps_script_sheets.AppsScriptExporter()


def _active_exporter():
    """Whichever transport is configured, preferring the Cloud one when both are.

    Both write the same rows with the same stable keys, so which one runs is an account-access
    question rather than a behavioural one.
    """
    if sheets_exporter.configured:
        return sheets_exporter, "service_account"
    if apps_script_exporter.configured:
        return apps_script_exporter, "apps_script"
    return sheets_exporter, "none"


def _season_path(season: int) -> Path:
    path = SEASONS_DIR / f"{season}.json"
    if not path.exists():
        raise RuntimeError(f"No season config at contracts/seasons/{season}.json")
    return path


def _clear_analysis_output(job_id: str) -> None:
    """Remove exactly one job's analyzer directory before an overwrite run."""

    base = Path(analysis_orchestrator.output_base_dir).resolve()
    target = (base / job_id).resolve()
    # Job ids are UUIDs, but keep the guard here so a malformed database row can never turn a
    # user-requested re-run into a broad recursive delete.
    if target.parent != base:
        raise RuntimeError(f"Refusing to clear analyzer output outside {base}: {target}")
    if target.exists():
        if not target.is_dir():
            raise RuntimeError(f"Analyzer output is not a directory: {target}")
        shutil.rmtree(target)

database.init_db()

get_db = database.get_db

data_dir = os.environ.get("FRC_DATA_DIR", "./data")
REPO_ROOT = Path(__file__).resolve().parent.parent
video_downloader = downloader.VideoDownloader(download_dir=os.path.join(data_dir, "segments"))
native_analysis_orchestrator = orchestrator.AnalysisOrchestrator(
    binary_path=os.environ.get("ANALYSIS_BINARY", "./analysis/build/bin/analysis"),
    output_base_dir=os.path.join(data_dir, "jobs"),
)

_yolo_model_default = REPO_ROOT / "data" / "models" / "yolo11n-frc-robots-20260901" / "weights" / "best.pt"
yolo_analysis_orchestrator = yolo_orchestrator.YoloAnalysisOrchestrator(
    repo_root=REPO_ROOT,
    python_path=os.environ.get("YOLO_PYTHON"),
    model_path=os.environ.get("YOLO_MODEL_PATH") or str(_yolo_model_default),
    output_base_dir=os.path.join(data_dir, "jobs"),
    tracker=os.environ.get("FRC_YOLO_TRACKER", "bytetrack"),
    confidence=float(os.environ.get("FRC_YOLO_CONFIDENCE", "0.25")),
    image_size=int(os.environ.get("FRC_YOLO_IMAGE_SIZE", "960")),
    device=os.environ.get("FRC_YOLO_DEVICE", "0"),
    save_annotated=os.environ.get("FRC_YOLO_SAVE_ANNOTATED", "0").lower()
    in {"1", "true", "yes", "on"},
    snapshot_interval=float(os.environ.get("FRC_YOLO_SNAPSHOT_INTERVAL", "5")),
    reid_memory_seconds=float(os.environ.get("FRC_YOLO_REID_MEMORY_SECONDS", "5.0")),
    reid_appearance_threshold=float(
        os.environ.get("FRC_YOLO_REID_APPEARANCE_THRESHOLD", "0.60")
    ),
    reid_max_distance=float(os.environ.get("FRC_YOLO_REID_MAX_DISTANCE", "0.60")),
    reid_score_margin=float(os.environ.get("FRC_YOLO_REID_SCORE_MARGIN", "0.08")),
    reid_max_speed=float(os.environ.get("FRC_YOLO_REID_MAX_SPEED", "0.75")),
    reid_edge_threshold=float(os.environ.get("FRC_YOLO_REID_EDGE_THRESHOLD", "0.10")),
    reid_template_gallery_size=int(
        os.environ.get("FRC_YOLO_REID_TEMPLATE_GALLERY_SIZE", "5")
    ),
    reid_template_confirmation_frames=int(
        os.environ.get("FRC_YOLO_REID_TEMPLATE_CONFIRMATION_FRAMES", "3")
    ),
    reid_template_min_confidence=float(
        os.environ.get("FRC_YOLO_REID_TEMPLATE_MIN_CONFIDENCE", "0.50")
    ),
    reid_alliance_lock_seconds=float(
        os.environ.get("FRC_YOLO_REID_ALLIANCE_LOCK_SECONDS", "5.0")
    ),
    reid_alliance_lock_margin_seconds=float(
        os.environ.get("FRC_YOLO_REID_ALLIANCE_LOCK_MARGIN_SECONDS", "2.0")
    ),
    auto_homography=os.environ.get("FRC_AUTO_HOMOGRAPHY", "1").lower()
    in {"1", "true", "yes", "on"},
    homography_hfov_deg=float(os.environ.get("FRC_HOMOGRAPHY_HFOV_DEG", "70")),
)

_analysis_backend = os.environ.get("FRC_ANALYSIS_BACKEND", "auto").strip().lower()
if _analysis_backend not in {"auto", "native", "yolo"}:
    raise RuntimeError("FRC_ANALYSIS_BACKEND must be one of: auto, native, yolo")
if _analysis_backend == "yolo" or (
    _analysis_backend == "auto" and yolo_analysis_orchestrator.available
):
    analysis_orchestrator = yolo_analysis_orchestrator
else:
    analysis_orchestrator = native_analysis_orchestrator


def _media_window(url: str, info: dict) -> tuple[float, float, bool]:
    """Resolve the local segment window from metadata and an optional URL timestamp."""
    total_duration = info.get("duration")
    if not isinstance(total_duration, (int, float)) or total_duration <= 0:
        raise ValueError("Could not determine a finite video duration")

    start = info.get("section_start") or info.get("start_time")
    if not isinstance(start, (int, float)):
        start = downloader.start_time_from_url(url)
    start = max(0.0, float(start))
    if start >= float(total_duration):
        raise ValueError("The YouTube start time is beyond the end of the video")

    end = info.get("section_end") or info.get("end_time")
    if not isinstance(end, (int, float)) or end <= start:
        end = float(total_duration)
    end = min(float(total_duration), float(end))
    duration = end - start
    full_video = start == 0.0 and abs(duration - float(total_duration)) < 0.001
    return start, duration, full_video


def _analysis_stream_url(video_id: str, start_offset: float) -> str:
    """Build the seek point used by the lazy yt-dlp analysis pipe."""

    suffix = f"&t={start_offset:g}s" if start_offset > 0 else ""
    return f"https://www.youtube.com/watch?v={video_id}{suffix}"


# ---------------------------------------------------------------- jobs


@app.post("/api/jobs")
async def create_job(
    payload: dict, background_tasks: BackgroundTasks, db: Session = Depends(get_db)
):
    url = payload.get("url")
    # match_id is optional per Contract E. Absent means "resolve it for me".
    match_id = payload.get("match_id")
    # Optional per Contract E; component 2 defaults it. Selects the season config.
    season = payload.get("season") or DEFAULT_SEASON

    if not url:
        raise HTTPException(status_code=400, detail="URL is required")

    try:
        info = video_downloader.get_video_info(url)
        video_id = info.get("id")
        is_live = bool(info.get("is_live"))
        if payload.get("live_capture") and not is_live:
            raise ValueError("That YouTube link is not currently live; queue it as a normal video instead")
        if is_live:
            # Live sources have no stable end time. The completed MP4 is probed before this
            # job reaches downloaded, where Contract A requires all media metadata.
            start_offset, duration = 0.0, None
        else:
            start_offset, duration, _full_video = _media_window(url, info)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to get video info: {exc}")

    if not isinstance(video_id, str) or len(video_id) != 11:
        raise HTTPException(status_code=400, detail="Could not resolve a YouTube video ID")

    job_id = str(uuid.uuid4())
    db_job = models.Job(
        job_id=job_id,
        video_id=video_id,
        # Stays NULL when unresolved. Contract E: "returns the job with match_id: null if it
        # cannot". The string "unknown" is not a valid TBA key and would collide across every
        # unresolved job's events.
        match_id=match_id,
        season=int(season),
        capture_mode="live" if is_live else "recorded",
        status="queued",
        attempt=1,
        start_offset=start_offset,
        duration=duration,
        fps=info.get("fps"),
        width=info.get("width"),
        height=info.get("height"),
    )
    # Doc 1: the three teams per alliance are what make robot identification tractable,
    # and tba_score is the only thing the accuracy comparison can be scored against.
    # A missing key or an unplayed match leaves both null, which Contract A allows.
    if db_job.match_id:
        alliances, tba_score = tba_client.alliances_and_score(db_job.match_id)
        db_job.alliances = alliances
        db_job.tba_score = tba_score

    db.add(db_job)
    db.commit()
    db.refresh(db_job)

    background_tasks.add_task(process_job, job_id, url, is_live)

    return job_to_dict(db_job)


@app.get("/api/jobs")
def list_jobs(db: Session = Depends(get_db)):
    jobs = db.query(models.Job).order_by(models.Job.created_at.desc()).all()
    # Doc 0: "Collection endpoints return an object, never a bare array." That is what let
    # box_sample_rate land on the tracks response without a breaking change.
    return {"jobs": [job_to_dict(job) for job in jobs]}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str, db: Session = Depends(get_db)):
    job = db.query(models.Job).filter(models.Job.job_id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job_to_dict(job)


@app.delete("/api/jobs/{job_id}", status_code=204)
def delete_job(job_id: str, db: Session = Depends(get_db)):
    job = db.query(models.Job).filter(models.Job.job_id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    # Events and tracks are the product; the downloaded media is a cache (doc 2). Deleting a
    # job removes its rows, and the segment file with them.
    db.query(models.Event).filter(models.Event.job_id == job_id).delete()
    db.query(models.Track).filter(models.Track.job_id == job_id).delete()
    if job.local_path and os.path.exists(job.local_path):
        try:
            os.remove(job.local_path)
        except OSError:
            pass
    db.delete(job)
    db.commit()
    return None


@app.post("/api/jobs/{job_id}/retry")
async def retry_job(
    job_id: str, background_tasks: BackgroundTasks, db: Session = Depends(get_db)
):
    """Retry a failed job or replace a completed result with a fresh pipeline run.

    Reusing the same job_id keeps the selected video in place. The previous analyzer files are
    cleared before the worker starts, and import_results replaces the raw database rows when the
    new run succeeds.
    """
    job = db.query(models.Job).filter(models.Job.job_id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status in {"queued", "downloading", "downloaded", "analyzing"}:
        raise HTTPException(status_code=409, detail="That job is already running")
    try:
        _clear_analysis_output(job_id)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not clear the previous result: {exc}") from exc
    job.status = "queued"
    job.error = None
    job.error_code = None
    job.progress = None
    job.stage = None
    # Doc 0: retry reuses the job id and increments attempt -- a new job would orphan the
    # failed one's history, which is exactly what you want when a venue keeps failing.
    job.attempt = (job.attempt or 1) + 1
    db.commit()
    db.refresh(job)
    retry_url = f"https://www.youtube.com/watch?v={job.video_id}"
    if job.start_offset:
        retry_url += f"&t={job.start_offset}s"
    background_tasks.add_task(process_job, job_id, retry_url, job.capture_mode == "live")
    return job_to_dict(job)


def process_job(job_id: str, url: str, live_capture: bool = False):
    db = next(database.get_db())
    job = db.query(models.Job).filter(models.Job.job_id == job_id).first()
    if job is None:
        return

    def set_status(status: str, **fields):
        job.status = status
        for key, value in fields.items():
            setattr(job, key, value)
        db.commit()

    try:
        set_status("downloading", stage="downloading", progress=0.0)

        info = video_downloader.get_video_info(url)
        if live_capture:
            # Live streams have no finite duration at queue time. The stream analyzer publishes
            # the final duration when the broadcast ends.
            start_offset, duration = 0.0, None
        else:
            start_offset, duration, _full_video = _media_window(url, info)
        fps = info.get("fps") or 30.0
        width, height = info.get("width") or 1920, info.get("height") or 1080
        stream_url = _analysis_stream_url(job.video_id, start_offset)

        # Only metadata is persisted. Both the model and browser consume the yt-dlp stream;
        # no downloaded segment is created or attached to the job.
        set_status(
            "downloaded",
            local_path=None,
            start_offset=start_offset,
            duration=duration,
            fps=fps,
            width=width,
            height=height,
            stage=None,
        )

        set_status("analyzing", stage="detecting", progress=0.0)

        job_data = job_to_dict(job)
        job_data.pop("error", None)
        job_data.pop("progress", None)
        job_data.pop("stage", None)
        job_data.pop("created_at", None)
        job_data["stream_url"] = stream_url

        def on_progress(progress, stage):
            # Contract D streams this so a progress bar can exist; component 3 draws it, so
            # it has to reach the job record. See OPEN_QUESTIONS.md #4.
            job.progress = progress
            job.stage = stage
            db.commit()

        results = analysis_orchestrator.run_job(
            job_data, season_path=str(_season_path(job.season)), on_progress=on_progress
        )
        import_results(db, job, results)
        result_metadata = results.get("result") or {}
        final_duration = result_metadata.get("duration")
        if duration is None and isinstance(final_duration, (int, float)) and final_duration > 0:
            duration = float(final_duration)
        set_status(
            "complete",
            local_path=None,
            duration=duration,
            fps=result_metadata.get("box_sample_rate") or fps,
            width=width,
            height=height,
            progress=1.0,
            stage=None,
            error=None,
        )

    except Exception as exc:
        # Doc 2: "treat a failed download as an expected condition, not a crash." Keep the
        # reason -- a retry the user cannot reason about is not much of a retry path.
        # Component 1 reports its own error_code on the last line of stderr; trust it
        # over our string matching when it gave us one.
        set_status(
            "failed",
            error=str(exc)[:1000],
            error_code=getattr(exc, "error_code", None) or _classify(exc),
            progress=None,
            stage=None,
        )
    finally:
        db.close()


def import_results(db: Session, job, results: dict):
    """Replace this job's raw events and tracks with the latest analyzer output."""
    # Re-analysis is an overwrite operation. Events use fresh UUIDs on every run, so checking
    # whether an event id already exists would preserve the old result and leave stale rows in
    # the match. Corrections remain a separate layer and are still composed on read.
    db.query(models.Event).filter(models.Event.job_id == job.job_id).delete(synchronize_session=False)
    db.query(models.Track).filter(models.Track.job_id == job.job_id).delete(synchronize_session=False)

    with open(results["events_path"], "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            data = json.loads(line)
            db.add(
                models.Event(
                    event_id=data["event_id"],
                    schema_version=data.get("schema_version", 1),
                    job_id=data.get("job_id", job.job_id),
                    match_id=data.get("match_id", job.match_id),
                    team=data.get("team"),
                    track_id=data.get("track_id"),
                    t_seconds=data["t_seconds"],
                    phase=data["phase"],
                    event_type=data["event_type"],
                    confidence=data["confidence"],
                    field_x=data.get("field_x"),
                    field_y=data.get("field_y"),
                    goal=data.get("goal"),
                    source=data.get("source", "model"),
                )
            )

    with open(results["tracks_path"], "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            data = json.loads(line)
            db.add(
                models.Track(
                    job_id=job.job_id,
                    # Contract C has NO match_id field. Reading it off the track JSON yields
                    # NULL for every row, and the tracks endpoint then returns nothing
                    # forever. It has to come from the job.
                    match_id=job.match_id,
                    track_id=data["track_id"],
                    team=data.get("team"),
                    alliance=data.get("alliance"),
                    team_confidence=data.get("team_confidence"),
                    boxes=data.get("boxes") or [],
                    # Required by Contract C. Never interpolate across one.
                    gaps=data.get("gaps") or [],
                )
            )
    db.commit()


# ---------------------------------------------------------------- match data


def _corrections_for(db: Session, match_id: str):
    """Every correction affecting a match: event-scoped by match, track-scoped by job."""
    job_ids = [
        j.job_id for j in db.query(models.Job).filter(models.Job.match_id == match_id).all()
    ]
    return (
        db.query(models.Correction)
        .filter(
            (models.Correction.match_id == match_id)
            | (models.Correction.job_id.in_(job_ids) if job_ids else False)
        )
        .order_by(models.Correction.created_at)
        .all()
    )


def _events_for(db: Session, match_id: str, min_confidence: float, raw: bool) -> list[dict]:
    rows = (
        db.query(models.Event)
        .filter(models.Event.match_id == match_id)
        .order_by(models.Event.t_seconds)
        .all()
    )
    if raw:
        # Uncorrected model output: what the accuracy comparison and training export need.
        events = [event_to_dict(e) for e in rows]
    else:
        events = apply_corrections(rows, _corrections_for(db, match_id))
    return [e for e in events if (e.get("confidence") or 0.0) >= min_confidence]


@app.get("/api/matches/{match_id}/events")
def get_match_events(
    match_id: str,
    min_confidence: float = 0.0,
    raw: bool = Query(False, description="Return uncorrected model output."),
    db: Session = Depends(get_db),
):
    return {"events": _events_for(db, match_id, min_confidence, raw)}


@app.get("/api/matches/{match_id}/tracks")
def get_match_tracks(
    match_id: str,
    raw: bool = Query(False, description="Return uncorrected track attribution."),
    db: Session = Depends(get_db),
):
    rows = db.query(models.Track).filter(models.Track.match_id == match_id).all()
    tracks = (
        [track_to_dict(t) for t in rows]
        if raw
        else apply_track_corrections(rows, _corrections_for(db, match_id))
    )
    if not raw:
        tracks = normalize_public_track_labels(tracks)
    # Contract C: the sample rate is stated in result.json and served here, so component 3
    # knows how much to interpolate instead of inferring it from sample spacing.
    job = db.query(models.Job).filter(models.Job.match_id == match_id).first()
    return {"box_sample_rate": _box_sample_rate(job), "tracks": tracks}


@app.get("/api/jobs/{job_id}/tracks")
def get_job_tracks(
    job_id: str,
    raw: bool = Query(False, description="Return every preserved legacy/raw track fragment."),
    db: Session = Depends(get_db),
):
    """Serve a job's YOLO tracks, including the partial file while inference is running."""
    job = db.query(models.Job).filter(models.Job.job_id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    job_dir = Path(analysis_orchestrator.output_base_dir) / job.job_id
    final_path = job_dir / "tracks.jsonl"
    partial_path = job_dir / "tracks.partial.jsonl"
    path = final_path if final_path.exists() else partial_path
    tracks = []
    if path.exists():
        try:
            tracks = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (OSError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=503, detail=f"Could not read YOLO tracks: {exc}")

    result = _read_result(job) or {}
    sample_rate = float(result.get("box_sample_rate") or job.fps or 0.0)
    public_tracks = tracks if raw else normalize_public_track_labels(tracks)
    return {
        "box_sample_rate": sample_rate,
        "tracks": public_tracks,
        "suppressed_track_count": 0,
        "complete": final_path.exists(),
    }


def _result_path(job) -> Path | None:
    if job is None:
        return None
    path = Path(analysis_orchestrator.output_base_dir) / job.job_id / "result.json"
    return path if path.exists() else None


def _read_result(job) -> dict | None:
    path = _result_path(job)
    if path is None:
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def _box_sample_rate(job) -> float:
    result = _read_result(job)
    return float(result.get("box_sample_rate", 0.0)) if result else 0.0


@app.get("/api/jobs/{job_id}/result")
def get_job_result(job_id: str, db: Session = Depends(get_db)):
    """Contract D's result.json, so component 3 can reach box_sample_rate and frame counts."""
    job = db.query(models.Job).filter(models.Job.job_id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    result = _read_result(job)
    if result is None:
        raise HTTPException(status_code=404, detail="No analysis result for that job yet")
    return result


@app.get("/api/matches/{match_id}/corrections")
def get_match_corrections(match_id: str, db: Session = Depends(get_db)):
    """Not in Contract E -- see OPEN_QUESTIONS.md #3.

    Without it a client has to fetch raw and corrected and diff them to find out which rows a
    human touched, which costs an extra request and still loses created_at.
    """
    return {"corrections": [correction_to_dict(c) for c in _corrections_for(db, match_id)]}


@app.get("/api/matches/{match_id}/accuracy")
def get_match_accuracy(match_id: str, db: Session = Depends(get_db)):
    """Doc 1: "If the pipeline's reconstructed score does not match TBA's official score for
    the same match, the pipeline is wrong. That comparison is the main evaluation loop."

    Scored from RAW events on purpose. Using the corrected stream would measure the reviewers
    and the number would improve every time somebody fixed a row by hand.
    """
    job = db.query(models.Job).filter(models.Job.match_id == match_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="No job for that match")

    events = _events_for(db, match_id, 0.0, raw=True)
    cfg = stats.season_config(job.season)
    reconstructed = stats.reconstruct_score(events, job.alliances, cfg)
    tba = job.tba_score
    delta = (
        {"red": reconstructed["red"] - tba["red"], "blue": reconstructed["blue"] - tba["blue"]}
        if tba
        else None
    )
    return {
        "match_id": match_id,
        # Contract A already allows alliances/tba_score to be null, and the UI needs to
        # tell "TBA had nothing" apart from "the score matched".
        "tba_available": tba is not None,
        "reconstructed": reconstructed,
        "tba": tba,
        "delta": delta,
    }


# ---------------------------------------------------------------- corrections


@app.post("/api/events")
def create_manual_event(event_data: dict, db: Session = Depends(get_db)):
    """Contract E: create a manual event. Recorded as a 'create' correction, not as raw output.

    The events table holds model output only, so a human-authored event lives in the
    corrections layer and is composed in on read.
    """
    payload = {k: v for k, v in event_data.items() if k not in ("event_id", "schema_version")}
    payload["source"] = "manual"

    match_id = payload.get("match_id")
    if not match_id:
        raise HTTPException(status_code=400, detail="match_id is required")

    checkable = {k: v for k, v in payload.items() if k not in ("job_id", "match_id")}
    job = db.query(models.Job).filter(models.Job.match_id == match_id).first()
    goals = stats.legal_goals(stats.season_config(job.season)) if job else None
    problems = validate_event_fields(checkable, legal_goals=goals)
    if problems:
        raise HTTPException(status_code=400, detail="; ".join(problems))

    # event_id is a UUIDv4 per doc 0's identifier table.
    event_id = str(uuid.uuid4())
    fields = {**payload, "event_id": event_id, "schema_version": SCHEMA_VERSION}
    correction_id = str(uuid.uuid4())

    db.add(
        models.Correction(
            correction_id=correction_id,
            scope="event",
            job_id=payload.get("job_id"),
            target_id=event_id,
            match_id=match_id,
            action="create",
            fields=fields,
        )
    )
    db.commit()
    return {**fields, "corrected": True, "correction_id": correction_id}


@app.patch("/api/events/{event_id}")
def update_event(event_id: str, updates: dict, db: Session = Depends(get_db)):
    """Records a correction. Does NOT modify the event row.

    Doc 0: "Corrections never overwrite model output... Keeping both is the whole point:
    overwriting destroys the ability to measure whether the model is improving." Mutating the
    row here would permanently destroy the model's original prediction -- it is not
    recoverable from the corrections table, which stores the new value, not the old one.
    """
    event = db.query(models.Event).filter(models.Event.event_id == event_id).first()

    # The target may be a manually created event, which lives only in the corrections layer.
    origin = (
        db.query(models.Correction)
        .filter(models.Correction.target_id == event_id)
        .first()
    )
    if not event and not origin:
        raise HTTPException(status_code=404, detail="Event not found")

    match_id = event.match_id if event else (origin.match_id if origin else None)
    job = (
        db.query(models.Job).filter(models.Job.match_id == match_id).first()
        if match_id
        else None
    )
    goals = stats.legal_goals(stats.season_config(job.season)) if job else None
    # An edit that only sets `goal` has no event_type in the patch, so fall back to the
    # stored one -- otherwise the shot check silently passes on any event.
    checkable = dict(updates)
    if "goal" in checkable and "event_type" not in checkable and event is not None:
        checkable["event_type"] = event.event_type
    problems = validate_event_fields(checkable, legal_goals=goals)
    if problems:
        raise HTTPException(status_code=400, detail="; ".join(problems))
    db.add(
        models.Correction(
            scope="event",
            job_id=event.job_id if event else (origin.job_id if origin else None),
            target_id=event_id,
            match_id=match_id,
            action="edit",
            fields=updates,
        )
    )
    db.commit()

    corrected = _events_for(db, match_id, 0.0, raw=False) if match_id else []
    for row in corrected:
        if row.get("event_id") == event_id:
            return row
    raise HTTPException(status_code=404, detail="Event not found after correction")


@app.delete("/api/events/{event_id}", status_code=204)
def delete_event(event_id: str, db: Session = Depends(get_db)):
    """A delete is a correction too. The raw event stays in the table for evaluation."""
    event = db.query(models.Event).filter(models.Event.event_id == event_id).first()
    origin = (
        db.query(models.Correction)
        .filter(models.Correction.target_id == event_id)
        .first()
    )
    if not event and not origin:
        raise HTTPException(status_code=404, detail="Event not found")

    db.add(
        models.Correction(
            scope="event",
            job_id=event.job_id if event else origin.job_id,
            target_id=event_id,
            match_id=event.match_id if event else origin.match_id,
            action="delete",
            fields=None,
        )
    )
    db.commit()
    return None


@app.patch("/api/jobs/{job_id}/tracks/{track_id}")
def patch_track(job_id: str, track_id: int, updates: dict, db: Session = Depends(get_db)):
    """Re-attribute a whole track, and every event on it, as one action.

    Doc 3: "The most common correction is a misread bumper, and it is a track-level fix, not
    an event-level one. One bad OCR read mislabels forty-odd events and every box on that
    robot." Scoped by job because track_id is job-local -- there is no global track address.
    """
    track = (
        db.query(models.Track)
        .filter(models.Track.job_id == job_id, models.Track.track_id == track_id)
        .first()
    )
    if not track:
        raise HTTPException(status_code=404, detail="Track not found on that job")

    team = updates.get("team")
    if team is not None and not isinstance(team, int):
        raise HTTPException(status_code=400, detail="team must be an integer or null")

    correction_id = str(uuid.uuid4())
    db.add(
        models.Correction(
            correction_id=correction_id,
            scope="track",
            job_id=job_id,
            target_id=str(track_id),
            match_id=track.match_id,
            action="edit",
            fields={"team": team},
            created_by=updates.get("created_by"),
        )
    )
    db.commit()
    rows = db.query(models.Track).filter(models.Track.match_id == track.match_id).all()
    corrected = apply_track_corrections(rows, _corrections_for(db, track.match_id))
    return next((t for t in corrected if t["track_id"] == track_id), None)


@app.delete("/api/corrections/{correction_id}", status_code=204)
def delete_correction(correction_id: str, db: Session = Depends(get_db)):
    """Undo. Doc 0: "Deleting a correction undoes it." Raw output was never touched."""
    row = (
        db.query(models.Correction)
        .filter(models.Correction.correction_id == correction_id)
        .first()
    )
    if not row:
        raise HTTPException(status_code=404, detail="Correction not found")
    db.delete(row)
    db.commit()
    return None


# ---------------------------------------------------------------- stats and export


@app.get("/api/teams/{team}/stats")
def get_team_stats(
    team: int,
    event_key: str | None = None,
    min_confidence: float = 0.0,
    db: Session = Depends(get_db),
):
    """Aggregates, computed per request. Doc 0: never stored, only queried."""
    jobs_query = db.query(models.Job).filter(models.Job.match_id.isnot(None))
    if event_key:
        jobs_query = jobs_query.filter(models.Job.match_id.like(f"{event_key}_%"))
    jobs = jobs_query.all()

    events: list[dict] = []
    played = 0
    cfg = None
    for job in jobs:
        rows = _events_for(db, job.match_id, min_confidence, raw=False)
        if any(e.get("team") == team for e in rows):
            played += 1
        events.extend(rows)
        cfg = cfg or stats.season_config(job.season)

    return stats.team_stats(team, events, cfg, event_key, played, min_confidence)


@app.post("/api/export/sheets")
def export_to_sheets(payload: dict, db: Session = Depends(get_db)):
    """Doc 3: "Sheets is an export destination, not storage."

    One row per team per match in aggregate mode, one row per event in raw mode. Writes are
    batched into a single API call, and every row carries a stable key so re-exporting the
    same matches updates in place instead of duplicating.
    """
    match_ids = payload.get("match_ids", [])
    mode = payload.get("mode", "aggregate")
    if mode not in ("raw", "aggregate"):
        raise HTTPException(status_code=400, detail="mode must be 'raw' or 'aggregate'")
    if not match_ids:
        raise HTTPException(status_code=400, detail="match_ids is required")

    per_match = []
    for match_id in match_ids:
        events = _events_for(db, match_id, 0.0, raw=False)
        job = db.query(models.Job).filter(models.Job.match_id == match_id).first()
        cfg = stats.season_config(job.season) if job else None
        teams = sorted({e["team"] for e in events if e.get("team") is not None})
        stats_list = [
            stats.team_stats(team, events, cfg, tba.event_key_of(match_id), 1)
            for team in teams
        ]
        per_match.append((match_id, events, stats_list))

    headers, rows = sheets.build_rows(mode, per_match)

    if not rows:
        # Writing nothing and returning 200 is the same failure this endpoint already refuses
        # further down: a success that did not happen. It is also the exact shape of the first
        # real end-to-end run, where 45 tracks and 11 team attributions produced zero rows
        # because a row needs a team-attributed EVENT and the analyzer emits only match_start
        # and match_end. Say which of those is missing rather than making the user guess.
        total = sum(len(events) for _, events, _ in per_match)
        attributed = sum(1 for _, events, _ in per_match for e in events
                         if e.get("team") is not None)
        raise HTTPException(
            status_code=422,
            detail=(
                f"Nothing to export: {total} events across those matches, {attributed} of them "
                f"attributed to a team, so there are no rows. Action events (shots, cycles) are "
                f"not extracted yet, and match_start/match_end belong to no team. Attributing a "
                f"track in the web app moves its events onto a team; if a match has only "
                f"match-level events, there is nothing to attribute."
            ),
        )

    exporter, transport = _active_exporter()
    result = exporter.export(mode, headers, rows)

    if not result.get("configured"):
        # Be honest rather than reporting a successful write that did not happen.
        raise HTTPException(
            status_code=503,
            detail=(
                "Sheets export is not configured. Either set SHEETS_SPREADSHEET_ID and "
                "GOOGLE_APPLICATION_CREDENTIALS and share the sheet with the service account, "
                "or -- if Google Cloud is disabled on the account, as it is on most school "
                "accounts -- deploy tools/apps-script/Code.gs as a Web App and set "
                "APPS_SCRIPT_URL and APPS_SCRIPT_SECRET. See tools/apps-script/README.md."
            ),
        )

    if result.get("error"):
        # The write was attempted and refused. A wrong shared secret lands here, and saying so
        # beats reporting a success that wrote nothing.
        raise HTTPException(status_code=502, detail=f"Sheets export failed: {result['error']}")

    return {
        "spreadsheet_id": getattr(exporter, "spreadsheet_id", ""),
        "spreadsheet_url": (
            exporter.spreadsheet_url() if callable(getattr(exporter, "spreadsheet_url", None))
            else getattr(exporter, "spreadsheet_url", "")
        ),
        "transport": transport,
        "mode": mode,
        "rows_written": result["rows_written"],
        "rows_skipped": result["rows_skipped"],
    }


# ---------------------------------------------------------------- media


@app.get("/api/video/{job_id}")
def get_video(job_id: str, db: Session = Depends(get_db)):
    job = db.query(models.Job).filter(models.Job.job_id == job_id).first()
    if not job or not job.local_path or not os.path.exists(job.local_path):
        raise HTTPException(status_code=404, detail="Video file not found")
    # FileResponse answers HTTP range requests, which <video> needs in order to seek.
    return FileResponse(job.local_path, media_type="video/mp4")


@app.get("/api/stream/{job_id}/{kind}")
async def stream_video(
    job_id: str, kind: str, request: Request, db: Session = Depends(get_db)
):
    """Proxy a byte-range video or audio stream resolved by yt-dlp.

    The browser still receives normal byte-range media, so its native <video> element can seek
    and requestVideoFrameCallback can keep the detection canvas aligned. Current YouTube media
    is usually split into DASH video and audio files; the web player synchronizes the two local
    proxy endpoints. No YouTube iframe or ad player is involved.
    """
    if kind not in {"video", "audio"}:
        raise HTTPException(status_code=404, detail="Stream kind must be video or audio")
    job = db.query(models.Job).filter(models.Job.job_id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    try:
        streams = await run_in_threadpool(video_downloader.resolve_stream, job.video_id)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Could not resolve yt-dlp stream: {exc}")

    resolved = streams.get(kind)
    if not resolved:
        raise HTTPException(
            status_code=404,
            detail=f"This yt-dlp format has no separate {kind} stream",
        )

    upstream_headers = dict(resolved.get("headers") or {})
    if request.headers.get("range"):
        upstream_headers["Range"] = request.headers["range"]

    client = httpx.AsyncClient(follow_redirects=True, timeout=None)
    try:
        upstream_request = client.build_request(
            "GET", resolved["url"], headers=upstream_headers
        )
        upstream = await client.send(upstream_request, stream=True)
    except Exception as exc:
        await client.aclose()
        raise HTTPException(status_code=502, detail=f"yt-dlp media stream failed: {exc}")

    if upstream.status_code >= 400 and upstream.status_code != 416:
        await upstream.aclose()
        await client.aclose()
        raise HTTPException(
            status_code=502,
            detail=f"yt-dlp media host returned HTTP {upstream.status_code}",
        )

    response_headers = {}
    for header in (
        "accept-ranges", "content-length", "content-range", "etag", "last-modified"
    ):
        if header in upstream.headers:
            response_headers[header] = upstream.headers[header]

    async def close_upstream():
        await upstream.aclose()
        await client.aclose()

    return StreamingResponse(
        upstream.aiter_raw(),
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type", resolved.get("content_type", "video/mp4")),
        headers=response_headers,
        background=BackgroundTask(close_upstream),
    )


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        # Read the constant, never a literal -- this one silently stayed at 1
        # through two version bumps.
        "schema_version": SCHEMA_VERSION,
        "statuses": sorted(JOB_STATUSES),
        "dependencies": video_downloader.dependency_status(),
        "analysis": {
            "selected_backend": "yolo"
            if analysis_orchestrator is yolo_analysis_orchestrator
            else "native",
            "native_binary": native_analysis_orchestrator.binary_path,
            "yolo": yolo_analysis_orchestrator.health(),
        },
    }


# ---------------------------------------------------------------- serving the web app
#
# Hosting answer: there is nothing to deploy. `npm run build` produces static files, and this
# mounts them so ONE process on ONE port serves both the API and the UI. At a competition you
# run this on a laptop and everyone else opens http://<that-laptop>:8080 over the venue wifi.
#
# Publishing it on the open internet would be a mistake anyway: it needs the database, the
# downloaded video, yt-dlp and the analysis binary, and doc 2 already notes that bulk
# downloading is against YouTube's terms. Keep it on your own network.
#
# Mounted last, so every /api route above wins. html=True gives SPA fallback.
_web_dist = Path(__file__).resolve().parent.parent / "web" / "dist"
if _web_dist.is_dir():
    app.mount("/", StaticFiles(directory=str(_web_dist), html=True), name="web")
