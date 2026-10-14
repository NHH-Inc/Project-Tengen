"""What the API needs to run on the internet: access keys, published runs, and strategy.

Tengen's analysis is heavy (YOLO on every frame) and runs on a team laptop or desktop. The
hosted API does not analyse anything: a machine that analysed a match *publishes* the result,
and the API stores it in the database and serves it to anyone with the key. So the website keeps
working with every team computer switched off, and no team computer is ever a server.

Three pieces, installed onto the FastAPI app by `install()`:

* **Access keys.** With TENGEN_REQUIRE_KEYS=1 (set on a hosted server) nothing works until the
  keys exist, so a misconfigured deploy is closed rather than open. With TENGEN_WRITE_KEY set, anything that changes data (POST, PATCH, PUT,
  DELETE) needs it in the `X-Tengen-Key` header. With TENGEN_READ_KEY also set, reading needs
  either key too, so scouting data is not public. Media URLs, which a <video> element cannot add
  headers to, may carry it as `?key=`. A local install with neither set behaves as before.
* **Publishing.** POST /api/publish takes a run's files (tracks, events, shots, result, clock)
  and its job record, stores events and tracks in their tables and the rest as RunArtifact rows.
  Tracks are thinned to 10 Hz first: plenty for the overlay and the strategy layer, and a tenth
  of the size of a 30 fps file.
* **Video.** YouTube refuses cloud servers ("confirm you're not a bot"), so a hosted API cannot
  stream a match from YouTube. Publishing uploads a small 480p copy instead, kept privately in
  Supabase Storage (SUPABASE_URL, SUPABASE_SERVICE_KEY); /api/video sends a signed link that
  expires. Media URLs carry a short-lived token for one job, never a key, so keys stay out of
  request logs.
* **Strategy.** /api/jobs/{id}/strategy, /api/strategy/teams and /api/strategy/teams/{team}
  run the same strategy layer as the desktop app, over published runs, with the web's own team
  corrections applied.
"""

from __future__ import annotations

import gzip
import hashlib
import hmac
import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Callable

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse
from sqlalchemy.orm import Session

from . import models

#: Files a published run may carry, beyond the job record. Anything else is refused.
PUBLISHABLE = {
    "tracks.jsonl", "events.jsonl", "shots.jsonl", "result.json", "match_clock.json",
    "goal_entries.jsonl", "ball_scouting.config.json",
}
#: Stored as artifacts; events and tracks go to their tables instead.
ARTIFACTS = PUBLISHABLE - {"tracks.jsonl", "events.jsonl"}
VIDEO_FILE = "preview.mp4"
VIDEO_BUCKET = os.environ.get("TENGEN_VIDEO_BUCKET", "tengen-video")
MEDIA_TOKEN_SECONDS = 6 * 3600
MEDIA_PATH = re.compile(r"^/api/(?:video|stream)/([^/]+)")
PUBLISHED_RATE_HZ = 10.0
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
READ_METHODS = {"GET", "HEAD", "OPTIONS"}


# ----------------------------------------------------------------------------- access keys

def _keys() -> tuple[str, str]:
    return os.environ.get("TENGEN_READ_KEY", ""), os.environ.get("TENGEN_WRITE_KEY", "")


def _matches(given: str, expected: str) -> bool:
    return bool(expected) and hmac.compare_digest(given.encode(), expected.encode())


def access_middleware(get_key_exempt: Callable[[str], bool]):
    async def middleware(request: Request, call_next):
        read_key, write_key = _keys()
        path = request.url.path
        if os.environ.get("TENGEN_REQUIRE_KEYS") == "1" and not write_key and path.startswith("/api")                 and not get_key_exempt(path):
            # A server on the internet that lost its keys must close, not quietly open to anyone.
            return JSONResponse(status_code=503, content={
                "error_code": "not_configured",
                "error": "This Tengen server has no team keys set yet, so it refuses every request.",
            })
        if os.environ.get("TENGEN_PUBLISH_ONLY") == "1" and request.method == "POST" and (
                path in ("/api/jobs", "/api/jobs/upload") or path.endswith("/retry")):
            # A hosted server has no GPU and half a gigabyte of memory: it shows published runs,
            # and analysis happens on the laptop or desktop that publishes them.
            return JSONResponse(status_code=409, content={
                "error_code": "publish_only",
                "error": "This server only shows published matches. Analyse the video in the Tengen "
                         "desktop app, then use Publish to web.",
            })
        if (read_key or write_key) and path.startswith("/api") and not get_key_exempt(path) \
                and request.method != "OPTIONS":
            given = request.headers.get("x-tengen-key") or ""
            writing = request.method not in READ_METHODS
            allowed = _matches(given, write_key) or (not writing and (not read_key or _matches(given, read_key)))
            media = MEDIA_PATH.match(path)
            if not allowed and media and not writing:
                # A <video> cannot send headers, so media URLs carry a token for that one job.
                allowed = check_media_token(media.group(1), request.query_params.get("t", ""))
            if not allowed:
                return JSONResponse(status_code=401, content={
                    "error_code": "unauthorized",
                    "error": "This Tengen server needs the team key for that: enter it under Team key in the sidebar (editing needs the write key).",
                })
        return await call_next(request)
    return middleware


def _token_secret() -> bytes:
    read_key, write_key = _keys()
    return (write_key or read_key).encode()


def media_token(job_id: str, now: float | None = None) -> str:
    """A token that opens one job's media for a few hours. Never the key itself."""
    expires = int((now or time.time()) + MEDIA_TOKEN_SECONDS)
    signature = hmac.new(_token_secret(), f"{job_id}:{expires}".encode(), hashlib.sha256).hexdigest()[:32]
    return f"{expires}.{signature}"


def check_media_token(job_id: str, token: str, now: float | None = None) -> bool:
    try:
        expires_text, signature = token.split(".", 1)
        expires = int(expires_text)
    except ValueError:
        return False
    if expires < (now or time.time()) or not _token_secret():
        return False
    expected = hmac.new(_token_secret(), f"{job_id}:{expires}".encode(), hashlib.sha256).hexdigest()[:32]
    return hmac.compare_digest(signature, expected)


# ----------------------------------------------------------------------------- video storage

def _storage() -> tuple[str, str] | None:
    url, key = os.environ.get("SUPABASE_URL", "").rstrip("/"), os.environ.get("SUPABASE_SERVICE_KEY", "")
    return (url, key) if url and key else None


def store_video(job_id: str, data: bytes) -> str:
    """Upload a published run's video copy; returns its storage reference."""
    import httpx

    storage = _storage()
    if storage is None:
        raise HTTPException(status_code=503, detail="Video storage is not configured on this server "
                                                    "(SUPABASE_URL and SUPABASE_SERVICE_KEY)")
    url, key = storage
    path = f"{job_id}.mp4"
    response = httpx.post(f"{url}/storage/v1/object/{VIDEO_BUCKET}/{path}", content=data, timeout=120,
                          headers={"Authorization": f"Bearer {key}", "apikey": key,
                                   "Content-Type": "video/mp4", "x-upsert": "true"})
    if response.status_code >= 300:
        raise HTTPException(status_code=502, detail=f"Video storage refused the upload: {response.text[:200]}")
    return f"{VIDEO_BUCKET}/{path}"


def video_redirect(job) -> RedirectResponse:
    """A short-lived signed link to the stored copy; the browser seeks against storage directly."""
    import httpx

    storage = _storage()
    if storage is None:
        raise HTTPException(status_code=503, detail="Video storage is not configured on this server")
    url, key = storage
    response = httpx.post(f"{url}/storage/v1/object/sign/{job.media_ref}", json={"expiresIn": MEDIA_TOKEN_SECONDS},
                          timeout=30, headers={"Authorization": f"Bearer {key}", "apikey": key})
    if response.status_code >= 300:
        raise HTTPException(status_code=502, detail="Could not sign the stored video")
    signed = response.json().get("signedURL") or response.json().get("signedUrl")
    return RedirectResponse(f"{url}/storage/v1{signed}", status_code=307)


def delete_video(job) -> None:
    import httpx

    storage = _storage()
    if storage is None or not getattr(job, "media_ref", None):
        return
    url, key = storage
    try:
        httpx.request("DELETE", f"{url}/storage/v1/object/{job.media_ref}", timeout=30,
                      headers={"Authorization": f"Bearer {key}", "apikey": key})
    except httpx.HTTPError:
        pass  # an orphaned copy costs storage, not correctness


# ----------------------------------------------------------------------------- artifacts

def put_artifact(db: Session, job_id: str, name: str, data: bytes) -> None:
    packed = gzip.compress(data)
    row = db.get(models.RunArtifact, (job_id, name))
    if row is None:
        db.add(models.RunArtifact(job_id=job_id, name=name, content=packed, size=len(data)))
    else:
        row.content, row.size = packed, len(data)


def artifact_text(db: Session, base_dir: str | Path, job_id: str, name: str) -> str | None:
    """A run's file: from the job folder when this machine analysed it, else from the database."""
    path = Path(base_dir) / job_id / name
    if path.is_file():
        return path.read_text(encoding="utf-8")
    row = db.get(models.RunArtifact, (job_id, name))
    return gzip.decompress(row.content).decode("utf-8") if row is not None else None


def _jsonl(text: str | None) -> list[dict]:
    if not text:
        return []
    return [json.loads(line) for line in text.splitlines() if line.strip()]


# ----------------------------------------------------------------------------- publishing

def thin_boxes(boxes: list[dict], rate_hz: float = PUBLISHED_RATE_HZ) -> list[dict]:
    """Keep at most `rate_hz` boxes per second, and always both sides of any hole.

    Dropping the box on either side of a hole would make the hole look shorter than it was, and
    a consumer that bridges short holes would then draw a robot through footage nobody saw.
    """
    if not boxes:
        return []
    step = 1.0 / rate_hz
    ordered = sorted(boxes, key=lambda b: b["t"])
    kept = [ordered[0]]
    for i, box in enumerate(ordered[1:], 1):
        after_hole = box["t"] - ordered[i - 1]["t"] > 2 * step
        before_hole = i + 1 < len(ordered) and ordered[i + 1]["t"] - box["t"] > 2 * step
        if box["t"] - kept[-1]["t"] >= step - 1e-6 or after_hole or before_hole or i == len(ordered) - 1:
            kept.append(box)
    return kept


def _publish(db: Session, job_record: dict, files: dict[str, bytes], local_video_id: Callable[[str], str]) -> dict:
    job_id = str(job_record.get("job_id") or "")
    if len(job_id) < 8:
        raise HTTPException(status_code=422, detail="job.job_id is required")
    if "tracks.jsonl" not in files or "events.jsonl" not in files:
        raise HTTPException(status_code=422, detail="A published run needs tracks.jsonl and events.jsonl")
    youtube_id = job_record.get("youtube_id")
    job = db.get(models.Job, job_id) or models.Job(job_id=job_id)
    job.match_id = job_record.get("match_id") or job.match_id
    job.season = int(job_record.get("season") or 2026)
    job.video_id = youtube_id or local_video_id(job_id)
    # A YouTube source can be streamed by the web player; a local recording stays on its laptop.
    job.capture_mode = "recorded" if youtube_id else "local"
    job.local_path = None
    job.start_offset = float(job_record.get("start_offset") or 0.0)
    for key in ("duration", "fps", "width", "height", "alliances", "tba_score"):
        if job_record.get(key) is not None:
            setattr(job, key, job_record[key])
    job.status, job.stage, job.progress, job.error, job.error_code = "complete", None, 1.0, None, None
    db.merge(job)
    db.flush()

    db.query(models.Event).filter(models.Event.job_id == job_id).delete(synchronize_session=False)
    db.query(models.Track).filter(models.Track.job_id == job_id).delete(synchronize_session=False)
    for data in _jsonl(files["events.jsonl"].decode("utf-8")):
        db.add(models.Event(
            event_id=data["event_id"], schema_version=data.get("schema_version", 3), job_id=job_id,
            match_id=data.get("match_id") or job.match_id, team=data.get("team"), track_id=data.get("track_id"),
            t_seconds=data["t_seconds"], phase=data["phase"], event_type=data["event_type"],
            confidence=data["confidence"], field_x=data.get("field_x"), field_y=data.get("field_y"),
            goal=data.get("goal"), source=data.get("source", "model")))
    box_count = 0
    for data in _jsonl(files["tracks.jsonl"].decode("utf-8")):
        boxes = thin_boxes(data.get("boxes") or [])
        box_count += len(boxes)
        db.add(models.Track(
            job_id=job_id, match_id=job.match_id, track_id=data["track_id"], team=data.get("team"),
            alliance=data.get("alliance"), team_confidence=data.get("team_confidence"),
            boxes=boxes, gaps=data.get("gaps") or []))

    for name, data in files.items():
        if name not in ARTIFACTS:
            continue
        if name == "result.json":
            result = json.loads(data)
            result["box_sample_rate"] = min(float(result.get("box_sample_rate") or PUBLISHED_RATE_HZ), PUBLISHED_RATE_HZ)
            result["published"] = True
            data = json.dumps(result).encode()
        put_artifact(db, job_id, name, data)
    db.commit()
    return {"job_id": job_id, "match_id": job.match_id, "boxes": box_count,
            "artifacts": sorted(n for n in files if n in ARTIFACTS)}


# ----------------------------------------------------------------------------- strategy

_feature_cache: dict[tuple, list[dict]] = {}


def _track_teams(db: Session, job_id: str) -> dict[int, int | None]:
    """The web's own team corrections for a job's tracks, latest last."""
    teams = {}
    rows = (db.query(models.Correction)
            .filter(models.Correction.job_id == job_id, models.Correction.scope == "track")
            .order_by(models.Correction.created_at).all())
    for row in rows:
        if row.action == "delete":
            teams.pop(int(row.target_id), None)
        elif "team" in (row.fields or {}):
            teams[int(row.target_id)] = row.fields["team"]
    return teams


def job_features(db: Session, base_dir: str | Path, job: models.Job) -> list[dict]:
    """Strategy features for one job, from whatever this server holds for it."""
    from .strategy.features import match_features
    from .strategy.timeline import SEASONS_DIR, build_timelines, load_geometry, read_json

    teams = _track_teams(db, job.job_id)
    key = (job.job_id, str(job.updated_at), tuple(sorted(teams.items())))
    if key in _feature_cache:
        return _feature_cache[key]
    with tempfile.TemporaryDirectory(prefix="tengen-strategy-") as tmp:
        folder = Path(tmp)
        (folder / "job.json").write_text(json.dumps({
            "job_id": job.job_id, "match_id": job.match_id, "season": job.season,
            "alliances": job.alliances}), encoding="utf-8")
        tracks = db.query(models.Track).filter(models.Track.job_id == job.job_id).all()
        (folder / "tracks.jsonl").write_text("".join(json.dumps({
            "track_id": t.track_id, "team": teams.get(t.track_id, t.team), "alliance": t.alliance,
            "boxes": t.boxes, "gaps": t.gaps or []}) + "\n" for t in tracks), encoding="utf-8")
        events = db.query(models.Event).filter(models.Event.job_id == job.job_id,
                                               models.Event.event_type == "match_start").all()
        (folder / "events.jsonl").write_text("".join(json.dumps({
            "event_type": e.event_type, "t_seconds": e.t_seconds}) + "\n" for e in events), encoding="utf-8")
        for name in ("shots.jsonl", "match_clock.json"):
            text = artifact_text(db, base_dir, job.job_id, name)
            if text is not None:
                (folder / name).write_text(text, encoding="utf-8")
        timelines = build_timelines(folder)
    season = read_json(SEASONS_DIR / f"{job.season or 2026}.json")
    features = match_features(timelines, load_geometry(job.season or 2026), season) if timelines else []
    _feature_cache[key] = features
    return features


def _all_features(db: Session, base_dir) -> list[dict]:
    jobs = db.query(models.Job).filter(models.Job.status == "complete").all()
    out = []
    for job in jobs:
        try:
            out.extend(job_features(db, base_dir, job))
        except Exception:
            continue  # one unreadable run must not take the team list down
    return out


def router(get_db, base_dir: Callable[[], str | Path], local_video_id: Callable[[str], str]) -> APIRouter:
    api = APIRouter()

    @api.post("/api/publish")
    async def publish(job: str = Form(...), files: list[UploadFile] = File(...), db: Session = Depends(get_db)):
        if not _keys()[1]:
            raise HTTPException(status_code=403, detail="Publishing needs TENGEN_WRITE_KEY set on the server")
        try:
            record = json.loads(job)
        except ValueError:
            raise HTTPException(status_code=422, detail="job must be JSON")
        payload, total = {}, 0
        for upload in files:
            name = Path(upload.filename or "").name
            if name not in PUBLISHABLE and name != VIDEO_FILE:
                raise HTTPException(status_code=422, detail=f"{name} is not a publishable run file")
            data = await upload.read()
            total += len(data)
            if total > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="Run is too large to publish")
            payload[name] = data
        video = payload.pop(VIDEO_FILE, None)
        summary = _publish(db, record, payload, local_video_id)
        if video:
            job_row = db.get(models.Job, summary["job_id"])
            job_row.media_ref = store_video(summary["job_id"], video)
            db.commit()
            summary["video"] = True
        _feature_cache.clear()
        return summary

    @api.get("/api/jobs/{job_id}/media-token")
    def get_media_token(job_id: str, db: Session = Depends(get_db)):
        """A token for this job's media URLs, fetched with the key in a header."""
        if db.get(models.Job, job_id) is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return {"token": media_token(job_id), "expires_in": MEDIA_TOKEN_SECONDS}

    @api.get("/api/jobs/{job_id}/strategy")
    def job_strategy(job_id: str, db: Session = Depends(get_db)):
        job = db.get(models.Job, job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        clock = artifact_text(db, base_dir(), job_id, "match_clock.json")
        return {"job_id": job_id, "match_id": job.match_id,
                "clock": json.loads(clock) if clock else None,
                "robots": job_features(db, base_dir(), job)}

    @api.get("/api/strategy/teams")
    def strategy_teams(db: Session = Depends(get_db)):
        teams: dict[int, dict] = {}
        for f in _all_features(db, base_dir()):
            if not f.get("team"):
                continue
            entry = teams.setdefault(f["team"], {"team": f["team"], "matches": [], "with_positions": 0})
            entry["matches"].append(f.get("match_id"))
            entry["with_positions"] += 1 if f.get("positions", True) is not None and "movement" in f else 0
        return {"teams": sorted(teams.values(), key=lambda t: t["team"])}

    @api.get("/api/strategy/teams/{team}")
    def strategy_team(team: int, db: Session = Depends(get_db)):
        from .strategy.profile import build_profile
        matches = [f for f in _all_features(db, base_dir()) if f.get("team") == team]
        if not matches:
            raise HTTPException(status_code=404, detail=f"No analysed matches with team {team} yet")
        return {"profile": build_profile(matches), "matches": matches}

    return api
