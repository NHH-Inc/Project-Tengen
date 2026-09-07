"""ORM rows -> the exact shapes in /contracts/, SCHEMA_VERSION 3.

Returning SQLAlchemy models straight out of FastAPI works, but the response shape then
silently tracks whatever the columns happen to be. These functions pin it to the contracts
instead, so a stray column cannot leak into a response and a missing one fails loudly here
rather than in the browser.

snake_case throughout, per doc 0. Component 3 converts to camelCase at its own boundary.
"""

import datetime

SCHEMA_VERSION = 3
MAX_PUBLIC_ROBOTS = 6


def cap_public_tracks(tracks: list[dict], limit: int = MAX_PUBLIC_ROBOTS) -> list[dict]:
    """Return at most six strongest tracks without modifying preserved analyzer output.

    Current trackers emit stable IDs 1-6. This ranking is a compatibility guard for legacy jobs
    containing dozens of raw fragments: prefer confirmed alliance tracks and longer histories,
    then return the selected records in deterministic track-id order. Raw API requests can bypass
    this view when exact historical analyzer output is needed.
    """

    if len(tracks) <= limit:
        return tracks
    ranked = sorted(
        tracks,
        key=lambda track: (
            track.get("alliance") not in {"red", "blue"},
            -len(track.get("boxes") or []),
            int(track.get("track_id") or 0),
        ),
    )
    selected = ranked[:limit]
    return sorted(selected, key=lambda track: int(track.get("track_id") or 0))


def iso_z(dt: datetime.datetime | None) -> str | None:
    """ISO 8601, UTC, Z suffix -- doc 0's stated format for metadata timestamps."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return dt.astimezone(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def job_to_dict(job) -> dict:
    """Contract A."""
    return {
        "schema_version": SCHEMA_VERSION,
        "job_id": job.job_id,
        "match_id": job.match_id,
        "season": job.season,
        "video_id": job.video_id,
        # Older web clients ignore this additive field; the current one labels live capture.
        "capture_mode": job.capture_mode or "recorded",
        "local_path": job.local_path,
        "start_offset": job.start_offset,
        "duration": job.duration,
        "fps": job.fps,
        "width": job.width,
        "height": job.height,
        "status": job.status,
        "stage": job.stage,
        "progress": job.progress,
        "error_code": job.error_code,
        "error": job.error,
        "attempt": job.attempt,
        "created_at": iso_z(job.created_at),
        "updated_at": iso_z(job.updated_at),
        "alliances": job.alliances,
        "tba_score": job.tba_score,
    }


def event_to_dict(event, corrected: bool = False, correction_id: str | None = None) -> dict:
    """Contract B, plus the two read-only annotations Contract E adds to API responses.

    `corrected` and `source` are independent: a model event a human fixed keeps
    source 'model' and gains corrected: true.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "job_id": event.job_id,
        "match_id": event.match_id,
        "event_id": event.event_id,
        "team": event.team,
        "track_id": event.track_id,
        "t_seconds": event.t_seconds,
        "phase": event.phase,
        "event_type": event.event_type,
        "confidence": event.confidence,
        "field_x": event.field_x,
        "field_y": event.field_y,
        "goal": event.goal,
        "source": event.source,
        "corrected": corrected,
        "correction_id": correction_id,
    }


def track_to_dict(track) -> dict:
    """Contract C. No match_id: that is a storage detail, not part of the contract."""
    return {
        "schema_version": SCHEMA_VERSION,
        "track_id": track.track_id,
        "team": track.team,
        "alliance": track.alliance,
        "team_confidence": track.team_confidence,
        "boxes": track.boxes or [],
        "gaps": track.gaps or [],
    }


def correction_to_dict(correction) -> dict:
    """Contract F."""
    return {
        "schema_version": SCHEMA_VERSION,
        "correction_id": correction.correction_id,
        "scope": correction.scope,
        "job_id": correction.job_id,
        "target_id": correction.target_id,
        "action": correction.action,
        "fields": correction.fields,
        "created_at": iso_z(correction.created_at),
        "created_by": correction.created_by,
    }


# Contract B field names a correction may patch.
EVENT_FIELDS = {
    "team", "track_id", "t_seconds", "phase", "event_type",
    "confidence", "field_x", "field_y", "goal", "source",
}

# Only a shot can have gone into a goal.
SHOT_EVENTS = {"shot_attempt", "shot_made"}

# Closed sets from /contracts/enums.md. Doc 0: "Anything unrecognized is a bug, not a
# fallback" -- so these are validated on the way in rather than stored and discovered later.
PHASES = {"auto", "teleop", "endgame", "unknown"}
EVENT_TYPES = {
    "match_start", "match_end", "phase_change",
    "shot_attempt", "shot_made", "reload",
    "defense_start", "defense_end",
    "immobile_start", "immobile_end", "foul",
}
MATCH_LEVEL_EVENTS = {"match_start", "match_end", "phase_change"}
SOURCES = {"model", "scoreboard_ocr", "tba", "manual"}
JOB_STATUSES = {"queued", "downloading", "downloaded", "analyzing", "complete", "failed"}
STAGES = {"downloading", "decoding", "detecting", "tracking", "ocr", "events"}
ERROR_CODES = {
    "video_unavailable", "download_failed", "rate_limited",
    "no_match_data", "analysis_failed", "timeout", "internal",
}
GAP_REASONS = {"shot_change", "occlusion", "out_of_frame", "detection_lost"}
CORRECTION_SCOPES = {"event", "track"}
CORRECTION_ACTIONS = {"edit", "delete", "create"}


def validate_event_fields(fields: dict, legal_goals: set[str] | None = None) -> list[str]:
    """Returns a list of problems; empty means the patch is contract-legal.

    `legal_goals` comes from the season config, not from this module -- goal names change
    every season, which is exactly why doc 0 keeps them out of the enum list. Pass None to
    skip the membership check when the season is not known.
    """
    problems = []
    for key, value in fields.items():
        if key not in EVENT_FIELDS:
            problems.append(f"'{key}' is not a correctable event field")
            continue
        if key == "phase" and value not in PHASES:
            problems.append(f"phase '{value}' is not one of {sorted(PHASES)}")
        if key == "event_type" and value not in EVENT_TYPES:
            problems.append(f"event_type '{value}' is not a known event type")
        if key == "source" and value not in SOURCES:
            problems.append(f"source '{value}' is not one of {sorted(SOURCES)}")
        if key == "confidence" and not (
            isinstance(value, (int, float)) and 0.0 <= float(value) <= 1.0
        ):
            problems.append("confidence must be a float 0..1, never a percentage")
        if key == "team" and value is not None and not isinstance(value, int):
            problems.append("team must be an integer with no 'frc' prefix, or null")
        if key == "goal" and value is not None:
            if legal_goals is not None and value not in legal_goals:
                problems.append(
                    f"goal '{value}' is not one of this season's goals: "
                    + ", ".join(sorted(legal_goals))
                )
            event_type = fields.get("event_type")
            if event_type is not None and event_type not in SHOT_EVENTS:
                problems.append(f"'{event_type}' cannot have a goal; only shots can")
    return problems
