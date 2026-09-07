import datetime
import uuid

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, JSON, String
from sqlalchemy.orm import declarative_base

from .serializers import SCHEMA_VERSION

Base = declarative_base()


def utcnow():
    """Timezone-aware UTC. Doc 0 wants ISO 8601 with a Z suffix, which needs real tzinfo."""
    return datetime.datetime.now(datetime.timezone.utc)


class Job(Base):
    """Contract A."""

    __tablename__ = "jobs"

    job_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    schema_version = Column(Integer, default=SCHEMA_VERSION, nullable=False)
    # Nullable on purpose: Contract E says an unresolved match comes back as null. Never the
    # string "unknown" -- that is not a valid TBA key and every unresolved job would collide.
    match_id = Column(String, nullable=True, index=True)
    # Selects /contracts/seasons/<year>.json, so old footage stays analyzable.
    season = Column(Integer, nullable=False, default=2026)
    video_id = Column(String(11), nullable=False)
    # Additive annotation: live jobs use the normal lifecycle once recording ends.
    capture_mode = Column(String, nullable=False, default="recorded", server_default="recorded")
    local_path = Column(String, nullable=True)
    start_offset = Column(Float, default=0.0, nullable=False)
    # Stream jobs never materialize a media file, so this remains NULL. Media dimensions are
    # still written to the job before analysis completes.
    duration = Column(Float, nullable=True)
    fps = Column(Float, nullable=True)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    status = Column(String, default="queued", nullable=False)
    alliances = Column(JSON, nullable=True)
    tba_score = Column(JSON, nullable=True)
    # Doc 2 treats a failed download as an expected condition, and doc 3 wants a retry path.
    # A retry the user cannot reason about is not much of a path, so keep the reason.
    error = Column(String, nullable=True)
    # Closed enum so the UI knows whether a retry is worth offering:
    # rate_limited is, video_unavailable is not.
    error_code = Column(String, nullable=True)
    attempt = Column(Integer, nullable=False, default=1)
    # Contract D has component 1 stream progress so "component 2 can show a progress bar" --
    # but component 3 is what draws it, so it has to survive to the job record.
    progress = Column(Float, nullable=True)
    stage = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)


class Event(Base):
    """Contract B. Raw model output ONLY.

    Doc 0: "events stores raw model output only" and "Corrections never overwrite model
    output." Nothing in this service may UPDATE a row in this table. Corrections go to the
    corrections table and are applied on read.
    """

    __tablename__ = "events"

    event_id = Column(String, primary_key=True)
    schema_version = Column(Integer, default=SCHEMA_VERSION, nullable=False)
    job_id = Column(String, ForeignKey("jobs.job_id"), index=True)
    match_id = Column(String, index=True)
    team = Column(Integer, nullable=True)
    # Nullable: match_start / match_end / phase_change belong to no track.
    # See contracts/OPEN_QUESTIONS.md #1.
    track_id = Column(Integer, nullable=True)
    t_seconds = Column(Float, nullable=False)
    phase = Column(String, nullable=False)
    event_type = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    field_x = Column(Float, nullable=True)
    field_y = Column(Float, nullable=True)
    # v3. Which goal a shot went into; null when unknown or not a shot. NOT a closed
    # set -- legal values are the season config's `goals`, which change every season.
    goal = Column(String, nullable=True)
    source = Column(String, nullable=False)


class Track(Base):
    """Contract C.

    match_id is denormalised from the job so the tracks endpoint can filter by it. Contract C
    itself has no match_id field, so it MUST be taken from the job record -- reading it off
    the track JSON yields NULL for every row and the endpoint then returns nothing forever.
    """

    __tablename__ = "tracks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String, ForeignKey("jobs.job_id"), index=True)
    match_id = Column(String, index=True)
    track_id = Column(Integer, nullable=False)
    team = Column(Integer, nullable=True)
    alliance = Column(String, nullable=True)
    # How sure bumper OCR is about the WHOLE track, separate from event confidence.
    team_confidence = Column(Float, nullable=True)
    boxes = Column(JSON, nullable=False)
    # Required by Contract C, possibly empty. Consumers must not interpolate across one.
    gaps = Column(JSON, nullable=False, default=list)


class Correction(Base):
    """Contract F. A correction references what it changes; it never replaces it.

    scope='event'  -> target_id is an event_id.
    scope='track'  -> target_id is a track_id and job_id is required, because track ids are
                      job-local and there is no global track address. A track-scoped
                      correction re-attributes the track AND every event on it, as one action.
    """

    __tablename__ = "corrections"

    correction_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    schema_version = Column(Integer, default=SCHEMA_VERSION, nullable=False)
    scope = Column(String, nullable=False, default="event")  # event | track
    job_id = Column(String, index=True, nullable=True)
    target_id = Column(String, index=True, nullable=False)
    # Denormalised so corrections for a match can be fetched without joining every event.
    match_id = Column(String, index=True, nullable=True)
    action = Column(String, nullable=False)  # edit | delete | create
    fields = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    created_by = Column(String, nullable=True)
