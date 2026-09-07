"""Loading analyzer output into the database, twice.

Re-analysis is a normal thing to do -- there is a retry endpoint for it, and every improvement to
the detector is a reason to re-run a match. It has to replace what was there, not add to it.
Duplicated tracks double every per-team count downstream and nothing says so.
"""

import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("DATABASE_URL", "sqlite://")
if "FRC_DATA_DIR" not in os.environ:
    os.environ["FRC_DATA_DIR"] = tempfile.mkdtemp(prefix="frc-import-tests-")

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from ingest import main, models


@pytest.fixture
def db():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    models.Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()


@pytest.fixture
def job(db):
    row = models.Job(job_id="j1", match_id="m1", season=2026, video_id="v" * 11,
                     status="complete", attempt=1)
    db.add(row)
    db.commit()
    return row


def write_output(tmp_path, tracks, events):
    (tmp_path / "tracks.jsonl").write_text(
        "".join(json.dumps(t) + "\n" for t in tracks), encoding="utf-8")
    (tmp_path / "events.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    (tmp_path / "result.json").write_text("{}", encoding="utf-8")
    return {"tracks_path": str(tmp_path / "tracks.jsonl"),
            "events_path": str(tmp_path / "events.jsonl"),
            "result_path": str(tmp_path / "result.json")}


TRACKS = [{"track_id": 0, "team": 254, "alliance": "red", "team_confidence": 0.9,
           "boxes": [{"t": 1.0, "x": 0.1, "y": 0.1, "w": 0.05, "h": 0.05}], "gaps": []},
          {"track_id": 1, "team": None, "alliance": "blue", "team_confidence": None,
           "boxes": [{"t": 1.0, "x": 0.5, "y": 0.5, "w": 0.05, "h": 0.05}], "gaps": []}]

EVENTS = [{"event_id": "e1", "schema_version": 3, "job_id": "j1", "match_id": "m1",
           "team": None, "track_id": None, "t_seconds": 0.0, "phase": "auto",
           "event_type": "match_start", "confidence": 1.0, "field_x": None, "field_y": None,
           "goal": None, "source": "model"}]


def test_a_first_import_loads_everything(db, job, tmp_path):
    main.import_results(db, job, write_output(tmp_path, TRACKS, EVENTS))
    assert db.query(models.Track).count() == 2
    assert db.query(models.Event).count() == 1


def test_re_importing_replaces_tracks_rather_than_doubling_them(db, job, tmp_path):
    # The bug this exists for: a real re-import turned 45 tracks into 90, and every per-team
    # count with them. Nothing downstream could tell.
    results = write_output(tmp_path, TRACKS, EVENTS)
    main.import_results(db, job, results)
    main.import_results(db, job, results)
    assert db.query(models.Track).count() == 2


def test_re_importing_does_not_duplicate_events(db, job, tmp_path):
    results = write_output(tmp_path, TRACKS, EVENTS)
    main.import_results(db, job, results)
    main.import_results(db, job, results)
    assert db.query(models.Event).count() == 1


def test_a_second_run_replaces_old_events(db, job, tmp_path):
    main.import_results(db, job, write_output(tmp_path, TRACKS, EVENTS))
    replacement = [dict(EVENTS[0], event_id="e2", event_type="match_end")]
    main.import_results(db, job, write_output(tmp_path, TRACKS, replacement))
    rows = db.query(models.Event).all()
    assert [(row.event_id, row.event_type) for row in rows] == [("e2", "match_end")]


def test_a_second_run_supersedes_the_first(db, job, tmp_path):
    # A better detector re-run must not leave the old attribution behind next to the new one.
    main.import_results(db, job, write_output(tmp_path, TRACKS, EVENTS))
    improved = [dict(TRACKS[0], team=1678), TRACKS[1]]
    main.import_results(db, job, write_output(tmp_path, improved, EVENTS))
    teams = {t.team for t in db.query(models.Track).all()}
    assert teams == {1678, None}


def test_another_job_s_tracks_are_untouched(db, job, tmp_path):
    # Tracks are cleared by job, never by match: two camera angles of one match are two jobs, and
    # re-running one must not wipe the other.
    other = models.Job(job_id="j2", match_id="m1", season=2026, video_id="w" * 11,
                       status="complete", attempt=1)
    db.add(other)
    db.commit()
    results = write_output(tmp_path, TRACKS, EVENTS)
    main.import_results(db, other, results)
    main.import_results(db, job, results)
    main.import_results(db, job, results)
    assert db.query(models.Track).filter(models.Track.job_id == "j2").count() == 2
    assert db.query(models.Track).count() == 4
