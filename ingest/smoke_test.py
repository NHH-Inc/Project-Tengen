"""Contract E smoke test, SCHEMA_VERSION 3.

Seeds the golden fixtures straight into the database and drives every endpoint component 3
calls. No network, no yt-dlp, no analysis binary -- doc 0: "Component 2 tests the pipeline
with a stub binary that copies the fixture output."

Run from the repo root:

    ingest\\.venv\\Scripts\\python -m ingest.smoke_test        (Windows / PowerShell)
    ingest/.venv/bin/python -m ingest.smoke_test              (POSIX)
"""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "fixtures"
MAIN = FIXTURES / "2026casf_qm42"
NO_TBA = FIXTURES / "2026casf_qm43_no_tba"

# Must be set before ingest.database is imported -- the engine is built at import time.
_tmp = Path(tempfile.mkdtemp())
os.environ["DATABASE_URL"] = f"sqlite:///{(_tmp / 'smoke.db').as_posix()}"
os.environ["FRC_DATA_DIR"] = str(_tmp / "data")

# Both exporters are built at import time from the environment, and load_dotenv runs with
# override=False, so setting these empty here wins over ingest/.env.
#
# This is not tidiness. Once a real export is configured -- and one now is -- an unguarded run
# of this file POSTS the fixture rows to the live spreadsheet, because the endpoint under test
# really does export. A test suite must not write to the team's sheet, and the assertion below
# is about the unconfigured path anyway, so the only correct environment for it is one with no
# credentials at all.
for _credential in ("SHEETS_SPREADSHEET_ID", "GOOGLE_APPLICATION_CREDENTIALS",
                    "APPS_SCRIPT_URL", "APPS_SCRIPT_SECRET"):
    os.environ[_credential] = ""

from fastapi.testclient import TestClient  # noqa: E402

from ingest import database, models  # noqa: E402
from ingest import main as main_module  # noqa: E402
from ingest.main import app  # noqa: E402

# The retry endpoint schedules its worker after returning the response. This smoke test verifies
# the retry state transition, not YouTube or whichever production analyzer happens to be configured
# in ingest/.env; letting that background task run makes an offline contract test download and
# analyze a real match. Keep the documented no-network boundary explicit.
main_module.process_job = lambda *_args, **_kwargs: None

MATCH = "2026casf_qm42"
NO_TBA_MATCH = "2026casf_qm43"

passed = 0
failed = 0


def check(label, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  ok   {label}")
    else:
        failed += 1
        print(f"  FAIL {label}" + (f" -- {detail}" if detail else ""))


def jsonl(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _segment_copy() -> Path:
    """A COPY: deleting a job removes its cached segment (doc 2: media is a cache, not a
    record), and pointing at the fixture would delete the golden video."""
    dest = _tmp / "segment.mp4"
    src = MAIN / "segment.mp4"
    if src.exists():
        shutil.copy2(src, dest)
    else:
        dest.write_bytes(b"")
    return dest


def seed_dir(db, base: Path, local_path: str | None):
    job_data = json.loads((base / "job.json").read_text(encoding="utf-8"))
    db.add(
        models.Job(
            job_id=job_data["job_id"],
            match_id=job_data["match_id"],
            season=job_data["season"],
            video_id=job_data["video_id"],
            local_path=local_path,
            start_offset=job_data["start_offset"],
            duration=job_data["duration"],
            fps=job_data["fps"],
            width=job_data["width"],
            height=job_data["height"],
            status=job_data["status"],
            error_code=job_data["error_code"],
            error=job_data["error"],
            attempt=job_data["attempt"],
            alliances=job_data["alliances"],
            tba_score=job_data["tba_score"],
        )
    )
    if job_data["status"] == "failed":
        return job_data
    for row in jsonl(base / "events.jsonl"):
        db.add(models.Event(**{k: row[k] for k in (
            "event_id", "schema_version", "job_id", "match_id", "team", "track_id",
            "t_seconds", "phase", "event_type", "confidence", "field_x", "field_y",
            "goal", "source")}))
    for row in jsonl(base / "tracks.jsonl"):
        db.add(
            models.Track(
                job_id=job_data["job_id"],
                match_id=job_data["match_id"],  # from the JOB, never the track JSON
                track_id=row["track_id"],
                team=row["team"],
                alliance=row["alliance"],
                team_confidence=row["team_confidence"],
                boxes=row["boxes"],
                gaps=row["gaps"],
            )
        )
    # result.json is read off disk by the API, so put it where the orchestrator would.
    out = Path(os.environ["FRC_DATA_DIR"]) / "jobs" / job_data["job_id"]
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(base / "result.json", out / "result.json")
    return job_data


def stats_mod_goals():
    """Legal goals for the fixture's season, read from the config -- never hardcoded."""
    from ingest import stats as _stats

    return _stats.legal_goals(_stats.season_config(2026)) or set()


def main():
    db = database.SessionLocal()
    main_job = seed_dir(db, MAIN, str(_segment_copy()))
    seed_dir(db, NO_TBA, None)
    seed_dir(db, FIXTURES / "failed_download", None)
    db.commit()
    db.close()

    client = TestClient(app)
    raw_count = len(jsonl(MAIN / "events.jsonl"))
    track_count = len(jsonl(MAIN / "tracks.jsonl"))

    print("\nContract E envelopes and job records")
    r = client.get("/api/jobs")
    body = r.json()
    check("GET /api/jobs returns an object, not a bare array", isinstance(body, dict) and "jobs" in body)
    check("all three fixture jobs load", len(body["jobs"]) == 3, str(len(body.get("jobs", []))))
    job = next(j for j in body["jobs"] if j["job_id"] == main_job["job_id"])
    check("job carries season", job["season"] == 2026)
    check("job carries attempt + timestamps", job["attempt"] >= 1 and job["updated_at"].endswith("Z"))
    check("schema_version is 3", job["schema_version"] == 3)

    # The health endpoint used to hardcode its version and silently stayed at 1 through two
    # bumps. Assert against the contracts file so it cannot drift again.
    declared = int((ROOT / "contracts" / "SCHEMA_VERSION").read_text(encoding="utf-8").strip())
    h = client.get("/api/health").json()
    check("health reports the contracts SCHEMA_VERSION",
          h.get("schema_version") == declared, f"{h.get('schema_version')} != {declared}")
    check("every serialized record matches it too", job["schema_version"] == declared)

    failed_job = next(j for j in body["jobs"] if j["status"] == "failed")
    check("failed job carries an error_code", failed_job["error_code"] == "rate_limited",
          str(failed_job.get("error_code")))

    print("\nEvents and tracks")
    r = client.get(f"/api/matches/{MATCH}/events")
    check("events envelope", "events" in r.json() and len(r.json()["events"]) == raw_count)
    ev = r.json()["events"][0]
    check("event carries corrected + correction_id", "corrected" in ev and "correction_id" in ev)

    r = client.get(f"/api/matches/{MATCH}/tracks")
    tr = r.json()
    check("tracks envelope carries box_sample_rate", tr.get("box_sample_rate") == 5.0,
          str(tr.get("box_sample_rate")))
    check("all tracks returned", len(tr["tracks"]) == track_count, str(len(tr.get("tracks", []))))
    check("track carries team_confidence", "team_confidence" in tr["tracks"][0])
    gapped = [t for t in tr["tracks"] if t.get("gaps")]
    check("gaps survive to the API", len(gapped) > 0 and gapped[0]["gaps"][0]["reason"] in
          {"shot_change", "occlusion", "out_of_frame", "detection_lost"})

    r = client.get(f"/api/jobs/{main_job['job_id']}/result")
    check("GET /api/jobs/:id/result", r.status_code == 200 and r.json()["box_sample_rate"] == 5.0)

    print("\nGoal field (v3)")
    evs = client.get(f"/api/matches/{MATCH}/events").json()["events"]
    shots = [e for e in evs if e["event_type"] in ("shot_attempt", "shot_made")]
    check("every event carries a goal key", all("goal" in e for e in evs))
    check("some shots name a goal", any(e["goal"] for e in shots))
    check("some shots have an unknown goal", any(e["goal"] is None for e in shots))
    legal = set(stats_mod_goals())
    check("every goal is legal for the season",
          all(e["goal"] in legal for e in shots if e["goal"]),
          str({e["goal"] for e in shots}))
    check("non-shots never carry a goal",
          all(e["goal"] is None for e in evs
              if e["event_type"] not in ("shot_attempt", "shot_made")))

    # Goal names are season-scoped, so validation reads the season config, not a fixed enum.
    a_shot = shots[0]
    r = client.patch(f"/api/events/{a_shot['event_id']}", json={"goal": "trench"})
    check("a goal not in this season's config is rejected", r.status_code == 400,
          str(r.status_code))
    r = client.patch(f"/api/events/{a_shot['event_id']}", json={"goal": "low"})
    check("a legal goal is accepted", r.status_code == 200 and r.json()["goal"] == "low",
          r.text[:140])
    raw_shot = next(
        e for e in client.get(f"/api/matches/{MATCH}/events?raw=true").json()["events"]
        if e["event_id"] == a_shot["event_id"]
    )
    check("correcting a goal leaves raw untouched", raw_shot["goal"] == a_shot["goal"],
          f"raw says {raw_shot['goal']}, was {a_shot['goal']}")

    a_reload = next(e for e in evs if e["event_type"] == "reload")
    r = client.patch(f"/api/events/{a_reload['event_id']}", json={"goal": "high"})
    check("a goal on a non-shot is rejected", r.status_code == 400, str(r.status_code))

    print("\nAccuracy")
    r = client.get(f"/api/matches/{MATCH}/accuracy")
    acc = r.json()
    check("accuracy reports tba_available", acc.get("tba_available") is True)
    r = client.get(f"/api/matches/{NO_TBA_MATCH}/accuracy")
    check("no-TBA match reports tba_available false", r.json().get("tba_available") is False)
    check("no-TBA match has null delta", r.json().get("delta") is None)

    print("\nCorrections never overwrite raw output")
    target = client.get(f"/api/matches/{MATCH}/events").json()["events"][1]
    original_team = target["team"]
    new_team = 1678 if original_team != 1678 else 254

    r = client.patch(f"/api/events/{target['event_id']}", json={"team": new_team})
    check("PATCH event", r.status_code == 200 and r.json()["team"] == new_team, r.text[:120])
    check("patched event is flagged corrected", r.json().get("corrected") is True)

    raw_rows = {e["event_id"]: e for e in client.get(f"/api/matches/{MATCH}/events?raw=true").json()["events"]}
    check("?raw=true still reports the ORIGINAL team",
          raw_rows[target["event_id"]]["team"] == original_team,
          f"raw says {raw_rows[target['event_id']]['team']}, expected {original_team}")

    print("\nTrack-level correction (doc 3's primary path)")
    tracks = client.get(f"/api/matches/{MATCH}/tracks").json()["tracks"]
    victim = next(t for t in tracks if t["team"] is not None)
    before_events = [
        e for e in client.get(f"/api/matches/{MATCH}/events").json()["events"]
        if e["track_id"] == victim["track_id"]
    ]
    r = client.patch(
        f"/api/jobs/{main_job['job_id']}/tracks/{victim['track_id']}", json={"team": 9999}
    )
    check("PATCH track", r.status_code == 200, r.text[:140])
    after_tracks = client.get(f"/api/matches/{MATCH}/tracks").json()["tracks"]
    moved = next(t for t in after_tracks if t["track_id"] == victim["track_id"])
    check("track itself is re-attributed", moved["team"] == 9999, str(moved["team"]))
    after_events = [
        e for e in client.get(f"/api/matches/{MATCH}/events").json()["events"]
        if e["track_id"] == victim["track_id"]
    ]
    check(f"all {len(before_events)} events on the track moved in one action",
          len(after_events) == len(before_events)
          and all(e["team"] == 9999 for e in after_events),
          str({e["team"] for e in after_events}))
    raw_after = [
        e for e in client.get(f"/api/matches/{MATCH}/events?raw=true").json()["events"]
        if e["track_id"] == victim["track_id"]
    ]
    check("raw output is untouched by the track correction",
          all(e["team"] == victim["team"] for e in raw_after))
    raw_tracks = client.get(f"/api/matches/{MATCH}/tracks?raw=true").json()["tracks"]
    check("raw=true on /tracks returns the original attribution",
          next(t for t in raw_tracks if t["track_id"] == victim["track_id"])["team"] == victim["team"])

    print("\nCorrection listing and undo")
    r = client.get(f"/api/matches/{MATCH}/corrections")
    corrections = r.json()["corrections"]
    # Count is not asserted: earlier checks in this run legitimately add corrections.
    check("GET corrections envelope",
          isinstance(corrections, list) and len(corrections) >= 2,
          str(len(corrections) if isinstance(corrections, list) else corrections))
    check("every correction carries a scope",
          all(c["scope"] in ("event", "track") for c in corrections))
    track_corr = next(c for c in corrections if c["scope"] == "track")
    check("track correction carries job_id + target_id",
          track_corr["job_id"] == main_job["job_id"] and track_corr["target_id"] == str(victim["track_id"]))

    r = client.delete(f"/api/corrections/{track_corr['correction_id']}")
    check("DELETE /api/corrections/:id", r.status_code == 204, str(r.status_code))
    restored = client.get(f"/api/matches/{MATCH}/tracks").json()["tracks"]
    check("undo restores the original attribution",
          next(t for t in restored if t["track_id"] == victim["track_id"])["team"] == victim["team"])

    print("\nManual events, deletes, enums")
    r = client.post("/api/events", json={
        "job_id": main_job["job_id"], "match_id": MATCH, "team": 254, "track_id": 7,
        "t_seconds": 63.5, "phase": "teleop", "event_type": "shot_made",
        "confidence": 1.0, "field_x": None, "field_y": None, "source": "manual",
    })
    check("POST /api/events", r.status_code == 200, r.text[:160])
    check("manual event id is a UUID", len(r.json()["event_id"]) == 36)
    check("manual event carries the current schema_version",
          r.json().get("schema_version") == declared, str(r.json())[:160])
    check("manual event carries its correction id",
          bool(r.json().get("correction_id")), str(r.json())[:160])
    check("manual event is NOT in raw output",
          len(client.get(f"/api/matches/{MATCH}/events?raw=true").json()["events"]) == raw_count)

    r = client.post("/api/events", json={"match_id": MATCH, "phase": "halftime"})
    check("unknown enum value is rejected", r.status_code == 400, str(r.status_code))
    check("error response carries error_code + error",
          "error" in r.json(), str(r.json())[:100])

    r = client.delete(f"/api/events/{target['event_id']}")
    check("DELETE /api/events/:id", r.status_code == 204, str(r.status_code))
    check("deleted event survives in raw",
          target["event_id"] in {e["event_id"] for e in
                                 client.get(f"/api/matches/{MATCH}/events?raw=true").json()["events"]})

    print("\nStats, export, retry, media")
    s = client.get("/api/teams/254/stats").json()
    check("GET team stats has the contract fields",
          all(k in s for k in ("cycles", "avg_cycle_seconds", "shot_accuracy",
                               "avg_shot_interval_seconds", "low_confidence_events")),
          str(sorted(s))[:140])
    check("cycle time computed", s["avg_cycle_seconds"] is not None)

    # Without credentials the export must refuse rather than report a write that never
    # happened. Doc 3 treats the spreadsheet URL as required, and there is no URL to give.
    r = client.post("/api/export/sheets", json={"match_ids": [MATCH], "mode": "aggregate"})
    check("unconfigured export returns 503, not a fake success", r.status_code == 503,
          str(r.status_code))
    r = client.post("/api/export/sheets", json={"match_ids": [], "mode": "aggregate"})
    check("export with no matches is rejected", r.status_code == 400)

    # A match whose events belong to no team produces no rows. Reporting 200 for that is the
    # same lie as reporting a write with no credentials -- and it is what the first real
    # end-to-end run got: 45 tracks, 11 attributed, rows_written 0, status 200.
    r = client.post("/api/export/sheets",
                    json={"match_ids": [NO_TBA_MATCH], "mode": "aggregate"})
    check("an export with no team-attributed events is rejected, not called a success",
          r.status_code == 422, str(r.status_code))
    # Errors leave this API as {error_code, error}, not FastAPI's raw {detail}.
    check("and it says why", "attributed to a team" in r.json().get("error", ""),
          str(r.json())[:120])

    print("\nSheets row building (pure, no API)")
    from ingest import sheets as sheets_mod
    from ingest import stats as stats_mod

    events = client.get(f"/api/matches/{MATCH}/events").json()["events"]
    cfg = stats_mod.season_config(2026)
    teams = sorted({e["team"] for e in events if e.get("team") is not None})
    stat_rows = [stats_mod.team_stats(t, events, cfg, "2026casf", 1) for t in teams]

    headers, rows = sheets_mod.build_rows("aggregate", [(MATCH, events, stat_rows)])
    check("aggregate mode is one row per team per match", len(rows) == len(teams),
          f"{len(rows)} rows for {len(teams)} teams")
    check("aggregate row key is match|team", rows[0][0] == f"{MATCH}|{teams[0]}", str(rows[0][0]))
    check("row width matches the header", all(len(r) == len(headers) for r in rows))

    raw_headers, raw_rows = sheets_mod.build_rows("raw", [(MATCH, events, stat_rows)])
    check("raw mode is one row per event", len(raw_rows) == len(events))
    check("raw row key is the event_id", raw_rows[0][0] == events[0]["event_id"])
    check("raw row width matches its header", all(len(r) == len(raw_headers) for r in raw_rows))
    # Idempotence rests on these being unique and stable.
    check("every row key is unique", len({r[0] for r in raw_rows}) == len(raw_rows))

    print("\nTBA client (injected fetch, no network)")
    from ingest import tba as tba_mod

    sample = {
        "key": MATCH,
        "alliances": {
            "red": {"score": 91, "team_keys": ["frc254", "frc1678", "frc971"]},
            "blue": {"score": 84, "team_keys": ["frc118", "frc148", "frc2056"]},
        },
        "videos": [{"type": "youtube", "key": "dQw4w9WgXcQ?t=120"}],
    }
    tc = tba_mod.TBAClient(api_key="test", fetch=lambda path: sample)
    alliances, score = tc.alliances_and_score(MATCH)
    check("frc prefix is stripped at the ingest boundary",
          alliances == {"red": [254, 1678, 971], "blue": [118, 148, 2056]}, str(alliances))
    check("tba_score comes through", score == {"red": 91, "blue": 84}, str(score))

    # An unplayed match reports -1; that is not a score.
    unplayed = {"alliances": {
        "red": {"score": -1, "team_keys": ["frc1", "frc2", "frc3"]},
        "blue": {"score": -1, "team_keys": ["frc4", "frc5", "frc6"]}}}
    tc2 = tba_mod.TBAClient(api_key="test", fetch=lambda path: unplayed)
    _, unplayed_score = tc2.alliances_and_score(MATCH)
    check("an unplayed match has no score", unplayed_score is None, str(unplayed_score))

    # A partial alliance is worse than none: component 1 uses it for elimination.
    partial = {"alliances": {
        "red": {"score": 10, "team_keys": ["frc1", "frc2"]},
        "blue": {"score": 9, "team_keys": ["frc4", "frc5", "frc6"]}}}
    tc3 = tba_mod.TBAClient(api_key="test", fetch=lambda path: partial)
    partial_alliances, _ = tc3.alliances_and_score(MATCH)
    check("a partial alliance is rejected", partial_alliances is None, str(partial_alliances))

    check("no API key means no calls and no crash",
          tba_mod.TBAClient(api_key="").alliances_and_score(MATCH) == (None, None))
    check("event_key_of splits the match key",
          tba_mod.event_key_of("2026casf_qm42") == "2026casf")

    tc4 = tba_mod.TBAClient(api_key="test", fetch=lambda path: [sample])
    check("video id resolves to a match key despite a &t= suffix",
          tc4.find_match_for_video("dQw4w9WgXcQ", "2026casf") == MATCH)

    r = client.post(f"/api/jobs/{failed_job['job_id']}/retry")
    check("POST retry reuses the job id", r.json()["job_id"] == failed_job["job_id"])
    check("retry increments attempt", r.json()["attempt"] == failed_job["attempt"] + 1,
          str(r.json()["attempt"]))
    check("retry clears error_code", r.json()["error_code"] is None)

    r = client.get(f"/api/video/{main_job['job_id']}")
    check("GET /api/video/:job_id", r.status_code == 200, str(r.status_code))

    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
