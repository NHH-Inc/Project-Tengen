import csv

import pytest

from desktop.core import AnalysisOptions, FIXTURE, Review, analyze, box_at, read_json


def sample(t, x):
    return dict(t=t, x=x, y=0.2, w=0.1, h=0.1)


def test_interpolation_never_bridges_camera_cuts():
    track = dict(boxes=[sample(0, 0), sample(1, 0.2), sample(5, 0.8)],
                 gaps=[dict(start=2, end=4, reason="shot_change")])
    assert box_at(track, 0.5)["x"] == pytest.approx(0.1)
    assert box_at(track, -1) is None
    assert box_at(track, 3) is None
    assert box_at(track, 1.5) is None
    assert box_at(track, 4.5) is None
    assert box_at(track, 5.1, hold=0.2)["x"] == 0.8
    assert box_at(track, 5.3, hold=0.2) is None


def test_review_corrections_persist_propagate_and_undo_without_changing_raw(tmp_path):
    correction_path = tmp_path / "corrections.json"
    review = Review(FIXTURE, correction_path)
    original = (FIXTURE / "tracks.jsonl").read_bytes()
    target = review.raw_tracks[0]["track_id"]
    review.correct("track", target, {"team": 1234})
    reopened = Review(FIXTURE, correction_path)
    tracks, events = reopened.rows()
    assert next(t for t in tracks if t["track_id"] == target)["team"] == 1234
    assert all(e["team"] == 1234 for e in events if e.get("track_id") == target)
    assert reopened.rows(raw=True)[0] == review.raw_tracks
    assert (FIXTURE / "tracks.jsonl").read_bytes() == original
    reopened.undo()
    assert reopened.rows()[0] == review.raw_tracks


def test_event_correction_export_and_undo(tmp_path):
    review = Review(FIXTURE, tmp_path / "corrections.json")
    event = dict(review.raw_events[3], event_id="manual-test", source="manual")
    review.correct("event", "manual-test", event, "create")
    review.correct("event", "manual-test", {"team": 5678})
    review.export(tmp_path / "events.csv")
    with (tmp_path / "events.csv").open() as file:
        rows = list(csv.DictReader(file))
    assert next(e for e in rows if e["event_id"] == "manual-test")["team"] == "5678"
    review.correct("event", "manual-test", {}, "delete")
    assert all(e["event_id"] != "manual-test" for e in review.rows()[1])
    review.undo()
    assert any(e["event_id"] == "manual-test" for e in review.rows()[1])
    review.export(tmp_path / "raw.json", raw=True)
    assert read_json(tmp_path / "raw.json")["events"] == review.raw_events


def test_analysis_fails_clearly_for_missing_model(tmp_path):
    with pytest.raises(ValueError, match="Robot model not found"):
        analyze(AnalysisOptions(video=str(FIXTURE / "segment.mp4"), model=str(tmp_path / "absent.pt")), runs=tmp_path / "runs")
    assert not (tmp_path / "runs").exists()


def test_failed_analysis_is_persisted(tmp_path, monkeypatch):
    from ingest.yolo_orchestrator import YoloAnalysisOrchestrator
    def fail(*args, **kwargs):
        raise RuntimeError("Deliberate detector failure")
    monkeypatch.setattr(YoloAnalysisOrchestrator, "run_job", fail)
    model = tmp_path / "model.pt"
    model.write_bytes(b"placeholder")
    with pytest.raises(RuntimeError, match="Deliberate"):
        analyze(AnalysisOptions(video=str(FIXTURE / "segment.mp4"), model=str(model)), runs=tmp_path / "runs")
    job = read_json(next((tmp_path / "runs").glob("*/job.json")))
    assert job["status"] == "failed"
    assert "Deliberate" in job["error"]
