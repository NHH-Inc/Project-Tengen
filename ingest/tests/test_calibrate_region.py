"""The calibration pass must abstain loudly. A wrong crop discards a third of every frame."""

import json

import pytest

from ingest.collection.calibrate_region import calibrate_source, main, sample_frames
from ingest.collection.view_region import CROP, FILTER


class FakeDetector:
    """Returns scripted boxes, one list per frame, cycling if asked for more."""

    def __init__(self, script):
        self.script = script
        self.calls = 0

    def detect(self, image):
        boxes = self.script[self.calls % len(self.script)]
        self.calls += 1
        return boxes


def box(y, h=0.1):
    return {"x": 0.1, "y": y, "w": 0.1, "h": h, "confidence": 0.9}


def band(y, n=3):
    return [box(y + i * 0.01) for i in range(n)]


def _clip(tmp_path, name="v.mp4", frames=30):
    """A real, tiny video, so the frame sampling is exercised rather than mocked."""
    import cv2
    import numpy as np

    path = tmp_path / name
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 64))
    for i in range(frames):
        writer.write(np.full((64, 64, 3), i * 5 % 255, dtype=np.uint8))
    writer.release()
    return path


def test_sampling_spreads_across_the_clip(tmp_path):
    frames = sample_frames(_clip(tmp_path), count=6)
    assert len(frames) == 6
    # Distinct frames, not the same one six times -- the clip's brightness ramps.
    assert len({int(f.mean()) for f in frames}) > 1


def test_sampling_a_missing_file_is_empty_not_a_crash(tmp_path):
    assert sample_frames(tmp_path / "nope.mp4") == []


def test_a_stacked_source_is_detected(tmp_path):
    detector = FakeDetector([band(0.35) + band(0.75)])
    result = calibrate_source(detector, _clip(tmp_path), count=8)
    assert result["stacked"]
    assert result["region"].h < 1.0


def test_a_single_view_source_keeps_the_whole_frame(tmp_path):
    detector = FakeDetector([band(0.45, n=6)])
    result = calibrate_source(detector, _clip(tmp_path), count=8)
    assert not result["stacked"]
    assert result["region"].is_full_frame


def test_an_unreadable_source_says_so(tmp_path):
    result = calibrate_source(FakeDetector([[]]), tmp_path / "missing.mp4")
    assert result["frames"] == 0
    assert "no readable frames" in result["reason"]
    assert result["region"].is_full_frame


def test_a_source_the_model_cannot_see_is_not_called_single_view(tmp_path):
    """Zero detections is absence of evidence, and must not be read as evidence of one view."""
    result = calibrate_source(FakeDetector([[]]), _clip(tmp_path), count=8)
    assert not result["stacked"]
    assert result["frames_usable"] == 0
    assert "too few detections" in result["reason"]


# --------------------------------------------------------------------------- the CLI


@pytest.fixture
def patched(monkeypatch):
    """Swap the ONNX detector for a scripted one so the CLI runs without a model file."""
    def install(script):
        import ingest.collection.calibrate_region as mod
        detector = FakeDetector(script)
        monkeypatch.setattr(mod, "OnnxDetector", lambda **kw: detector, raising=False)
        import ingest.collection.detect_runner as runner
        monkeypatch.setattr(runner, "OnnxDetector", lambda **kw: detector)
        return detector
    return install


def test_cli_writes_only_the_stacked_sources(tmp_path, patched, capsys):
    patched([band(0.35) + band(0.75)])
    segments = tmp_path / "segments"
    segments.mkdir()
    _clip(segments, "aaa_0_1.mp4")
    _clip(segments, "bbb_0_1.mp4")
    out = tmp_path / "regions.json"

    assert main(["--model", "m.onnx", "--segments", str(segments), "--out", str(out),
                 "--frames", "6"]) == 0
    written = json.loads(out.read_text(encoding="utf-8"))["regions"]
    assert set(written) == {"aaa_0_1", "bbb_0_1"}
    assert written["aaa_0_1"]["mode"] == FILTER


def test_cli_records_the_requested_mode(tmp_path, patched):
    patched([band(0.35) + band(0.75)])
    segments = tmp_path / "segments"
    segments.mkdir()
    _clip(segments, "aaa_0_1.mp4")
    out = tmp_path / "regions.json"
    main(["--model", "m.onnx", "--segments", str(segments), "--out", str(out),
          "--frames", "6", "--mode", CROP])
    assert json.loads(out.read_text(encoding="utf-8"))["regions"]["aaa_0_1"]["mode"] == CROP


def test_cli_omits_single_view_sources_entirely(tmp_path, patched):
    """An absent entry means the full frame, so writing one would be noise."""
    patched([band(0.45, n=6)])
    segments = tmp_path / "segments"
    segments.mkdir()
    _clip(segments, "aaa_0_1.mp4")
    out = tmp_path / "regions.json"
    main(["--model", "m.onnx", "--segments", str(segments), "--out", str(out), "--frames", "6"])
    assert json.loads(out.read_text(encoding="utf-8"))["regions"] == {}


def test_cli_warns_when_a_source_could_not_be_judged(tmp_path, patched, capsys):
    patched([[]])
    segments = tmp_path / "segments"
    segments.mkdir()
    _clip(segments, "aaa_0_1.mp4")
    main(["--model", "m.onnx", "--segments", str(segments), "--frames", "6"])
    out = capsys.readouterr().out
    assert "unknown" in out
    assert "could not be judged" in out


def test_cli_with_no_videos_fails_rather_than_writing_an_empty_file(tmp_path, patched):
    patched([[]])
    empty = tmp_path / "segments"
    empty.mkdir()
    out = tmp_path / "regions.json"
    assert main(["--model", "m.onnx", "--segments", str(empty), "--out", str(out)]) == 1
    assert not out.exists()
