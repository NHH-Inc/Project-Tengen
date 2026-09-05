"""The eval has to be honest about frames nobody has labelled, and about where it is blind.

The trap this guards is specific: a pack ships with the detector's own proposals sitting in
`labels/`, so the obvious implementation scores a model against boxes it drew itself. v2 measured
0.986 that way, which is self-agreement wearing recall's clothes.
"""

import json
from collections import Counter

import pytest

from ingest.collection.viewpoint_eval import (
    evaluate,
    report,
    reviewed_stems,
    segment_of,
    truth_counts,
)


def test_segment_strips_only_the_timestamp():
    assert segment_of("1EmXuPKOUkE_00000_00215_00021441") == "1EmXuPKOUkE_00000_00215"


def test_segment_survives_an_id_containing_underscores():
    assert segment_of("a_b_c_00000_00215_00021441") == "a_b_c_00000_00215"


def _pack(tmp_path, images, labels=None, manifest=None):
    import cv2
    import numpy as np

    (tmp_path / "images").mkdir(parents=True, exist_ok=True)
    (tmp_path / "labels").mkdir(parents=True, exist_ok=True)
    for stem in images:
        cv2.imwrite(str(tmp_path / "images" / f"{stem}.jpg"),
                    np.zeros((32, 32, 3), dtype=np.uint8))
    for stem, text in (labels or {}).items():
        (tmp_path / "labels" / f"{stem}.txt").write_text(text, encoding="utf-8")
    if manifest is not None:
        (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path


def _manifest(stems, reviewed=False):
    return {"items": [{"image": f"images/{s}.jpg",
                       "status": "reviewed" if reviewed else "proposed",
                       "human_review_required": not reviewed} for s in stems]}


class FakeDetector:
    """Returns a scripted count per frame stem, so the arithmetic is checkable."""

    def __init__(self, counts):
        self.counts = counts
        self.order = sorted(counts)
        self.seen = 0

    def detect(self, image):
        stem = self.order[self.seen]
        self.seen += 1
        return [{"x": 0, "y": 0, "w": 1, "h": 1}] * self.counts[stem]


def _run(tmp_path, counts, labels=None, manifest=None, expected=6, truth_dir=None):
    pack = _pack(tmp_path, sorted(counts), labels, manifest)
    return evaluate(FakeDetector(counts), pack, expected=expected, truth_dir=truth_dir)


def test_counts_and_mean(tmp_path):
    result = _run(tmp_path, {"v_0_1_001": 2, "v_0_1_002": 4})
    assert result["frames"] == 2
    assert result["mean"] == 3.0
    assert result["histogram"] == Counter({2: 1, 4: 1})


def test_blind_and_full_frames_are_counted(tmp_path):
    result = _run(tmp_path, {"v_0_1_001": 0, "v_0_1_002": 6, "v_0_1_003": 7})
    assert result["blind_frames"] == 1
    # A frame with more than expected still counts as full; extra boxes are a precision problem,
    # and this measure is deliberately only about what was found.
    assert result["full_frames"] == 2


def test_segments_are_averaged_separately(tmp_path):
    result = _run(tmp_path, {"a_0_1_001": 0, "a_0_1_002": 0, "b_0_1_001": 6})
    assert result["segments"] == {"a_0_1": 0.0, "b_0_1": 6.0}


def test_proposals_are_not_ground_truth(tmp_path):
    """The bug this exists to prevent: scoring a model against boxes it drew itself."""
    result = _run(tmp_path, {"v_0_1_001": 3},
                  labels={"v_0_1_001": "0 .5 .5 .1 .1\n" * 3},
                  manifest=_manifest(["v_0_1_001"], reviewed=False))
    assert "ceiling_recall" not in result
    assert "self-agreement" in report(result)


def test_reviewed_frames_do_count(tmp_path):
    result = _run(tmp_path, {"v_0_1_001": 3},
                  labels={"v_0_1_001": "0 .5 .5 .1 .1\n" * 6},
                  manifest=_manifest(["v_0_1_001"], reviewed=True))
    assert result["ceiling_recall"] == pytest.approx(0.5)


def test_a_pack_with_no_manifest_claims_no_truth(tmp_path):
    """Absent evidence that a human looked, assume none did."""
    result = _run(tmp_path, {"v_0_1_001": 3}, labels={"v_0_1_001": "0 .5 .5 .1 .1\n"})
    assert "ceiling_recall" not in result
    assert reviewed_stems(tmp_path) is None


def test_returned_labels_override_the_manifest(tmp_path):
    """Labellers edit label files and send them back; they do not edit the manifest."""
    returned = tmp_path / "returned"
    returned.mkdir()
    (returned / "v_0_1_001.txt").write_text("0 .5 .5 .1 .1\n" * 6, encoding="utf-8")
    result = _run(tmp_path / "pack", {"v_0_1_001": 3},
                  manifest=_manifest(["v_0_1_001"], reviewed=False), truth_dir=returned)
    assert result["labelled_frames"] == 1
    assert result["ceiling_recall"] == pytest.approx(0.5)


def test_empty_label_files_are_not_treated_as_ground_truth(tmp_path):
    """The reason merging drops empty labels: nobody looked is not the same as no robots."""
    (tmp_path / "labels").mkdir()
    (tmp_path / "labels" / "v_0_1_001.txt").write_text("\n \n", encoding="utf-8")
    assert truth_counts(tmp_path / "labels") == {}


def test_ceiling_does_not_exceed_one_when_over_detecting(tmp_path):
    """Nine boxes against six real robots is not 150% recall; the extras are false positives."""
    result = _run(tmp_path, {"v_0_1_001": 9},
                  labels={"v_0_1_001": "0 .5 .5 .1 .1\n" * 6},
                  manifest=_manifest(["v_0_1_001"], reviewed=True))
    assert result["ceiling_recall"] == 1.0


def test_empty_pack_reports_rather_than_dividing_by_zero(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    result = evaluate(FakeDetector({}), tmp_path)
    assert result == {"frames": 0}
    assert "no readable frames" in report(result)


def test_report_names_the_weakest_segments_first(tmp_path):
    result = _run(tmp_path, {"good_0_1_001": 6, "bad_0_1_001": 0})
    text = report(result)
    assert text.index("bad_0_1") < text.index("good_0_1")
