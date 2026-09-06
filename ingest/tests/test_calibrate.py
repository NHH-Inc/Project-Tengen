"""Rules that decide whether an auto-calibration may be trusted.

Every one of these guards a failure that produces confident nonsense rather than an error: a
homography fitted to a moving camera, to two cameras at once, or to tags at different heights
reports wrong field positions forever and never says so.
"""

from ingest.collection.calibrate import TagSighting, steady_tags


def sighting(tag_id, points):
    s = TagSighting(tag_id)
    for x, y in points:
        s.xs.append(x)
        s.ys.append(y)
    return s


class TestDrift:
    def test_a_static_camera_gives_almost_no_drift(self):
        # Real numbers from a real match: every usable tag came in under one pixel across 40
        # frames spanning the whole clip.
        s = sighting(2, [(597.7, 439.7), (597.9, 439.5), (597.5, 440.0), (598.0, 439.6)])
        assert s.drift() < 1.0

    def test_a_panning_camera_shows_up_as_drift(self):
        s = sighting(2, [(100.0, 400.0), (140.0, 402.0), (180.0, 404.0)])
        assert s.drift() > 12.0

    def test_the_median_resists_a_single_bad_detection(self):
        # One frame where the detector latched onto something else must not drag the point.
        s = sighting(2, [(600.0, 440.0), (601.0, 440.0), (599.0, 440.0), (1500.0, 900.0)])
        x, y = s.median()
        assert 599.0 <= x <= 601.0
        assert y == 440.0

    def test_one_sighting_has_no_drift_to_measure(self):
        assert sighting(2, [(10.0, 10.0)]).drift() == 0.0


class TestSteadyTags:
    def test_still_tags_are_kept(self):
        tags = {2: sighting(2, [(600.0, 440.0)] * 5)}
        kept, notes = steady_tags(tags)
        assert kept == {2: (600.0, 440.0)}
        assert notes == []

    def test_a_moving_tag_is_dropped_and_the_reason_recorded(self):
        # Silently averaging this would invent a point the camera never saw. The reason has to
        # reach the operator, because "camera is not static" changes what they should do next.
        tags = {2: sighting(2, [(100.0, 400.0), (300.0, 400.0), (500.0, 400.0)])}
        kept, notes = steady_tags(tags)
        assert kept == {}
        assert any("not static" in n for n in notes)

    def test_a_tag_seen_once_or_twice_is_a_coincidence(self):
        tags = {7: sighting(7, [(600.0, 440.0), (600.0, 440.0)])}
        kept, notes = steady_tags(tags)
        assert kept == {}
        assert any("sightings" in n for n in notes)

    def test_good_and_bad_tags_are_separated_not_all_or_nothing(self):
        tags = {
            2: sighting(2, [(600.0, 440.0)] * 5),
            9: sighting(9, [(100.0, 400.0), (400.0, 400.0), (800.0, 400.0)]),
        }
        kept, notes = steady_tags(tags)
        assert set(kept) == {2}
        assert len(notes) == 1

    def test_nothing_seen_is_not_a_crash(self):
        assert steady_tags({}) == ({}, [])


# --------------------------------------------------------------------------- carpet mode
#
# Measured across all 60 sources: neither tag-based method can calibrate this corpus. `pose` is
# ill-conditioned because every 2026 tag sits within 0.57 m of one plane while the field spans
# 16.5 m, and `plane` needs four non-collinear tags at one height, which no steady camera sees.
# Carpet mode is the remaining path -- hand-marked points, no tags, no focal length.


def _square(tmp_path):
    """A clean quadrilateral: the well-posed case, and the one a human is told to aim for."""
    return [
        {"image": [100, 900], "field": [0.0, 0.0]},
        {"image": [1800, 900], "field": [54.0, 0.0]},
        {"image": [1500, 500], "field": [54.0, 26.6]},
        {"image": [400, 500], "field": [0.0, 26.6]},
    ]


def test_carpet_mode_ignores_tags_entirely(monkeypatch, tmp_path):
    """No tags, no FOV, no camera pose -- only the marked points."""
    from ingest.collection import calibrate as mod

    monkeypatch.setattr(mod, "gather_sightings", lambda *a, **k: ({}, 40))
    result = mod.calibrate(tmp_path / "clip.mp4", extra_points=_square(tmp_path), method="carpet")
    assert result["mapping_source"] == "carpet_marked"
    assert result["plane_height_ft"] == 0.0
    assert result["point_count"] == 4
    assert result["solution"] is not None


def test_carpet_mode_maps_to_the_carpet_not_a_tag_plane(monkeypatch, tmp_path):
    """The whole point: a robot's footprint is on the carpet, so the mapping must be too."""
    from ingest.collection import calibrate as mod

    monkeypatch.setattr(mod, "gather_sightings", lambda *a, **k: ({}, 40))
    result = mod.calibrate(tmp_path / "clip.mp4", extra_points=_square(tmp_path), method="carpet")
    assert result["plane_height_ft"] == 0.0


def test_four_marked_points_cannot_be_checked(monkeypatch, tmp_path):
    """Any four fit exactly, so redundancy is false and the error means nothing yet."""
    from ingest.collection import calibrate as mod

    monkeypatch.setattr(mod, "gather_sightings", lambda *a, **k: ({}, 40))
    result = mod.calibrate(tmp_path / "clip.mp4", extra_points=_square(tmp_path), method="carpet")
    assert result["solution"]["has_redundancy"] is False


def test_a_fifth_point_makes_the_error_mean_something(monkeypatch, tmp_path):
    from ingest.collection import calibrate as mod

    monkeypatch.setattr(mod, "gather_sightings", lambda *a, **k: ({}, 40))
    points = _square(tmp_path) + [{"image": [950, 700], "field": [27.0, 13.3]}]
    result = mod.calibrate(tmp_path / "clip.mp4", extra_points=points, method="carpet")
    assert result["point_count"] == 5
    assert result["solution"]["has_redundancy"] is True


def test_too_few_marked_points_refuses_rather_than_fitting(monkeypatch, tmp_path):
    from ingest.collection import calibrate as mod

    monkeypatch.setattr(mod, "gather_sightings", lambda *a, **k: ({}, 40))
    result = mod.calibrate(tmp_path / "clip.mp4", extra_points=_square(tmp_path)[:3],
                           method="carpet")
    assert result["point_count"] == 3
    assert result["solution"] is None


def test_collinear_marked_points_are_refused(monkeypatch, tmp_path):
    """Four points along one edge define no plane, and must not silently produce a matrix."""
    from ingest.collection import calibrate as mod

    monkeypatch.setattr(mod, "gather_sightings", lambda *a, **k: ({}, 40))
    line = [{"image": [100 + 200 * i, 900], "field": [float(13 * i), 0.0]} for i in range(4)]
    result = mod.calibrate(tmp_path / "clip.mp4", extra_points=line, method="carpet")
    assert result["solution"] is None or not result["solution"]["trustworthy"]


def test_carpet_mode_still_reports_what_tags_were_seen(monkeypatch, tmp_path):
    """Read-only, for the operator's benefit -- they take no part in the fit."""
    from ingest.collection import calibrate as mod

    monkeypatch.setattr(mod, "gather_sightings",
                        lambda *a, **k: ({7: mod.TagSighting(7, [10.0] * 5, [20.0] * 5)}, 40))
    result = mod.calibrate(tmp_path / "clip.mp4", extra_points=_square(tmp_path), method="carpet")
    assert result["tags_detected"] == [7]
    assert result["point_count"] == 4      # the tag did not join the fit
