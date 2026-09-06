"""A wrong crop is silent: every box stays a valid normalized box and every robot moves.

So the geometry is pinned in both directions, and the calibration is made to abstain rather than
guess -- cropping on a seam that is not there throws away a third of the picture and nothing
raises.
"""

import pytest

from ingest.collection.view_region import (
    MIN_GAP,
    Region,
    calibrate,
    centres_of,
    crop,
    map_from_region,
    seam_from_centres,
)


def box(y, h=0.1, x=0.1, w=0.1):
    return {"x": x, "y": y, "w": w, "h": h, "confidence": 0.9}


def band(y, n=3):
    """n robots sitting at roughly one height, as one camera view would show them."""
    return [box(y + i * 0.01) for i in range(n)]


# --------------------------------------------------------------------------- Region


def test_default_region_is_the_whole_frame():
    assert Region().is_full_frame


def test_from_dict_round_trips():
    r = Region(0.0, 0.0, 1.0, 0.7)
    assert Region.from_dict(r.to_dict()) == r


def test_from_dict_of_nothing_is_the_full_frame():
    assert Region.from_dict(None).is_full_frame
    assert Region.from_dict({}).is_full_frame


def test_a_region_with_no_area_is_refused():
    with pytest.raises(ValueError, match="no area"):
        Region.from_dict({"x": 0, "y": 0, "w": 0, "h": 0.5})


def test_a_region_off_the_edge_is_refused():
    """Better to fail loudly than to crop somewhere the frame does not reach."""
    with pytest.raises(ValueError, match="outside the frame"):
        Region.from_dict({"x": 0.5, "y": 0, "w": 0.8, "h": 1.0})


# --------------------------------------------------------------------------- crop


def test_full_frame_crop_returns_the_same_array(monkeypatch):
    import numpy as np
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    assert crop(image, Region()) is image


def test_crop_takes_the_top_fraction():
    import numpy as np
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    assert crop(image, Region(0.0, 0.0, 1.0, 0.7)).shape[:2] == (70, 200)


def test_crop_never_returns_an_empty_array():
    """A degenerate region must not hand the model a zero-row image."""
    import numpy as np
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    out = crop(image, Region(0.0, 0.0, 0.001, 0.001))
    assert out.shape[0] >= 1 and out.shape[1] >= 1


# --------------------------------------------------------------------------- mapping back


def test_mapping_back_undoes_the_crop():
    region = Region(0.0, 0.0, 1.0, 0.7)
    # A robot halfway down a crop that is the top 70% is at 0.35 of the source frame.
    assert map_from_region(box(0.5, h=0.2), region)["y"] == pytest.approx(0.35)


def test_mapping_scales_height_but_not_width_for_a_vertical_crop():
    region = Region(0.0, 0.0, 1.0, 0.7)
    out = map_from_region(box(0.5, h=0.2, x=0.25, w=0.3), region)
    assert out["h"] == pytest.approx(0.14)
    assert out["w"] == pytest.approx(0.30)
    assert out["x"] == pytest.approx(0.25)


def test_mapping_offsets_by_the_regions_origin():
    out = map_from_region(box(0.0, h=0.1, x=0.0), Region(0.2, 0.3, 0.5, 0.5))
    assert out["x"] == pytest.approx(0.2)
    assert out["y"] == pytest.approx(0.3)


def test_mapping_a_full_frame_region_changes_nothing():
    b = box(0.4)
    assert map_from_region(b, Region()) == b


def test_confidence_survives_the_mapping():
    assert map_from_region(box(0.5), Region(0, 0, 1, 0.7))["confidence"] == 0.9


def test_forgetting_to_map_back_moves_a_robot_up_the_frame():
    """Pins the failure this function exists to prevent, so it cannot regress quietly."""
    region = Region(0.0, 0.0, 1.0, 0.7)
    raw, mapped = box(0.9, h=0.05), map_from_region(box(0.9, h=0.05), region)
    assert raw["y"] == pytest.approx(0.90)
    assert mapped["y"] == pytest.approx(0.63)


# --------------------------------------------------------------------------- seam


def test_two_separated_bands_give_a_seam():
    assert seam_from_centres([0.35, 0.36, 0.40, 0.75, 0.78, 0.80]) == pytest.approx(0.575)


def test_one_band_gives_no_seam():
    assert seam_from_centres([0.40, 0.42, 0.45, 0.48, 0.50]) is None


def test_a_gap_smaller_than_the_minimum_is_not_a_seam():
    """Robots near and far in one view span height; only a real void means two pictures."""
    centres = [0.40, 0.41, 0.42, 0.42 + MIN_GAP * 0.9]
    assert seam_from_centres(centres) is None


def test_too_few_boxes_abstains():
    assert seam_from_centres([0.3, 0.8]) is None


def test_two_boxes_in_the_scoreboard_are_not_a_camera_view():
    """Their midpoint with the field lands innocently mid-frame; only the shape gives it away."""
    assert seam_from_centres([0.02, 0.03, 0.60, 0.62, 0.64, 0.66]) is None


def test_two_boxes_on_a_caption_bar_are_not_a_camera_view():
    assert seam_from_centres([0.30, 0.32, 0.34, 0.36, 0.97, 0.98]) is None


def test_a_lone_box_below_the_gap_is_not_a_second_view():
    assert seam_from_centres([0.35, 0.37, 0.39, 0.41, 0.78]) is None


def test_centres_are_box_middles_not_tops():
    assert centres_of([box(0.4, h=0.2)]) == [pytest.approx(0.5)]


# --------------------------------------------------------------------------- calibrate


def _stacked_frame():
    return band(0.35) + band(0.75)


def test_a_stacked_source_is_cropped_to_the_top_panel():
    result = calibrate([_stacked_frame() for _ in range(10)])
    assert result["stacked"]
    assert result["region"].y == 0.0
    # Crops just below the seam, keeping the whole top panel.
    assert result["region"].h == pytest.approx(0.63, abs=0.05)


def test_a_single_view_source_is_left_alone():
    result = calibrate([band(0.45, n=6) for _ in range(10)])
    assert not result["stacked"]
    assert result["region"].is_full_frame


def test_a_minority_of_stacked_frames_does_not_crop():
    """One frame with everyone at the far end looks like two bands. One frame is not evidence."""
    frames = [_stacked_frame()] + [band(0.45, n=6) for _ in range(9)]
    assert not calibrate(frames)["stacked"]


def test_a_clear_majority_does_crop():
    frames = [_stacked_frame() for _ in range(7)] + [band(0.45, n=6) for _ in range(3)]
    assert calibrate(frames)["stacked"]


def test_frames_with_too_few_boxes_do_not_count_against_the_share():
    """A near-empty frame is not evidence of a single view; it is evidence of nothing."""
    frames = [_stacked_frame() for _ in range(4)] + [[box(0.4)] for _ in range(20)]
    result = calibrate(frames)
    assert result["frames_usable"] == 4
    assert result["stacked"]


def test_no_detections_at_all_says_so_rather_than_claiming_one_view():
    result = calibrate([[] for _ in range(10)])
    assert not result["stacked"]
    assert result["region"].is_full_frame
    assert "too few detections" in result["reason"]


def test_the_evidence_is_reported():
    result = calibrate([_stacked_frame() for _ in range(6)])
    assert result["frames"] == 6
    assert result["frames_stacked"] == 6
    assert "seam" in result and "seam_spread" in result
    assert "6 of 6" in result["reason"]


def test_the_seam_is_a_median_so_one_odd_frame_cannot_move_it():
    frames = [_stacked_frame() for _ in range(9)] + [band(0.31) + band(0.79)]
    assert calibrate(frames)["seam"] == pytest.approx(0.61, abs=0.02)


def test_calibrating_nothing_is_not_a_crash():
    result = calibrate([])
    assert result["region"].is_full_frame
    assert not result["stacked"]


# --------------------------------------------------------------------------- mode


def test_the_default_mode_is_filter():
    """Filtering measured better than cropping on the pack; the default follows the measurement."""
    from ingest.collection.view_region import FILTER
    assert Region().mode == FILTER


def test_an_unknown_mode_is_refused():
    with pytest.raises(ValueError, match="mode must be"):
        Region.from_dict({"x": 0, "y": 0, "w": 1, "h": 0.7, "mode": "sometimes"})


def test_mode_survives_a_round_trip():
    from ingest.collection.view_region import CROP
    r = Region(0.0, 0.0, 1.0, 0.7, CROP)
    assert Region.from_dict(r.to_dict()).mode == CROP


def test_a_box_inside_the_region_is_kept():
    assert Region(0.0, 0.0, 1.0, 0.7).contains_centre(box(0.3, h=0.1))


def test_a_box_below_the_region_is_dropped():
    assert not Region(0.0, 0.0, 1.0, 0.7).contains_centre(box(0.85, h=0.1))


def test_a_box_straddling_the_seam_goes_to_the_view_it_is_mostly_in():
    """Judged by the centre: testing edges would drop it from both panels or keep it in both."""
    region = Region(0.0, 0.0, 1.0, 0.7)
    assert region.contains_centre(box(0.60, h=0.16))       # centre 0.68, above
    assert not region.contains_centre(box(0.66, h=0.16))   # centre 0.74, below


def test_the_full_frame_region_keeps_everything():
    assert Region().contains_centre(box(0.99, h=0.01))


def test_a_low_composited_panel_is_still_a_camera_view():
    """Regression: a 0.92 ceiling called real stacked sources single-view.

    Measured on data/segments — composited lower panels put their robots between 0.87 and 0.97,
    with a centroid near 0.93, which is lower than a first guess at "too low to be a field".
    """
    centres = [0.36, 0.44, 0.48, 0.51, 0.57, 0.61, 0.88, 0.90, 0.93, 0.95, 0.97]
    assert seam_from_centres(centres) == pytest.approx(0.745, abs=0.01)


def test_scoreboard_avatars_do_not_become_the_upper_panel():
    """Several real sources detect team avatars in the alliance strip at about y=0.04."""
    assert seam_from_centres([0.04, 0.04, 0.05, 0.39, 0.41, 0.46]) is None
