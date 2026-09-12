"""Independent geometry, resized broadcasts, tag outliers and real entry regressions."""
import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from ingest.collection.apriltag_layout import load_layout
from ingest.collection.calibrate import TagSighting, fit_tag_corners
from ingest.collection.homography import solve_broadcast_corners
from training.auto_goals import project_goals, configure_auto_goals
from training.ball_scouting import BallShotAnalyzer, BallScoutingConfig, load_ball_scouting_config
from ingest.tests.test_goal_scoring import basket, feed, point, SIZE
from training.goal_scoring import GoalEntryCounter

LAYOUT = Path("contracts/fields/2026-apriltags.json")


def camera():
    return dict(trustworthy=True, point_count=32, image_size=[1920, 1080],
        mapping_source="carpet_tag_corners", tags_used=[5, 6, 8, 13, 14, 17, 18, 27],
        pose=dict(camera_matrix=[[1250, 0, 960], [0, 850, 375.3], [0, 0, 1]],
                  rvec=[2.0, -.01, .01], tvec=[-27, 4, 21]))


def observations(outlier=False):
    layout, cal = load_layout(LAYOUT), camera()
    p = cal["pose"]
    sightings = {}
    for i in cal["tags_used"]:
        pixels, _ = cv2.projectPoints(layout.tags[i].corners_ft(), np.array(p["rvec"]),
                np.array(p["tvec"], float), np.array(p["camera_matrix"], float), None)
        pixels = pixels.reshape(4, 2)
        if outlier and i == 6:
            pixels += [25, -15]
        x, y = pixels.mean(axis=0)
        sightings[i] = TagSighting(i, [x] * 4, [y] * 4, [pixels.tolist()] * 4)
    return layout, cal, sightings


def test_corner_solve_recovers_squeezed_video_and_elevated_rim():
    layout, original, sightings = observations()
    fit = fit_tag_corners(sightings, sorted(sightings), layout, (1920, 1080), (.035, .66))
    assert fit is not None and fit[0].trustworthy
    assert fit[1]["focal_aspect_ratio"] == pytest.approx(850 / 1250, abs=.001)
    recovered = dict(original, pose=fit[1])
    goals_a = project_goals(original, (1920, 1080), (.02, .035, .98, .66))
    goals_b = project_goals(recovered, (1920, 1080), (.02, .035, .98, .66))
    assert len(goals_b) == 2
    for a, b in zip(goals_a, goals_b):
        assert np.max(np.abs(np.array(a["polygon"]) - b["polygon"])) < 1e-4


def test_whole_inconsistent_tag_is_rejected_and_reported():
    layout, _, sightings = observations(outlier=True)
    fit = fit_tag_corners(sightings, sorted(sightings), layout, (1920, 1080), (.035, .66))
    assert fit is not None and fit[0].trustworthy
    assert 6 not in fit[2]
    assert fit[1]["rejected_tags"][0]["tag_id"] == 6


def test_goal_projection_respects_crop_resolution_and_requires_pose():
    cal = camera()
    full = project_goals(cal, (1920, 1080), (0, 0, 1, 1))
    half = project_goals(cal, (960, 540), (0, 0, 1, 1))
    assert np.allclose(full[0]["polygon"], half[0]["polygon"])
    cropped = project_goals(cal, (1920, 1080), (.02, .035, .98, .66))
    assert np.allclose(np.array(cropped[0]["polygon"]) * [1844, 675] + [38, 38],
                       np.array(full[0]["polygon"]) * [1920, 1080])
    assert project_goals(dict(cal, trustworthy=False), (1920, 1080), (0, 0, 1, 1)) == []
    assert project_goals(dict(trustworthy=True, matrix=np.eye(3).tolist()),
                         (1920, 1080), (0, 0, 1, 1)) == []


def test_generated_configuration_loads_and_manual_regions_are_preserved(tmp_path):
    cal = tmp_path / "camera.json"
    cal.write_text(json.dumps(camera()))
    source = tmp_path / "source.json"
    source.write_text(json.dumps(dict(schema_version=1, goals=[])))
    output = tmp_path / "effective.json"
    path = configure_auto_goals(source, cal, (1920, 1080), (.02, .035, .98, .66), output)
    config = load_ball_scouting_config(path)
    assert config.automatic_goals and len(config.goals) == 2
    assert all(g.expected_ball_radius > 0 for g in config.goals)
    assert configure_auto_goals(output, cal, (1920, 1080), (0, 0, 1, 1), source) == output


def test_late_visible_approach_needs_observed_depth_and_rejects_tiny_fragments():
    goal = basket(allow_partial_approach=True)
    assert len(feed(GoalEntryCounter([goal]), [(100, 45), (100, 55), (100, 65)])) == 1
    assert feed(GoalEntryCounter([goal]), [(100, 62), (100, 70), (100, 80)]) == []
    assert feed(GoalEntryCounter([replace(goal, expected_ball_radius=.04)]),
                [(100, 45), (100, 55), (100, 65)]) == []
    assert feed(GoalEntryCounter([goal]), [(100, y) for y in
        [55, 58, 53, 52, 52, 52, 52, 55, 61, 65]]) == []


def test_automatic_regions_pause_on_scene_change():
    config = BallScoutingConfig(goals=(basket(),), automatic_goals=True)
    analyzer = BallShotAnalyzer(config)
    analyzer.process_frame(np.zeros((100, 200, 3), np.uint8), 0, 0, [])
    analyzer.process_frame(np.full((100, 200, 3), 255, np.uint8), 1, 1 / 60, [])
    assert not analyzer.goals_valid
    assert not analyzer.goal_entries


def test_camera_guard_resumes_and_records_unavailable_interval(monkeypatch):
    responses = iter([True, False, True])
    monkeypatch.setattr('training.auto_goals.camera_matches', lambda *_: next(responses))
    analyzer = BallShotAnalyzer(BallScoutingConfig(goals=(basket(),), automatic_goals=True))
    for index in range(3):
        analyzer.process_frame(np.zeros((100, 200, 3), np.uint8), index, float(index), [])
    assert analyzer.goals_valid
    assert analyzer.goal_camera_gaps == [[1.0, 2.0]]


def test_camera_guard_requires_two_decoded_markers_at_expected_positions():
    from training.auto_goals import camera_matches
    frame = np.full((240, 480, 3), 255, np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    anchors = []
    for tag_id, x in [(5, 60), (18, 300)]:
        marker = cv2.aruco.generateImageMarker(dictionary, tag_id, 60)
        frame[80:140, x:x + 60] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        anchors.append(dict(tag_id=tag_id, corners=(np.array(
            [[x, 80], [x+60, 80], [x+60, 140], [x, 140]]) / [480, 240]).tolist()))
    assert camera_matches(frame, anchors)
    assert not camera_matches(frame, anchors[:1])
    assert not camera_matches(np.roll(frame, 25, axis=1), anchors)


def test_brief_occlusion_keeps_regions_and_retries_without_resetting_entries(monkeypatch):
    now = [0.0]
    checks = []
    def visible(*_):
        checks.append(now[0])
        return not (1 <= now[0] < 2.7)
    monkeypatch.setattr('training.auto_goals.camera_matches', visible)
    analyzer = BallShotAnalyzer(BallScoutingConfig(goals=(basket(),), automatic_goals=True))
    frame = np.zeros((100, 200, 3), np.uint8)
    for index in range(181):
        now[0] = index / 60
        analyzer.process_frame(frame, index, now[0], [])
        assert analyzer.goals_valid
    assert analyzer.goal_camera_gaps == []
    assert len([t for t in checks if 1 <= t < 2.7]) >= 8


def test_sustained_tag_loss_expires_and_a_cut_bypasses_occlusion_grace(monkeypatch):
    now = [0.0]
    monkeypatch.setattr('training.auto_goals.camera_matches', lambda *_: now[0] < .5)
    frame = np.zeros((100, 200, 3), np.uint8)
    for cut in (False, True):
        analyzer = BallShotAnalyzer(BallScoutingConfig(goals=(basket(),), automatic_goals=True))
        for index in range(211):
            now[0] = index / 60
            current = np.full_like(frame, 255) if cut and now[0] >= 1 else frame
            analyzer.process_frame(current, index, now[0], [])
            if now[0] == 1:
                assert analyzer.goals_valid == (not cut)
        assert not analyzer.goals_valid
        assert len(analyzer.goal_camera_gaps) == 1
        assert analyzer.goal_camera_gaps[0][0] == pytest.approx(1 if cut else 3.0, abs=.12)


def test_fast_ball_born_at_goal_keeps_identity_without_any_robot_launch():
    analyzer = BallShotAnalyzer(BallScoutingConfig(goals=(basket(),)))
    for index, y in enumerate([190, 217, 244, 271, 298, 325, 352]):
        frame = np.zeros((500, 1000, 3), np.uint8)
        cv2.circle(frame, (500, y), 6, (0, 255, 255), -1)
        analyzer.process_frame(frame, index, index / 60, [])
    assert len(analyzer.goal_entries) == 1
    assert analyzer.goal_entries[0].robot_track_id is None
    assert not analyzer.shots


def test_adjacent_duplicate_broadcast_images_do_not_double_count():
    from training.ball_scouting import BallTrack
    counter = GoalEntryCounter([basket()])
    tracks = [BallTrack(track_id=i, matched_this_frame=True) for i in (1, 2)]
    # Identical positions on alternating source frames, observed by two fragmented IDs.
    for frame in range(10):
        track = tracks[frame % 2]
        other = tracks[1 - frame % 2]
        other.matched_this_frame = False
        track.matched_this_frame = True
        track.points.append(point(frame, 100, [30, 45, 55, 65, 75][frame // 2]))
        counter.observe(tracks, frame, frame / 60, SIZE)
    assert len(counter.entries) == 1
