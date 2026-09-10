"""Video-level regressions: count physical launches, not yellow blobs or track IDs."""

import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from training.ball_scouting import (
    BallDetection, BallBlobTracker, BallShotAnalyzer, RobotObservation,
    YellowBallDetector, load_ball_scouting_config,
)
from training.launch_signals import GateSample, ProminentPulseDetector, LaunchSignalBank, LaunchSignalConfig
from ingest.tests.test_ball_scouting import analyzer_config, frame_with_balls


def burst(count=12, period=3, speed=9, visible_frames=7, missing=(), moving_robot=0):
    analyzer = BallShotAnalyzer(analyzer_config())
    for frame in range(count * period + visible_frames + 2):
        positions = []
        for launch in range(count):
            age = frame - launch * period - 2
            if 0 <= age < visible_frames and age not in missing:
                positions.append((68 + age * speed + moving_robot * frame, 40))
        analyzer.process_frame(
            frame_with_balls(*positions), frame, frame / 30,
            [RobotObservation(12, (30 + moving_robot * frame, 20,
                                   65 + moving_robot * frame, 80))],
        )
    return analyzer


@pytest.mark.parametrize("period,speed,visible", [(2, 12, 2), (3, 9, 7), (2, 4, 12)])
def test_same_lane_burst_counts_every_observable_launch(period, speed, visible):
    analyzer = burst(period=period, speed=speed, visible_frames=visible)
    assert len(analyzer.shots) == 12
    assert len({s.shot_id for s in analyzer.shots}) == 12
    assert len({s.ball_track_id for s in analyzer.shots}) == 12
    assert all(s.robot_track_id == 12 for s in analyzer.shots)


def test_short_launch_does_not_wait_for_global_track_confirmation():
    config = analyzer_config()
    analyzer = BallShotAnalyzer(replace(config, tracking=replace(config.tracking, minimum_confirmed_hits=6)))
    for i, positions in enumerate([[], [(68, 40)], [(80, 40)], [], [], []]):
        analyzer.process_frame(frame_with_balls(*positions), i, i / 30,
                               [RobotObservation(12, (30, 20, 65, 80))])
    assert len(analyzer.shots) == 1


def test_two_frame_launch_handles_one_missing_frame():
    assert len(burst(count=8, period=4, visible_frames=3, speed=6, missing=(1,)).shots) == 8


def test_touching_yellow_balls_split_without_splitting_one_smear_into_balls():
    detector = YellowBallDetector(analyzer_config().detector)
    assert len(detector.detect(frame_with_balls((50, 40), (58, 40), (66, 40)))) == 3
    frame = frame_with_balls()
    cv2.rectangle(frame, (30, 35), (70, 43), (0, 255, 255), -1)
    detector.detect(frame_with_balls())
    assert len(detector.detect(frame)) == 0 or len(detector.last_detections) == 1


def test_moving_blurred_ball_survives_while_static_yellow_bar_does_not():
    config = replace(analyzer_config().detector, max_area_px=400, max_radius_px=15)
    detector = YellowBallDetector(config)
    stationary = frame_with_balls()
    cv2.rectangle(stationary, (20, 40), (60, 45), (0, 255, 255), -1)
    assert detector.detect(stationary) == []
    assert detector.detect(stationary) == []
    moving = frame_with_balls()
    cv2.rectangle(moving, (35, 40), (75, 45), (0, 255, 255), -1)
    assert len(detector.detect(moving)) == 1


def test_moving_robot_burst_and_no_shots_for_carried_decorations():
    assert len(burst(count=6, moving_robot=1).shots) == 6
    analyzer = BallShotAnalyzer(analyzer_config())
    for i in range(20):
        dx = i * 3
        analyzer.process_frame(frame_with_balls((64 + dx, 40)), i, i / 30,
                               [RobotObservation(12, (30 + dx, 20, 65 + dx, 80))])
    assert analyzer.shots == []


@pytest.mark.parametrize("positions", [
    [(68, 78), (80, 78), (92, 78)],  # floor contact
    [(100, 40), (88, 40), (76, 40), (64, 40)],  # inbound external throw
    [(130, 40), (142, 40), (154, 40)],  # source not locally observed
    [(68, 40)],  # one-frame yellow flash
])
def test_nonlaunches_are_not_counted(positions):
    analyzer = BallShotAnalyzer(analyzer_config())
    for i, point in enumerate(positions):
        analyzer.process_frame(frame_with_balls(point), i, i / 30,
                               [RobotObservation(12, (30, 20, 65, 80))])
    assert analyzer.shots == []


def test_camera_cut_and_time_gap_do_not_bridge_a_launch():
    for gap in (0.5, 1 / 30):
        analyzer = BallShotAnalyzer(analyzer_config())
        analyzer.process_frame(frame_with_balls((68, 40)), 0, 0,
                               [RobotObservation(12, (30, 20, 65, 80))])
        frame = frame_with_balls((80, 40))
        if gap < 0.5:
            frame[frame.sum(axis=2) == 0] = 180
        analyzer.process_frame(frame, 1, gap, [RobotObservation(12, (30, 20, 65, 80))])
        assert analyzer.shots == []


def test_pulse_hysteresis_resolves_nonzero_valleys_and_does_not_count_plateau_twice():
    detector = ProminentPulseDetector(0.16, 0.22)
    values = [0, 0, 0.8, 1, 0.45, 0.85, 1, 0.5, 0.9, 1, 1, 1, 0, 0]
    peaks = [peak for i, value in enumerate(values)
             if (peak := detector.update(GateSample(i, i / 60, value, (70, 40), 5)))]
    assert [p.frame_index for p in peaks] == [3, 6, 9]
    constant = ProminentPulseDetector(0.16, 0.22)
    assert all(constant.update(GateSample(i, i / 60, 1, (70, 40), 5)) is None for i in range(50))


def test_gate_signals_count_without_any_contour_or_track_ids():
    config = analyzer_config()
    analyzer = BallShotAnalyzer(config)
    box = (30, 20, 65, 80)
    # Calibrate from one known observed departure, as the normal short-launch path does.
    analyzer.launch_signals.seed(12, box, (62, 40), (74, 40), 5, 120, 0)
    for i in range(48):
        positions = [(60 + (i - launch) * 4, 40) for launch in range(3, 35, 4)
                     if 0 <= i - launch <= 12]
        # Keep the real colour mask but deliberately remove all contour detections.
        real_detect = analyzer.detector.detect
        def mask_only(frame):
            real_detect(frame)
            return []
        with patch.object(analyzer.detector, "detect", side_effect=mask_only):
            analyzer.process_frame(frame_with_balls(*positions), i, i / 30,
                                   [RobotObservation(12, box)])
    assert len(analyzer.shots) == 8
    assert all(len(s.ball_track) == 2 for s in analyzer.shots)
    assert all(s.outcome == "unknown" for s in analyzer.shots)


def test_gate_pulse_needs_outward_order():
    analyzer = BallShotAnalyzer(analyzer_config())
    box = (30, 20, 65, 80)
    analyzer.launch_signals.seed(12, box, (62, 40), (74, 40), 5, 120, 0)
    for i in range(16):
        analyzer.process_frame(frame_with_balls((112 - i * 4, 40)), i, i / 30,
                               [RobotObservation(12, box)])
    assert analyzer.shots == []


def test_shot_records_still_satisfy_the_persisted_contract():
    schema = json.loads((Path(__file__).parents[2] / "contracts/shots.schema.json").read_text())
    for shot in burst(count=3).shots:
        record = json.loads(json.dumps(shot.to_dict()))
        assert set(record) == set(schema["required"])
        assert record["ball_track_id"] >= 1
        assert 0 <= record["confidence"] <= 1
        assert record["ball_track"]
        for point in record["ball_track"]:
            assert set(point) == set(schema["properties"]["ball_track"]["items"]["required"])
            assert 0 <= point["x"] <= 1 and 0 <= point["y"] <= 1
            assert point["observed"] is True


def test_global_assignment_does_not_steal_the_only_match_of_another_ball():
    config = replace(analyzer_config().tracking, base_link_distance_ratio=0,
                     acceleration_allowance_ratio_per_second=0.4)
    tracker = BallBlobTracker(config)
    def detection(x):
        return BallDetection((x, 40), 1, 3.14, 1, (x - 1, 39, x + 1, 41))
    tracker.update([detection(50), detection(53)], 0, 0, (200, 100))
    tracker.update([detection(52), detection(55)], 1, 1 / 30, (200, 100))
    assert len(tracker.active) == 2
    assert tracker.active[1].last.center == (52, 40)
    assert tracker.active[2].last.center == (55, 40)


def test_config_rejects_silent_signal_typos(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"launch_signals": {"pulse_minimim_prominence": 0.1}}))
    with pytest.raises(ValueError, match="unknown launch_signals"):
        load_ball_scouting_config(path)


def test_runner_defaults_to_every_frame():
    from training.track_yolo import parse_args
    with patch("sys.argv", ["track_yolo", "--model", "model.pt", "--video", "match.mp4",
                             "--output", "tracks.jsonl"]):
        assert parse_args().frame_stride == 1


def test_reference_matching_penalizes_duplicates_and_wrong_robots():
    from tools.replay_ball_scouting import match_shots
    reference = [dict(launch_t_seconds=1.0, robot_track_id=12),
                 dict(launch_t_seconds=1.1, robot_track_id=12)]
    predictions = [reference[0], reference[0], dict(launch_t_seconds=1.1, robot_track_id=3)]
    result = match_shots(predictions, reference, .04)
    assert result["true_positive"] == 1
    assert result["false_positive"] == 2
    assert result["false_negative"] == 1


def test_invalid_timestamps_fail_before_mutating_history():
    analyzer = BallShotAnalyzer(analyzer_config())
    analyzer.process_frame(frame_with_balls(), 0, 0, [])
    with pytest.raises(ValueError, match="increasing"):
        analyzer.process_frame(frame_with_balls(), 1, 0, [])
    assert analyzer._analysis_step == 0


def test_old_edge_heuristic_cannot_bypass_rejected_flow():
    analyzer = BallShotAnalyzer(analyzer_config())
    with patch.object(analyzer, "_flow_agrees", return_value=False):
        for i in range(5):
            analyzer.process_frame(frame_with_balls((68 + i * 12, 40)), i, i / 30,
                                   [RobotObservation(12, (30, 20, 65, 80))])
    assert analyzer.shots == []


def test_occluded_ball_does_not_capture_the_next_launch_at_its_old_position():
    analyzer = burst(count=8, period=2, speed=12, visible_frames=2)
    assert len(analyzer.shots) == 8
    for shot in analyzer.shots:
        assert shot.ball_track[-1]["frame_index"] <= shot.launch_frame + 1


def test_sixty_fps_broadcast_with_duplicated_thirty_fps_images():
    analyzer = BallShotAnalyzer(analyzer_config())
    for frame in range(90):
        step = frame // 2
        positions = [(68 + (step - 2 - launch * 3) * 9, 40) for launch in range(12)
                     if 0 <= step - 2 - launch * 3 < 7]
        analyzer.process_frame(frame_with_balls(*positions), frame, frame / 60,
                               [RobotObservation(12, (30, 20, 65, 80))])
    assert len(analyzer.shots) == 12


def test_duplicate_gate_observations_fuse_across_independently_learned_ports():
    from training.ball_scouting import ShotRecord
    analyzer = BallShotAnalyzer(analyzer_config())
    analyzer.frame_size = (200, 100)
    path = [dict(frame_index=i, t_seconds=i/30, x=.35+i*.05, y=.4,
                 radius=.02, observed=True) for i in range(3)]
    existing = ShotRecord("first", 0, 0, 12, 1, .9, .9, ball_track=path)
    analyzer.shots.append(existing)
    proposed = ShotRecord("second", 1, 1/30, 12, 2, .8, .8, ball_track=path[1:])
    assert analyzer._find_existing_launch(proposed) is existing
    proposed.ball_track = [dict(point, y=.7) for point in path[1:]]
    assert analyzer._find_existing_launch(proposed) is None
