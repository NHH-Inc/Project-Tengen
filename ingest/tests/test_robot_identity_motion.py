"""Identity recovery with hidden crossings, camera movement and uncertain motion.

These sequences specify the robots' actual paths independently of the tracker's estimates.
No model download or video decoder is needed to exercise the association decisions.
"""

from dataclasses import replace

import pytest

from training.track_yolo import AppearanceTrackMemory, ReIDConfig, ReIDDetection


def detection(raw_id, x, y=0.5, *, descriptor=(1.0, 0.0), field_center=None):
    return ReIDDetection(raw_id, list(descriptor) if descriptor is not None else None,
                         (x, y), "red", field_center=field_center)


@pytest.mark.parametrize("reused_ids", [False, True])
@pytest.mark.parametrize("reverse_order", [False, True])
def test_same_looking_robots_keep_direction_through_a_hidden_crossing(reused_ids, reverse_order):
    memory = AppearanceTrackMemory()
    for t in (0.0, 0.1, 0.2):
        assert memory.resolve_frame(t, [
            detection(10, 0.4 + 0.2 * t),
            detection(20, 0.6 - 0.2 * t),
        ]) == [1, 2]
    # Both vanish behind an obstacle and emerge on opposite sides. ByteTrack may issue new
    # IDs or swap the old IDs; detector order and raw continuity must not override the paths.
    returned = [detection(20 if reused_ids else 91, 0.54),
                detection(10 if reused_ids else 92, 0.46)]
    if reverse_order:
        returned.reverse()
    assert memory.resolve_frame(0.7, returned) == ([2, 1] if reverse_order else [1, 2])


def test_speed_distinguishes_same_looking_robots_moving_in_the_same_direction():
    memory = AppearanceTrackMemory()
    for t in (0.0, 0.1, 0.2):
        assert memory.resolve_frame(t, [
            detection(10, 0.3 + 0.3 * t),
            detection(20, 0.5 + 0.05 * t),
        ]) == [1, 2]
    assert memory.resolve_frame(1.0, [detection(91, 0.60), detection(92, 0.55)]) == [1, 2]


def test_calibrated_speed_and_direction_override_ambiguous_image_positions():
    memory = AppearanceTrackMemory()
    for t in (0.0, 0.1, 0.2):
        assert memory.resolve_frame(t, [
            detection(10, 0.4, field_center=(10 + 4 * t, 10)),
            detection(20, 0.6, field_center=(14 - 4 * t, 10)),
        ]) == [1, 2]
    # Image positions favor the opposite association; the field positions establish who crossed.
    assert memory.resolve_frame(0.8, [
        detection(91, 0.6, field_center=(13.2, 10)),
        detection(92, 0.4, field_center=(10.8, 10)),
    ]) == [1, 2]


def test_camera_pan_is_not_learned_as_robot_speed():
    memory = AppearanceTrackMemory()
    assert memory.resolve_frame(0.0, [detection(10, 0.2)]) == [1]
    for t, x in ((0.1, 0.3), (0.2, 0.4)):
        assert memory.resolve_frame(t, [detection(10, x)], camera_motion=(0.1, 0.0)) == [1]
    # The camera stops; the robot was stationary all along. Even without an appearance crop,
    # its new tracklet is at the predicted position rather than continuing the camera's pan.
    assert memory.resolve_frame(0.5, [detection(91, 0.4, descriptor=None)]) == [1]


def test_recent_path_reduces_detector_jitter_before_an_appearance_free_return():
    memory = AppearanceTrackMemory()
    for i in range(13):
        t = i / 30
        jitter = 0.004 if i % 2 else -0.004
        assert memory.resolve_frame(t, [detection(10, 0.3 + 0.15 * t + jitter)]) == [1]
    assert memory.resolve_frame(0.8, [detection(91, 0.42, descriptor=None)]) == [1]


def test_recovery_relearns_motion_after_a_robot_turns_while_hidden():
    memory = AppearanceTrackMemory()
    for t in (0.0, 0.1, 0.2):
        assert memory.resolve_frame(t, [detection(10, 0.3 + 0.2 * t)]) == [1]
    # The robot turned behind the obstruction. Strong appearance permits recovery, but the
    # displacement over that hidden second must not be treated as its new velocity.
    assert memory.resolve_frame(1.2, [detection(91, 0.38, 0.6)]) == [1]
    assert memory.states[1].velocity is None
    assert memory.resolve_frame(1.3, [detection(91, 0.38, 0.62)]) == [1]
    assert memory.resolve_frame(1.6, [detection(92, 0.38, 0.68, descriptor=None)]) == [1]


def test_long_hidden_gap_does_not_extrapolate_old_heading_indefinitely():
    memory = AppearanceTrackMemory()
    for t in (0.0, 0.1, 0.2):
        assert memory.resolve_frame(t, [detection(10, 0.3 + 0.4 * t)]) == [1]
    # Continuing at 0.4 frame-widths/s for three seconds would predict x=1.58, off-screen.
    # The bounded forecast and widened uncertainty allow a return at the plausible x=0.78.
    assert memory.resolve_frame(3.2, [detection(91, 0.78, descriptor=None)]) == [1]


def test_unobserved_positions_are_not_added_to_public_tracks():
    memory = AppearanceTrackMemory()
    for t in (0.0, 0.1, 0.2):
        memory.resolve_frame(t, [detection(10, 0.3 + 0.2 * t)])
    last_seen = memory.states[1].last_seen
    center = memory.states[1].center
    samples = list(memory.states[1].motion_samples)
    assert memory.resolve_frame(0.4, []) == []
    assert memory.resolve_frame(0.6, []) == []
    assert memory.states[1].last_seen == last_seen
    assert memory.states[1].center == center
    assert memory.states[1].motion_samples == samples
    assert memory.resolve_frame(0.7, [detection(91, 0.44)]) == [1]


def test_prediction_cannot_override_a_conflicting_appearance():
    memory = AppearanceTrackMemory()
    for t in (0.0, 0.1, 0.2):
        memory.resolve_frame(t, [detection(10, 0.3 + 0.2 * t)])
    assert memory.resolve_frame(0.7, [detection(91, 0.44, descriptor=(0.0, 1.0))]) != [1]


def test_prediction_does_not_force_an_ambiguous_hidden_identity():
    memory = AppearanceTrackMemory()
    for t in (0.0, 0.1, 0.2):
        assert memory.resolve_frame(t, [
            detection(10, 0.3 + 0.2 * t), detection(20, 0.7 - 0.2 * t),
        ]) == [1, 2]
    assert memory.resolve_frame(1.0, [detection(91, 0.5)]) == [None]


def test_lost_calibration_discards_field_velocity_until_observed_again():
    memory = AppearanceTrackMemory()
    for t in (0.0, 0.1, 0.2):
        assert memory.resolve_frame(t, [detection(10, 0.4, field_center=(10 + 4 * t, 10))]) == [1]
    assert memory.states[1].field_velocity is not None
    assert memory.resolve_frame(0.3, [detection(10, 0.4)]) == [1]
    assert memory.states[1].field_velocity is None
    assert memory.resolve_frame(0.4, [detection(10, 0.4, field_center=(13, 10))]) == [1]
    assert memory.states[1].field_velocity is None


def test_motion_prediction_still_rejects_physically_impossible_field_speed():
    memory = AppearanceTrackMemory()
    for t in (0.0, 0.1, 0.2):
        memory.resolve_frame(t, [detection(10, 0.4, field_center=(10 + 4 * t, 10))])
    assert memory.resolve_frame(0.3, [detection(91, 0.4, field_center=(40, 10))]) != [1]


def test_motion_state_does_not_grow_with_video_length():
    memory = AppearanceTrackMemory(config=replace(ReIDConfig(), motion_history_seconds=2.0))
    for i in range(600):
        assert memory.resolve_frame(i / 120, [detection(10, 0.3 + 0.01 * i / 120)]) == [1]
    assert len(memory.states[1].motion_samples) <= 64
