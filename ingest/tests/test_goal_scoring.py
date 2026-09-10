"""Observed basket entry, bounce rejection, and conservative source attribution."""
from dataclasses import replace
import json

import pytest

from training.ball_scouting import (
    BallScoutingConfig, BallShotAnalyzer, BallTrack, BallTrackPoint,
    DirectedBoundary, GoalGeometry, ShotRecord, load_ball_scouting_config,
)
from training.goal_scoring import (
    GoalEntry, GoalEntryCounter, goal_statistics, trajectory_link_cost, write_goal_entries,
)

SIZE = (200, 100)


def basket(**kwargs):
    goal = GoalGeometry(
        goal_id="high", polygon=None, entry_direction=(0, 1),
        made_line=DirectedBoundary(((.3, .6), (.7, .6)), (0, 1)),
        region_id="blue_hub",
        approach_line=DirectedBoundary(((.3, .4), (.7, .4)), (0, 1)),
        confirmation_polygon=((.3, .4), (.7, .4), (.7, .9), (.3, .9)),
        confirmation_frames=2, minimum_depth_radii=.5,
    )
    return replace(goal, **kwargs)


def point(index, x, y, timestamp=None):
    return BallTrackPoint(index, index / 60 if timestamp is None else timestamp,
                          (x, y), 3, 28, .9)


def feed(counter, positions, track_id=1):
    track = BallTrack(track_id=track_id)
    for i, position in enumerate(positions):
        track.matched_this_frame = position is not None
        if position is not None:
            track.points.append(point(i, *position))
        counter.observe([track], i, i / 60, SIZE)
    return counter.entries


def test_entry_requires_ordered_approach_and_depth_and_counts_once():
    counter = GoalEntryCounter([basket()])
    entries = feed(counter, [(100, y) for y in (30, 45, 55, 61, 65, 70, 80)])
    assert len(entries) == 1
    assert entries[0].crossing_frame == 3
    assert entries[0].frame_index == 4  # centre must move half a radius past line
    assert entries[0].robot_track_id is None
    assert goal_statistics(entries)["unassigned"] == 1


@pytest.mark.parametrize("positions", [
    [(100, y) for y in (30, 45, 55, 61, 55, 40, 25)],  # rim bounce
    [(100, y) for y in (80, 65, 55, 45, 30)],  # upward flight
    [(160, y) for y in (30, 45, 55, 65, 80)],  # beside basket
    [(100, y) for y in (55, 61, 65, 80)],  # first seen inside; no approach
    [(100, 30), (100, 45), (100, 55), None, None, None],  # disappears
    [(100, 30), (100, 45), (100, 55)] + [None] * 8 + [(100, 75)],
    [(100, 30), (100, 45), (150, 55), (150, 65)],  # exits side
    [(100, 30), (100, 45), (150, 55), (100, 65)],  # leaves and reenters side
])
def test_ambiguous_and_non_scoring_paths_stay_uncounted(positions):
    assert feed(GoalEntryCounter([basket()]), positions) == []


def test_duplicate_images_do_not_supply_inward_confirmation():
    counter = GoalEntryCounter([basket(confirmation_frames=3, minimum_depth_radii=0)])
    assert feed(counter, [(100, 30), (100, 45), (100, 65)] + [(100, 65)] * 8) == []


def test_scene_reset_does_not_join_a_crossing():
    counter = GoalEntryCounter([basket()])
    track = BallTrack(track_id=1, points=[point(0, 100, 30), point(1, 100, 45)],
                      matched_this_frame=True)
    counter.observe([track], 1, 1/60, SIZE)
    counter.reset_temporal(2/60)
    track.points.append(point(2, 100, 70))
    assert counter.observe([track], 2, 2/60, SIZE) == []


def test_two_close_balls_count_separately_without_goal_cooldown():
    counter = GoalEntryCounter([basket()])
    tracks = [BallTrack(track_id=i, matched_this_frame=True) for i in (1, 2)]
    for i, y in enumerate((30, 45, 55, 65, 75)):
        for track in tracks:
            track.points.append(point(i, 85 + 15 * track.track_id, y))
        counter.observe(tracks, i, i/60, SIZE)
    assert len(counter.entries) == 2


def test_fragment_with_same_observed_crossing_is_not_double_counted():
    counter = GoalEntryCounter([basket()])
    positions = [(100, y) for y in (30, 45, 55, 65, 75)]
    feed(counter, positions, 1)
    feed(counter, positions, 2)
    assert len(counter.entries) == 1


def make_entry(track_id=7, x=.5):
    evidence = [dict(frame_index=i, t_seconds=i/60, x=x, y=.1 + i*.02,
                     radius=.0134, observed=True) for i in (9, 10, 11)]
    return GoalEntry("entry", "high", "blue_hub", track_id, 11, 11/60, 10, 10/60, .92, evidence)


def make_shot(robot_id=3, x=.5):
    shot = ShotRecord(f"shot{robot_id}", 0, 0, robot_id, robot_id, .9, .95)
    shot.ball_track = [dict(frame_index=i, t_seconds=i/60, x=x, y=.1 + i*.02,
                           radius=.0134, observed=True) for i in range(8)]
    return shot


def test_same_track_goal_credits_robot_and_preserves_observations():
    analyzer = BallShotAnalyzer(BallScoutingConfig(goals=(basket(),)))
    shot = make_shot()
    analyzer.shots.append(shot)
    analyzer._shots_by_ball_track[7] = shot
    entry = make_entry()
    analyzer._associate_goal_entries([entry])
    assert shot.outcome == "made"
    assert shot.outcome_frame == entry.frame_index
    assert entry.robot_track_id == 3
    assert entry.association == "same_track"
    assert len(shot.ball_track) == 11


def test_local_ballistic_relink_and_ambiguous_sources():
    entry, shot = make_entry(), make_shot()
    assert trajectory_link_cost(shot, entry, SIZE) == pytest.approx(0, abs=1e-6)
    analyzer = BallShotAnalyzer(BallScoutingConfig(goals=(basket(),)))
    analyzer.frame_size = SIZE
    analyzer.shots = [shot]
    analyzer._associate_goal_entries([entry])
    assert entry.association == "trajectory_relink"
    assert shot.outcome == "made"
    analyzer.shots = [make_shot(3), make_shot(4)]
    ambiguous = make_entry(8)
    analyzer._associate_goal_entries([ambiguous])
    assert ambiguous.robot_track_id is None
    assert all(s.outcome == "unknown" for s in analyzer.shots)


def test_one_shot_cannot_credit_two_simultaneous_entries():
    analyzer = BallShotAnalyzer(BallScoutingConfig(goals=(basket(),)))
    analyzer.frame_size = SIZE
    analyzer.shots = [make_shot()]
    entries = [make_entry(7), make_entry(8)]
    analyzer._associate_goal_entries(entries)
    assert all(e.shot_id is None for e in entries)


def test_wrong_path_or_long_occlusion_has_no_robot_credit():
    assert trajectory_link_cost(make_shot(x=.2), make_entry(), SIZE) is None
    entry = make_entry()
    for p in entry.ball_track:
        p["t_seconds"] += .5
    assert trajectory_link_cost(make_shot(), entry, SIZE) is None


def test_sidecar_keeps_goal_totals_separate_from_launch_attempts(tmp_path):
    entry, unassigned = make_entry(), make_entry(8)
    entry.robot_track_id = 3
    path = tmp_path / "goal_entries.jsonl"
    write_goal_entries(path, [entry, unassigned])
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    totals = goal_statistics(rows)
    assert (totals["made"], totals["attributed"], totals["unassigned"]) == (2, 1, 1)
    assert totals["per_robot"] == {"3": 1}
    assert totals["per_region"]["blue_hub"]["made"] == 2


def test_goal_config_validation(tmp_path):
    goal = dict(id="high", region_id="blue", entry_direction=[0, 1],
                made_boundary=dict(line=[[.3, .6], [.7, .6]], direction=[0, 1]),
                confirmation_frames=0)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(dict(schema_version=1, goals=[goal])))
    with pytest.raises(ValueError, match="confirmation_frames"):
        load_ball_scouting_config(path)
    goal["confirmation_frames"] = 2
    path.write_text(json.dumps(dict(schema_version=1, goals=[goal, goal])))
    with pytest.raises(ValueError, match="region_id"):
        load_ball_scouting_config(path)
