"""The API's aggregation must agree with the producer's, exactly.

`ingest/shot_stats.py` exists so the web service does not import `training/`, which Doc 0 forbids.
The cost of that boundary is two implementations of one rule, and the way that goes wrong is
silent: a dashboard total that no longer matches the file it was computed from. So every case here
runs both and asserts they agree, which is the only reason the duplicate is safe to keep.

A test may import across components; the running service may not.
"""

import pytest

from ingest.shot_stats import goal_statistics, shot_statistics
from training.ball_scouting import shot_statistics as producer_shot_statistics
from training.goal_scoring import goal_statistics as producer_goal_statistics


SHOT_CASES = [
    pytest.param([], id="no shots"),
    pytest.param([{"robot_track_id": 7, "outcome": "made"}], id="one made"),
    pytest.param(
        [
            {"robot_track_id": 7, "outcome": "made"},
            {"robot_track_id": 7, "outcome": "missed"},
            {"robot_track_id": 9, "outcome": "unknown"},
        ],
        id="two robots, three outcomes",
    ),
    # A launch nobody could be credited with still happened, and still counts as an attempt.
    pytest.param([{"robot_track_id": None, "outcome": "made"}], id="unattributed"),
    pytest.param([{"outcome": "missed"}], id="no robot field at all"),
    # Outcome absent or from a newer producer: neither is a confirmed miss.
    pytest.param([{"robot_track_id": 3}], id="outcome missing"),
    pytest.param([{"robot_track_id": 3, "outcome": "bounced_out"}], id="unrecognised outcome"),
    # A track id that is not an int cannot index a robot, so it is nobody's shot.
    pytest.param([{"robot_track_id": "7", "outcome": "made"}], id="track id as a string"),
]

GOAL_CASES = [
    pytest.param([], id="no entries"),
    pytest.param([{"region_id": "red_hub", "goal": "high", "robot_track_id": 4}], id="one attributed"),
    pytest.param([{"region_id": "red_hub", "goal": "high", "robot_track_id": None}], id="one unassigned"),
    pytest.param(
        [
            {"region_id": "red_hub", "goal": "high", "robot_track_id": 4},
            {"region_id": "red_hub", "goal": "high", "robot_track_id": 4},
            {"region_id": "blue_hub", "goal": "high", "robot_track_id": None},
        ],
        id="two regions",
    ),
]


@pytest.mark.parametrize("shots", SHOT_CASES)
def test_shot_statistics_match_the_producer(shots):
    assert shot_statistics(shots) == producer_shot_statistics(shots)


@pytest.mark.parametrize("entries", GOAL_CASES)
def test_goal_statistics_match_the_producer(entries):
    assert goal_statistics(entries) == producer_goal_statistics(entries)


def test_unknown_stays_distinct_from_a_confirmed_miss():
    """The distinction the whole vocabulary rests on: no evidence is not evidence of a miss."""
    totals = shot_statistics([{"robot_track_id": 1}, {"robot_track_id": 1, "outcome": "missed"}])
    assert totals["unknown"] == 1
    assert totals["missed"] == 1
    assert totals["attempted"] == 2


def test_a_malformed_record_does_not_take_the_endpoint_down():
    """These come off disk. A bad line should cost one row, not a 500."""
    assert shot_statistics([None, "not a record", {"outcome": "made"}])["attempted"] == 1
    assert goal_statistics([None, 42, {"region_id": "red_hub"}])["made"] == 1


def test_an_entry_missing_its_region_is_still_counted():
    stats = goal_statistics([{"goal": "high", "robot_track_id": 2}])
    assert stats["made"] == 1
    assert "unknown" in stats["per_region"]
