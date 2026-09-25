"""Aggregate shot and goal records for the API, without reaching into `training/`.

Doc 0: "One repo, three top-level directories, no cross-imports." The API previously imported
`training.ball_scouting` at module scope and `training.goal_scoring` inside a handler, which put
the whole ball-scouting stack -- and its dependencies -- behind the web service starting. The
analyzer is launched as a subprocess and hands its results over as JSONL; that file is the
boundary, so this reads it and counts.

These are deliberately a line-for-line restatement of the producer's own aggregation rather than
an improvement on it. Two implementations of one rule are how the numbers on a dashboard come to
disagree with the numbers in a file, so `tests/test_shot_stats.py` runs both over the same input
and fails if they ever diverge.

Records come off disk, so a missing or misspelled field must not take the endpoint down with a
KeyError: an unreadable outcome counts as `unknown`, which is already a value this vocabulary
carries, and a record with no robot is `unassigned` rather than dropped.
"""

from __future__ import annotations

from typing import Iterable

#: Outcomes a shot may carry. Anything else -- absent, misspelled, a value from a newer producer --
#: becomes "unknown", which is distinct from a confirmed miss and must stay that way: a miss is
#: evidence the ball was seen leaving the goal, and "unknown" is the absence of evidence.
OUTCOMES = ("made", "missed", "unknown")


def _empty() -> dict[str, int]:
    return {"attempted": 0, "made": 0, "missed": 0, "unknown": 0}


def shot_statistics(shots: Iterable[dict]) -> dict[str, object]:
    """Count attempts and outcomes overall and per robot."""
    totals = _empty()
    per_robot: dict[str, dict[str, int]] = {}
    for shot in shots:
        if not isinstance(shot, dict):
            continue
        outcome = shot.get("outcome", "unknown")
        outcome = str(outcome) if outcome in OUTCOMES else "unknown"
        robot_id = shot.get("robot_track_id")
        key = str(robot_id) if isinstance(robot_id, int) else "unassigned"

        totals["attempted"] += 1
        totals[outcome] += 1
        values = per_robot.setdefault(key, _empty())
        values["attempted"] += 1
        values[outcome] += 1
    return {**totals, "per_robot": per_robot}


def goal_statistics(entries: Iterable[dict]) -> dict[str, object]:
    """Count observed goal entries by region and by the robot they were attributed to.

    Every entry here is already a ball seen crossing into a goal, so each one counts as `made`.
    What is uncertain is *whose* it was, which is why attributed and unassigned are reported
    separately instead of being summed into a single per-team number.
    """
    regions: dict[str, dict[str, object]] = {}
    robots: dict[str, int] = {}
    total = 0
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        total += 1
        region_id = str(entry.get("region_id", "unknown"))
        region = regions.setdefault(
            region_id, {"goal": entry.get("goal"), "made": 0, "attributed": 0, "unassigned": 0}
        )
        robot_id = entry.get("robot_track_id")
        attributed = isinstance(robot_id, int)
        region["made"] += 1
        region["attributed" if attributed else "unassigned"] += 1
        if attributed:
            key = str(robot_id)
            robots[key] = robots.get(key, 0) + 1

    assigned = sum(robots.values())
    return {
        "made": total,
        "attributed": assigned,
        "unassigned": total - assigned,
        "per_region": regions,
        "per_robot": robots,
    }
