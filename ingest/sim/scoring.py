"""Scoring (T3): turn robot trajectories + shooting behaviour into points.

A shooter scores while its trajectory keeps it within its goal's effective range: it fires a volley
every `volley_interval_s`, and the expected made shots (`shots_per_volley * accuracy`) earn the
goal's per-phase point value. Expected value, not dice, so a run is reproducible like the rest of the
strategy layer.

This is where the sim's tiers pay off together: feed it the trajectories from `simulate_match`
(T2) and a **defended** attacker -- pushed off its spot and out of range -- scores less, with no
special-casing. Feed it real tracks instead and it scores an actual match. Point values come from the
game model (game.py), which reads them from the season config, so no scoring rule is hard-coded here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .game import GameModel
from .model import DEFAULT_DT_S, Pose


@dataclass(frozen=True)
class Shooter:
    robot_id: str
    alliance: str
    goal: str  # which goal name this robot shoots at
    volley_interval_s: float
    shots_per_volley: float
    accuracy: float  # 0..1, expected fraction of shots made


def shooter_from_profile(profile: dict, alliance: str, goal: str, *, accuracy: float = 0.6,
                         robot_id: str | None = None) -> Shooter:
    """A shooter from a team profile's measured cadence. Accuracy is a parameter -- we count shots
    from video, not makes vs misses reliably -- so it defaults and can be set per team as data allows."""
    shooting = (profile.get("tendencies") or {}).get("shooting") or {}
    timing = profile.get("timing") or {}
    interval = shooting.get("median_volley_interval_s") or timing.get("median_volley_interval_s") or 8.0
    per_volley = timing.get("median_launches_per_volley") or 3.0
    return Shooter(robot_id or str(profile.get("team")), alliance, goal,
                   float(interval), float(per_volley), accuracy)


def score_tracks(tracks: dict[str, list[Pose]], shooters: list[Shooter], game: GameModel,
                 *, dt: float = DEFAULT_DT_S) -> dict:
    """Score a set of trajectories. Returns points by alliance, the total, and a per-shooter
    breakdown (points and how long each spent in scoring range)."""
    by_alliance: dict[str, float] = {}
    breakdown = []
    for shooter in shooters:
        goal = game.goal(shooter.goal)
        poses = tracks.get(shooter.robot_id)
        if goal is None or not poses:
            continue
        time_in_phase = {"auto": 0.0, "teleop": 0.0, "endgame": 0.0}
        for pose in poses:
            if math.hypot(pose.x - goal.x, pose.y - goal.y) <= goal.effective_range_ft:
                time_in_phase[game.phase_at(pose.t)] += dt
        points = 0.0
        for phase, seconds in time_in_phase.items():
            volleys = seconds / shooter.volley_interval_s if shooter.volley_interval_s > 0 else 0.0
            made = volleys * shooter.shots_per_volley * shooter.accuracy
            points += made * goal.points.get(phase, 0.0)
        by_alliance[shooter.alliance] = by_alliance.get(shooter.alliance, 0.0) + points
        breakdown.append({"robot_id": shooter.robot_id, "alliance": shooter.alliance,
                          "points": round(points, 1),
                          "time_in_range_s": round(sum(time_in_phase.values()), 1)})
    return {
        "by_alliance": {k: round(v, 1) for k, v in by_alliance.items()},
        "total": round(sum(by_alliance.values()), 1),
        "shooters": sorted(breakdown, key=lambda b: -b["points"]),
    }
