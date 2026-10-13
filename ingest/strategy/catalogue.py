"""The strategy feature catalogue from docs/STRATEGY-PLAN.md, and which features exist.

The status treemap and the history chart read this, so "how much of the strategy layer is built"
is a list someone can check line by line, not a percentage someone typed. A feature counts as
built when `features.py` or `profile.py` computes it from real data; `added` is the date it
landed. Partial work does not count: routes between zones are not built just because lanes are.
"""

from __future__ import annotations

#: (category, short label, description, added or None)
FEATURES: list[tuple[str, str, str, str | None]] = [
    ("Positioning", "Zones", "Zone dwell map: share of teleop in each zone", "2026-10-12"),
    ("Positioning", "Heat", "Heat map per team, aggregated across matches", None),
    ("Positioning", "Lanes", "Preferred lane", "2026-10-12"),
    ("Positioning", "Speed", "Distance driven, typical and top speed", "2026-10-12"),
    ("Positioning", "Still", "Time stationary vs moving", "2026-10-12"),
    ("Positioning", "Half", "Own half vs opponent half", "2026-10-12"),
    ("Scoring", "Spots", "Shooting spots, clustered", "2026-10-12"),
    ("Scoring", "Dwell", "Dwell at the shooting spot", "2026-10-12"),
    ("Scoring", "Volley", "Shots per volley", "2026-10-12"),
    ("Scoring", "Moving", "Shoot on the move vs stop to shoot", "2026-10-12"),
    ("Scoring", "Range", "Shooting range from the hub", "2026-10-12"),
    ("Scoring", "Made", "Made/missed per spot, where ball tracking is confident", None),
    ("Cycling", "Interval", "Volley interval (cycle-time proxy)", "2026-10-12"),
    ("Cycling", "Source", "Pick-up source preference", None),
    ("Cycling", "Fatigue", "Cycles per match and late-match slowdown", None),
    ("Auto", "Start", "Starting position", "2026-10-12"),
    ("Auto", "Path", "Auto path fingerprint", "2026-10-12"),
    ("Auto", "Routines", "Auto variants across matches", "2026-10-12"),
    ("Defense", "Time", "Defense time near opponents", "2026-10-12"),
    ("Defense", "Marked", "Who it defends", "2026-10-12"),
    ("Defense", "Response", "Cycle time when defended against", None),
    ("Defense", "Contact", "Contact and pin events", None),
    ("Endgame", "Leaves", "When it stops scoring for the endgame", "2026-10-12"),
    ("Endgame", "Climb", "Endgame position and success", None),
    ("Endgame", "Dead", "Immobile periods", None),
    ("Endgame", "Steady", "Consistency across matches", None),
    ("Alliance", "Role", "Role: scorer, feeder, defender", None),
    ("Alliance", "Coord", "Alliance coordination and spacing", None),
    ("Alliance", "Crowd", "Congestion at shared spots", None),
]


def built(on: str | None = None) -> int:
    """Features built, optionally as of a date (YYYY-MM-DD)."""
    return sum(1 for *_, added in FEATURES if added and (on is None or added <= on))
