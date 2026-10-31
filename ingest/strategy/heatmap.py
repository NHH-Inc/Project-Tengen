"""Scoring heatmap and effective zones: where a team scores on the field, across matches.

Built from scoring locations only, so it holds for practice matches too. In a practice match a
team is testing systems, so its *movement* is noise -- but *where it scores* is still real signal
(docs/PLAN-2026.md, section 5: "if we see where they score, we are developing an effective zone").

Recent matches count for more. Over an event, robots and the field "get smarter" (Noah's note):
what a team does in its last qualifier predicts the playoffs better than its first match, so the
weight of a match halves every `RECENCY_HALF_LIFE` matches back from the latest. Practice matches
are treated as the oldest context -- their scoring spots are trusted, but weighted least.

"Our own" team is no special case: build this for our own number to get the drivers' map of where
we are most effective.

Everything is in the alliance frame (feet, origin at the team's own wall corner), so a team's red
and blue matches stack directly -- see contracts/strategy/<season>.json.
"""

from __future__ import annotations

from .profile import _launch_points, match_order
from .timeline import lane_of, zone_of

#: A match this many matches before the latest counts half as much.
RECENCY_HALF_LIFE = 4.0
#: Heatmap grid cell, in feet.
CELL_FT = 3.0
#: A zone/lane region holding at least this share of a team's weighted scoring is "effective"...
EFFECTIVE_SHARE = 0.15
#: ...provided it has been seen scoring there in at least this many matches.
MIN_REGION_MATCHES = 2


def is_practice(features: dict) -> bool:
    """A match whose id is not an official qual/playoff code (or is flagged) is practice/testing."""
    if features.get("practice"):
        return True
    return match_order(features)[1] == 9  # match_order's sentinel level for an unrecognised id


def ordered_with_recency(team_matches: list[dict]) -> list[tuple[dict, float]]:
    """Matches oldest -> newest (practice first, then official by match order), each with its
    recency weight. The newest match has weight 1.0; weight halves every RECENCY_HALF_LIFE back."""
    practice = [m for m in team_matches if is_practice(m)]
    official = sorted((m for m in team_matches if not is_practice(m)), key=match_order)
    ordered = practice + official
    n = len(ordered)
    return [(m, 0.5 ** ((n - 1 - i) / RECENCY_HALF_LIFE)) for i, m in enumerate(ordered)]


def official_with_recency(team_matches: list[dict]) -> list[tuple[dict, float]]:
    """Official matches only, oldest -> newest, each with its recency weight. Used where practice
    data cannot be trusted at all (movement/auto paths), unlike scoring, which survives practice."""
    official = sorted((m for m in team_matches if not is_practice(m)), key=match_order)
    n = len(official)
    return [(m, 0.5 ** ((n - 1 - i) / RECENCY_HALF_LIFE)) for i, m in enumerate(official)]


def _grid(samples: list[tuple[dict, float, float, object]], geometry: dict, cell_ft: float,
          count_key: str) -> list[dict]:
    """Bin (weight, ax, ay, match_id) samples into a field grid, recency-weighted, with a 0..1
    intensity for rendering. Shared by the scoring and auto heatmaps so they stay comparable."""
    length, width = float(geometry["field_length_ft"]), float(geometry["field_width_ft"])
    max_ix, max_iy = int(length // cell_ft), int(width // cell_ft)
    cells: dict[tuple[int, int], dict] = {}
    for weight, ax, ay, mid in samples:
        if ax is None or ay is None:
            continue
        key = (min(int(ax // cell_ft), max_ix), min(int(ay // cell_ft), max_iy))
        cell = cells.setdefault(key, {"count": 0, "weight": 0.0, "matches": set()})
        cell["count"] += 1
        cell["weight"] += weight
        cell["matches"].add(mid)
    out = [{"x": round((ix + 0.5) * cell_ft, 1), "y": round((iy + 0.5) * cell_ft, 1),
            count_key: cell["count"], "weight": round(cell["weight"], 3),
            "matches": len(cell["matches"])}
           for (ix, iy), cell in cells.items()]
    peak = max((c["weight"] for c in out), default=0.0)
    for c in out:
        c["intensity"] = round(c["weight"] / peak, 3) if peak else 0.0
    return sorted(out, key=lambda c: -c["weight"])


def scoring_heatmap(team_matches: list[dict], geometry: dict, cell_ft: float = CELL_FT) -> list[dict]:
    """A grid of where the team scores. Each cell carries the raw launch count, the recency-
    weighted weight, how many matches it spans, and a 0..1 intensity for rendering. Practice
    matches are included -- where a team scores is real signal even while it tests systems."""
    samples = [(weight, point.get("ax"), point.get("ay"), features.get("match_id"))
               for features, weight in ordered_with_recency(team_matches)
               for point in _launch_points(features)]
    return _grid(samples, geometry, cell_ft, count_key="launches")


def _auto_points(features: dict) -> list[list]:
    return [p for p in (features.get("auto_path") or {}).get("points", []) if p]


def auto_heatmap(team_matches: list[dict], geometry: dict, cell_ft: float = CELL_FT) -> list[dict]:
    """A grid of where the team drives during autonomous, over official matches, recency-weighted.

    Practice matches are excluded, not merely down-weighted: in practice a team tests autos, so its
    auto *movement* is noise. That is the same honesty rule that lets the scoring heatmap keep
    practice data -- scoring location survives practice, driving behaviour does not."""
    samples = [(weight, ax, ay, features.get("match_id"))
               for features, weight in official_with_recency(team_matches)
               for ax, ay in _auto_points(features)]
    return _grid(samples, geometry, cell_ft, count_key="samples")


def effective_zones(team_matches: list[dict], geometry: dict) -> list[dict]:
    """The team's scoring broken down by named zone x lane region, recency-weighted. A region is
    `effective` when it holds a real share of the team's scoring across enough matches -- the
    places worth contesting. Regions the team never scores in simply do not appear."""
    regions: dict[tuple[str, str], dict] = {}
    total = 0.0
    for features, weight in ordered_with_recency(team_matches):
        mid = features.get("match_id")
        for point in _launch_points(features):
            zone, lane = zone_of(point.get("ax"), geometry), lane_of(point.get("ay"), geometry)
            if zone is None or lane is None:
                continue
            region = regions.setdefault((zone, lane), {"launches": 0, "weight": 0.0, "matches": set()})
            region["launches"] += 1
            region["weight"] += weight
            region["matches"].add(mid)
            total += weight
    out = []
    for (zone, lane), region in regions.items():
        share = region["weight"] / total if total else 0.0
        out.append({"zone": zone, "lane": lane, "launches": region["launches"],
                    "weight": round(region["weight"], 3), "matches": len(region["matches"]),
                    "share": round(share, 3),
                    "effective": share >= EFFECTIVE_SHARE and len(region["matches"]) >= MIN_REGION_MATCHES})
    return sorted(out, key=lambda r: -r["weight"])


def scoring_map(team_matches: list[dict], geometry: dict) -> dict:
    """Both views, plus a one-line summary of where the team is most dangerous."""
    heatmap = scoring_heatmap(team_matches, geometry)
    zones = effective_zones(team_matches, geometry)
    hottest = heatmap[0] if heatmap else None
    return {
        "cell_ft": CELL_FT,
        "recency_half_life_matches": RECENCY_HALF_LIFE,
        "n_matches": len(team_matches),
        "n_practice": sum(1 for m in team_matches if is_practice(m)),
        "hot_spot": {"x": hottest["x"], "y": hottest["y"]} if hottest else None,
        "heatmap": heatmap,
        "effective_zones": [r for r in zones if r["effective"]],
        "zones": zones,
    }


def auto_map(team_matches: list[dict], geometry: dict) -> dict:
    """Where the team goes in autonomous: the grid, plus how many matches stand behind it. Built
    from official matches only (see auto_heatmap); a team with only practice data yields an empty
    grid rather than a heatmap of noise."""
    heatmap = auto_heatmap(team_matches, geometry)
    n_official = sum(1 for m in team_matches if not is_practice(m))
    busiest = heatmap[0] if heatmap else None
    return {
        "cell_ft": CELL_FT,
        "recency_half_life_matches": RECENCY_HALF_LIFE,
        "n_matches": n_official,
        "n_excluded_practice": sum(1 for m in team_matches if is_practice(m)),
        "busiest_cell": {"x": busiest["x"], "y": busiest["y"]} if busiest else None,
        "heatmap": heatmap,
    }
