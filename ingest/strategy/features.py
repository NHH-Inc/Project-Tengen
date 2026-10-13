"""Behaviour features: what one robot did in one match, from its timeline.

Everything is computed in the alliance frame (see `timeline.py`), so "left lane" and "shooting
spot at (12, 20)" mean the same thing whichever alliance the team was on. A feature that needs
field positions is `None` when the match had none, never zero: "it never moved" and "we could not
see where it was" must stay distinguishable, exactly as `unknown` stays distinct from a miss.
"""

from __future__ import annotations

import math
from statistics import median

from .timeline import samples_as_rows

#: Below this a robot counts as stopped. Homography jitter on a parked robot reads ~0.3-0.6 ft/s.
STATIONARY_FTPS = 1.0
#: Launches within this distance of each other belong to one shooting spot.
SPOT_RADIUS_FT = 4.0
#: Launches separated by less than this are one volley (one visit to a spot).
VOLLEY_GAP_S = 3.0
#: An opponent this close, in the opponent's half, counts as being defended / defending.
DEFENSE_RADIUS_FT = 6.0
#: The starting position is read from samples this close to match time zero.
START_WINDOW_S = 1.5
#: The auto path is summarised as one point per second.
AUTO_STEP_S = 1.0
#: A sample is matched to another robot's sample at the same instant within this tolerance.
SAME_INSTANT_S = 0.051


def match_features(timelines: list[dict], geometry: dict, season: dict) -> list[dict]:
    """Features for every robot in one match. Robots see each other for defense measures."""
    rows = {t["robot_key"]: samples_as_rows(t) for t in timelines}
    return [robot_features(t, rows[t["robot_key"]], timelines, rows, geometry, season)
            for t in timelines]


def robot_features(timeline, rows, all_timelines, all_rows, geometry, season) -> dict:
    match_seconds = float(season.get("auto_seconds", 0)) + float(season.get("teleop_seconds", 0)) \
        + float(season.get("endgame_seconds", 0))
    # The pause between auto and teleop is part of the match (robots sit or drift), but it is
    # neither phase, so it counts toward coverage and movement and not toward teleop zones.
    in_match = [r for r in rows if r["phase"] in ("auto", "transition", "teleop", "endgame")]
    positioned = [r for r in in_match if r["ax"] is not None]
    step = 1.0 / timeline["rate_hz"]

    features = {
        "job_id": timeline.get("job_id"),
        "match_id": timeline.get("match_id"),
        "robot_key": timeline["robot_key"],
        "team": timeline.get("team"),
        "alliance": timeline.get("alliance"),
        "identity_issues": timeline.get("identity_issues", []),
        "coverage": {
            "match_seconds": match_seconds,
            "observed_s": round(len(in_match) * step, 1),
            "positioned_s": round(len(positioned) * step, 1),
            "positioned_share": _share(len(positioned) * step, match_seconds),
        },
        "launches": _launch_counts(timeline["launches"]),
    }
    if not positioned:
        features["positions"] = None
        return features

    teleop = [r for r in positioned if r["phase"] in ("teleop", "endgame")]
    auto = [r for r in positioned if r["phase"] == "auto"]
    features["movement"] = _movement(positioned, step)
    features["zones"] = {
        "auto": _distribution(r["zone"] for r in auto),
        "teleop": _distribution(r["zone"] for r in teleop),
        "teleop_lanes": _distribution(r["lane"] for r in teleop),
        "own_half_share": _share(sum(1 for r in teleop if r["ax"] < geometry["field_length_ft"] / 2),
                                 len(teleop)),
    }
    features["start"] = _start(rows)
    features["auto_path"] = _auto_path(rows, season)
    features["shooting"] = _shooting(timeline["launches"], rows, geometry)
    features["defense"] = _defense(timeline, rows, all_timelines, all_rows, geometry, step)
    features["endgame"] = _endgame(timeline["launches"], positioned, season, match_seconds)
    return features


# ----------------------------------------------------------------------------- pieces

def _share(part, whole):
    return round(part / whole, 3) if whole else None


def _distribution(values) -> dict:
    counts: dict[str, int] = {}
    for value in values:
        if value is not None:
            counts[value] = counts.get(value, 0) + 1
    total = sum(counts.values())
    return {key: round(count / total, 3) for key, count in sorted(counts.items())} if total else {}


def _launch_counts(launches) -> dict:
    by_phase: dict[str, int] = {}
    for launch in launches:
        by_phase[launch["phase"]] = by_phase.get(launch["phase"], 0) + 1
    return {"total": sum(by_phase.values()), "by_phase": by_phase}


def _movement(positioned, step) -> dict:
    speeds = [r["speed"] for r in positioned if r["speed"] is not None]
    moving = [s for s in speeds if s >= STATIONARY_FTPS]
    longest = current = 0
    previous_t = None
    for r in positioned:
        still = r["speed"] is not None and r["speed"] < STATIONARY_FTPS
        contiguous = previous_t is not None and abs(r["t"] - previous_t - step) < 1e-6
        current = current + 1 if still and contiguous else (1 if still else 0)
        longest = max(longest, current)
        previous_t = r["t"]
    return {
        # Smoothed speed x time; summing raw step distances would count jitter as driving.
        "distance_ft": round(sum(moving) * step, 1),
        "median_moving_speed_ftps": round(median(moving), 2) if moving else None,
        "top_speed_ftps": round(_percentile(speeds, 0.95), 2) if speeds else None,
        "stationary_share": _share(len(speeds) - len(moving), len(speeds)),
        "longest_stop_s": round(longest * step, 1),
    }


def _percentile(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1) + 0.5))]


def _nearest(rows, match_time, tolerance):
    best = None
    for r in rows:
        if r["ax"] is None:
            continue
        gap = abs(r["mt"] - match_time)
        if gap <= tolerance and (best is None or gap < abs(best["mt"] - match_time)):
            best = r
    return best


def _start(rows) -> dict | None:
    """Where the robot stood when the match began: the positioned sample nearest time zero."""
    sample = _nearest(rows, 0.0, START_WINDOW_S)
    if sample is None:
        return None
    return {"ax": sample["ax"], "ay": sample["ay"], "zone": sample["zone"], "lane": sample["lane"]}


def _auto_path(rows, season) -> dict:
    auto_seconds = int(season.get("auto_seconds", 0))
    points = []
    for second in range(0, auto_seconds + 1, int(AUTO_STEP_S)):
        sample = _nearest(rows, float(second), 0.3)
        points.append([sample["ax"], sample["ay"]] if sample else None)
    known = [p for p in points if p]
    distance = sum(math.dist(a, b) for a, b in zip(points, points[1:]) if a and b)
    return {"points": points, "observed_share": _share(len(known), len(points)),
            "distance_ft": round(distance, 1)}


def cluster_points(points, radius=SPOT_RADIUS_FT) -> list[dict]:
    """Greedy leader clustering, then one centroid refinement. Deterministic for a given order.

    Shooting spots are few and well separated in practice, so a simple method that a person can
    check by eye beats a tuned one nobody can explain.
    """
    clusters: list[dict] = []
    for point in points:
        home = next((c for c in clusters if math.dist(c["centre"], point) <= radius), None)
        if home is None:
            clusters.append({"centre": list(point), "members": [point]})
        else:
            home["members"].append(point)
            n = len(home["members"])
            home["centre"] = [sum(p[0] for p in home["members"]) / n,
                              sum(p[1] for p in home["members"]) / n]
    return sorted(clusters, key=lambda c: -len(c["members"]))


def _volleys(launches) -> list[list[dict]]:
    volleys: list[list[dict]] = []
    for launch in launches:
        if volleys and launch["t"] - volleys[-1][-1]["t"] <= VOLLEY_GAP_S:
            volleys[-1].append(launch)
        else:
            volleys.append([launch])
    return volleys


def _dwell(volley, rows, radius) -> float | None:
    """How long the robot stayed at the spot it fired this volley from, before and during it."""
    located = [l for l in volley if l["ax"] is not None]
    if not located:
        return None
    centre = (sum(l["ax"] for l in located) / len(located), sum(l["ay"] for l in located) / len(located))
    times = [r["t"] for r in rows]
    first, last = volley[0]["t"], volley[-1]["t"]
    inside = [i for i, r in enumerate(rows)
              if r["ax"] is not None and math.dist((r["ax"], r["ay"]), centre) <= radius]
    if not inside:
        return None
    inside_set = set(inside)
    # Expand outward from the sample nearest the first launch while the robot stays in radius.
    anchor = min(inside, key=lambda i: abs(times[i] - first))
    lo = hi = anchor
    while lo - 1 in inside_set:
        lo -= 1
    while hi + 1 in inside_set:
        hi += 1
    # A volley fired while driving through the radius still lasted until its last launch.
    return round(max(times[hi], last) - min(times[lo], first), 1)


def _shooting(launches, rows, geometry) -> dict:
    hub = geometry["hubs"]["own_hub"]
    located = [l for l in launches if l["ax"] is not None]
    spots = []
    for cluster in cluster_points([(l["ax"], l["ay"]) for l in located]):
        cx, cy = cluster["centre"]
        spots.append({"ax": round(cx, 2), "ay": round(cy, 2), "launches": len(cluster["members"]),
                      "share": _share(len(cluster["members"]), len(located)),
                      "distance_to_hub_ft": round(math.dist((cx, cy), (hub["x"], hub["y"])), 1)})
    volleys = _volleys(launches)
    starts = [v[0]["t"] for v in volleys]
    dwell = [d for d in (_dwell(v, rows, SPOT_RADIUS_FT) for v in volleys) if d is not None]
    speeds = [l["speed"] for l in located if l["speed"] is not None]
    distances = [math.dist((l["ax"], l["ay"]), (hub["x"], hub["y"])) for l in located]
    return {
        "launches": len(launches),
        "located_launches": len(located),
        "points": [[l["ax"], l["ay"], l["mt"]] for l in located],
        "spots": spots,
        "volleys": len(volleys),
        "median_launches_per_volley": median(len(v) for v in volleys) if volleys else None,
        # Start-to-start time between visits to shoot. Not a Doc 0 "cycle", which is reload to
        # reload and needs intake events; this needs only launches.
        "median_volley_interval_s": round(median(b - a for a, b in zip(starts, starts[1:])), 1)
        if len(starts) > 1 else None,
        "median_dwell_s": round(median(dwell), 1) if dwell else None,
        "on_the_move_share": _share(sum(1 for s in speeds if s >= STATIONARY_FTPS), len(speeds)),
        "median_distance_to_hub_ft": round(median(distances), 1) if distances else None,
    }


def _defense(timeline, rows, all_timelines, all_rows, geometry, step) -> dict:
    """Time spent in the opponent's half close to an opponent robot, and whom it was."""
    alliance = timeline.get("alliance")
    half = geometry["field_length_ft"] / 2
    deep = sum(1 for r in rows if r["phase"] in ("teleop", "endgame") and r["ax"] is not None
               and r["ax"] > half)
    if alliance is None:
        return {"opponent_half_s": round(deep * step, 1), "near_opponent_s": None, "marked": None}
    opponents = [t for t in all_timelines if t.get("alliance") not in (None, alliance)]
    indexed = {t["robot_key"]: {round(r["t"], 1): r for r in all_rows[t["robot_key"]] if r["x"] is not None}
               for t in opponents}
    close: dict[str, int] = {}
    for r in rows:
        if r["phase"] not in ("teleop", "endgame") or r["ax"] is None or r["ax"] <= half:
            continue
        for opponent in opponents:
            other = indexed[opponent["robot_key"]].get(round(r["t"], 1))
            if other and math.dist((r["x"], r["y"]), (other["x"], other["y"])) <= DEFENSE_RADIUS_FT:
                label = str(opponent["team"]) if opponent.get("team") else opponent["robot_key"]
                close[label] = close.get(label, 0) + 1
    marked = max(close, key=close.get) if close else None
    return {
        "opponent_half_s": round(deep * step, 1),
        "near_opponent_s": round(sum(close.values()) * step, 1),
        "marked": marked,
        "marked_s": round(close[marked] * step, 1) if marked else None,
    }


def _endgame(launches, positioned, season, match_seconds) -> dict:
    endgame_from = match_seconds - float(season.get("endgame_seconds", 0))
    last = max((l["mt"] for l in launches if l["mt"] is not None), default=None)
    final = positioned[-1]
    return {
        "last_launch_mt": last,
        "stopped_scoring_before_endgame_s": round(endgame_from - last, 1) if last is not None else None,
        "final_position": {"ax": final["ax"], "ay": final["ay"], "zone": final["zone"],
                           "lane": final["lane"], "mt": final["mt"]},
    }
