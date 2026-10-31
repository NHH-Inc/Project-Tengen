"""Team profiles: patterns across matches, a prediction for the next one, and counters.

Statistics and rules, not a learned model: a team plays 10-12 qualification matches per event,
far too few to fit anything per team. Every tendency carries the number of matches behind it,
and every counter names the evidence that triggered it, so a strategist can check the claim
against the video instead of trusting a number.
"""

from __future__ import annotations

import math
import re
from statistics import median

from .features import SPOT_RADIUS_FT, cluster_points

#: Two auto paths whose per-second points are this close on average are the same routine.
AUTO_SAME_PATH_FT = 3.0
#: A match counts as "played defense" with at least this much time marking an opponent.
DEFENSE_MATCH_S = 15.0
#: Below this many matches with positions, tendencies are reported but no counter is offered.
MIN_MATCHES_FOR_COUNTERS = 3


def match_order(features: dict):
    """Sort key: event, then qualification before playoffs, then match number."""
    match_id = features.get("match_id") or ""
    found = re.match(r"(\d{4}[a-z0-9]+?)_(qm|ef|qf|sf|f)(\d+)(?:m(\d+))?$", match_id)
    if not found:
        return (match_id, 9, 0, 0)
    event, level, number, game = found.groups()
    return (event, ["qm", "ef", "qf", "sf", "f"].index(level), int(number), int(game or 0))


def _mode(values):
    counts: dict = {}
    for value in values:
        if value is not None:
            counts[value] = counts.get(value, 0) + 1
    if not counts:
        return None, 0
    best = max(counts, key=lambda k: (counts[k], str(k)))
    return best, counts[best]


def _median(values):
    values = [v for v in values if v is not None]
    return round(median(values), 2) if values else None


def _path_distance(a, b) -> float | None:
    pairs = [(p, q) for p, q in zip(a, b) if p and q]
    if len(pairs) < 3:
        return None
    return sum(math.dist(p, q) for p, q in pairs) / len(pairs)


def auto_variants(paths: list[list]) -> list[dict]:
    """Group a team's auto paths into routines. Each routine keeps its member indices."""
    variants: list[dict] = []
    for index, path in enumerate(paths):
        home = None
        for variant in variants:
            distance = _path_distance(variant["path"], path)
            if distance is not None and distance <= AUTO_SAME_PATH_FT:
                home = variant
                break
        if home is None:
            variants.append({"path": path, "matches": [index]})
        else:
            home["matches"].append(index)
    return sorted(variants, key=lambda v: -len(v["matches"]))


def build_profile(team_matches: list[dict], geometry: dict | None = None) -> dict:
    """Profile one team from its per-match feature dicts (any order; sorted here).

    Pass `geometry` (contracts/strategy/<season>.json) to also attach a scoring heatmap and
    effective-zone map; without it the profile is unchanged, so older callers keep working.
    """
    matches = sorted(team_matches, key=match_order)
    located = [m for m in matches if m.get("positions", True) is not None and "movement" in m]
    team = matches[0].get("team") if matches else None
    profile = {
        "team": team,
        "matches": [m.get("match_id") for m in matches],
        "n_matches": len(matches),
        "n_with_positions": len(located),
        "launches_per_match": _median(m["launches"]["total"] for m in matches),
        # Timing needs no calibration, so it is summarised over every match, located or not.
        "timing": {
            "median_volley_interval_s": _median((m.get("shooting") or {}).get("median_volley_interval_s")
                                                for m in matches),
            "median_launches_per_volley": _median((m.get("shooting") or {}).get("median_launches_per_volley")
                                                  for m in matches),
            "median_stopped_scoring_before_endgame_s": _median(
                (m.get("endgame") or {}).get("stopped_scoring_before_endgame_s") for m in matches),
        },
    }
    if geometry is not None:
        from .heatmap import auto_map, scoring_map
        profile["scoring_map"] = scoring_map(matches, geometry)
        profile["auto_map"] = auto_map(matches, geometry)
    if not located:
        profile["tendencies"] = None
        profile["prediction"] = None
        profile["counters"] = []
        return profile

    starts = [m.get("start") for m in located]
    start_lane, start_lane_n = _mode(s["lane"] for s in starts if s)
    known_starts = [s for s in starts if s]
    paths = [m["auto_path"]["points"] for m in located]
    variants = auto_variants(paths)
    launches = [(s["ax"], s["ay"], m.get("match_id"))
                for m in located for s in _launch_points(m)]
    spots = []
    for cluster in cluster_points([(x, y) for x, y, _ in launches], SPOT_RADIUS_FT):
        cx, cy = cluster["centre"]
        in_matches = {mid for x, y, mid in launches if math.dist((x, y), (cx, cy)) <= SPOT_RADIUS_FT}
        spots.append({"ax": round(cx, 1), "ay": round(cy, 1), "launches": len(cluster["members"]),
                      "matches": len(in_matches)})
    defense = [m["defense"].get("near_opponent_s") for m in located]
    defended_matches = [d for d in defense if d is not None and d >= DEFENSE_MATCH_S]
    zone_names = sorted({z for m in located for z in m["zones"]["teleop"]})
    lanes = sorted({z for m in located for z in m["zones"]["teleop_lanes"]})

    tendencies = {
        "start": {
            "lane": start_lane, "lane_matches": start_lane_n, "known": len(known_starts),
            "mean_ax": _median(s["ax"] for s in known_starts),
            "mean_ay": _median(s["ay"] for s in known_starts),
        },
        "auto": {
            "routines": [{"matches": len(v["matches"]),
                          "match_ids": [located[i].get("match_id") for i in v["matches"]],
                          "path": v["path"]} for v in variants],
            "median_distance_ft": _median(m["auto_path"]["distance_ft"] for m in located),
        },
        "shooting": {
            "spots": spots[:3],
            "median_volley_interval_s": _median(m["shooting"]["median_volley_interval_s"] for m in located),
            "median_dwell_s": _median(m["shooting"]["median_dwell_s"] for m in located),
            "on_the_move_share": _median(m["shooting"]["on_the_move_share"] for m in located),
            "median_distance_to_hub_ft": _median(m["shooting"]["median_distance_to_hub_ft"] for m in located),
        },
        "zones": {z: _median(m["zones"]["teleop"].get(z, 0.0) for m in located) for z in zone_names},
        "lanes": {z: _median(m["zones"]["teleop_lanes"].get(z, 0.0) for m in located) for z in lanes},
        "movement": {
            "top_speed_ftps": _median(m["movement"]["top_speed_ftps"] for m in located),
            "stationary_share": _median(m["movement"]["stationary_share"] for m in located),
        },
        "defense": {
            "matches_played": len(defended_matches),
            "of": sum(1 for d in defense if d is not None),
            "median_near_opponent_s": _median(defense),
        },
        "endgame": {
            "median_stopped_scoring_before_endgame_s":
                _median(m["endgame"]["stopped_scoring_before_endgame_s"] for m in located),
        },
    }
    profile["tendencies"] = tendencies
    profile["prediction"] = predict(tendencies, profile)
    profile["counters"] = counters(tendencies, profile)
    return profile


def _launch_points(features: dict) -> list[dict]:
    return [{"ax": x, "ay": y} for x, y, *_ in features.get("shooting", {}).get("points", [])]


def predict(tendencies: dict, profile: dict) -> dict:
    routines = tendencies["auto"]["routines"]
    spots = tendencies["shooting"]["spots"]
    defense = tendencies["defense"]
    return {
        "start_lane": tendencies["start"]["lane"],
        "auto_routine": 0 if routines else None,
        "auto_routine_confidence": _confidence(routines[0]["matches"], profile["n_with_positions"])
        if routines else None,
        "primary_spot": {"ax": spots[0]["ax"], "ay": spots[0]["ay"]} if spots else None,
        "launches": profile["launches_per_match"],
        "plays_defense": (defense["matches_played"] / defense["of"] >= 0.5) if defense["of"] else None,
    }


def _confidence(hits: int, n: int) -> float | None:
    """Laplace-smoothed share, so one match out of one is not reported as certainty."""
    return round((hits + 1) / (n + 2), 2) if n else None


def counters(tendencies: dict, profile: dict) -> list[dict]:
    """Rule library: tendency -> advice, each with the evidence that triggered it."""
    n = profile["n_with_positions"]
    if n < MIN_MATCHES_FOR_COUNTERS:
        return []
    advice = []
    routines = tendencies["auto"]["routines"]
    if routines and routines[0]["matches"] >= max(2, math.ceil(0.7 * n)):
        advice.append({
            "id": "predictable_auto",
            "advice": "Their auto is predictable: plan your auto to contest the game pieces on its path.",
            "evidence": f"Same auto routine in {routines[0]['matches']} of {n} matches.",
            "matches": routines[0]["match_ids"],
        })
    spots = tendencies["shooting"]["spots"]
    if spots and spots[0]["matches"] >= max(2, math.ceil(0.6 * n)):
        spot = spots[0]
        dwell = tendencies["shooting"]["median_dwell_s"]
        moving = tendencies["shooting"]["on_the_move_share"]
        stops = moving is not None and moving < 0.3
        advice.append({
            "id": "favourite_spot",
            "advice": (f"Deny their favourite shooting spot (~{spot['ax']:.0f} ft out, "
                       f"{spot['ay']:.0f} ft across in their frame)"
                       + (" -- they stop to shoot there, so defense at the spot costs them time."
                          if stops else ".")),
            "evidence": (f"Shot from it in {spot['matches']} of {n} matches"
                         + (f"; median dwell {dwell:.1f} s" if dwell is not None else "") + "."),
        })
    defense = tendencies["defense"]
    if defense["of"] and defense["matches_played"] / defense["of"] >= 0.5:
        advice.append({
            "id": "expect_defense",
            "advice": "Expect them to play defense: keep a scoring route that avoids their half, "
                      "or give your best scorer a screen.",
            "evidence": f"Marked an opponent for {DEFENSE_MATCH_S:.0f}+ s in "
                        f"{defense['matches_played']} of {defense['of']} matches.",
        })
    stopped = tendencies["endgame"]["median_stopped_scoring_before_endgame_s"]
    if stopped is not None and stopped > 10:
        advice.append({
            "id": "early_endgame",
            "advice": "They leave scoring early for the endgame: the last stretch of teleop is "
                      "yours to contest.",
            "evidence": f"Median last shot {stopped:.0f} s before the endgame period.",
        })
    lanes = tendencies["lanes"]
    if lanes:
        lane, share = max(lanes.items(), key=lambda item: item[1])
        if share >= 0.6:
            advice.append({
                "id": "lane_bias",
                "advice": f"They live in the {lane} lane (their view): block it, or route around it.",
                "evidence": f"Median {share:.0%} of teleop in that lane over {n} matches.",
            })
    return advice


# ----------------------------------------------------------------------------- evaluation

def next_match_test(team_matches: dict[int, list[dict]], min_history: int = 2) -> dict:
    """Predict each team's match k+1 from matches 1..k and score it. The strategy yardstick.

    Each prediction is also scored against a baseline that ignores the team's history (the most
    common answer across all teams), so a profile that only restates the field's habits does not
    get credit for knowing the team.
    """
    trials = {"start_lane": [], "primary_spot": [], "plays_defense": []}
    baseline = {"start_lane": [], "plays_defense": []}
    all_lanes = [m["start"]["lane"] for ms in team_matches.values() for m in ms
                 if m.get("start")]
    field_lane, _ = _mode(all_lanes)
    for matches in team_matches.values():
        ordered = sorted(matches, key=match_order)
        for k in range(min_history, len(ordered)):
            history, actual = ordered[:k], ordered[k]
            prediction = build_profile(history).get("prediction")
            if not prediction or "movement" not in actual:
                continue
            if prediction["start_lane"] and actual.get("start"):
                trials["start_lane"].append(prediction["start_lane"] == actual["start"]["lane"])
                baseline["start_lane"].append(field_lane == actual["start"]["lane"])
            spots = actual["shooting"]["spots"]
            if prediction["primary_spot"] and spots:
                trials["primary_spot"].append(
                    math.dist((prediction["primary_spot"]["ax"], prediction["primary_spot"]["ay"]),
                              (spots[0]["ax"], spots[0]["ay"])) <= 2 * SPOT_RADIUS_FT)
            near = actual["defense"].get("near_opponent_s")
            if prediction["plays_defense"] is not None and near is not None:
                trials["plays_defense"].append(prediction["plays_defense"] == (near >= DEFENSE_MATCH_S))
                baseline["plays_defense"].append(near < DEFENSE_MATCH_S)
    report = {}
    for key, outcomes in trials.items():
        report[key] = {"n": len(outcomes),
                       "accuracy": round(sum(outcomes) / len(outcomes), 3) if outcomes else None}
        if key in baseline:
            base = baseline[key]
            report[key]["baseline_accuracy"] = round(sum(base) / len(base), 3) if base else None
    return report
