"""Alliance-selection engine: our own pick list and pick calls (PLAN-2026 section 7).

Ignores the official rankings and builds our own, from the profiles: rank every available team as a
partner by offensive power minus how much it would conflict with us, simulate a pick, and judge
whether to accept an offer. The conflict model is the same one the match engine uses (engine.py):
two robots "conflict" when they want the same auto cell or the same scoring zone, which we already
measure.

Speed is the hard requirement -- this runs live during a draft -- so everything here is arithmetic
over the cached profile dicts (no re-analysis, no I/O). Ranking an 18-team pool against a 3-team
alliance is a few hundred dict lookups; it is sub-second with room to spare.
"""

from __future__ import annotations

from .engine import AUTO_HOT_INTENSITY

#: A unit of conflict is worth avoiding this many shots/match of raw power. Tunable: higher means we
#: weight "they stay out of each other's way" more heavily against raw scoring.
CONFLICT_PENALTY = 2.0


def offensive_power(profile: dict) -> float:
    """One number for ranking: shots per match, the clearest offensive proxy we have. 0 when
    unknown, so a team with no data sorts last instead of erroring."""
    return float(profile.get("launches_per_match") or 0.0)


def _auto_footprint(profile: dict) -> dict:
    cells = (profile.get("auto_map") or {}).get("heatmap") or []
    return {(c["x"], c["y"]): c.get("intensity", 0.0)
            for c in cells if c.get("intensity", 0.0) >= AUTO_HOT_INTENSITY}


def _effective_zones(profile: dict) -> dict:
    return {(z["zone"], z["lane"]): z["weight"]
            for z in (profile.get("scoring_map") or {}).get("effective_zones") or []}


def conflict(a: dict, b: dict) -> dict:
    """How much two robots would step on each other: shared hot auto cells, plus scoring zones they
    both own (counted by the overlap they actually contest -- the smaller of the two weights)."""
    fa, fb = _auto_footprint(a), _auto_footprint(b)
    auto = sum(min(fa[k], fb[k]) for k in fa.keys() & fb.keys())
    za, zb = _effective_zones(a), _effective_zones(b)
    zones = sum(min(za[k], zb[k]) for k in za.keys() & zb.keys())
    return {"auto": round(auto, 3), "zones": round(zones, 3), "total": round(auto + zones, 3)}


def partner_value(our: list[dict], candidate: dict) -> dict:
    """Score one candidate as a partner for our current alliance: its power, its total conflict with
    each of our robots, and the combined score we rank on."""
    power = offensive_power(candidate)
    total_conflict = sum(conflict(member, candidate)["total"] for member in our)
    return {
        "team": candidate.get("team"),
        "power": round(power, 1),
        "conflict": round(total_conflict, 3),
        "score": round(power - CONFLICT_PENALTY * total_conflict, 2),
    }


def rank_partners(our_teams: list[int], available_teams: list[int], profiles: dict[int, dict],
                  limit: int = 18) -> list[dict]:
    """Our pick list: available teams ranked best partner first, by power minus conflict with us.
    Teams already on our alliance, and teams we have no profile for, are left out."""
    our = [profiles[t] for t in our_teams if t in profiles]
    ranked = [partner_value(our, profiles[t]) for t in available_teams
              if t not in our_teams and t in profiles]
    ranked.sort(key=lambda v: (-v["score"], -v["power"]))
    return ranked[:limit]


def best_pick(our_teams: list[int], available_teams: list[int], profiles: dict[int, dict]) -> dict | None:
    """The single team to call next, or None if the pool has nobody we have data on."""
    ranked = rank_partners(our_teams, available_teams, profiles, limit=1)
    return ranked[0] if ranked else None


def alliance_strength(teams: list[int], profiles: dict[int, dict]) -> dict:
    """A hypothetical alliance's combined power minus its internal conflict -- for roleplay ("1
    picks 2, 3 picks 4") and for comparing accept-vs-decline."""
    members = [profiles[t] for t in teams if t in profiles]
    power = sum(offensive_power(m) for m in members)
    internal = 0.0
    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            internal += conflict(members[i], members[j])["total"]
    return {"teams": list(teams), "power": round(power, 1),
            "internal_conflict": round(internal, 3),
            "score": round(power - CONFLICT_PENALTY * internal, 2)}


def should_accept(our_team: int, offering_alliance: list[int], available_teams: list[int],
                  profiles: dict[int, dict]) -> dict:
    """Offered a spot on `offering_alliance`: accept (join them) or decline (stay captain and pick
    our best available)? Compares the two resulting alliances by the same score."""
    accept = alliance_strength([*offering_alliance, our_team], profiles)
    pool = [t for t in available_teams if t not in offering_alliance and t != our_team]
    best = best_pick([our_team], pool, profiles)
    decline_teams = [our_team, best["team"]] if best else [our_team]
    decline = alliance_strength(decline_teams, profiles)
    return {
        "accept": accept,
        "decline": decline,
        "decline_best_pick": best["team"] if best else None,
        "recommendation": "accept" if accept["score"] >= decline["score"] else "decline",
    }
