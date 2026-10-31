"""Match strategy recommendations: turn team profiles into an alliance plan (PLAN-2026 section 6).

Rules and statistics on top of the profiles (profile.py) and the heatmaps (heatmap.py), never a
learned model -- the same reason as profile.py: a few matches per team is too little to fit
anything, and a drive coach has to be able to check every call against the video. So every
recommendation names the evidence that produced it.

Three questions, from the chat and Noah's notes:
  - Auto: which auto puts our three robots on distinct, high-value paths -- least conflict with each
    other, most contesting the opponents? 50-75% of teams run only 2-3 autos, so auto is predictable
    enough to plan against.
  - Teleop: where do the opponents actually score (deny there), and where are we effective (feed
    there)?
  - Defense: who should we expect to come after us?

It reads what the profile already carries -- auto_map, scoring_map.effective_zones, and the defense
tendency -- so a team with no field positions simply contributes nothing instead of erroring.
"""

from __future__ import annotations

#: A team "runs auto" in a cell this bright or brighter (auto_map intensity is 0..1).
AUTO_HOT_INTENSITY = 0.4
#: One auto cell holding this share of a team's auto weight, over 2+ matches, is a predictable auto.
AUTO_PREDICTABLE_SHARE = 0.4
#: A team that marked an opponent in at least this share of its matches is a defender.
DEFENSE_SHARE = 0.5


def _auto_hot_cells(profile: dict) -> dict:
    cells = (profile.get("auto_map") or {}).get("heatmap") or []
    return {(c["x"], c["y"]): c for c in cells if c.get("intensity", 0.0) >= AUTO_HOT_INTENSITY}


def auto_conflicts(profiles: list[dict]) -> list[dict]:
    """Field cells where two or more alliance partners both run auto. They would contest the same
    space or duplicate the same job, so the auto plan should split them up."""
    by_cell: dict[tuple, list[tuple]] = {}
    for profile in profiles:
        for key, cell in _auto_hot_cells(profile).items():
            by_cell.setdefault(key, []).append((profile.get("team"), cell))
    conflicts = []
    for (x, y), hits in by_cell.items():
        teams = sorted(t for t, _ in hits if t is not None)
        if len(teams) >= 2:
            conflicts.append({"x": x, "y": y, "teams": teams,
                              "intensity": round(max(c["intensity"] for _, c in hits), 3)})
    return sorted(conflicts, key=lambda c: -c["intensity"])


def predictable_autos(profiles: list[dict]) -> list[dict]:
    """Opponents whose auto concentrates in one place across matches -- the ones worth contesting,
    because you can predict where their auto goes."""
    out = []
    for profile in profiles:
        cells = (profile.get("auto_map") or {}).get("heatmap") or []
        total = sum(c["weight"] for c in cells)
        if not total:
            continue
        top = cells[0]  # heatmap is sorted by weight
        share = top["weight"] / total
        if share >= AUTO_PREDICTABLE_SHARE and top["matches"] >= 2:
            out.append({"team": profile.get("team"), "x": top["x"], "y": top["y"],
                        "share": round(share, 2), "matches": top["matches"]})
    return sorted(out, key=lambda a: -a["share"])


def aggregate_zones(profiles: list[dict]) -> list[dict]:
    """Effective scoring zones pooled across a set of teams, by combined recency weight. Used both
    for 'deny the opponents here' and 'we are strong here', so the two read the same way."""
    agg: dict[tuple, dict] = {}
    for profile in profiles:
        for zone in (profile.get("scoring_map") or {}).get("effective_zones") or []:
            key = (zone["zone"], zone["lane"])
            entry = agg.setdefault(key, {"zone": zone["zone"], "lane": zone["lane"],
                                         "weight": 0.0, "teams": []})
            entry["weight"] += zone["weight"]
            if profile.get("team") is not None:
                entry["teams"].append(profile["team"])
    for entry in agg.values():
        entry["weight"] = round(entry["weight"], 2)
        entry["teams"] = sorted(set(entry["teams"]))
    return sorted(agg.values(), key=lambda z: -z["weight"])


def defense_watch(profiles: list[dict]) -> list[dict]:
    """Teams that play defense often enough to plan around."""
    out = []
    for profile in profiles:
        defense = (profile.get("tendencies") or {}).get("defense") or {}
        of = defense.get("of") or 0
        if of and defense.get("matches_played", 0) / of >= DEFENSE_SHARE:
            out.append({"team": profile.get("team"), "matches_played": defense["matches_played"], "of": of})
    return out


def _zone_label(zone: str, lane: str) -> str:
    return f"{zone.replace('_', ' ')} / {lane} lane"


def _advice(plan: dict) -> list[dict]:
    """Human calls with the evidence behind each, ready for the drive coach's card."""
    advice: list[dict] = []
    for conflict in plan["auto"]["partner_conflicts"]:
        teams = " and ".join(str(t) for t in conflict["teams"])
        advice.append({
            "id": "auto_partner_conflict",
            "advice": f"Split {teams} in auto: both camp ~({conflict['x']}, {conflict['y']}) ft, "
                      "so give one that spot and send the other elsewhere.",
            "evidence": f"Both run auto in the same cell (intensity {conflict['intensity']:.2f}).",
        })
    for auto in plan["auto"]["contest_opponent_autos"]:
        advice.append({
            "id": "contest_opponent_auto",
            "advice": f"Contest team {auto['team']}'s auto near ({auto['x']}, {auto['y']}) ft -- it is "
                      "predictable, so plan your auto onto its game pieces.",
            "evidence": f"{auto['share'] * 100:.0f}% of its auto there over {auto['matches']} matches.",
        })
    deny = plan["teleop"]["deny"]
    if deny:
        top = deny[0]
        teams = ", ".join(str(t) for t in top["teams"])
        advice.append({
            "id": "deny_scoring",
            "advice": f"Deny {_zone_label(top['zone'], top['lane'])}: it is the opponents' best scoring "
                      "ground, so contesting it prevents the most points.",
            "evidence": f"Opponents ({teams}) score most there (weight {top['weight']}).",
        })
    for watch in plan["teleop"]["defense_watch"]:
        advice.append({
            "id": "expect_defense",
            "advice": f"Expect defense from team {watch['team']}: keep a scoring route that avoids it.",
            "evidence": f"Played defense in {watch['matches_played']} of {watch['of']} matches.",
        })
    strengths = plan["teleop"]["our_strengths"]
    if strengths:
        top = strengths[0]
        advice.append({
            "id": "play_to_strength",
            "advice": f"Feed our scorers at {_zone_label(top['zone'], top['lane'])} -- that is where we "
                      "convert best.",
            "evidence": f"Our alliance scores most there (weight {top['weight']}).",
        })
    return advice


def build_matchplan(our_teams: list[int], their_teams: list[int], profiles: dict[int, dict]) -> dict:
    """An alliance plan for one match: auto separation + opponent-auto contests, teleop deny/feed,
    and defense to expect. `profiles` maps team -> profile built WITH geometry (so it carries
    auto_map and scoring_map). Teams missing from `profiles` are simply skipped."""
    our = [profiles[t] for t in our_teams if t in profiles]
    their = [profiles[t] for t in their_teams if t in profiles]
    plan = {
        "alliance": list(our_teams),
        "opponents": list(their_teams),
        "missing": [t for t in [*our_teams, *their_teams] if t not in profiles],
        "auto": {
            "partner_conflicts": auto_conflicts(our),
            "contest_opponent_autos": predictable_autos(their),
        },
        "teleop": {
            "deny": aggregate_zones(their),
            "our_strengths": aggregate_zones(our),
            "defense_watch": defense_watch(their),
        },
    }
    plan["advice"] = _advice(plan)
    return plan
