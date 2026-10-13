"""Command line for the strategy layer.

    python -m ingest.strategy clock     RUN [RUN ...]   (read the scoreboard clock: needs Tesseract)
    python -m ingest.strategy timeline  RUN [RUN ...]
    python -m ingest.strategy features  RUN [RUN ...]
    python -m ingest.strategy profile   RUN [RUN ...] [--team 254] [--json out.json]
    python -m ingest.strategy evaluate  RUN [RUN ...]

RUN is an analysed match folder (it contains job.json and tracks.jsonl), or a folder of them such
as data/desktop/runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .clock import fit_clock, read_clock
from .features import match_features
from .profile import build_profile, match_order, next_match_test
from .timeline import SEASONS_DIR, build_timelines, load_geometry, read_json, write_timelines


def discover(paths: list[str]) -> list[Path]:
    runs = []
    for raw in paths:
        path = Path(raw)
        if (path / "tracks.jsonl").exists():
            runs.append(path)
        else:
            runs.extend(sorted(p.parent for p in path.glob("*/tracks.jsonl")))
    return runs


def analyse(run: Path) -> list[dict]:
    timelines = build_timelines(run)
    write_timelines(run, timelines)
    season_number = timelines[0]["season"] if timelines else 2026
    features = match_features(timelines, load_geometry(season_number),
                              read_json(SEASONS_DIR / f"{season_number}.json"))
    (run / "strategy_features.json").write_text(json.dumps(features, indent=1) + "\n", encoding="utf-8")
    return features


def _fmt(value, suffix=""):
    return "-" if value is None else f"{value}{suffix}"


def print_match(run: Path, features: list[dict]) -> None:
    print(f"\n{run.name}")
    print(f"  {'robot':<16}{'all':<6}{'seen':>6}{'xy':>6}{'launch':>7}{'spot':>14}{'volley':>7}{'dwell':>7}{'def':>6}  start")
    for f in features:
        cov = f["coverage"]
        shooting = f.get("shooting") or {}
        spot = (shooting.get("spots") or [None])[0]
        start = f.get("start")
        print(f"  {f['robot_key']:<16}{_fmt(f['alliance']):<6}{cov['observed_s']:>6.0f}{cov['positioned_s']:>6.0f}"
              f"{f['launches']['total']:>7}"
              f"{(f'({spot['ax']:.0f},{spot['ay']:.0f})x{spot['launches']}' if spot else '-'):>14}"
              f"{_fmt(shooting.get('median_volley_interval_s')):>7}{_fmt(shooting.get('median_dwell_s')):>7}"
              f"{_fmt((f.get('defense') or {}).get('near_opponent_s')):>6}  "
              f"{(start['lane'] + ' ' + str(start['zone'])) if start else '-'}")


def print_profile(profile: dict) -> None:
    print(f"\nTeam {profile['team']}  --  {profile['n_matches']} matches "
          f"({profile['n_with_positions']} with field positions)")
    tendencies = profile.get("tendencies")
    if not tendencies:
        print("  No field positions in any match: tendencies unavailable.")
        return
    t = tendencies
    print(f"  Start lane: {_fmt(t['start']['lane'])} ({t['start']['lane_matches']}/{t['start']['known']})")
    routines = t["auto"]["routines"]
    if routines:
        print(f"  Auto routines: " + ", ".join(f"#{i} x{r['matches']}" for i, r in enumerate(routines)))
    for spot in t["shooting"]["spots"]:
        print(f"  Shooting spot ({spot['ax']}, {spot['ay']}) ft: {spot['launches']} launches in {spot['matches']} matches")
    s = t["shooting"]
    print(f"  Volley interval {_fmt(s['median_volley_interval_s'], ' s')}, dwell {_fmt(s['median_dwell_s'], ' s')}, "
          f"on the move {_fmt(s['on_the_move_share'])}, range {_fmt(s['median_distance_to_hub_ft'], ' ft')}")
    print(f"  Defense in {t['defense']['matches_played']} of {t['defense']['of']} matches")
    for counter in profile["counters"]:
        print(f"  COUNTER: {counter['advice']}\n           ({counter['evidence']})")
    if not profile["counters"]:
        print("  No counters yet (needs 3+ matches with positions and a clear tendency).")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ingest.strategy", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["clock", "timeline", "features", "profile", "evaluate"])
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--team", type=int)
    parser.add_argument("--json", type=Path, help="also write the result here as JSON")
    args = parser.parse_args(argv)

    runs = discover(args.runs)
    if not runs:
        print("No analysed matches found (looked for tracks.jsonl).", file=sys.stderr)
        return 1

    if args.command == "clock":
        for run in runs:
            job = read_json(run / "job.json") if (run / "job.json").exists() else {}
            video = next((p for p in (run / "segment.mp4", Path(job.get("local_path") or ""))
                          if str(p) not in ("", ".") and p.is_file()), None)
            if video is None:
                print(f"{run.name}: video not found")
                continue
            season = read_json(SEASONS_DIR / f"{int(job.get('season') or 2026)}.json")
            readings = read_clock(video)
            clock = fit_clock(readings, season)
            (run / "match_clock.json").write_text(json.dumps(
                {**clock.as_dict(), "readings": readings}, indent=1) + "
", encoding="utf-8")
            print(f"{run.name}: auto starts {clock.auto_start_t} s ({clock.auto_seconds:.0f} s), "
                  f"teleop starts {clock.teleop_start_t} s ({clock.teleop_seconds:.0f} s); "
                  f"{clock.support['auto']}+{clock.support['teleop']} of "
                  f"{clock.support['readings']} readings agree"
                  + "".join(f"
  note: {n}" for n in clock.notes))
        return 0

    if args.command == "timeline":
        for run in runs:
            timelines = build_timelines(run)
            write_timelines(run, timelines)
            positioned = sum(1 for t in timelines if t["coverage"]["positioned_s"] > 0)
            print(f"{run.name}: {len(timelines)} robots, {positioned} with field positions")
        return 0

    per_match = [(run, analyse(run)) for run in runs]
    if args.command == "features":
        for run, features in per_match:
            print_match(run, features)
        output = [f for _, features in per_match for f in features]
    else:
        by_team: dict[int, list[dict]] = {}
        for _, features in per_match:
            for f in features:
                if f.get("team"):
                    by_team.setdefault(f["team"], []).append(f)
        if args.team:
            by_team = {args.team: by_team.get(args.team, [])}
        if args.command == "profile":
            output = [build_profile(matches) for matches in by_team.values() if matches]
            for profile in output:
                print_profile(profile)
            if not output:
                print("No robot has a team number yet. Assign teams in review first.")
        else:
            output = next_match_test(by_team)
            print(json.dumps(output, indent=2))
    if args.json:
        args.json.write_text(json.dumps(output, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
