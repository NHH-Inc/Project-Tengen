"""Robot timelines: one canonical record per robot per match, for everything strategic.

Every consumer used to read raw `tracks.jsonl` boxes at whatever rate, in whatever coordinates,
the analyzer happened to write. A timeline fixes three things once, so features never touch
pixels or analyzer quirks:

* **Rate.** Positions are resampled to a fixed grid (10 Hz). Interpolation only bridges boxes
  that are close in time and not separated by a listed gap: a hole stays a hole, because a robot
  drawn gliding through footage nobody analysed is a fabricated observation.
* **Frame.** FRC fields are rotationally symmetric, so a red robot's position is rotated 180
  degrees into the ALLIANCE FRAME (own wall at x=0, x toward the opponent, y to the driver's
  left). A team's red and blue matches then line up, which is what makes a pattern visible.
* **Identity.** Team corrections from either review UI are applied, and fragments of one team
  are merged into one robot when they do not overlap in time.

Nothing here invents a position. A track without field coordinates yields samples with null
positions, so coverage is reported honestly and position features say "unavailable".
"""

from __future__ import annotations

import json
import math
from bisect import bisect_right
from pathlib import Path
from typing import Iterable

from .clock import MatchClock, fixed_clock

SCHEMA_VERSION = 1
RATE_HZ = 10.0
#: Two boxes further apart than this are not interpolated between. At 2-30 Hz analysis rates a
#: real robot is sampled well inside this; anything longer is a detection hole, not a gap in rate.
MAX_BRIDGE_S = 0.6
#: Positions are smoothed over this window before speed is taken. Homography jitter of a few
#: inches per frame would otherwise read as a robot that never stops moving.
SMOOTH_S = 0.5
#: A launch is placed at the robot's sampled position if one exists this close in time.
LAUNCH_POSITION_TOLERANCE_S = 0.3

REPO_ROOT = Path(__file__).resolve().parents[2]
STRATEGY_DIR = REPO_ROOT / "contracts" / "strategy"
SEASONS_DIR = REPO_ROOT / "contracts" / "seasons"


# ----------------------------------------------------------------------------- file helpers

def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def load_geometry(season: int) -> dict:
    path = STRATEGY_DIR / f"{season}.json"
    if not path.exists():
        raise FileNotFoundError(f"No strategy geometry for season {season}: {path}")
    return read_json(path)


# ----------------------------------------------------------------------------- geometry

def to_alliance_frame(x: float, y: float, alliance: str | None, geometry: dict):
    """Field feet -> alliance frame. Blue owns the x=0 wall; red is rotated about the centre."""
    if alliance == "blue":
        return x, y
    if alliance == "red":
        return geometry["field_length_ft"] - x, geometry["field_width_ft"] - y
    return None, None


def _band(value: float | None, bands: Iterable[dict], low: str, high: str) -> str | None:
    if value is None:
        return None
    bands = list(bands)
    for band in bands:
        if band[low] <= value < band[high]:
            return band["id"]
    # Positions a few inches outside the carpet (footprint error at the walls) belong to the
    # nearest band rather than to no band at all.
    nearest = min(bands, key=lambda b: min(abs(value - b[low]), abs(value - b[high])))
    return nearest["id"]


def zone_of(ax: float | None, geometry: dict) -> str | None:
    return _band(ax, geometry.get("zones", []), "x_min", "x_max")


def lane_of(ay: float | None, geometry: dict) -> str | None:
    return _band(ay, geometry.get("lanes", []), "y_min", "y_max")


# ----------------------------------------------------------------------------- identity

def team_corrections(directory: Path) -> dict[int, int | None]:
    """Track-scope team corrections from the web (`corrections.jsonl`) and the desktop app
    (`desktop-corrections.json`), applied in order so the latest decision wins."""
    teams: dict[int, int | None] = {}
    for row in read_jsonl(directory / "corrections.jsonl"):
        if row.get("scope") == "track" and "team" in (row.get("fields") or {}):
            try:
                teams[int(row["target_id"])] = row["fields"]["team"]
            except (KeyError, TypeError, ValueError):
                continue
    desktop = directory / "desktop-corrections.json"
    if desktop.exists():
        try:
            rows = read_json(desktop)
        except ValueError:
            rows = []
        for row in rows if isinstance(rows, list) else []:
            if isinstance(row, dict) and row.get("scope") == "track" and "team" in (row.get("fields") or {}):
                try:
                    teams[int(row["target"])] = row["fields"]["team"]
                except (KeyError, TypeError, ValueError):
                    continue
    return teams


def alliance_for(team: int | None, job: dict) -> str | None:
    alliances = job.get("alliances") or {}
    for colour in ("red", "blue"):
        members = alliances.get(colour) or []
        if team is not None and any(_team_number(m) == team for m in members):
            return colour
    return None


def _team_number(value) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        digits = value.removeprefix("frc")
        return int(digits) if digits.isdigit() else None
    return None


def match_clock(directory: Path, events: list[dict], season: dict) -> MatchClock:
    """The run's fitted scoreboard clock if one was read, else a fixed start.

    `python -m ingest.strategy clock RUN` reads the clock and writes `match_clock.json`. Without
    it, match time zero is the first match_start event, or the start of the clip -- which is
    seconds wrong on any broadcast upload, so the source is recorded with the timeline.
    """
    path = Path(directory) / "match_clock.json"
    if path.exists():
        try:
            clock = MatchClock.from_dict(read_json(path))
            if clock.source == "scoreboard_clock":
                return clock
        except (ValueError, TypeError):
            pass
    starts = [e for e in events if e.get("event_type") == "match_start"]
    source = "match_start_event" if starts else "clip_start"
    return fixed_clock(match_start_seconds(events), season, source)


def match_start_seconds(events: list[dict]) -> float:
    """Match time zero in video seconds: the first match_start event, or the clip start."""
    starts = [e.get("t_seconds") for e in events if e.get("event_type") == "match_start"]
    starts = [t for t in starts if isinstance(t, (int, float))]
    return float(min(starts)) if starts else 0.0


# ----------------------------------------------------------------------------- resampling

def _in_gap(a: float, b: float, gaps: list[dict]) -> bool:
    return any(g.get("start", 0) < b and g.get("end", 0) > a for g in gaps)


def resample_track(track: dict, rate_hz: float = RATE_HZ) -> list[dict]:
    """Sample a track on a fixed grid. Returns only observed instants; holes are omitted."""
    boxes = sorted((b for b in track.get("boxes", []) if isinstance(b.get("t"), (int, float))),
                   key=lambda b: b["t"])
    if not boxes:
        return []
    gaps = [g for g in track.get("gaps", []) if isinstance(g, dict)]
    times = [b["t"] for b in boxes]
    step = 1.0 / rate_hz
    first = math.ceil(times[0] * rate_hz - 1e-9) / rate_hz
    samples = []
    k = 0
    while True:
        t = round(first + k * step, 6)
        k += 1
        if t > times[-1] + 1e-9:
            break
        i = bisect_right(times, t + 1e-9) - 1
        a = boxes[i]
        if abs(a["t"] - t) < 1e-6:
            fx, fy = a.get("field_x"), a.get("field_y")
        else:
            if i + 1 >= len(boxes):
                break
            b = boxes[i + 1]
            if b["t"] - a["t"] > MAX_BRIDGE_S or _in_gap(a["t"], b["t"], gaps):
                continue
            u = (t - a["t"]) / (b["t"] - a["t"])
            fx = _lerp(a.get("field_x"), b.get("field_x"), u)
            fy = _lerp(a.get("field_y"), b.get("field_y"), u)
        samples.append({"t": t, "x": fx, "y": fy})
    return samples


def _lerp(a, b, u):
    if a is None or b is None:
        return None
    return a + (b - a) * u


def _smooth_and_speed(samples: list[dict], rate_hz: float) -> None:
    """Add smoothed alliance-frame positions' speed, within runs of consecutive samples only."""
    step = 1.0 / rate_hz
    half = max(1, int(round(SMOOTH_S * rate_hz / 2)))
    runs: list[list[int]] = []
    for index, sample in enumerate(samples):
        positioned = sample["x"] is not None
        contiguous = (index > 0 and runs and runs[-1][-1] == index - 1
                      and abs(sample["t"] - samples[index - 1]["t"] - step) < 1e-6)
        if positioned and contiguous and samples[index - 1]["x"] is not None:
            runs[-1].append(index)
        elif positioned:
            runs.append([index])
    for sample in samples:
        sample["speed"] = None
    for run in runs:
        xs = [samples[i]["x"] for i in run]
        ys = [samples[i]["y"] for i in run]
        sx = [_mean(xs[max(0, j - half): j + half + 1]) for j in range(len(run))]
        sy = [_mean(ys[max(0, j - half): j + half + 1]) for j in range(len(run))]
        if len(run) < 2:
            continue
        for j, index in enumerate(run):
            lo, hi = max(0, j - half), min(len(run) - 1, j + half)
            if hi == lo:
                continue
            distance = math.hypot(sx[hi] - sx[lo], sy[hi] - sy[lo])
            samples[index]["speed"] = round(distance / ((hi - lo) * step), 3)


def _mean(values):
    return sum(values) / len(values)


# ----------------------------------------------------------------------------- building

def build_timelines(directory: Path, rate_hz: float = RATE_HZ) -> list[dict]:
    """Build every robot's timeline for one analysed match directory."""
    directory = Path(directory)
    job = read_json(directory / "job.json") if (directory / "job.json").exists() else {}
    season_number = int(job.get("season") or 2026)
    season = read_json(SEASONS_DIR / f"{season_number}.json")
    geometry = load_geometry(season_number)
    tracks = read_jsonl(directory / "tracks.jsonl")
    shots = read_jsonl(directory / "shots.jsonl")
    events = read_jsonl(directory / "events.jsonl")
    corrections = team_corrections(directory)
    clock = match_clock(directory, events, season)
    result = read_json(directory / "result.json") if (directory / "result.json").exists() else {}

    # Identity first: corrected team, then alliance from the track or the match roster.
    robots: dict[str, dict] = {}
    for track in tracks:
        track_id = track.get("track_id")
        if not isinstance(track_id, int):
            continue
        team = corrections.get(track_id, track.get("team"))
        team = _team_number(team)
        alliance = track.get("alliance") or alliance_for(team, job)
        key = f"team:{team}" if team else f"track:{track_id}"
        samples = resample_track(track, rate_hz)
        robot = robots.get(key)
        if robot and _overlaps(robot["samples"], samples):
            # Two simultaneous tracks claiming one team is an identity error, not one robot.
            # Keep them apart and say so rather than averaging two robots into one.
            key = f"{key}#track:{track_id}"
            robot = None
        if robot is None:
            robot = robots[key] = dict(key=key, team=team, alliance=alliance, track_ids=[],
                                       samples=[], issues=[],
                                       position_source=track.get("position_source"))
            if "#" in key:
                robot["issues"].append("overlapping_team_claim")
        robot["track_ids"].append(track_id)
        robot["alliance"] = robot["alliance"] or alliance
        robot["samples"] = sorted(robot["samples"] + samples, key=lambda s: s["t"])

    timelines = []
    for robot in robots.values():
        samples = robot["samples"]
        alliance = robot["alliance"]
        for sample in samples:
            ax, ay = (to_alliance_frame(sample["x"], sample["y"], alliance, geometry)
                      if sample["x"] is not None else (None, None))
            sample["ax"], sample["ay"] = _round(ax), _round(ay)
            sample["x"], sample["y"] = _round(sample["x"]), _round(sample["y"])
        _smooth_and_speed(samples, rate_hz)
        launches = _launches(robot["track_ids"], shots, samples, clock)
        positioned = sum(1 for s in samples if s["x"] is not None)
        timelines.append({
            "schema_version": SCHEMA_VERSION,
            "job_id": job.get("job_id"),
            "match_id": job.get("match_id"),
            "season": season_number,
            "robot_key": robot["key"],
            "team": robot["team"],
            "alliance": alliance,
            "track_ids": robot["track_ids"],
            "identity_issues": robot["issues"],
            "rate_hz": rate_hz,
            "match_start_t": clock.match_start_t,
            "clock_source": clock.source,
            "frame": "alliance" if alliance else "field",
            "position_source": robot["position_source"] or result.get("position_source"),
            "coverage": {
                "observed_s": round(len(samples) / rate_hz, 1),
                "positioned_s": round(positioned / rate_hz, 1),
            },
            "samples": _columns(samples, clock, geometry),
            "launches": launches,
        })
    timelines.sort(key=lambda r: (r["alliance"] or "z", r["team"] or 0, r["robot_key"]))
    return timelines


def _round(value):
    return None if value is None else round(value, 3)


def _overlaps(existing: list[dict], new: list[dict]) -> bool:
    if not existing or not new:
        return False
    return new[0]["t"] <= existing[-1]["t"] and existing[0]["t"] <= new[-1]["t"]


def _columns(samples: list[dict], clock: MatchClock, geometry: dict) -> dict:
    """Columnar samples: a 150 s match at 10 Hz is 1,500 rows, so keys are written once."""
    columns = {key: [] for key in ("t", "mt", "phase", "x", "y", "ax", "ay", "speed", "zone", "lane")}
    for sample in samples:
        match_time, phase = clock.at(sample["t"])
        columns["t"].append(sample["t"])
        columns["mt"].append(match_time)
        columns["phase"].append(phase)
        for key in ("x", "y", "ax", "ay", "speed"):
            columns[key].append(sample[key])
        columns["zone"].append(zone_of(sample["ax"], geometry))
        columns["lane"].append(lane_of(sample["ay"], geometry))
    return columns


def _launches(track_ids, shots, samples, clock: MatchClock) -> list[dict]:
    times = [s["t"] for s in samples]
    launches = []
    for shot in shots:
        if shot.get("robot_track_id") not in track_ids:
            continue
        t = shot.get("launch_t_seconds")
        if not isinstance(t, (int, float)):
            continue
        nearest = None
        if times:
            i = bisect_right(times, t)
            candidates = [samples[j] for j in (i - 1, i) if 0 <= j < len(samples)]
            nearest = min(candidates, key=lambda s: abs(s["t"] - t))
            if abs(nearest["t"] - t) > LAUNCH_POSITION_TOLERANCE_S:
                nearest = None
        match_time, phase = clock.at(t)
        launches.append({
            "shot_id": shot.get("shot_id"),
            "t": t,
            "mt": match_time,
            "phase": phase,
            "ax": nearest.get("ax") if nearest else None,
            "ay": nearest.get("ay") if nearest else None,
            "speed": nearest.get("speed") if nearest else None,
            "confidence": shot.get("confidence"),
        })
    return sorted(launches, key=lambda launch: launch["t"])


def write_timelines(directory: Path, timelines: list[dict]) -> Path:
    path = Path(directory) / "timeline.jsonl"
    temp = path.with_suffix(".jsonl.tmp")
    temp.write_text("".join(json.dumps(t, separators=(",", ":")) + "\n" for t in timelines),
                    encoding="utf-8")
    temp.replace(path)
    return path


def samples_as_rows(timeline: dict) -> list[dict]:
    """Columnar samples back to row dicts, for code that reads more naturally per instant."""
    columns = timeline["samples"]
    return [dict(zip(columns, values)) for values in zip(*columns.values())]
