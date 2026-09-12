"""Count observed goal entries independently of launch-track survival.

Physical goal regions are camera calibration, not detections of a nearby field object.
An optional approach gate and interior polygon require ordered motion into the basket.
Disappearance, a predicted intersection, and a yellow blob near the goal are never makes.
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class GoalEntry:
    entry_id: str
    goal: str
    region_id: str
    ball_track_id: int
    frame_index: int
    t_seconds: float
    crossing_frame: int
    crossing_t_seconds: float
    confidence: float
    ball_track: list[dict]
    shot_id: str | None = None
    robot_track_id: int | None = None
    association: str = "unassigned"
    association_confidence: float = 0.0

    def to_dict(self):
        return dict(schema_version=1, entry_id=self.entry_id, goal=self.goal,
                    region_id=self.region_id, ball_track_id=self.ball_track_id,
                    frame_index=self.frame_index, t_seconds=round(self.t_seconds, 6),
                    crossing_frame=self.crossing_frame,
                    crossing_t_seconds=round(self.crossing_t_seconds, 6),
                    confidence=round(self.confidence, 6), shot_id=self.shot_id,
                    robot_track_id=self.robot_track_id, association=self.association,
                    association_confidence=round(self.association_confidence, 6),
                    coordinate_space="model_crop_normalized", ball_track=self.ball_track)


@dataclass
class _EntryState:
    armed_at: float | None = None
    crossing: object | None = None
    inside_hits: int = 0
    last_frame: int = -1


def _inside(point, polygon):
    from training.ball_scouting import _point_in_polygon
    return _point_in_polygon(point, polygon)


def _crosses(first, second, boundary):
    from training.ball_scouting import crosses_directed_boundary
    return crosses_directed_boundary(first, second, boundary)


def _normalized(point, size):
    width, height = size
    return dict(frame_index=point.frame_index, t_seconds=round(point.t_seconds, 6),
                x=round(point.center[0] / width, 6), y=round(point.center[1] / height, 6),
                radius=round(point.radius / math.hypot(width, height), 6), observed=True)


class GoalEntryCounter:
    def __init__(self, goals):
        self.goals = goals
        self.entries: list[GoalEntry] = []
        self.states: dict[tuple[int, int], _EntryState] = {}
        self.counted_tracks: set[int] = set()
        self.scene_start = 0.0

    def reset_temporal(self, t_seconds):
        self.states.clear()
        self.counted_tracks.clear()
        self.scene_start = t_seconds

    @staticmethod
    def _depth(point, goal, size):
        """Signed distance beyond the scoring line, in physical ball radii."""
        if goal.made_line is None:
            return math.inf
        width, height = size
        a, b = goal.made_line.line
        a = (a[0] * width, a[1] * height)
        b = (b[0] * width, b[1] * height)
        normal = (-(b[1] - a[1]), b[0] - a[0])
        length = math.hypot(*normal)
        if length <= 1e-9:
            return -math.inf
        direction = goal.made_line.direction
        sign = 1 if normal[0] * direction[0] * width + normal[1] * direction[1] * height > 0 else -1
        return sign * sum((point.center[i] - a[i]) * normal[i] for i in (0, 1)) / (
            length * max(point.radius, 1.0))

    def observe(self, tracks, frame_index, t_seconds, size):
        tracks = list(tracks)
        width, height = size
        new_entries = []
        active_ids = {track.track_id for track in tracks}
        self.states = {key: state for key, state in self.states.items() if key[0] in active_ids}
        for track in tracks:
            if (not track.matched_this_frame or len(track.points) < 2
                    or track.track_id in self.counted_tracks):
                continue
            first, last = track.points[-2:]
            previous = (first.center[0] / width, first.center[1] / height)
            current = (last.center[0] / width, last.center[1] / height)
            for index, goal in enumerate(self.goals):
                key = (track.track_id, index)
                state = self.states.setdefault(key, _EntryState())
                if state.last_frame == last.frame_index:
                    continue
                state.last_frame = last.frame_index
                gap = last.t_seconds - first.t_seconds
                if (gap <= 0 or gap > goal.maximum_gap_seconds
                        or first.t_seconds < self.scene_start):
                    self.states[key] = _EntryState(last_frame=last.frame_index)
                    continue
                movement = (current[0] - previous[0], current[1] - previous[1])
                directed = sum(movement[i] * goal.entry_direction[i] for i in (0, 1))
                # Repeated broadcast images may confirm continued presence, but cannot
                # arm a gate or manufacture a crossing. A measurable reversal cancels it.
                reversal = directed < -max(first.radius, last.radius) / math.hypot(width, height) * .25
                expired = state.armed_at is not None and last.t_seconds - state.armed_at > goal.maximum_confirmation_seconds
                if reversal or expired:
                    state.armed_at = None
                    state.crossing = None
                    state.inside_hits = 0
                if goal.approach_line is None:
                    if state.armed_at is None:
                        state.armed_at = first.t_seconds
                elif _crosses(previous, current, goal.approach_line):
                    state.armed_at = last.t_seconds
                    state.inside_hits = 0
                    state.crossing = None
                if state.armed_at is None:
                    # Broadcast graphics can hide the rim crossing. Two observed
                    # inward positions above the deeper scoring gate still establish
                    # approach. Never arm from a ball born below the scoring line.
                    if (goal.allow_partial_approach and goal.confirmation_polygon
                            and first.t_seconds - track.points[0].t_seconds <= goal.maximum_gap_seconds
                            and _inside(previous, goal.confirmation_polygon)
                            and self._depth(first, goal, size) <= -.5 and directed > 0):
                        state.armed_at = first.t_seconds
                        state.inside_hits = 1
                    else:
                        continue
                interior = goal.confirmation_polygon
                inside_now = interior is None or _inside(current, interior)
                if inside_now:
                    # A duplicate frame can preserve evidence but not supply an extra
                    # independent inward-motion observation.
                    if directed > 1e-6:
                        state.inside_hits += 1
                elif state.crossing is not None or state.inside_hits > 0:
                    state.crossing = None
                    state.inside_hits = 0
                    state.armed_at = None
                    continue
                crossed = (_crosses(previous, current, goal.made_line) if goal.made_line else
                           (not _inside(previous, goal.polygon) and _inside(current, goal.polygon)
                            and directed > 0))
                if crossed and directed > 0 and state.crossing is None:
                    state.crossing = last
                if (state.crossing is None or not inside_now
                        or state.inside_hits < goal.confirmation_frames
                        or self._depth(last, goal, size) < goal.minimum_depth_radii
                        or max(p.radius for p in track.points[-12:]) <
                        goal.expected_ball_radius * math.hypot(width, height) * .5):
                    continue
                entry = GoalEntry(
                    str(uuid.uuid4()), goal.goal_id, goal.region_id or f"{goal.goal_id}_{index + 1}",
                    track.track_id, frame_index, t_seconds, state.crossing.frame_index,
                    state.crossing.t_seconds, .92 if interior and goal.approach_line else .8,
                    [_normalized(p, size) for p in track.points[-24:]
                     if last.t_seconds - p.t_seconds <= .5 and p.t_seconds >= self.scene_start],
                )
                self.counted_tracks.add(track.track_id)
                if not self._duplicate(entry, size):
                    self.entries.append(entry)
                    new_entries.append(entry)
                break
        return new_entries

    def _duplicate(self, proposed, size):
        width, height = size
        diagonal = math.hypot(width, height)
        for old in reversed(self.entries):
            if old.region_id != proposed.region_id or abs(old.t_seconds - proposed.t_seconds) > .15:
                continue
            matches = 0
            for p in proposed.ball_track:
                q = next((q for q in old.ball_track if q["frame_index"] == p["frame_index"]), None)
                if q is not None and math.hypot((p["x"] - q["x"]) * width,
                                               (p["y"] - q["y"]) * height) <= max(
                        2, min(p["radius"], q["radius"]) * diagonal * .8):
                    matches += 1
            if matches >= 2:
                return True
            # 30 Hz images encoded at 60 Hz can split one flight into alternating
            # odd/even track IDs. Require three almost identical observed positions
            # on adjacent source frames; proximity at the basket alone is insufficient.
            aliases = set()
            for p in proposed.ball_track:
                for qi, q in enumerate(old.ball_track):
                    if (abs(p["frame_index"] - q["frame_index"]) <= 1
                            and abs(p["t_seconds"] - q["t_seconds"]) <= .022
                            and math.hypot((p["x"] - q["x"]) * width,
                                           (p["y"] - q["y"]) * height)
                            <= max(1, min(p["radius"], q["radius"]) * diagonal * .25)):
                        aliases.add(qi)
                        break
            if len(aliases) >= 3:
                return True
        return False


def trajectory_link_cost(shot, entry, size, maximum_gap=.25):
    """Local ballistic continuation into observed goal evidence, never a predicted make.

    Long unobserved flights deliberately have no association. A launch-only gate pulse
    cannot identify which of several balls at the goal belongs to that robot.
    """
    width, height = size
    observed = shot.ball_track
    if (len(observed) < 4 or len(entry.ball_track) < 2
            or observed[-1]["t_seconds"] >= entry.ball_track[0]["t_seconds"]):
        return None
    old = observed[-8:]
    first = entry.ball_track[0]
    gap = first["t_seconds"] - old[-1]["t_seconds"]
    duration = old[-1]["t_seconds"] - old[0]["t_seconds"]
    if not 0 < gap <= maximum_gap or duration < .04 or gap > duration * 1.5:
        return None
    times = np.array([p["t_seconds"] - old[-1]["t_seconds"] for p in old]) / duration
    matrix = np.stack([np.ones_like(times), times, times * times], axis=1)
    positions = np.array([[p["x"] * width, p["y"] * height] for p in old])
    coefficients, *_ = np.linalg.lstsq(matrix, positions, rcond=None)
    radius = max(2, float(np.median([p["radius"] for p in old])) * math.hypot(width, height))
    error = float(np.sqrt(np.mean(np.sum((matrix @ coefficients - positions) ** 2, axis=1))))
    if error > radius * .75:
        return None
    checks = entry.ball_track[:3]
    target_times = np.array([p["t_seconds"] - old[-1]["t_seconds"] for p in checks]) / duration
    predicted = np.stack([np.ones_like(target_times), target_times, target_times ** 2], axis=1) @ coefficients
    actual = np.array([[p["x"] * width, p["y"] * height] for p in checks])
    residual = float(np.max(np.linalg.norm(actual - predicted, axis=1)))
    allowed = radius * 2 + error
    if residual > allowed:
        return None
    movement = actual[-1] - actual[0]
    velocity = predicted[-1] - predicted[0]
    cosine = float(np.dot(movement, velocity)) / max(1e-6, float(np.linalg.norm(movement) * np.linalg.norm(velocity)))
    if cosine < .8:
        return None
    return residual / allowed + (1 - cosine) * .5


def goal_statistics(entries):
    rows = [entry.to_dict() if isinstance(entry, GoalEntry) else entry for entry in entries]
    regions = {}
    robots = {}
    for row in rows:
        region = regions.setdefault(row["region_id"], dict(goal=row["goal"], made=0, attributed=0, unassigned=0))
        attributed = isinstance(row.get("robot_track_id"), int)
        region["made"] += 1
        region["attributed" if attributed else "unassigned"] += 1
        if attributed:
            key = str(row["robot_track_id"])
            robots[key] = robots.get(key, 0) + 1
    assigned = sum(robots.values())
    return dict(made=len(rows), attributed=assigned, unassigned=len(rows) - assigned,
                per_region=regions, per_robot=robots)


def write_goal_entries(path, entries):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for entry in entries:
            handle.write(json.dumps(entry.to_dict(), sort_keys=True) + "\n")
    temporary.replace(output)
