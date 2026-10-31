"""Multi-robot simulation with blocking (T2): several robots on the field at once, so defense runs.

Each robot either follows its own waypoints or **shadows** another robot (a defender chasing an
attacker). When two robots touch -- centres within the sum of their radii -- they are pushed apart
and the velocity driving them together is removed, so a defender that catches an attacker *blocks*
it: the attacker can no longer drive straight through. That is the whole point of T2, and it is the
honest minimum -- a soft-body separation, not a rigid-body engine (see docs/SIM-DESIGN.md section 3).

The step order each tick is: decide every robot's intended velocity from its *current* view of the
field, integrate them all, then resolve overlaps. Deciding before anyone moves keeps the result
independent of robot order.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .model import DEFAULT_DT_S, Pose, RobotParams, _apply_limits, _desired_velocity


@dataclass
class SimRobot:
    """One robot in a match sim: it either follows `waypoints`, or `shadow`s another robot's id."""
    robot_id: str
    start: tuple[float, float]
    params: RobotParams = field(default_factory=RobotParams)
    waypoints: list[tuple[float, float]] | None = None
    shadow: str | None = None  # id of a robot to chase/defend; overrides waypoints


@dataclass
class _State:
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    heading: float = 0.0
    target: int = 0  # index into this robot's waypoints


def _goal(robot: SimRobot, state: dict[str, _State]) -> tuple[tuple[float, float], bool] | None:
    """Where this robot wants to be next, and whether that is its final target. A shadow aims at the
    robot it is covering; a follower walks its waypoint list. None means 'nothing to do, brake'."""
    me = state[robot.robot_id]
    if robot.shadow is not None:
        other = state.get(robot.shadow)
        return ((other.x, other.y), False) if other is not None else None
    if robot.waypoints:
        while me.target < len(robot.waypoints) and \
                math.hypot(robot.waypoints[me.target][0] - me.x,
                           robot.waypoints[me.target][1] - me.y) <= robot.params.waypoint_tol_ft:
            me.target += 1
        if me.target < len(robot.waypoints):
            return robot.waypoints[me.target], me.target == len(robot.waypoints) - 1
    return None


def _resolve_collisions(robots: list[SimRobot], state: dict[str, _State]) -> None:
    """Push overlapping robots apart and cancel the velocity that drove them together, so contact
    blocks motion instead of letting robots pass through each other."""
    for i in range(len(robots)):
        for j in range(i + 1, len(robots)):
            a, b = state[robots[i].robot_id], state[robots[j].robot_id]
            dx, dy = b.x - a.x, b.y - a.y
            distance = math.hypot(dx, dy)
            min_sep = robots[i].params.radius_ft + robots[j].params.radius_ft
            if distance >= min_sep:
                continue
            if distance < 1e-6:  # exactly overlapping: nudge along x so there is a normal to use
                dx, dy, distance = 1.0, 0.0, 1.0
            nx, ny = dx / distance, dy / distance
            push = (min_sep - distance) / 2.0
            a.x -= nx * push; a.y -= ny * push
            b.x += nx * push; b.y += ny * push
            # Remove the components of velocity pushing the pair together (inelastic along normal).
            va, vb = a.vx * nx + a.vy * ny, b.vx * nx + b.vy * ny
            if va > 0:
                a.vx -= va * nx; a.vy -= va * ny
            if vb < 0:
                b.vx -= vb * nx; b.vy -= vb * ny


def simulate_match(robots: list[SimRobot], *, dt: float = DEFAULT_DT_S,
                   duration_s: float = 15.0) -> dict[str, list[Pose]]:
    """Step every robot together for `duration_s`, returning each robot's pose trajectory."""
    state = {r.robot_id: _State(float(r.start[0]), float(r.start[1])) for r in robots}
    out = {r.robot_id: [Pose(0.0, round(r.start[0], 3), round(r.start[1], 3), 0.0)] for r in robots}
    steps = int(round(duration_s / dt))
    for step in range(1, steps + 1):
        intended: dict[str, tuple[float, float]] = {}
        for robot in robots:  # decide all velocities before anyone moves (order-independent)
            me = state[robot.robot_id]
            goal = _goal(robot, state)
            if goal is None:
                desired_vx, desired_vy = 0.0, 0.0
            else:
                desired_vx, desired_vy = _desired_velocity(me.x, me.y, goal[0], robot.params, goal[1])
            intended[robot.robot_id] = _apply_limits(me.vx, me.vy, desired_vx, desired_vy, robot.params, dt)
        for robot in robots:
            me = state[robot.robot_id]
            me.vx, me.vy = intended[robot.robot_id]
            me.x += me.vx * dt
            me.y += me.vy * dt
            if me.vx * me.vx + me.vy * me.vy > 1e-6:
                me.heading = math.atan2(me.vy, me.vx)
        _resolve_collisions(robots, state)
        for robot in robots:
            me = state[robot.robot_id]
            out[robot.robot_id].append(Pose(round(step * dt, 4), round(me.x, 3), round(me.y, 3),
                                            round(me.heading, 4)))
    return out


def reached(trajectory: list[Pose], target: tuple[float, float], tol_ft: float = 1.5) -> float | None:
    """The time a trajectory first comes within `tol_ft` of `target`, or None if it never does. Used
    to score defense: how much longer an attacker takes to reach its goal when covered."""
    for pose in trajectory:
        if math.hypot(pose.x - target[0], pose.y - target[1]) <= tol_ft:
            return pose.t
    return None
