"""The robot motion model (T1): drive a point mass to a sequence of waypoints under real limits.

A robot is a point mass with a top speed and an acceleration limit. A simple proportional controller
steers it toward the next waypoint; it advances through the list as it arrives. That is enough to
answer the questions the sim is for -- *where* a robot goes and *how long* it takes -- with realistic
arrival times, which a constant-speed replay cannot give (a robot cannot teleport to top speed).

Holonomic simplification (T1): the robot may accelerate in any direction, so a tank-drive robot that
must turn before it strafes is modelled a little optimistically. At strategy altitude -- which zone,
how many seconds -- this holds up, and T2 adds a turn-rate limit. See docs/SIM-DESIGN.md.

Deterministic and dependency-free (stdlib math only), like the rest of the strategy layer, so a run
is reproducible and checkable by eye. Units: feet, seconds, feet/second.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: Defaults when a team's measured motion is unknown. A typical FRC robot tops out around 12 ft/s
#: and accelerates at roughly 10 ft/s^2; conservative, and overridden per team where we have data.
DEFAULT_MAX_SPEED_FTPS = 12.0
DEFAULT_MAX_ACCEL_FTPS2 = 10.0
#: Within this of a waypoint, count it reached and aim at the next one.
DEFAULT_WAYPOINT_TOL_FT = 1.0
#: Integrator step. 50 Hz is well below the robot's dynamics and matches a fine replay.
DEFAULT_DT_S = 0.02
#: An FRC robot is ~3 ft across the bumpers, so ~1.5 ft centre-to-edge. Two robots are in contact
#: when their centres are within the sum of their radii (T2 blocking).
ROBOT_RADIUS_FT = 1.5
#: How fast the velocity heading may rotate. The default is large enough to be effectively
#: holonomic (T1 behaviour is unchanged); lower it to model a tank drive that must turn to redirect.
DEFAULT_MAX_TURN_RADPS = 12.0


@dataclass(frozen=True)
class RobotParams:
    max_speed_ftps: float = DEFAULT_MAX_SPEED_FTPS
    max_accel_ftps2: float = DEFAULT_MAX_ACCEL_FTPS2
    waypoint_tol_ft: float = DEFAULT_WAYPOINT_TOL_FT
    max_turn_radps: float = DEFAULT_MAX_TURN_RADPS
    radius_ft: float = ROBOT_RADIUS_FT


@dataclass(frozen=True)
class Pose:
    t: float
    x: float
    y: float
    heading: float = 0.0  # radians, direction of travel; 0 when stopped


def params_from_profile(profile: dict) -> RobotParams:
    """A team's motion limits from its profile. We measure top speed from video; acceleration we do
    not yet measure, so it keeps the default until a better estimate exists."""
    top = ((profile.get("tendencies") or {}).get("movement") or {}).get("top_speed_ftps")
    return RobotParams(max_speed_ftps=float(top) if top else DEFAULT_MAX_SPEED_FTPS)


def _desired_velocity(x: float, y: float, target: tuple[float, float], params: RobotParams,
                      final: bool) -> tuple[float, float]:
    """Velocity we *want*: toward the target at top speed, but slowed to stop on a final waypoint
    (v = sqrt(2*a*d) is the fastest speed we can still brake from within the remaining distance)."""
    dx, dy = target[0] - x, target[1] - y
    distance = math.hypot(dx, dy)
    if distance <= 1e-9:
        return 0.0, 0.0
    speed = min(params.max_speed_ftps, math.sqrt(2.0 * params.max_accel_ftps2 * distance)) \
        if final else params.max_speed_ftps
    return speed * dx / distance, speed * dy / distance


def _apply_limits(vx: float, vy: float, desired_vx: float, desired_vy: float,
                  params: RobotParams, dt: float) -> tuple[float, float]:
    """The velocity we can actually reach this step: capped change (acceleration) and capped
    rotation of the heading (turn rate). Turn capping is skipped near rest, where heading is ill-
    defined; this keeps the holonomic T1 behaviour when max_turn_radps is left at its large default."""
    dvx, dvy = desired_vx - vx, desired_vy - vy
    dv = math.hypot(dvx, dvy)
    max_dv = params.max_accel_ftps2 * dt
    if dv > max_dv and dv > 1e-9:
        dvx *= max_dv / dv
        dvy *= max_dv / dv
    nvx, nvy = vx + dvx, vy + dvy
    speed0, speed1 = math.hypot(vx, vy), math.hypot(nvx, nvy)
    if speed0 > 0.5 and speed1 > 1e-9:
        a0, a1 = math.atan2(vy, vx), math.atan2(nvy, nvx)
        delta = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
        max_delta = params.max_turn_radps * dt
        if abs(delta) > max_delta:
            angle = a0 + math.copysign(max_delta, delta)
            nvx, nvy = speed1 * math.cos(angle), speed1 * math.sin(angle)
    return nvx, nvy


def simulate_path(start: tuple[float, float], waypoints: list[tuple[float, float]],
                  params: RobotParams | None = None, *, dt: float = DEFAULT_DT_S,
                  max_time_s: float = 20.0, start_t: float = 0.0) -> list[Pose]:
    """Drive from `start` through `waypoints`, returning the pose every `dt` until the last waypoint
    is reached or `max_time_s` elapses. The first pose is the start; the robot begins at rest."""
    params = params or RobotParams()
    x, y = float(start[0]), float(start[1])
    vx = vy = 0.0
    heading = 0.0
    poses = [Pose(round(start_t, 4), round(x, 3), round(y, 3), 0.0)]
    target = 0
    steps = int(round(max_time_s / dt))
    for step in range(1, steps + 1):
        while target < len(waypoints) and \
                math.hypot(waypoints[target][0] - x, waypoints[target][1] - y) <= params.waypoint_tol_ft:
            target += 1
        if target >= len(waypoints):
            break
        desired_vx, desired_vy = _desired_velocity(x, y, waypoints[target], params,
                                                   final=target == len(waypoints) - 1)
        vx, vy = _apply_limits(vx, vy, desired_vx, desired_vy, params, dt)
        x += vx * dt
        y += vy * dt
        if vx * vx + vy * vy > 1e-6:
            heading = math.atan2(vy, vx)
        poses.append(Pose(round(start_t + step * dt, 4), round(x, 3), round(y, 3), round(heading, 4)))
    return poses


def pose_at(poses: list[Pose], t: float) -> tuple[float, float] | None:
    """Linear-interpolated (x, y) at time `t`, or None if `t` is outside the trajectory."""
    if not poses or t < poses[0].t or t > poses[-1].t:
        return None
    lo, hi = 0, len(poses) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if poses[mid].t <= t:
            lo = mid
        else:
            hi = mid
    a, b = poses[lo], poses[hi]
    span = b.t - a.t
    u = 0.0 if span <= 1e-9 else (t - a.t) / span
    return (a.x + (b.x - a.x) * u, a.y + (b.y - a.y) * u)
