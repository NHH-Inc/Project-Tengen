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


@dataclass(frozen=True)
class RobotParams:
    max_speed_ftps: float = DEFAULT_MAX_SPEED_FTPS
    max_accel_ftps2: float = DEFAULT_MAX_ACCEL_FTPS2
    waypoint_tol_ft: float = DEFAULT_WAYPOINT_TOL_FT


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
        if target >= len(waypoints):
            break
        tx, ty = waypoints[target]
        dx, dy = tx - x, ty - y
        distance = math.hypot(dx, dy)
        if distance <= params.waypoint_tol_ft:
            target += 1
            if target >= len(waypoints):
                break
            tx, ty = waypoints[target]
            dx, dy = tx - x, ty - y
            distance = math.hypot(dx, dy)
        # Desired velocity: toward the waypoint at top speed, but slow down so we can stop on the
        # final one instead of overshooting (v = sqrt(2*a*d) is the fastest speed we can still brake
        # from within the remaining distance).
        if distance > 1e-9:
            brake_speed = math.sqrt(2.0 * params.max_accel_ftps2 * distance)
            speed = min(params.max_speed_ftps, brake_speed) if target == len(waypoints) - 1 \
                else params.max_speed_ftps
            desired_vx = speed * dx / distance
            desired_vy = speed * dy / distance
        else:
            desired_vx = desired_vy = 0.0
        # Accelerate toward the desired velocity, capped by the acceleration limit this step.
        dvx, dvy = desired_vx - vx, desired_vy - vy
        dv = math.hypot(dvx, dvy)
        max_dv = params.max_accel_ftps2 * dt
        if dv > max_dv and dv > 1e-9:
            dvx *= max_dv / dv
            dvy *= max_dv / dv
        vx += dvx
        vy += dvy
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
