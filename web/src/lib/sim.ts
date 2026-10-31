// A TypeScript port of the sim's motion model + multi-robot match (ingest/sim/model.py, match.py),
// run client-side so the Sim view animates scenarios with no backend round-trip -- the same pattern
// as the strategy-engine port (lib/strategyEngine.ts). Scoring (game.py/scoring.py) stays Python for
// now; this covers motion and defense, which is what the interactive view shows.
//
// Mirrors the Python field-for-field; the Python keeps the authoritative tests. Feet, seconds, ft/s;
// everything is in the alliance frame (0..field_length x 0..field_width), same as the profiles.

export const DEFAULT_MAX_SPEED_FTPS = 12.0;
export const DEFAULT_MAX_ACCEL_FTPS2 = 10.0;
export const DEFAULT_WAYPOINT_TOL_FT = 1.0;
export const DEFAULT_DT_S = 0.02;
export const ROBOT_RADIUS_FT = 1.5;
export const DEFAULT_MAX_TURN_RADPS = 12.0;

export interface RobotParams {
  maxSpeedFtps: number;
  maxAccelFtps2: number;
  waypointTolFt: number;
  maxTurnRadps: number;
  radiusFt: number;
}

export const defaultParams = (over: Partial<RobotParams> = {}): RobotParams => ({
  maxSpeedFtps: DEFAULT_MAX_SPEED_FTPS,
  maxAccelFtps2: DEFAULT_MAX_ACCEL_FTPS2,
  waypointTolFt: DEFAULT_WAYPOINT_TOL_FT,
  maxTurnRadps: DEFAULT_MAX_TURN_RADPS,
  radiusFt: ROBOT_RADIUS_FT,
  ...over,
});

export interface Pose { t: number; x: number; y: number; heading: number; }

export interface SimRobot {
  id: string;
  start: [number, number];
  params: RobotParams;
  waypoints?: Array<[number, number]>;
  shadow?: string; // id of a robot to chase/defend; overrides waypoints
  alliance?: 'us' | 'them';
}

function desiredVelocity(x: number, y: number, target: [number, number], params: RobotParams,
                         final: boolean): [number, number] {
  const dx = target[0] - x, dy = target[1] - y;
  const distance = Math.hypot(dx, dy);
  if (distance <= 1e-9) return [0, 0];
  const speed = final
    ? Math.min(params.maxSpeedFtps, Math.sqrt(2 * params.maxAccelFtps2 * distance))
    : params.maxSpeedFtps;
  return [speed * dx / distance, speed * dy / distance];
}

function applyLimits(vx: number, vy: number, dvxWant: number, dvyWant: number,
                     params: RobotParams, dt: number): [number, number] {
  let dvx = dvxWant - vx, dvy = dvyWant - vy;
  const dv = Math.hypot(dvx, dvy);
  const maxDv = params.maxAccelFtps2 * dt;
  if (dv > maxDv && dv > 1e-9) { dvx *= maxDv / dv; dvy *= maxDv / dv; }
  let nvx = vx + dvx, nvy = vy + dvy;
  const speed0 = Math.hypot(vx, vy), speed1 = Math.hypot(nvx, nvy);
  if (speed0 > 0.5 && speed1 > 1e-9) {
    const a0 = Math.atan2(vy, vx), a1 = Math.atan2(nvy, nvx);
    const delta = ((a1 - a0 + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
    const maxDelta = params.maxTurnRadps * dt;
    if (Math.abs(delta) > maxDelta) {
      const angle = a0 + Math.sign(delta) * maxDelta;
      nvx = speed1 * Math.cos(angle); nvy = speed1 * Math.sin(angle);
    }
  }
  return [nvx, nvy];
}

interface State { x: number; y: number; vx: number; vy: number; heading: number; target: number; nvx: number; nvy: number; }

function goal(robot: SimRobot, state: Map<string, State>): { at: [number, number]; final: boolean } | null {
  const me = state.get(robot.id)!;
  if (robot.shadow !== undefined) {
    const other = state.get(robot.shadow);
    return other ? { at: [other.x, other.y], final: false } : null;
  }
  const wps = robot.waypoints;
  if (wps && wps.length) {
    while (me.target < wps.length &&
           Math.hypot(wps[me.target][0] - me.x, wps[me.target][1] - me.y) <= robot.params.waypointTolFt) {
      me.target += 1;
    }
    if (me.target < wps.length) return { at: wps[me.target], final: me.target === wps.length - 1 };
  }
  return null;
}

function resolveCollisions(robots: SimRobot[], state: Map<string, State>): void {
  for (let i = 0; i < robots.length; i++) {
    for (let j = i + 1; j < robots.length; j++) {
      const a = state.get(robots[i].id)!, b = state.get(robots[j].id)!;
      let dx = b.x - a.x, dy = b.y - a.y;
      let distance = Math.hypot(dx, dy);
      const minSep = robots[i].params.radiusFt + robots[j].params.radiusFt;
      if (distance >= minSep) continue;
      if (distance < 1e-6) { dx = 1; dy = 0; distance = 1; }
      const nx = dx / distance, ny = dy / distance, push = (minSep - distance) / 2;
      a.x -= nx * push; a.y -= ny * push; b.x += nx * push; b.y += ny * push;
      const va = a.vx * nx + a.vy * ny, vb = b.vx * nx + b.vy * ny;
      if (va > 0) { a.vx -= va * nx; a.vy -= va * ny; }
      if (vb < 0) { b.vx -= vb * nx; b.vy -= vb * ny; }
    }
  }
}

/** Step every robot together for `durationS`, returning each robot's pose trajectory. */
export function simulateMatch(robots: SimRobot[], { dt = DEFAULT_DT_S, durationS = 15.0 } = {}): Map<string, Pose[]> {
  const state = new Map<string, State>();
  const out = new Map<string, Pose[]>();
  for (const r of robots) {
    state.set(r.id, { x: r.start[0], y: r.start[1], vx: 0, vy: 0, heading: 0, target: 0, nvx: 0, nvy: 0 });
    out.set(r.id, [{ t: 0, x: r.start[0], y: r.start[1], heading: 0 }]);
  }
  const steps = Math.round(durationS / dt);
  for (let step = 1; step <= steps; step++) {
    for (const r of robots) {
      const me = state.get(r.id)!;
      const g = goal(r, state);
      const [dvx, dvy] = g ? desiredVelocity(me.x, me.y, g.at, r.params, g.final) : [0, 0];
      [me.nvx, me.nvy] = applyLimits(me.vx, me.vy, dvx, dvy, r.params, dt);
    }
    for (const r of robots) {
      const me = state.get(r.id)!;
      me.vx = me.nvx; me.vy = me.nvy;
      me.x += me.vx * dt; me.y += me.vy * dt;
      if (me.vx * me.vx + me.vy * me.vy > 1e-6) me.heading = Math.atan2(me.vy, me.vx);
    }
    resolveCollisions(robots, state);
    for (const r of robots) {
      const me = state.get(r.id)!;
      out.get(r.id)!.push({ t: step * dt, x: me.x, y: me.y, heading: me.heading });
    }
  }
  return out;
}

/** First time a trajectory comes within `tolFt` of `target`, or null. */
export function reached(trajectory: Pose[], target: [number, number], tolFt = 1.5): number | null {
  for (const p of trajectory) if (Math.hypot(p.x - target[0], p.y - target[1]) <= tolFt) return p.t;
  return null;
}

export function poseAt(trajectory: Pose[], t: number): Pose | null {
  if (!trajectory.length || t < trajectory[0].t || t > trajectory[trajectory.length - 1].t) return null;
  let lo = 0, hi = trajectory.length - 1;
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (trajectory[mid].t <= t) lo = mid; else hi = mid; }
  const a = trajectory[lo], b = trajectory[hi], span = b.t - a.t;
  const u = span <= 1e-9 ? 0 : (t - a.t) / span;
  return { t, x: a.x + (b.x - a.x) * u, y: a.y + (b.y - a.y) * u, heading: a.heading };
}
