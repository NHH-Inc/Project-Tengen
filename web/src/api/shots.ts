export type ShotOutcome = 'made' | 'missed' | 'unknown';
export type NormalizedPoint = [number, number];

export interface ShotTrackPoint {
  frameIndex: number;
  tSeconds: number;
  x: number;
  y: number;
  radius: number;
}

export interface ShotRecord {
  shotId: string;
  launchFrame: number;
  launchTSeconds: number;
  robotTrackId: number | null;
  ballTrackId: number;
  confidence: number;
  attributionConfidence: number;
  outcome: ShotOutcome;
  goal: string | null;
  outcomeFrame: number | null;
  outcomeTSeconds: number | null;
  ballTrack: ShotTrackPoint[];
}

export interface DirectedBoundary {
  line: [NormalizedPoint, NormalizedPoint];
  direction: NormalizedPoint;
}

export interface ShotGoal {
  id: string;
  regionId: string;
  polygon: NormalizedPoint[] | null;
  entryDirection: NormalizedPoint;
  madeBoundary: DirectedBoundary | null;
  missBoundaries: DirectedBoundary[];
  approachBoundary: DirectedBoundary | null;
  confirmationPolygon: NormalizedPoint[] | null;
}

export interface GoalEntry {
  entryId: string;
  regionId: string;
  goal: string;
  tSeconds: number;
  frameIndex: number;
  shotId: string | null;
  robotTrackId: number | null;
  confidence: number;
  association: string;
  ballTrack: ShotTrackPoint[];
}

export interface ShotCounts {
  attempted: number;
  made: number;
  missed: number;
  unknown: number;
}

export interface ShotResponse {
  shots: ShotRecord[];
  statistics: ShotCounts & { perRobot: Record<string, ShotCounts> };
  goals: ShotGoal[];
  goalEntries: GoalEntry[];
}

interface WireShotPoint {
  frame_index: number;
  t_seconds: number;
  x: number;
  y: number;
  radius: number;
}

interface WireShot {
  shot_id: string;
  launch_frame: number;
  launch_t_seconds: number;
  robot_track_id: number | null;
  ball_track_id: number;
  confidence: number;
  attribution_confidence: number;
  outcome: string;
  goal: string | null;
  outcome_frame: number | null;
  outcome_t_seconds: number | null;
  ball_track: WireShotPoint[];
}

interface WireBoundary {
  line: [NormalizedPoint, NormalizedPoint];
  direction: NormalizedPoint;
}

interface WireGoal {
  id: string;
  region_id?: string;
  polygon?: NormalizedPoint[];
  entry_direction: NormalizedPoint;
  made_boundary?: WireBoundary;
  miss_boundaries?: WireBoundary[];
  approach_boundary?: WireBoundary;
  confirmation_polygon?: NormalizedPoint[];
}

interface WireGoalEntry {
  entry_id: string;
  region_id: string;
  goal: string;
  t_seconds: number;
  frame_index: number;
  shot_id: string | null;
  robot_track_id: number | null;
  confidence: number;
  association: string;
  ball_track: WireShotPoint[];
}

interface WireCounts extends ShotCounts {
  per_robot: Record<string, ShotCounts>;
}

export interface WireShotResponse {
  shots: WireShot[];
  statistics: WireCounts;
  goals: WireGoal[];
  goal_entries?: WireGoalEntry[];
}

const EMPTY_COUNTS = (): ShotCounts => ({ attempted: 0, made: 0, missed: 0, unknown: 0 });

export const EMPTY_SHOT_RESPONSE: ShotResponse = {
  shots: [],
  statistics: { ...EMPTY_COUNTS(), perRobot: {} },
  goals: [],
  goalEntries: [],
};

function validPoint(value: unknown): value is NormalizedPoint {
  return Array.isArray(value) && value.length === 2
    && value.every((component) => typeof component === 'number' && Number.isFinite(component));
}

function parseBoundary(raw: WireBoundary | undefined): DirectedBoundary | null {
  if (!raw || !Array.isArray(raw.line) || raw.line.length !== 2
      || !validPoint(raw.line[0]) || !validPoint(raw.line[1]) || !validPoint(raw.direction)) {
    return null;
  }
  return { line: raw.line, direction: raw.direction };
}

function parsePoints(points: WireShotPoint[]): ShotTrackPoint[] {
  return points.flatMap((p) => Number.isInteger(p.frame_index)
    && [p.t_seconds, p.x, p.y, p.radius].every((v) => typeof v === 'number' && Number.isFinite(v))
      ? [{ frameIndex: p.frame_index, tSeconds: p.t_seconds, x: p.x, y: p.y, radius: p.radius }]
      : []);
}

/** Include legacy made shots, but never count a linked goal entry a second time. */
export function scoringCountsAt(shots: ShotRecord[], entries: GoalEntry[], t = Infinity) {
  const visible = entries.filter((entry) => entry.tSeconds <= t);
  const linked = new Set(visible.map((entry) => entry.shotId));
  const legacy = shots.filter((shot) => shot.outcome === 'made'
    && shot.outcomeTSeconds != null && shot.outcomeTSeconds <= t && !linked.has(shot.shotId));
  return {
    made: visible.length + legacy.length,
    unassigned: visible.filter((entry) => entry.robotTrackId == null).length
      + legacy.filter((shot) => shot.robotTrackId == null).length,
  };
}

export function parseShotResponse(raw: WireShotResponse): ShotResponse {
  const shots = (raw.shots ?? []).flatMap((shot): ShotRecord[] => {
    if (
      typeof shot.shot_id !== 'string'
      || !Number.isInteger(shot.launch_frame)
      || typeof shot.launch_t_seconds !== 'number'
      || !Number.isInteger(shot.ball_track_id)
      || !['made', 'missed', 'unknown'].includes(shot.outcome)
      || !Array.isArray(shot.ball_track)
    ) return [];
    return [{
      shotId: shot.shot_id,
      launchFrame: shot.launch_frame,
      launchTSeconds: shot.launch_t_seconds,
      robotTrackId: Number.isInteger(shot.robot_track_id) ? shot.robot_track_id : null,
      ballTrackId: shot.ball_track_id,
      confidence: shot.confidence,
      attributionConfidence: shot.attribution_confidence,
      outcome: shot.outcome as ShotOutcome,
      goal: typeof shot.goal === 'string' ? shot.goal : null,
      outcomeFrame: Number.isInteger(shot.outcome_frame) ? shot.outcome_frame : null,
      outcomeTSeconds: typeof shot.outcome_t_seconds === 'number' ? shot.outcome_t_seconds : null,
      ballTrack: shot.ball_track.flatMap((point): ShotTrackPoint[] => (
        Number.isInteger(point.frame_index)
        && [point.t_seconds, point.x, point.y, point.radius]
          .every((value) => typeof value === 'number' && Number.isFinite(value))
          ? [{
              frameIndex: point.frame_index,
              tSeconds: point.t_seconds,
              x: point.x,
              y: point.y,
              radius: point.radius,
            }]
          : []
      )),
    }];
  });
  const goals = (raw.goals ?? []).flatMap((goal, index): ShotGoal[] => {
    if (typeof goal.id !== 'string' || !validPoint(goal.entry_direction)) return [];
    const madeBoundary = parseBoundary(goal.made_boundary);
    const polygon = Array.isArray(goal.polygon) && goal.polygon.every(validPoint)
      ? goal.polygon
      : null;
    if (!madeBoundary && (!polygon || polygon.length < 3)) return [];
    return [{
      id: goal.id,
      regionId: goal.region_id ?? `${goal.id}_${index + 1}`,
      polygon,
      entryDirection: goal.entry_direction,
      madeBoundary,
      approachBoundary: parseBoundary(goal.approach_boundary),
      confirmationPolygon: Array.isArray(goal.confirmation_polygon)
        && goal.confirmation_polygon.length >= 3 && goal.confirmation_polygon.every(validPoint)
        ? goal.confirmation_polygon : null,
      missBoundaries: (goal.miss_boundaries ?? [])
        .map(parseBoundary)
        .filter((boundary): boundary is DirectedBoundary => boundary != null),
    }];
  });
  const counts = raw.statistics ?? { ...EMPTY_COUNTS(), per_robot: {} };
  return {
    shots,
    statistics: {
      attempted: counts.attempted ?? 0,
      made: counts.made ?? 0,
      missed: counts.missed ?? 0,
      unknown: counts.unknown ?? 0,
      perRobot: counts.per_robot ?? {},
    },
    goals,
    goalEntries: (raw.goal_entries ?? []).flatMap((entry): GoalEntry[] => {
      if (typeof entry.entry_id !== 'string' || typeof entry.region_id !== 'string'
          || typeof entry.goal !== 'string' || !Number.isFinite(entry.t_seconds)
          || !Number.isInteger(entry.frame_index) || !Array.isArray(entry.ball_track)) return [];
      return [{
        entryId: entry.entry_id, regionId: entry.region_id, goal: entry.goal,
        tSeconds: entry.t_seconds, frameIndex: entry.frame_index,
        shotId: typeof entry.shot_id === 'string' ? entry.shot_id : null,
        robotTrackId: Number.isInteger(entry.robot_track_id) ? entry.robot_track_id : null,
        confidence: entry.confidence, association: entry.association,
        ballTrack: parsePoints(entry.ball_track),
      }];
    }),
  };
}
