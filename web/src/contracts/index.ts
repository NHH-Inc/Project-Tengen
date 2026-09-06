// The contract layer, SCHEMA_VERSION 3. Everything component 2 sends crosses through this
// file and nothing else in web/ is allowed to touch a snake_case key.
//
// Doc 0: "snake_case in JSON, SQL, C++, and Python. camelCase only inside TypeScript,
// converted at the API boundary." Wire* types mirror /contracts/*.schema.json exactly, the
// domain types are camelCase, and parse*/serialize* are the only crossing points.
//
// Doc 0 also says: "Anything unrecognized is a bug, not a fallback." These parsers do not
// coerce or default unknown enum values. They collect violations, hand them back for the UI
// to show, and drop the offending row rather than pretending it was fine.

// ---------------------------------------------------------------- closed sets

export const PHASES = ['auto', 'teleop', 'endgame', 'unknown'] as const;
export const ALLIANCES = ['red', 'blue'] as const;
export const JOB_STATUSES = [
  'queued', 'downloading', 'downloaded', 'analyzing', 'complete', 'failed',
] as const;
export const STAGES = [
  'downloading', 'decoding', 'detecting', 'tracking', 'ocr', 'events',
] as const;
export const ERROR_CODES = [
  'video_unavailable', 'download_failed', 'rate_limited',
  'no_match_data', 'analysis_failed', 'timeout', 'internal',
] as const;
export const EVENT_TYPES = [
  'match_start', 'match_end', 'phase_change',
  'shot_attempt', 'shot_made',
  'reload',
  'defense_start', 'defense_end',
  'immobile_start', 'immobile_end',
  'foul',
] as const;
export const SOURCES = ['model', 'scoreboard_ocr', 'tba', 'manual'] as const;
export const GAP_REASONS = ['shot_change', 'occlusion', 'out_of_frame', 'detection_lost'] as const;
export const CORRECTION_SCOPES = ['event', 'track'] as const;
export const CORRECTION_ACTIONS = ['edit', 'delete', 'create'] as const;

export type Phase = (typeof PHASES)[number];
export type Alliance = (typeof ALLIANCES)[number];
export type JobStatus = (typeof JOB_STATUSES)[number];
export type Stage = (typeof STAGES)[number];
export type ErrorCode = (typeof ERROR_CODES)[number];
export type EventType = (typeof EVENT_TYPES)[number];
export type Source = (typeof SOURCES)[number];
export type GapReason = (typeof GAP_REASONS)[number];
export type CorrectionScope = (typeof CORRECTION_SCOPES)[number];
export type CorrectionAction = (typeof CORRECTION_ACTIONS)[number];

export const SCHEMA_VERSION = 3;

/** Events that describe the match rather than a robot. team/track_id/field_* are all null. */
export const MATCH_LEVEL_EVENTS: ReadonlySet<EventType> = new Set<EventType>([
  'match_start', 'match_end', 'phase_change',
]);

/**
 * Whether a failed job is worth retrying. Doc 0 made error_code a closed enum specifically
 * so the UI can tell: rate_limited is worth retrying, video_unavailable is not.
 */
export const RETRYABLE_ERRORS: ReadonlySet<ErrorCode> = new Set<ErrorCode>([
  'rate_limited', 'download_failed', 'timeout', 'internal', 'analysis_failed',
]);

// ---------------------------------------------------------------- wire types

export interface WireJob {
  schema_version: number;
  job_id: string;
  match_id: string | null;
  season: number;
  video_id: string;
  /** Additive API field; omitted by fixture files and older servers. */
  capture_mode?: 'recorded' | 'live';
  local_path: string | null;
  start_offset: number;
  duration: number | null;
  fps: number | null;
  width: number | null;
  height: number | null;
  status: string;
  stage: string | null;
  progress: number | null;
  error_code: string | null;
  error: string | null;
  attempt: number;
  created_at: string;
  updated_at: string;
  alliances: { red: number[]; blue: number[] } | null;
  tba_score: { red: number; blue: number } | null;
}

export interface WireEvent {
  schema_version: number;
  job_id: string;
  match_id: string;
  event_id: string;
  team: number | null;
  track_id: number | null;
  t_seconds: number;
  phase: string;
  event_type: string;
  confidence: number;
  field_x: number | null;
  field_y: number | null;
  /** v3, optional. Absent from v2 producers; legal values come from the season config. */
  goal?: string | null;
  source: string;
  /** Read-only annotations the API adds; not part of the stored Contract B record. */
  corrected?: boolean;
  correction_id?: string | null;
}

export interface WireBox {
  t: number;
  x: number;
  y: number;
  w: number;
  h: number;
  field_x?: number | null;
  field_y?: number | null;
  velocity_x_ftps?: number | null;
  velocity_y_ftps?: number | null;
  speed_ftps?: number | null;
  motion_heading_rad?: number | null;
}
export interface WireGap { start: number; end: number; reason: string }

export interface WireTrack {
  schema_version: number;
  track_id: number;
  team: number | null;
  alliance: string | null;
  team_confidence: number | null;
  position_source?: string | null;
  boxes: WireBox[];
  gaps: WireGap[];
}

export interface WireCorrection {
  schema_version: number;
  correction_id: string;
  scope: string;
  job_id: string | null;
  target_id: string;
  action: string;
  fields: Partial<WireEvent> | null;
  created_at: string;
  created_by: string | null;
}

// ---------------------------------------------------------------- domain types

export interface Job {
  jobId: string;
  matchId: string | null;
  season: number;
  videoId: string;
  /** A live recording is analyzed only after its YouTube stream ends. */
  captureMode: 'recorded' | 'live';
  localPath: string | null;
  /** Seconds. Add to an event's tSeconds to get a position in the ORIGINAL video. */
  startOffset: number;
  duration: number | null;
  fps: number | null;
  width: number | null;
  height: number | null;
  status: JobStatus;
  stage: Stage | null;
  progress: number | null;
  errorCode: ErrorCode | null;
  error: string | null;
  attempt: number;
  createdAt: string;
  updatedAt: string;
  alliances: { red: number[]; blue: number[] } | null;
  tbaScore: { red: number; blue: number } | null;
}

export interface ScoutEvent {
  jobId: string;
  matchId: string;
  eventId: string;
  team: number | null;
  trackId: number | null;
  /** Float seconds relative to the start of the SEGMENT, never the original video. */
  tSeconds: number;
  phase: Phase;
  eventType: EventType;
  /** 0..1. Never a percentage. */
  confidence: number;
  fieldX: number | null;
  fieldY: number | null;
  /**
   * Which goal a shot went into. Null when unknown or when the event is not a shot.
   * NOT a closed set -- legal values are the season config's `goals`, because they change
   * every season. Validate against that, never against a hardcoded list.
   */
  goal: string | null;
  source: Source;
  /** Whether a human has touched this row. Independent of `source`. */
  corrected: boolean;
  correctionId: string | null;
}

export interface Box {
  t: number;
  x: number;
  y: number;
  w: number;
  h: number;
  fieldX: number | null;
  fieldY: number | null;
  velocityXFtps: number | null;
  velocityYFtps: number | null;
  speedFtps: number | null;
  motionHeadingRad: number | null;
}

export interface Gap {
  start: number;
  end: number;
  reason: GapReason;
}

export interface Track {
  trackId: number;
  team: number | null;
  alliance: Alliance | null;
  positionSource: string | null;
  /** Confidence in the WHOLE track's identity, not any one event. Flags misattributions. */
  teamConfidence: number | null;
  boxes: Box[];
  /** Required, possibly empty. Consumers must not interpolate across a listed gap. */
  gaps: Gap[];
}

export interface Correction {
  correctionId: string;
  scope: CorrectionScope;
  jobId: string | null;
  targetId: string;
  action: CorrectionAction;
  fields: Partial<ScoutEvent> | null;
  createdAt: string;
  createdBy: string | null;
}

/**
 * A job whose media metadata has arrived. A yt-dlp stream can be playable before the merged
 * local file exists, so localPath remains nullable while the timeline fields narrow to numbers.
 */
export interface PlayableJob extends Job {
  duration: number;
  fps: number;
  width: number;
  height: number;
}

export function isPlayable(job: Job | null): job is PlayableJob {
  return (
    job != null &&
    job.duration != null && job.fps != null &&
    job.width != null && job.height != null &&
    job.duration > 0
  );
}

// ---------------------------------------------------------------- violations

export interface ContractViolation {
  what: string;
  detail: string;
  raw: unknown;
}

export class ViolationLog {
  readonly items: ContractViolation[] = [];
  add(what: string, detail: string, raw: unknown) {
    this.items.push({ what, detail, raw });
  }
  get ok() {
    return this.items.length === 0;
  }
}

function oneOf<T extends string>(
  allowed: readonly T[], value: unknown, field: string, raw: unknown, log: ViolationLog
): T | null {
  if (typeof value === 'string' && (allowed as readonly string[]).includes(value)) {
    return value as T;
  }
  log.add(field, `${JSON.stringify(value)} is not one of ${allowed.join(' | ')}`, raw);
  return null;
}

/** Nullable enum: null passes, an unrecognised string does not. */
function oneOfOrNull<T extends string>(
  allowed: readonly T[], value: unknown, field: string, raw: unknown, log: ViolationLog
): T | null {
  if (value == null) return null;
  return oneOf(allowed, value, field, raw, log);
}

function checkVersion(v: unknown, kind: string, raw: unknown, log: ViolationLog) {
  // Additive bumps stay backward compatible, so this is a warning, not a drop.
  if (v !== SCHEMA_VERSION) {
    log.add(kind, `schema_version ${JSON.stringify(v)} != ${SCHEMA_VERSION}`, raw);
  }
}

// ---------------------------------------------------------------- parsers

export function parseJob(raw: WireJob, log: ViolationLog): Job | null {
  checkVersion(raw.schema_version, 'job.schema_version', raw, log);
  const status = oneOf(JOB_STATUSES, raw.status, 'job.status', raw, log);
  if (!status) return null;
  return {
    jobId: raw.job_id,
    matchId: raw.match_id ?? null,
    season: raw.season,
    videoId: raw.video_id,
    captureMode: raw.capture_mode === 'live' ? 'live' : 'recorded',
    localPath: raw.local_path ?? null,
    startOffset: raw.start_offset,
    duration: raw.duration ?? null,
    fps: raw.fps ?? null,
    width: raw.width ?? null,
    height: raw.height ?? null,
    status,
    stage: oneOfOrNull(STAGES, raw.stage, 'job.stage', raw, log),
    progress: typeof raw.progress === 'number' ? raw.progress : null,
    errorCode: oneOfOrNull(ERROR_CODES, raw.error_code, 'job.error_code', raw, log),
    error: raw.error ?? null,
    attempt: raw.attempt ?? 1,
    createdAt: raw.created_at,
    updatedAt: raw.updated_at,
    alliances: raw.alliances ?? null,
    tbaScore: raw.tba_score ?? null,
  };
}

export function parseEvent(raw: WireEvent, log: ViolationLog): ScoutEvent | null {
  checkVersion(raw.schema_version, 'event.schema_version', raw, log);
  const phase = oneOf(PHASES, raw.phase, 'event.phase', raw, log);
  const eventType = oneOf(EVENT_TYPES, raw.event_type, 'event.event_type', raw, log);
  const source = oneOf(SOURCES, raw.source, 'event.source', raw, log);
  if (!phase || !eventType || !source) return null;
  if (typeof raw.confidence !== 'number' || raw.confidence < 0 || raw.confidence > 1) {
    log.add('event.confidence', `${JSON.stringify(raw.confidence)} is not a float 0..1`, raw);
    return null;
  }
  return {
    jobId: raw.job_id,
    matchId: raw.match_id,
    eventId: raw.event_id,
    team: raw.team ?? null,
    trackId: raw.track_id ?? null,
    tSeconds: raw.t_seconds,
    phase,
    eventType,
    confidence: raw.confidence,
    fieldX: raw.field_x ?? null,
    fieldY: raw.field_y ?? null,
    goal: raw.goal ?? null,
    source,
    corrected: raw.corrected === true,
    correctionId: raw.correction_id ?? null,
  };
}

export function parseTrack(raw: WireTrack, log: ViolationLog): Track | null {
  checkVersion(raw.schema_version, 'track.schema_version', raw, log);
  let alliance: Alliance | null = null;
  if (raw.alliance != null) {
    alliance = oneOf(ALLIANCES, raw.alliance, 'track.alliance', raw, log);
    if (!alliance) return null;
  }
  const gaps: Gap[] = [];
  if (!Array.isArray(raw.gaps)) {
    // Contract C makes gaps required. Missing means we cannot tell a hole from a low sample
    // rate, which is exactly the failure the field exists to prevent.
    log.add('track.gaps', 'required array is missing', raw);
  } else {
    for (const g of raw.gaps) {
      const reason = oneOf(GAP_REASONS, g.reason, 'track.gaps[].reason', g, log);
      if (!reason) continue;
      gaps.push({ start: g.start, end: g.end, reason });
    }
  }
  return {
    trackId: raw.track_id,
    team: raw.team ?? null,
    alliance,
    positionSource: raw.position_source ?? null,
    teamConfidence: typeof raw.team_confidence === 'number' ? raw.team_confidence : null,
    boxes: (raw.boxes ?? []).map((box) => ({
      ...box,
      fieldX: box.field_x ?? null,
      fieldY: box.field_y ?? null,
      velocityXFtps: box.velocity_x_ftps ?? null,
      velocityYFtps: box.velocity_y_ftps ?? null,
      speedFtps: box.speed_ftps ?? null,
      motionHeadingRad: box.motion_heading_rad ?? null,
    })),
    gaps,
  };
}

export function parseCorrection(raw: WireCorrection, log: ViolationLog): Correction | null {
  const scope = oneOf(CORRECTION_SCOPES, raw.scope, 'correction.scope', raw, log);
  const action = oneOf(CORRECTION_ACTIONS, raw.action, 'correction.action', raw, log);
  if (!scope || !action) return null;
  if (scope === 'track' && !raw.job_id) {
    // track_id is job-local, so a track-scoped correction without a job cannot be addressed.
    log.add('correction.job_id', 'required when scope is "track"', raw);
    return null;
  }
  return {
    correctionId: raw.correction_id,
    scope,
    jobId: raw.job_id ?? null,
    targetId: raw.target_id,
    action,
    fields: raw.fields ? eventFieldsToDomain(raw.fields) : null,
    createdAt: raw.created_at,
    createdBy: raw.created_by ?? null,
  };
}

// ---------------------------------------------------------------- serializers

const EVENT_KEY_MAP: Record<string, keyof WireEvent> = {
  jobId: 'job_id',
  matchId: 'match_id',
  eventId: 'event_id',
  team: 'team',
  trackId: 'track_id',
  tSeconds: 't_seconds',
  phase: 'phase',
  eventType: 'event_type',
  confidence: 'confidence',
  fieldX: 'field_x',
  fieldY: 'field_y',
  goal: 'goal',
  source: 'source',
};
const WIRE_KEY_MAP = Object.fromEntries(
  Object.entries(EVENT_KEY_MAP).map(([k, v]) => [v, k])
) as Record<string, keyof ScoutEvent>;

/** camelCase patch -> snake_case patch, for PATCH /api/events/:event_id. */
export function eventFieldsToWire(fields: Partial<ScoutEvent>): Partial<WireEvent> {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(fields)) {
    const wireKey = EVENT_KEY_MAP[k];
    if (wireKey) out[wireKey] = v;
  }
  return out as Partial<WireEvent>;
}

export function eventFieldsToDomain(fields: Partial<WireEvent>): Partial<ScoutEvent> {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(fields)) {
    const domainKey = WIRE_KEY_MAP[k];
    if (domainKey) out[domainKey] = v;
  }
  return out as Partial<ScoutEvent>;
}

export function eventToWire(e: ScoutEvent): WireEvent {
  return {
    schema_version: SCHEMA_VERSION,
    job_id: e.jobId,
    match_id: e.matchId,
    event_id: e.eventId,
    team: e.team,
    track_id: e.trackId,
    t_seconds: e.tSeconds,
    phase: e.phase,
    event_type: e.eventType,
    confidence: e.confidence,
    field_x: e.fieldX,
    field_y: e.fieldY,
    goal: e.goal,
    source: e.source,
  };
}
