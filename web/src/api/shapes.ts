// Contract E response shapes, SCHEMA_VERSION 3.
//
// These were assumptions in v1 and are now written out in doc 0, so this file transcribes
// rather than guesses. Collection endpoints return an OBJECT, never a bare array -- that is
// what let `box_sample_rate` land on the tracks response without a breaking change, and it
// leaves room for pagination later.

// ---- envelopes

export interface WireJobList<J> { jobs: J[] }
export interface WireEventList<E> { events: E[] }
export interface WireCorrectionList<C> { corrections: C[] }
export interface WireTrackList<T> {
  box_sample_rate: number;
  tracks: T[];
}

// ---- GET /api/matches/:match_id/accuracy

export interface ScoreBreakdown { red: number; blue: number }

export interface WireAccuracy {
  match_id: string;
  tba_available: boolean;
  reconstructed: ScoreBreakdown;
  tba: ScoreBreakdown | null;
  delta: ScoreBreakdown | null;
}

export interface Accuracy {
  matchId: string;
  /** Distinguishes "no TBA data for this match" from "the score matched exactly". */
  tbaAvailable: boolean;
  reconstructed: ScoreBreakdown;
  tba: ScoreBreakdown | null;
  delta: ScoreBreakdown | null;
}

export function parseAccuracy(raw: WireAccuracy): Accuracy {
  return {
    matchId: raw.match_id,
    tbaAvailable: raw.tba_available === true,
    reconstructed: raw.reconstructed,
    tba: raw.tba ?? null,
    delta: raw.delta ?? null,
  };
}

// ---- GET /api/teams/:team/stats
//
// Doc 0: "Fields may be added to this object additively. None may be renamed or removed."

export interface WireTeamStats {
  team: number;
  event_key: string | null;
  min_confidence: number;
  matches_played: number;
  cycles: number;
  avg_cycle_seconds: number | null;
  shot_attempts: number;
  shots_made: number;
  shot_accuracy: number | null;
  avg_shot_interval_seconds: number | null;
  reloads: number;
  defense_seconds: number;
  immobile_seconds: number;
  fouls: number;
  low_confidence_events: number;
}

export interface TeamStatsSummary {
  team: number;
  eventKey: string | null;
  minConfidence: number;
  matchesPlayed: number;
  cycles: number;
  avgCycleSeconds: number | null;
  shotAttempts: number;
  shotsMade: number;
  shotAccuracy: number | null;
  avgShotIntervalSeconds: number | null;
  reloads: number;
  defenseSeconds: number;
  immobileSeconds: number;
  fouls: number;
  lowConfidenceEvents: number;
}

export function parseTeamStats(raw: WireTeamStats): TeamStatsSummary {
  return {
    team: raw.team,
    eventKey: raw.event_key ?? null,
    minConfidence: raw.min_confidence ?? 0,
    matchesPlayed: raw.matches_played,
    cycles: raw.cycles,
    avgCycleSeconds: raw.avg_cycle_seconds ?? null,
    shotAttempts: raw.shot_attempts,
    shotsMade: raw.shots_made,
    shotAccuracy: raw.shot_accuracy ?? null,
    avgShotIntervalSeconds: raw.avg_shot_interval_seconds ?? null,
    reloads: raw.reloads,
    defenseSeconds: raw.defense_seconds,
    immobileSeconds: raw.immobile_seconds,
    fouls: raw.fouls,
    lowConfidenceEvents: raw.low_confidence_events ?? 0,
  };
}

// ---- POST /api/export/sheets

export interface WireExportResult {
  spreadsheet_id: string;
  spreadsheet_url: string;
  mode: 'raw' | 'aggregate';
  rows_written: number;
  rows_skipped: number;
}

export interface ExportResult {
  spreadsheetId: string;
  spreadsheetUrl: string;
  mode: 'raw' | 'aggregate';
  rowsWritten: number;
  /** Doc 3 wants re-export idempotent: a second run skips rather than duplicating. */
  rowsSkipped: number;
}

export function parseExportResult(raw: WireExportResult): ExportResult {
  return {
    spreadsheetId: raw.spreadsheet_id,
    spreadsheetUrl: raw.spreadsheet_url,
    mode: raw.mode,
    rowsWritten: raw.rows_written,
    rowsSkipped: raw.rows_skipped ?? 0,
  };
}

// ---- GET /api/jobs/:job_id/result  (Contract D result.json)

export interface WireRunResult {
  schema_version: number;
  job_id: string;
  model_version: string;
  box_sample_rate: number;
  homography_ok: boolean;
  homography_source?: string | null;
  frames_total: number;
  frames_analyzed: number;
  frames_skipped_shot_change: number;
  tracks_emitted: number;
  events_emitted: number;
  reconstructed_score: ScoreBreakdown | null;
  started_at: string;
  finished_at: string;
}

export interface RunResult {
  jobId: string;
  modelVersion: string;
  boxSampleRate: number;
  homographyOk: boolean;
  homographySource: string | null;
  framesTotal: number;
  framesAnalyzed: number;
  framesSkippedShotChange: number;
  tracksEmitted: number;
  eventsEmitted: number;
  reconstructedScore: ScoreBreakdown | null;
  startedAt: string;
  finishedAt: string;
}

export function parseRunResult(raw: WireRunResult): RunResult {
  return {
    jobId: raw.job_id,
    modelVersion: raw.model_version,
    boxSampleRate: raw.box_sample_rate,
    homographyOk: raw.homography_ok,
    homographySource: raw.homography_source ?? null,
    framesTotal: raw.frames_total,
    framesAnalyzed: raw.frames_analyzed,
    framesSkippedShotChange: raw.frames_skipped_shot_change,
    tracksEmitted: raw.tracks_emitted,
    eventsEmitted: raw.events_emitted,
    reconstructedScore: raw.reconstructed_score ?? null,
    startedAt: raw.started_at,
    finishedAt: raw.finished_at,
  };
}
