// The only surface component 3 uses to reach component 2.
//
// Doc 0: "Component 3 only ever talks to component 2 over HTTP." Nothing in web/ imports from
// analysis/ or ingest/, shells out, or calls TBA or YouTube directly. Two implementations of
// one interface: the HTTP client, and a fixture client so the whole UI runs with no backend.

import type { ContractViolation, Correction, Job, ScoutEvent, Track } from '../contracts';
import type { Accuracy, ExportResult, RunResult, TeamStatsSummary } from './shapes';
import type { ShotResponse } from './shots';
import type { MatchStrategy, StrategyTeam, TeamProfileResponse } from './strategy';

export interface Parsed<T> {
  data: T;
  violations: ContractViolation[];
}

export interface EventQuery {
  /** Contract E: ?min_confidence=0.5 */
  minConfidence?: number;
  /** Contract E: ?raw=true returns uncorrected model output. Honoured on /events and /tracks only. */
  raw?: boolean;
}

export interface CreateJobInput {
  url: string;
  /** Optional per Contract E. Component 2 resolves it from video metadata if omitted. */
  matchId?: string | null;
  /** Optional. Component 2 defaults it when omitted. */
  season?: number | null;
  /** Record an active stream until it ends, then analyze the finished MP4. */
  liveCapture?: boolean;
}

/** POST /api/jobs/upload: a recording made on this machine rather than a YouTube link. */
export interface UploadJobInput {
  file: File;
  matchId?: string | null;
  season?: number | null;
}

export interface ExportInput {
  matchIds: string[];
  mode: 'raw' | 'aggregate';
}

/** Contract C's tracks response: the sample rate rides along with the tracks. */
export interface TracksResponse {
  boxSampleRate: number;
  tracks: Track[];
}

export interface ScoutingApi {
  readonly mode: 'http' | 'fixture';

  listJobs(): Promise<Parsed<Job[]>>;
  getJob(jobId: string): Promise<Parsed<Job | null>>;
  createJob(input: CreateJobInput): Promise<Parsed<Job>>;
  /** Upload a phone or webcam recording; it is analyzed from disk, never fetched from YouTube. */
  uploadJob(input: UploadJobInput): Promise<Parsed<Job>>;
  deleteJob(jobId: string): Promise<void>;
  /**
   * Reuses the job id for both failed retries and completed-job re-runs. Resets status to queued,
   * clears error_code/error, increments attempt, and replaces the raw result on success.
   */
  retryJob(jobId: string): Promise<Parsed<Job>>;
  /** Contract D's result.json, including box_sample_rate and the frame counts. */
  getResult(jobId: string): Promise<RunResult | null>;
  /** Reviewable HSV shot trajectories and on-demand per-robot outcome counts. */
  getShots(jobId: string): Promise<ShotResponse>;

  getEvents(matchId: string, query?: EventQuery): Promise<Parsed<ScoutEvent[]>>;
  getTracks(matchId: string, query?: EventQuery): Promise<Parsed<TracksResponse>>;
  getJobTracks(jobId: string): Promise<Parsed<TracksResponse>>;
  getCorrections(matchId: string): Promise<Parsed<Correction[]>>;
  getAccuracy(matchId: string): Promise<Accuracy>;

  createEvent(event: Omit<ScoutEvent, 'eventId' | 'corrected' | 'correctionId'>): Promise<Parsed<ScoutEvent>>;
  patchEvent(eventId: string, fields: Partial<ScoutEvent>): Promise<Parsed<ScoutEvent>>;
  deleteEvent(eventId: string): Promise<void>;
  /**
   * Doc 3: "The most common correction is a misread bumper, and it is a track-level fix...
   * Build that path first; per-event editing is the exception." One action re-attributes the
   * track and every event on it. Scoped by job because track_id is job-local.
   */
  patchTrack(jobId: string, trackId: number, fields: { team: number | null }): Promise<void>;
  /** Undo. Doc 0: "Deleting a correction undoes it." */
  deleteCorrection(correctionId: string): Promise<void>;

  getTeamStats(team: number, eventKey?: string, minConfidence?: number): Promise<TeamStatsSummary>;
  exportSheets(input: ExportInput): Promise<ExportResult>;

  /** The strategy layer over one job: every robot's behaviour (ingest/hosted.py). */
  getStrategy(jobId: string): Promise<MatchStrategy | null>;
  /** Every team with at least one analysed match. */
  listStrategyTeams(): Promise<StrategyTeam[]>;
  /** One team's profile across its matches: tendencies, prediction, counters. */
  getTeamProfile(team: number): Promise<TeamProfileResponse | null>;

  /**
   * A token for one job's media URLs, or null where none is needed. A <video> cannot send the
   * team key as a header, and a key in a URL ends up in request logs, so media URLs carry this.
   */
  mediaToken(jobId: string): Promise<string | null>;
  /** GET /api/video/:job_id -- the local segment file, or a published run's stored copy. */
  videoUrl(job: Job, token?: string | null): string;
  /** yt-dlp-resolved DASH video and audio, proxied locally without a YouTube player. */
  streamVideoUrl(job: Job, token?: string | null): string;
  streamAudioUrl(job: Job, token?: string | null): string;
}

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly url: string,
    readonly code: string | null = null
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

// A promise, not the instance: concurrent callers must not each construct their own client
// while the dynamic import is still in flight.
let cached: Promise<ScoutingApi> | null = null;

/**
 * Picks the client from VITE_API_MODE. Defaults to fixture so a fresh clone runs with
 * `npm install && npm run dev` and nothing else -- doc 0: "Component 3 builds the whole UI
 * against fixture data with no backend running."
 */
export function getApi(): Promise<ScoutingApi> {
  cached ??= (async () => {
    const mode = (import.meta.env.VITE_API_MODE as string) ?? 'fixture';
    if (mode === 'http') {
      const { HttpApi } = await import('./http');
      return new HttpApi((import.meta.env.VITE_API_BASE as string) ?? '/api');
    }
    const { FixtureApi } = await import('./fixture');
    return new FixtureApi();
  })();
  return cached;
}
