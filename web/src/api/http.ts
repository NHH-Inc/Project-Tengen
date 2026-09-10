// Contract E, spoken over HTTP to component 2, SCHEMA_VERSION 3.
//
// Every endpoint here is one from doc 0. Errors come back as standard status codes with
// {"error_code": "...", "error": "message"}. Collection endpoints return objects, never bare
// arrays.

import {
  ViolationLog,
  eventFieldsToWire,
  parseCorrection,
  parseEvent,
  parseJob,
  parseTrack,
  SCHEMA_VERSION,
  type Correction,
  type Job,
  type ScoutEvent,
  type Track,
  type WireCorrection,
  type WireEvent,
  type WireJob,
  type WireTrack,
} from '../contracts';
import {
  ApiError,
  type CreateJobInput,
  type EventQuery,
  type ExportInput,
  type Parsed,
  type ScoutingApi,
  type TracksResponse,
} from './index';
import {
  parseAccuracy,
  parseExportResult,
  parseRunResult,
  parseTeamStats,
  type Accuracy,
  type ExportResult,
  type RunResult,
  type TeamStatsSummary,
  type WireAccuracy,
  type WireCorrectionList,
  type WireEventList,
  type WireExportResult,
  type WireJobList,
  type WireRunResult,
  type WireTeamStats,
  type WireTrackList,
} from './shapes';
import { EMPTY_SHOT_RESPONSE, parseShotResponse, type ShotResponse, type WireShotResponse } from './shots';

export class HttpApi implements ScoutingApi {
  readonly mode = 'http' as const;

  constructor(private readonly base: string) {}

  private async request<T>(path: string, init?: RequestInit): Promise<T> {
    const url = `${this.base}${path}`;
    let res: Response;
    try {
      res = await fetch(url, {
        ...init,
        headers: {
          Accept: 'application/json',
          ...(init?.body ? { 'Content-Type': 'application/json' } : {}),
          ...init?.headers,
        },
      });
    } catch {
      throw new ApiError(`Could not reach the ingest service at ${url}`, 0, url);
    }
    if (!res.ok) {
      let message = `${res.status} ${res.statusText}`;
      let code: string | null = null;
      try {
        const body = (await res.json()) as { error?: string; error_code?: string };
        if (body?.error) message = body.error;
        if (body?.error_code) code = body.error_code;
      } catch {
        // non-JSON error body; the status line is all we have
      }
      throw new ApiError(message, res.status, url, code);
    }
    if (res.status === 204) return undefined as T;
    return (await res.json()) as T;
  }

  // ---- jobs

  async listJobs(): Promise<Parsed<Job[]>> {
    const log = new ViolationLog();
    const raw = await this.request<WireJobList<WireJob>>('/jobs');
    const data = (raw.jobs ?? []).map((j) => parseJob(j, log)).filter((j): j is Job => j !== null);
    return { data, violations: log.items };
  }

  async getJob(jobId: string): Promise<Parsed<Job | null>> {
    const log = new ViolationLog();
    const raw = await this.request<WireJob>(`/jobs/${encodeURIComponent(jobId)}`);
    return { data: parseJob(raw, log), violations: log.items };
  }

  async createJob(input: CreateJobInput): Promise<Parsed<Job>> {
    const log = new ViolationLog();
    const body: Record<string, unknown> = { url: input.url };
    // Optional per Contract E; omit rather than send null so component 2 can tell
    // "resolve it for me" from "it is definitively unknown".
    if (input.matchId) body.match_id = input.matchId;
    if (input.season) body.season = input.season;
    if (input.liveCapture) body.live_capture = true;
    const raw = await this.request<WireJob>('/jobs', { method: 'POST', body: JSON.stringify(body) });
    const job = parseJob(raw, log);
    if (!job) throw new ApiError('Ingest returned a job that failed contract validation', 200, '/jobs');
    return { data: job, violations: log.items };
  }

  async deleteJob(jobId: string): Promise<void> {
    await this.request<void>(`/jobs/${encodeURIComponent(jobId)}`, { method: 'DELETE' });
  }

  async retryJob(jobId: string): Promise<Parsed<Job>> {
    const log = new ViolationLog();
    const raw = await this.request<WireJob>(`/jobs/${encodeURIComponent(jobId)}/retry`, {
      method: 'POST',
    });
    const job = parseJob(raw, log);
    if (!job) throw new ApiError('Retry returned a job that failed contract validation', 200, '/jobs');
    return { data: job, violations: log.items };
  }

  async getResult(jobId: string): Promise<RunResult | null> {
    try {
      const raw = await this.request<WireRunResult>(`/jobs/${encodeURIComponent(jobId)}/result`);
      return parseRunResult(raw);
    } catch (e) {
      // A job that has not finished analysis has no result yet; that is not an error.
      if (e instanceof ApiError && e.status === 404) return null;
      throw e;
    }
  }

  async getShots(jobId: string): Promise<ShotResponse> {
    try {
      const raw = await this.request<WireShotResponse>(`/jobs/${encodeURIComponent(jobId)}/shots`);
      return parseShotResponse(raw);
    } catch (e) {
      // Jobs analyzed before ball scouting was enabled have no sidecar.
      if (e instanceof ApiError && e.status === 404) return EMPTY_SHOT_RESPONSE;
      throw e;
    }
  }

  // ---- match data

  async getEvents(matchId: string, query: EventQuery = {}): Promise<Parsed<ScoutEvent[]>> {
    const log = new ViolationLog();
    const qs = new URLSearchParams();
    if (query.minConfidence != null) qs.set('min_confidence', String(query.minConfidence));
    if (query.raw) qs.set('raw', 'true');
    const suffix = qs.toString() ? `?${qs}` : '';
    const raw = await this.request<WireEventList<WireEvent>>(
      `/matches/${encodeURIComponent(matchId)}/events${suffix}`
    );
    const data = (raw.events ?? [])
      .map((e) => parseEvent(e, log))
      .filter((e): e is ScoutEvent => e !== null)
      .sort((a, b) => a.tSeconds - b.tSeconds);
    return { data, violations: log.items };
  }

  async getTracks(matchId: string, query: EventQuery = {}): Promise<Parsed<TracksResponse>> {
    const log = new ViolationLog();
    const suffix = query.raw ? '?raw=true' : '';
    const raw = await this.request<WireTrackList<WireTrack>>(
      `/matches/${encodeURIComponent(matchId)}/tracks${suffix}`
    );
    const tracks = (raw.tracks ?? [])
      .map((t) => parseTrack(t, log))
      .filter((t): t is Track => t !== null);
    if (typeof raw.box_sample_rate !== 'number' || raw.box_sample_rate <= 0) {
      log.add('tracks.box_sample_rate', 'missing or not a positive number', raw);
    }
    return {
      data: { boxSampleRate: raw.box_sample_rate, tracks },
      violations: log.items,
    };
  }

  async getJobTracks(jobId: string): Promise<Parsed<TracksResponse>> {
    const log = new ViolationLog();
    const raw = await this.request<WireTrackList<WireTrack>>(
      `/jobs/${encodeURIComponent(jobId)}/tracks`
    );
    const tracks = (raw.tracks ?? [])
      .map((t) => parseTrack(t, log))
      .filter((t): t is Track => t !== null);
    if (typeof raw.box_sample_rate !== 'number' || raw.box_sample_rate <= 0) {
      log.add('tracks.box_sample_rate', 'missing or not a positive number', raw);
    }
    return {
      data: { boxSampleRate: raw.box_sample_rate, tracks },
      violations: log.items,
    };
  }

  async getCorrections(matchId: string): Promise<Parsed<Correction[]>> {
    const log = new ViolationLog();
    const raw = await this.request<WireCorrectionList<WireCorrection>>(
      `/matches/${encodeURIComponent(matchId)}/corrections`
    );
    const data = (raw.corrections ?? [])
      .map((c) => parseCorrection(c, log))
      .filter((c): c is Correction => c !== null);
    return { data, violations: log.items };
  }

  async getAccuracy(matchId: string): Promise<Accuracy> {
    const raw = await this.request<WireAccuracy>(`/matches/${encodeURIComponent(matchId)}/accuracy`);
    return parseAccuracy(raw);
  }

  // ---- corrections

  async createEvent(
    event: Omit<ScoutEvent, 'eventId' | 'corrected' | 'correctionId'>
  ): Promise<Parsed<ScoutEvent>> {
    const log = new ViolationLog();
    const raw = await this.request<WireEvent>('/events', {
      method: 'POST',
      body: JSON.stringify({
        ...eventFieldsToWire(event as Partial<ScoutEvent>),
        schema_version: SCHEMA_VERSION,
        source: 'manual',
      }),
    });
    const parsed = parseEvent(raw, log);
    if (!parsed) throw new ApiError('Created event failed contract validation', 200, '/events');
    return { data: parsed, violations: log.items };
  }

  async patchEvent(eventId: string, fields: Partial<ScoutEvent>): Promise<Parsed<ScoutEvent>> {
    const log = new ViolationLog();
    const raw = await this.request<WireEvent>(`/events/${encodeURIComponent(eventId)}`, {
      method: 'PATCH',
      body: JSON.stringify(eventFieldsToWire(fields)),
    });
    const parsed = parseEvent(raw, log);
    if (!parsed) throw new ApiError('Patched event failed contract validation', 200, '/events');
    return { data: parsed, violations: log.items };
  }

  async deleteEvent(eventId: string): Promise<void> {
    await this.request<void>(`/events/${encodeURIComponent(eventId)}`, { method: 'DELETE' });
  }

  async patchTrack(jobId: string, trackId: number, fields: { team: number | null }): Promise<void> {
    await this.request<unknown>(
      `/jobs/${encodeURIComponent(jobId)}/tracks/${encodeURIComponent(String(trackId))}`,
      { method: 'PATCH', body: JSON.stringify({ team: fields.team }) }
    );
  }

  async deleteCorrection(correctionId: string): Promise<void> {
    await this.request<void>(`/corrections/${encodeURIComponent(correctionId)}`, {
      method: 'DELETE',
    });
  }

  // ---- stats and export

  async getTeamStats(
    team: number,
    eventKey?: string,
    minConfidence?: number
  ): Promise<TeamStatsSummary> {
    const qs = new URLSearchParams();
    if (eventKey) qs.set('event_key', eventKey);
    if (minConfidence != null) qs.set('min_confidence', String(minConfidence));
    const suffix = qs.toString() ? `?${qs}` : '';
    const raw = await this.request<WireTeamStats>(`/teams/${team}/stats${suffix}`);
    return parseTeamStats(raw);
  }

  async exportSheets(input: ExportInput): Promise<ExportResult> {
    const raw = await this.request<WireExportResult>('/export/sheets', {
      method: 'POST',
      body: JSON.stringify({ match_ids: input.matchIds, mode: input.mode }),
    });
    return parseExportResult(raw);
  }

  videoUrl(job: Job): string {
    return `${this.base}/video/${encodeURIComponent(job.jobId)}`;
  }

  streamVideoUrl(job: Job): string {
    return `${this.base}/stream/${encodeURIComponent(job.jobId)}/video`;
  }

  streamAudioUrl(job: Job): string {
    return `${this.base}/stream/${encodeURIComponent(job.jobId)}/audio`;
  }
}
