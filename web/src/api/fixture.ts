// The fixture client, SCHEMA_VERSION 3.
//
// Doc 0: "Component 3 builds the whole UI against fixture data with no backend running."
// Same ScoutingApi as the HTTP client, backed by /fixtures/. Not a mock in the testing sense
// -- it serves the real golden data, so anything that renders here renders against
// component 2 too. It loads all three fixture jobs, including the awkward ones (a match with
// no TBA data, and a failed download carrying an error_code).

import {
  ViolationLog,
  eventToWire,
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
import { applyCorrections, applyTrackCorrections } from '../lib/corrections';
import { parseVideoStartTime } from '../lib/format';
import { computeTeamStats, reconstructScore } from '../lib/stats';
import { seasonConfig, type SeasonConfig } from '../season';
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
  parseRunResult,
  type Accuracy,
  type ExportResult,
  type RunResult,
  type TeamStatsSummary,
  type WireRunResult,
} from './shapes';

/** Vite serves /fixtures as its publicDir, so the golden set is at the site root. */
const FIXTURE_DIRS = ['2026casf_qm42', '2026casf_qm43_no_tba', 'failed_download'];
const MAIN_FIXTURE = '2026casf_qm42';
const CORRECTIONS_KEY = 'frc-scouting.fixture-corrections.v2';

interface Bundle {
  job: Job;
  events: ScoutEvent[];
  tracks: Track[];
  result: RunResult | null;
}

async function loadJson<T>(dir: string, name: string): Promise<T> {
  const url = `/${dir}/${name}`;
  const res = await fetch(url);
  if (!res.ok) throw new ApiError(`Fixture ${dir}/${name} not found`, res.status, url);
  return (await res.json()) as T;
}

async function loadJsonl<T>(dir: string, name: string): Promise<T[]> {
  const url = `/${dir}/${name}`;
  const res = await fetch(url);
  if (!res.ok) throw new ApiError(`Fixture ${dir}/${name} not found`, res.status, url);
  const text = await res.text();
  return text.split('\n').filter((l) => l.trim()).map((l) => JSON.parse(l) as T);
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

const PIPELINE: Array<[Job['status'], Job['stage'], number]> = [
  ['queued', null, 1000],
  ['downloading', 'downloading', 3500],
  ['downloaded', null, 700],
  ['analyzing', 'detecting', 2000],
  ['analyzing', 'tracking', 2000],
  ['analyzing', 'events', 1500],
  ['complete', null, 0],
];

export class FixtureApi implements ScoutingApi {
  readonly mode = 'fixture' as const;

  private jobs: Job[] = [];
  private bundles = new Map<string, Bundle>(); // by match_id
  private corrections: Correction[] = [];
  private shippedCorrections = 0;
  private violations = new ViolationLog();
  // A promise, not a boolean. useMatch fires four calls concurrently; a flag set before
  // the awaits complete lets the other three skip seeding and read empty state.
  private seeding: Promise<void> | null = null;

  private seed(): Promise<void> {
    this.seeding ??= this.loadAll();
    return this.seeding;
  }

  private async loadAll(): Promise<void> {
    const log = new ViolationLog();

    for (const dir of FIXTURE_DIRS) {
      const wireJob = await loadJson<WireJob>(dir, 'job.json');
      const job = parseJob(wireJob, log);
      if (!job) continue;
      this.jobs.push(job);

      // A failed job never produced analysis output.
      if (job.status === 'failed' || !job.matchId) continue;

      const [wireEvents, wireTracks, wireResult] = await Promise.all([
        loadJsonl<WireEvent>(dir, 'events.jsonl'),
        loadJsonl<WireTrack>(dir, 'tracks.jsonl'),
        loadJson<WireRunResult>(dir, 'result.json'),
      ]);
      this.bundles.set(job.matchId, {
        job,
        events: wireEvents
          .map((e) => parseEvent(e, log))
          .filter((e): e is ScoutEvent => e !== null)
          .sort((a, b) => a.tSeconds - b.tSeconds),
        tracks: wireTracks.map((t) => parseTrack(t, log)).filter((t): t is Track => t !== null),
        result: parseRunResult(wireResult),
      });

      try {
        const wire = await loadJsonl<WireCorrection>(dir, 'corrections.jsonl');
        this.corrections.push(
          ...wire.map((c) => parseCorrection(c, log)).filter((c): c is Correction => c !== null)
        );
      } catch {
        // corrections.jsonl is optional
      }
    }
    this.shippedCorrections = this.corrections.length;
    this.violations = log;

    try {
      const saved = localStorage.getItem(CORRECTIONS_KEY);
      if (saved) this.corrections.push(...(JSON.parse(saved) as Correction[]));
    } catch {
      // private window, or storage disabled -- session-only corrections are fine
    }
  }

  private persist() {
    try {
      localStorage.setItem(
        CORRECTIONS_KEY,
        JSON.stringify(this.corrections.slice(this.shippedCorrections))
      );
    } catch {
      // non-fatal
    }
  }

  private forMatch(matchId: string): Bundle | null {
    return this.bundles.get(matchId) ?? null;
  }

  private correctionsFor(matchId: string): Correction[] {
    const bundle = this.forMatch(matchId);
    if (!bundle) return [];
    const eventIds = new Set(bundle.events.map((e) => e.eventId));
    return this.corrections.filter(
      (c) =>
        (c.scope === 'track' && c.jobId === bundle.job.jobId) ||
        (c.scope === 'event' && (eventIds.has(c.targetId) || c.jobId === bundle.job.jobId))
    );
  }

  private configFor(job: Job | undefined): SeasonConfig | null {
    return job ? seasonConfig(job.season) : null;
  }

  // ---- jobs

  async listJobs(): Promise<Parsed<Job[]>> {
    await this.seed();
    return { data: [...this.jobs], violations: this.violations.items };
  }

  async getJob(jobId: string): Promise<Parsed<Job | null>> {
    await this.seed();
    return { data: this.jobs.find((j) => j.jobId === jobId) ?? null, violations: [] };
  }

  async createJob(input: CreateJobInput): Promise<Parsed<Job>> {
    await this.seed();
    const template = this.jobs[0];
    const now = new Date().toISOString();
    const job: Job = {
      ...template,
      jobId: uuid(),
      matchId: input.matchId ?? null,
      season: input.season ?? template.season,
      videoId: extractId(input.url) ?? template.videoId,
      captureMode: input.liveCapture ? 'live' : 'recorded',
      startOffset: parseVideoStartTime(input.url),
      status: 'queued',
      stage: null,
      progress: null,
      errorCode: null,
      error: null,
      attempt: 1,
      createdAt: now,
      updatedAt: now,
    };
    this.jobs = [job, ...this.jobs];
    void this.advance(job.jobId);
    return { data: job, violations: [] };
  }

  /** Walk a simulated job through the pipeline so the queue UI has something to show. */
  private async advance(jobId: string): Promise<void> {
    for (const [status, stage, dwell] of PIPELINE) {
      await sleep(dwell);
      const i = this.jobs.findIndex((j) => j.jobId === jobId);
      if (i < 0) return; // deleted mid-flight
      this.jobs[i] = {
        ...this.jobs[i],
        status,
        stage,
        progress: stage ? Math.round(Math.random() * 80 + 10) / 100 : null,
        updatedAt: new Date().toISOString(),
        matchId:
          status === 'complete' ? (this.jobs[i].matchId ?? MAIN_FIXTURE) : this.jobs[i].matchId,
      };
      this.jobs = [...this.jobs];
    }
  }

  async deleteJob(jobId: string): Promise<void> {
    await this.seed();
    this.jobs = this.jobs.filter((j) => j.jobId !== jobId);
  }

  async retryJob(jobId: string): Promise<Parsed<Job>> {
    await this.seed();
    const i = this.jobs.findIndex((j) => j.jobId === jobId);
    if (i < 0) throw new ApiError(`No job ${jobId}`, 404, '/jobs');
    // Doc 0: retry REUSES the job id and increments attempt. A new job would orphan history.
    this.jobs[i] = {
      ...this.jobs[i],
      status: 'queued',
      stage: null,
      progress: null,
      errorCode: null,
      error: null,
      attempt: this.jobs[i].attempt + 1,
      updatedAt: new Date().toISOString(),
    };
    this.jobs = [...this.jobs];
    const job = this.jobs[i];
    void this.advance(jobId);
    return { data: job, violations: [] };
  }

  async getResult(jobId: string): Promise<RunResult | null> {
    await this.seed();
    for (const bundle of this.bundles.values()) {
      if (bundle.job.jobId === jobId) return bundle.result;
    }
    return null;
  }

  // ---- match data

  async getEvents(matchId: string, query: EventQuery = {}): Promise<Parsed<ScoutEvent[]>> {
    await this.seed();
    const bundle = this.forMatch(matchId);
    if (!bundle) return { data: [], violations: [] };
    const base = query.raw
      ? bundle.events
      : applyCorrections(bundle.events, this.correctionsFor(matchId));
    const min = query.minConfidence ?? 0;
    return {
      data: base.filter((e) => e.confidence >= min),
      violations: this.violations.items,
    };
  }

  async getTracks(matchId: string, query: EventQuery = {}): Promise<Parsed<TracksResponse>> {
    await this.seed();
    const bundle = this.forMatch(matchId);
    if (!bundle) return { data: { boxSampleRate: 0, tracks: [] }, violations: [] };
    // A track-scoped correction has to reach the tracks too, or the overlay keeps the old
    // label even though every event was re-attributed.
    const tracks = query.raw
      ? bundle.tracks
      : applyTrackCorrections(bundle.tracks, this.correctionsFor(matchId));
    return {
      data: { boxSampleRate: bundle.result?.boxSampleRate ?? 0, tracks },
      violations: [],
    };
  }

  async getJobTracks(_jobId: string): Promise<Parsed<TracksResponse>> {
    return { data: { boxSampleRate: 0, tracks: [] }, violations: [] };
  }

  async getCorrections(matchId: string): Promise<Parsed<Correction[]>> {
    await this.seed();
    return { data: this.correctionsFor(matchId), violations: [] };
  }

  async getAccuracy(matchId: string): Promise<Accuracy> {
    await this.seed();
    const bundle = this.forMatch(matchId);
    // Scored from RAW output: the corrected stream would measure the reviewers, not the model.
    const { data } = await this.getEvents(matchId, { raw: true });
    const cfg = this.configFor(bundle?.job);
    const reconstructed = cfg
      ? reconstructScore(data, bundle?.job.alliances ?? null, cfg)
      : { red: 0, blue: 0 };
    const tba = bundle?.job.tbaScore ?? null;
    return {
      matchId,
      tbaAvailable: tba != null,
      reconstructed,
      tba,
      delta: tba
        ? { red: reconstructed.red - tba.red, blue: reconstructed.blue - tba.blue }
        : null,
    };
  }

  // ---- corrections

  async createEvent(
    event: Omit<ScoutEvent, 'eventId' | 'corrected' | 'correctionId'>
  ): Promise<Parsed<ScoutEvent>> {
    await this.seed();
    const eventId = uuid();
    const correctionId = uuid();
    const created: ScoutEvent = {
      ...event,
      eventId,
      source: 'manual',
      corrected: true,
      correctionId,
    };
    this.corrections.push({
      correctionId,
      scope: 'event',
      jobId: event.jobId,
      targetId: eventId,
      action: 'create',
      fields: created,
      createdAt: new Date().toISOString(),
      createdBy: 'local',
    });
    this.persist();
    const log = new ViolationLog();
    const parsed = parseEvent({ ...eventToWire(created), schema_version: SCHEMA_VERSION }, log);
    if (!parsed) throw new ApiError('Manual event failed contract validation', 400, '/events');
    return { data: { ...parsed, corrected: true, correctionId }, violations: log.items };
  }

  private bundleForEvent(eventId: string): Bundle | undefined {
    return [...this.bundles.values()].find(
      (b) =>
        b.events.some((e) => e.eventId === eventId) ||
        this.corrections.some((c) => c.targetId === eventId && c.jobId === b.job.jobId)
    );
  }

  async patchEvent(eventId: string, fields: Partial<ScoutEvent>): Promise<Parsed<ScoutEvent>> {
    await this.seed();
    const bundle = this.bundleForEvent(eventId);
    this.corrections.push({
      correctionId: uuid(),
      scope: 'event',
      jobId: bundle?.job.jobId ?? null,
      targetId: eventId,
      action: 'edit',
      fields,
      createdAt: new Date().toISOString(),
      createdBy: 'local',
    });
    this.persist();
    if (!bundle?.job.matchId) throw new ApiError(`No event ${eventId}`, 404, '/events');
    const { data } = await this.getEvents(bundle.job.matchId);
    const found = data.find((e) => e.eventId === eventId);
    if (!found) throw new ApiError(`No event ${eventId}`, 404, '/events');
    return { data: found, violations: [] };
  }

  async deleteEvent(eventId: string): Promise<void> {
    await this.seed();
    const bundle = this.bundleForEvent(eventId);
    this.corrections.push({
      correctionId: uuid(),
      scope: 'event',
      jobId: bundle?.job.jobId ?? null,
      targetId: eventId,
      action: 'delete',
      fields: null,
      createdAt: new Date().toISOString(),
      createdBy: 'local',
    });
    this.persist();
  }

  async patchTrack(jobId: string, trackId: number, fields: { team: number | null }): Promise<void> {
    await this.seed();
    // One action re-attributes the track AND every event on it -- doc 3's primary path.
    this.corrections.push({
      correctionId: uuid(),
      scope: 'track',
      jobId,
      targetId: String(trackId),
      action: 'edit',
      fields: { team: fields.team },
      createdAt: new Date().toISOString(),
      createdBy: 'local',
    });
    this.persist();
  }

  async deleteCorrection(correctionId: string): Promise<void> {
    await this.seed();
    this.corrections = this.corrections.filter((c) => c.correctionId !== correctionId);
    this.persist();
  }

  // ---- stats and export

  async getTeamStats(
    team: number,
    eventKey?: string,
    minConfidence = 0
  ): Promise<TeamStatsSummary> {
    await this.seed();
    let events: ScoutEvent[] = [];
    let played = 0;
    let cfg: SeasonConfig | null = null;
    for (const bundle of this.bundles.values()) {
      if (!bundle.job.matchId) continue;
      if (eventKey && !bundle.job.matchId.startsWith(`${eventKey}_`)) continue;
      const { data } = await this.getEvents(bundle.job.matchId, { minConfidence });
      if (data.some((e) => e.team === team)) played++;
      events = events.concat(data);
      cfg = cfg ?? this.configFor(bundle.job);
    }
    const s = computeTeamStats(team, events, null, cfg, minConfidence);
    return {
      team,
      eventKey: eventKey ?? null,
      minConfidence,
      matchesPlayed: played,
      cycles: s.cycleCount,
      avgCycleSeconds: s.medianCycleSeconds,
      shotAttempts: s.shotAttempts,
      shotsMade: s.shotsMade,
      shotAccuracy: s.accuracy,
      avgShotIntervalSeconds: s.avgShotIntervalSeconds,
      reloads: s.reloads,
      defenseSeconds: s.defenseSeconds,
      immobileSeconds: s.immobileSeconds,
      fouls: s.fouls,
      lowConfidenceEvents: s.lowConfidenceEvents,
    };
  }

  async exportSheets(input: ExportInput): Promise<ExportResult> {
    await this.seed();
    await sleep(600);
    let rows = 0;
    for (const matchId of input.matchIds) {
      const { data } = await this.getEvents(matchId);
      rows +=
        input.mode === 'raw'
          ? data.length
          : new Set(data.map((e) => e.team).filter((t): t is number => t != null)).size;
    }
    return {
      spreadsheetId: '1FIXTUREsheetIdNotARealSpreadsheet',
      spreadsheetUrl: 'https://docs.google.com/spreadsheets/d/1FIXTUREsheetIdNotARealSpreadsheet',
      mode: input.mode,
      rowsWritten: rows,
      rowsSkipped: 0,
    };
  }

  videoUrl(_job: Job): string {
    // Only the main fixture ships a real segment.
    return `/${MAIN_FIXTURE}/segment.mp4`;
  }

  streamVideoUrl(job: Job): string {
    return this.videoUrl(job);
  }

  streamAudioUrl(job: Job): string {
    return this.videoUrl(job);
  }
}

function uuid(): string {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) return crypto.randomUUID();
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    return (c === 'x' ? r : (r & 0x3) | 0x8).toString(16);
  });
}

function extractId(url: string): string | null {
  const m = url.match(/[?&]v=([A-Za-z0-9_-]{11})|youtu\.be\/([A-Za-z0-9_-]{11})/);
  return m ? (m[1] ?? m[2]) : /^[A-Za-z0-9_-]{11}$/.test(url.trim()) ? url.trim() : null;
}
