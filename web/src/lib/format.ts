// Formatting, labels, and the one time conversion component 3 owns.

import type { EventType, JobStatus, Phase, Source } from '../contracts';

/** mm:ss.mmm -- segment time, three decimal places, matching the contract's precision. */
export function fmtTime(t: number): string {
  const sign = t < 0 ? '-' : '';
  const abs = Math.abs(t);
  const m = Math.floor(abs / 60);
  const s = abs - m * 60;
  return `${sign}${m}:${s.toFixed(3).padStart(6, '0')}`;
}

export function fmtClock(t: number): string {
  const m = Math.floor(t / 60);
  const s = Math.floor(t - m * 60);
  return `${m}:${String(s).padStart(2, '0')}`;
}

export function fmtSeconds(s: number | null, digits = 2): string {
  return s == null ? '--' : `${s.toFixed(digits)}s`;
}

export function fmtPercent(x: number | null, digits = 0): string {
  return x == null ? '--' : `${(x * 100).toFixed(digits)}%`;
}

export function fmtSigned(n: number): string {
  return n > 0 ? `+${n}` : String(n);
}

/**
 * Segment time -> position in the original YouTube video.
 *
 * Doc 0: "t_seconds is always relative to the start of the segment... To get a position in
 * the original video, add start_offset from the job record. Component 3 does that
 * conversion when linking to YouTube; nothing else ever should."
 *
 * This function and youtubeUrlAt() are the only places in web/ that add startOffset.
 */
export function toOriginalVideoTime(tSeconds: number, startOffset: number): number {
  return tSeconds + startOffset;
}

export function youtubeUrlAt(videoId: string, tSeconds: number, startOffset: number): string {
  const at = Math.max(0, Math.floor(toOriginalVideoTime(tSeconds, startOffset)));
  return `https://www.youtube.com/watch?v=${encodeURIComponent(videoId)}&t=${at}s`;
}

export const EVENT_LABEL: Record<EventType, string> = {
  match_start: 'Match start',
  match_end: 'Match end',
  phase_change: 'Phase change',
  shot_attempt: 'Shot attempt',
  shot_made: 'Shot made',
  reload: 'Reload',
  defense_start: 'Defense start',
  defense_end: 'Defense end',
  immobile_start: 'Immobile start',
  immobile_end: 'Immobile end',
  foul: 'Foul',
};

export const PHASE_LABEL: Record<Phase, string> = {
  auto: 'Auto',
  teleop: 'Teleop',
  endgame: 'Endgame',
  unknown: 'Unknown',
};

export const SOURCE_LABEL: Record<Source, string> = {
  model: 'Model',
  scoreboard_ocr: 'Scoreboard OCR',
  tba: 'TBA',
  manual: 'Manual',
};

export const STATUS_LABEL: Record<JobStatus, string> = {
  queued: 'Queued',
  downloading: 'Downloading',
  downloaded: 'Downloaded',
  analyzing: 'Analyzing',
  complete: 'Complete',
  failed: 'Failed',
};

/** Statuses where the job is still moving and the queue should keep polling. */
export const ACTIVE_STATUSES: ReadonlySet<JobStatus> = new Set<JobStatus>([
  'queued',
  'downloading',
  'downloaded',
  'analyzing',
]);

/** A short glyph per event type, for the dense timeline rows. */
export const EVENT_GLYPH: Record<EventType, string> = {
  match_start: '|',
  match_end: '|',
  phase_change: '|',
  shot_attempt: 'o',
  shot_made: '*',
  reload: '+',
  defense_start: 'D',
  defense_end: 'd',
  immobile_start: 'X',
  immobile_end: 'x',
  foul: '!',
};

/**
 * Extract an 11-character YouTube ID from whatever the user pasted.
 * Returns null rather than guessing, so the form can say so instead of queueing garbage.
 */
export function parseVideoId(input: string): string | null {
  const s = input.trim();
  if (/^[A-Za-z0-9_-]{11}$/.test(s)) return s;
  const patterns = [
    /[?&]v=([A-Za-z0-9_-]{11})/,
    /youtu\.be\/([A-Za-z0-9_-]{11})/,
    /youtube\.com\/embed\/([A-Za-z0-9_-]{11})/,
    /youtube\.com\/live\/([A-Za-z0-9_-]{11})/,
    /youtube\.com\/shorts\/([A-Za-z0-9_-]{11})/,
  ];
  for (const p of patterns) {
    const m = s.match(p);
    if (m) return m[1];
  }
  return null;
}

/** Extract a YouTube URL's optional t/start/time_continue value as seconds. */
export function parseVideoStartTime(input: string): number {
  let parsed: URL;
  try {
    parsed = new URL(input.trim());
  } catch {
    return 0;
  }
  const params = [parsed.searchParams, new URLSearchParams(parsed.hash.replace(/^#/, ''))];
  for (const values of params) {
    for (const key of ['t', 'start', 'time_continue']) {
      const raw = values.get(key)?.trim().toLowerCase();
      if (!raw) continue;
      const numeric = Number(raw);
      if (Number.isFinite(numeric)) return Math.max(0, numeric);
      const match = raw.match(/^(?:(\d+(?:\.\d+)?)h)?(?:(\d+(?:\.\d+)?)m)?(?:(\d+(?:\.\d+)?)s)?$/);
      if (match && (match[1] || match[2] || match[3])) {
        return Number(match[1] ?? 0) * 3600 + Number(match[2] ?? 0) * 60 + Number(match[3] ?? 0);
      }
    }
  }
  return 0;
}

/** TBA match keys are lowercase, e.g. 2026casf_qm42. */
export function isMatchKey(s: string): boolean {
  return /^[0-9]{4}[a-z0-9]+_[a-z0-9]+$/.test(s.trim());
}

/** The match levels, qualification before playoffs, as the strategy layer orders them. */
const MATCH_LEVELS = ['qm', 'ef', 'qf', 'sf', 'f'];
const LEVEL_LABEL: Record<string, string> = {
  qm: 'Qual', ef: 'Eighth', qf: 'Quarter', sf: 'Semi', f: 'Final',
};

export interface MatchKeyParts {
  event: string; // e.g. 2026casf
  level: string; // qm | ef | qf | sf | f
  levelIndex: number; // for sorting, qualification (0) before playoffs
  number: number;
  label: string; // "Qual 42", "Semi 1-2"
}

/**
 * Split a TBA match key into event, level and number, mirroring the strategy layer's match_order
 * (ingest/strategy/profile.py) so the web library sorts matches the same way the Python side does.
 */
export function parseMatchKey(matchId: string | null): MatchKeyParts | null {
  if (!matchId) return null;
  const m = matchId.match(/^(\d{4}[a-z0-9]+?)_(qm|ef|qf|sf|f)(\d+)(?:m(\d+))?$/);
  if (!m) return null;
  const [, event, level, number, game] = m;
  const label = `${LEVEL_LABEL[level]} ${number}${game ? `-${game}` : ''}`;
  return { event, level, levelIndex: MATCH_LEVELS.indexOf(level), number: Number(number), label };
}

/** A short, human "when" for a job's createdAt (ISO string). Falls back to a local date. */
export function fmtAgo(iso: string | null | undefined): string {
  if (!iso) return '';
  const then = Date.parse(iso);
  if (!Number.isFinite(then)) return '';
  const seconds = Math.max(0, (Date.now() - then) / 1000);
  if (seconds < 60) return 'just now';
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  if (days < 7) return `${days}d ago`;
  return new Date(then).toLocaleDateString();
}
