// Season config, per year.
//
// Doc 0: "/contracts/seasons/<year>.json. Selected by the `season` field on the job record,
// so old footage stays analyzable after the game changes." A single current-season config
// breaks the first time someone loads 2025 footage, and they will.
//
// Reading from /contracts/ is not a cross-import -- that directory is explicitly shared.

import season2025 from '../../contracts/seasons/2025.json';
import season2026 from '../../contracts/seasons/2026.json';
import type { Phase } from './contracts';

export interface SeasonConfig {
  season: number;
  fieldLengthFt: number;
  fieldWidthFt: number;
  autoSeconds: number;
  teleopSeconds: number;
  endgameSeconds: number;
  gamePieces: string[];
  goals: string[];
  pointValues: Record<string, Record<string, number>>;
}

interface RawSeason {
  season: number;
  field_length_ft: number;
  field_width_ft: number;
  auto_seconds: number;
  teleop_seconds: number;
  endgame_seconds: number;
  game_pieces: string[];
  goals: string[];
  point_values: Record<string, Record<string, number>>;
}

function toConfig(raw: RawSeason): SeasonConfig {
  return {
    season: raw.season,
    fieldLengthFt: raw.field_length_ft,
    fieldWidthFt: raw.field_width_ft,
    autoSeconds: raw.auto_seconds,
    teleopSeconds: raw.teleop_seconds,
    endgameSeconds: raw.endgame_seconds,
    gamePieces: raw.game_pieces,
    goals: raw.goals,
    pointValues: raw.point_values,
  };
}

const REGISTRY: Record<number, SeasonConfig> = {
  2025: toConfig(season2025 as RawSeason),
  2026: toConfig(season2026 as RawSeason),
};

export const SEASONS = Object.keys(REGISTRY).map(Number).sort();

/** Null rather than a fallback: an unknown season is a bug, not something to guess through. */
export function seasonConfig(year: number): SeasonConfig | null {
  return REGISTRY[year] ?? null;
}

export interface PhaseBounds {
  autoStart: number;
  autoEnd: number;
  teleopEnd: number;
  matchEnd: number;
}

/** Phase boundaries in match-relative seconds, derived from the season config. */
export function phaseBounds(cfg: SeasonConfig): PhaseBounds {
  return {
    autoStart: 0,
    autoEnd: cfg.autoSeconds,
    teleopEnd: cfg.autoSeconds + cfg.teleopSeconds - cfg.endgameSeconds,
    matchEnd: cfg.autoSeconds + cfg.teleopSeconds,
  };
}

/**
 * Doc 0: "phase is a pure function of match-relative time and the season config. Both
 * component 1 and component 3 compute it with the same function from the same file, so they
 * cannot disagree."
 *
 * `tMatch` is segment time minus the time of match_start. Nobody hardcodes 15, 135 or 20.
 */
export function phaseAt(tMatch: number, cfg: SeasonConfig): Phase {
  const b = phaseBounds(cfg);
  if (tMatch < 0) return 'unknown';
  if (tMatch < b.autoEnd) return 'auto';
  if (tMatch < b.teleopEnd) return 'teleop';
  if (tMatch <= b.matchEnd) return 'endgame';
  return 'unknown';
}

/**
 * Legal `goal` values for a season. Doc 0 deliberately keeps these out of the enum list,
 * because they change every January -- a closed set would need editing every season.
 */
export function legalGoals(cfg: SeasonConfig): ReadonlySet<string> {
  return new Set(cfg.goals);
}

/** Field extents in feet, matching WPILib/AprilTag coordinates from one field corner. */
export function fieldExtents(cfg: SeasonConfig) {
  return {
    minX: 0,
    maxX: cfg.fieldLengthFt,
    minY: 0,
    maxY: cfg.fieldWidthFt,
    lengthFt: cfg.fieldLengthFt,
    widthFt: cfg.fieldWidthFt,
  };
}

/**
 * True when every point value in the season config is still zero.
 *
 * Doc 0: "Point values are zero placeholders until the game is public. Score reconstruction
 * is not meaningful until they are filled in, and that is expected. Do not invent values to
 * make a test pass." The UI says so out loud rather than showing a confident 0.
 */
export function pointValuesArePlaceholders(cfg: SeasonConfig): boolean {
  return Object.values(cfg.pointValues).every((group) =>
    Object.values(group ?? {}).every((v) => v === 0)
  );
}
