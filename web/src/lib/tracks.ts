// Box interpolation for the overlay.
//
// Contract C: "Box sampling rate does not need to match video frame rate. Component 3
// interpolates between samples." The fixture samples at 5 Hz against 30 fps video, so five
// out of every six rendered frames are interpolated. If this is wrong the boxes visibly
// stair-step, which is the fastest way to notice a regression here.
//
// v2 removed the threshold heuristic that used to stand in for gap detection. Contract C now
// carries an explicit `gaps` array, so "not observed" is a fact from the pipeline rather than
// something inferred from sample spacing.

import type { Box, Gap, Track } from '../contracts';

const STATIONARY_HIDE_SECONDS = 5;
const STATIONARY_WINDOW_SECONDS = 1.25;
const STATIONARY_MIN_DISPLACEMENT = 0.012;
const stationaryCache = new WeakMap<object, number | null>();

/**
 * Return the stable human-facing name for a track. New Contract C output carries this from the
 * tracker; old jobs did not, so order their job-local ids once rather than showing raw ByteTrack
 * ids such as "track 1427".
 */
export function robotName(track: Pick<Track, 'trackId' | 'robotName'>, tracks: Track[]): string {
  if (track.robotName) return track.robotName;
  const ids = [...new Set(tracks.map((candidate) => candidate.trackId))].sort((a, b) => a - b);
  const index = ids.indexOf(track.trackId);
  return `robot${index >= 0 ? index + 1 : 1}`;
}

function median(values: number[]): number {
  const ordered = [...values].sort((a, b) => a - b);
  const middle = Math.floor(ordered.length / 2);
  return ordered.length % 2 === 1
    ? ordered[middle]
    : (ordered[middle - 1] + ordered[middle]) / 2;
}

/** The same robust short-window movement test used by the YOLO runner. */
function imageWindowHasMotion(boxes: Box[]): boolean {
  if (boxes.length < 4) return false;
  const midpoint = Math.floor(boxes.length / 2);
  const first = boxes.slice(0, midpoint);
  const second = boxes.slice(midpoint);
  const firstX = median(first.map((box) => box.x + box.w / 2));
  const firstY = median(first.map((box) => box.y + box.h / 2));
  const secondX = median(second.map((box) => box.x + box.w / 2));
  const secondY = median(second.map((box) => box.y + box.h / 2));
  const diagonal = Math.hypot(
    median(boxes.map((box) => box.w)),
    median(boxes.map((box) => box.h)),
  );
  const required = Math.max(STATIONARY_MIN_DISPLACEMENT, diagonal * 0.10);
  return Math.hypot(secondX - firstX, secondY - firstY) >= required;
}

/**
 * Find the first time a track has remained still for five seconds. Once returned, the track is
 * hidden permanently for this recording, including if a stale detector box continues afterward.
 * This client-side guard also fixes already-persisted jobs that were generated before the runner
 * learned the stationary-object rule.
 */
export function stationarySuppressionAt(track: Pick<Track, 'boxes'>): number | null {
  const cached = stationaryCache.get(track);
  if (cached !== undefined) return cached;
  const boxes = [...track.boxes].sort((a, b) => a.t - b.t);
  if (boxes.length < 4) {
    stationaryCache.set(track, null);
    return null;
  }

  let lastMotionAt = boxes[0].t;
  const window: Box[] = [];
  for (const box of boxes) {
    window.push(box);
    while (window.length > 0 && window[0].t < box.t - STATIONARY_WINDOW_SECONDS) {
      window.shift();
    }
    if (imageWindowHasMotion(window)) lastMotionAt = box.t;
    if (box.t - lastMotionAt >= STATIONARY_HIDE_SECONDS) {
      const suppressedAt = lastMotionAt + STATIONARY_HIDE_SECONDS;
      stationaryCache.set(track, suppressedAt);
      return suppressedAt;
    }
  }
  stationaryCache.set(track, null);
  return null;
}

/** True when t falls inside a declared gap: the robot was not observed, so draw nothing. */
export function inGap(gaps: Gap[], t: number): Gap | null {
  for (const g of gaps) {
    if (t >= g.start && t <= g.end) return g;
  }
  return null;
}

/** Index of the last box at or before t, or -1 if t precedes every sample. */
function lastAtOrBefore(boxes: Box[], t: number): number {
  let lo = 0;
  let hi = boxes.length - 1;
  let ans = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (boxes[mid].t <= t) {
      ans = mid;
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  return ans;
}

/**
 * The track's box at time t, linearly interpolated between samples.
 *
 * Returns null when t is outside the track's lifetime, or inside a declared gap. Doc 0:
 * "Consumers must not interpolate across a listed gap." A robot gliding smoothly through
 * footage nobody analyzed is fabricated data rendered at the same confidence as real data.
 *
 * `holdSeconds` is one sample period, so a box does not strobe out at the last sample.
 */
export function boxAt(track: Track, t: number, holdSeconds = 0): Box | null {
  const b = track.boxes;
  if (b.length === 0) return null;
  if (inGap(track.gaps, t)) return null;
  if (t < b[0].t - 1e-9) return null;
  if (t > b[b.length - 1].t + holdSeconds + 1e-9) return null;

  const i = lastAtOrBefore(b, t);
  if (i < 0) return null;
  if (i === b.length - 1) return b[i];

  const a = b[i];
  const c = b[i + 1];
  const span = c.t - a.t;
  if (span <= 1e-9) return a;

  // Two consecutive samples can still straddle a gap, because the samples inside it were
  // never emitted. Interpolating across that span would recreate exactly the bug `gaps`
  // exists to prevent.
  for (const g of track.gaps) {
    if (g.start < c.t && g.end > a.t) return t <= a.t + holdSeconds ? a : null;
  }

  const u = (t - a.t) / span;
  const interpolate = (left: number | null, right: number | null): number | null =>
    left != null && right != null ? left + (right - left) * u : null;
  return {
    t,
    x: a.x + (c.x - a.x) * u,
    y: a.y + (c.y - a.y) * u,
    w: a.w + (c.w - a.w) * u,
    h: a.h + (c.h - a.h) * u,
    fieldX: interpolate(a.fieldX, c.fieldX),
    fieldY: interpolate(a.fieldY, c.fieldY),
    velocityXFtps: interpolate(a.velocityXFtps, c.velocityXFtps),
    velocityYFtps: interpolate(a.velocityYFtps, c.velocityYFtps),
    speedFtps: interpolate(a.speedFtps, c.speedFtps),
    motionHeadingRad: interpolate(a.motionHeadingRad, c.motionHeadingRad),
  };
}

/** Every track visible at time t, with its interpolated box. */
export function visibleBoxes(
  tracks: Track[],
  t: number,
  holdSeconds = 0
): Array<{ track: Track; box: Box }> {
  const out: Array<{ track: Track; box: Box }> = [];
  for (const track of tracks) {
    const suppressedAt = stationarySuppressionAt(track);
    if (suppressedAt != null && t >= suppressedAt) continue;
    const box = boxAt(track, t, holdSeconds);
    if (box) out.push({ track, box });
  }
  // Painter's order: boxes lower on screen are nearer the camera and drawn last.
  out.sort((p, q) => p.box.y - q.box.y);
  return out;
}

/** Any gap covering t, across all tracks -- used to tell the viewer why the overlay is empty. */
export function activeGaps(tracks: Track[], t: number): Gap[] {
  const found: Gap[] = [];
  for (const track of tracks) {
    const g = inGap(track.gaps, t);
    if (g) found.push(g);
  }
  return found;
}
