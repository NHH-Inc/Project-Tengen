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
