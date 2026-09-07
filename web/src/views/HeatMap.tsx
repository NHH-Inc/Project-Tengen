import { useEffect, useMemo, useRef } from 'react';
import type { Alliance, Track } from '../contracts';
import { robotName, stationarySuppressionAt, visibleBoxes } from '../lib/tracks';
import { fieldExtents, type SeasonConfig } from '../season';

// Contract C already carries optional field_x/field_y on every box. The YOLO runner fills them
// from its AprilTag camera-pose calibration, so this view is dwell density and robot paths rather
// than the old event-only placeholder.

const GRID_X = 72;
const GRID_Y = 36;
const SIGMA = 1.6; // grid cells

export interface HeatMapProps {
  season: SeasonConfig;
  tracks: Track[];
  selectedTeam: number | null;
  currentTime: number;
}

export function HeatMap({
  season,
  tracks,
  selectedTeam,
  currentTime,
}: HeatMapProps) {
  const FIELD = fieldExtents(season);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const { points, missing, sampleCount, positionedTracks, positionSources } = useMemo(() => {
    const points: Array<{
      fieldX: number;
      fieldY: number;
      trackId: number;
      alliance: Alliance | null;
      t: number;
    }> = [];
    const positioned = new Set<number>();
    const sources = new Set<string>();
    let missing = 0;
    let sampleCount = 0;
    for (const track of tracks) {
      if (selectedTeam != null && track.team !== selectedTeam) continue;
      const suppressedAt = stationarySuppressionAt(track);
      if (track.positionSource) sources.add(track.positionSource);
      let lastIncluded = -Infinity;
      for (const box of track.boxes) {
        // The heat map is a playback view, not a full-match report: future samples must not
        // reveal where a robot will travel later in the match.
        if (box.t > currentTime) continue;
        if (track.alliance == null && suppressedAt != null && box.t >= suppressedAt) continue;
        sampleCount++;
        if (box.fieldX == null || box.fieldY == null) {
          missing++;
          continue;
        }
        if (
          box.fieldX < FIELD.minX || box.fieldX > FIELD.maxX ||
          box.fieldY < FIELD.minY || box.fieldY > FIELD.maxY
        ) {
          missing++;
          continue;
        }
        positioned.add(track.trackId);
        // Four points per second preserves dwell time while keeping a 60 fps match cheap to draw.
        if (box.t - lastIncluded < 0.25) continue;
        lastIncluded = box.t;
        points.push({
          fieldX: box.fieldX,
          fieldY: box.fieldY,
          trackId: track.trackId,
          alliance: track.alliance,
          t: box.t,
        });
      }
    }
    return {
      points,
      missing,
      sampleCount,
      positionedTracks: positioned.size,
      positionSources: [...sources].sort(),
    };
  }, [tracks, selectedTeam, currentTime, FIELD.minX, FIELD.maxX, FIELD.minY, FIELD.maxY]);

  const liveRobots = useMemo(
    () => visibleBoxes(tracks, currentTime, 0.25).filter(({ track, box }) =>
      (selectedTeam == null || track.team === selectedTeam) &&
      box.fieldX != null && box.fieldY != null &&
      box.fieldX >= FIELD.minX && box.fieldX <= FIELD.maxX &&
      box.fieldY >= FIELD.minY && box.fieldY <= FIELD.maxY
    ),
    [tracks, currentTime, selectedTeam, FIELD.minX, FIELD.maxX, FIELD.minY, FIELD.maxY]
  );

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const cssW = canvas.clientWidth;
    const cssH = Math.round((cssW * FIELD.widthFt) / FIELD.lengthFt);
    canvas.style.height = `${cssH}px`;
    canvas.width = Math.round(cssW * dpr);
    canvas.height = Math.round(cssH * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);

    // Field. +x points at the blue alliance wall, so blue is on the right.
    ctx.fillStyle = '#171a21';
    ctx.fillRect(0, 0, cssW, cssH);
    ctx.fillStyle = 'rgba(224,85,95,0.10)';
    ctx.fillRect(0, 0, cssW * 0.16, cssH);
    ctx.fillStyle = 'rgba(76,140,240,0.10)';
    ctx.fillRect(cssW * 0.84, 0, cssW * 0.16, cssH);

    // Density grid with a small gaussian splat per quarter-second observation.
    const grid = new Float32Array(GRID_X * GRID_Y);
    const radius = Math.ceil(SIGMA * 2.5);
    for (const e of points) {
      const gx = ((e.fieldX! - FIELD.minX) / FIELD.lengthFt) * (GRID_X - 1);
      const gy = ((e.fieldY! - FIELD.minY) / FIELD.widthFt) * (GRID_Y - 1);
      const x0 = Math.max(0, Math.floor(gx - radius));
      const x1 = Math.min(GRID_X - 1, Math.ceil(gx + radius));
      const y0 = Math.max(0, Math.floor(gy - radius));
      const y1 = Math.min(GRID_Y - 1, Math.ceil(gy + radius));
      for (let y = y0; y <= y1; y++) {
        for (let x = x0; x <= x1; x++) {
          const d2 = (x - gx) ** 2 + (y - gy) ** 2;
          grid[y * GRID_X + x] += Math.exp(-d2 / (2 * SIGMA * SIGMA));
        }
      }
    }

    let peak = 0;
    for (const v of grid) if (v > peak) peak = v;
    if (peak > 0) {
      const cw = cssW / GRID_X;
      const ch = cssH / GRID_Y;
      for (let y = 0; y < GRID_Y; y++) {
        for (let x = 0; x < GRID_X; x++) {
          const v = grid[y * GRID_X + x] / peak;
          if (v < 0.02) continue;
          ctx.fillStyle = heatColour(v);
          ctx.fillRect(x * cw, y * ch, cw + 0.5, ch + 0.5);
        }
      }
    }

    // Draw each stable robot's path over the dwell heat. Gaps naturally break because the
    // pipeline emits no samples while the robot is unobserved.
    const byTrack = new Map<number, typeof points>();
    for (const point of points) {
      const group = byTrack.get(point.trackId) ?? [];
      group.push(point);
      byTrack.set(point.trackId, group);
    }
    ctx.lineWidth = 1.4;
    for (const path of byTrack.values()) {
      if (path.length < 2) continue;
      ctx.strokeStyle = path[0].alliance === 'red'
        ? 'rgba(255,105,115,0.42)'
        : path[0].alliance === 'blue'
          ? 'rgba(105,165,255,0.42)'
          : 'rgba(235,240,248,0.24)';
      ctx.beginPath();
      for (let i = 0; i < path.length; i++) {
        const x = ((path[i].fieldX - FIELD.minX) / FIELD.lengthFt) * cssW;
        const y = ((path[i].fieldY - FIELD.minY) / FIELD.widthFt) * cssH;
        if (i === 0 || path[i].t - path[i - 1].t > 0.75) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }

    // A live marker makes the field view follow playback instead of looking like a static
    // report. It is intentionally separate from dwell density so a robot can be visible even
    // before it has accumulated enough samples to make a bright heat spot.
    for (const { track, box } of liveRobots) {
      const x = ((box.fieldX! - FIELD.minX) / FIELD.lengthFt) * cssW;
      const y = ((box.fieldY! - FIELD.minY) / FIELD.widthFt) * cssH;
      const colour = track.alliance === 'red' ? '#ff6973' : track.alliance === 'blue' ? '#69a5ff' : '#f0f3f8';
      ctx.fillStyle = colour;
      ctx.strokeStyle = '#0d0f14';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(x, y, 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.font = '600 10px ui-monospace, SFMono-Regular, Menlo, monospace';
      ctx.fillStyle = '#f4f6fb';
      ctx.fillText(robotName(track, tracks), x + 7, y + 3);
    }

    // Field markings on top of the heat.
    ctx.strokeStyle = 'rgba(255,255,255,0.22)';
    ctx.lineWidth = 1;
    ctx.strokeRect(0.5, 0.5, cssW - 1, cssH - 1);
    ctx.beginPath();
    ctx.moveTo(cssW / 2, 0);
    ctx.lineTo(cssW / 2, cssH);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(cssW / 2, cssH / 2, Math.min(cssW, cssH) * 0.08, 0, Math.PI * 2);
    ctx.stroke();
  }, [points, liveRobots, tracks, FIELD.lengthFt, FIELD.widthFt, FIELD.minX, FIELD.minY]);

  return (
    <div className="panel">
      <div className="panel-head">
        <h2>Field heat map</h2>
        <span className="muted">
          {selectedTeam ? `team ${selectedTeam}` : 'all robots'} · through {formatTime(currentTime)}
        </span>
      </div>
      <canvas ref={canvasRef} className="heatmap" />
      <div className="heat-axis">
        <span className="muted">x = 0 ft</span>
        <span className="muted">WPILib AprilTag field coordinates · +y is drawn lower</span>
        <span className="muted">x = {season.fieldLengthFt.toFixed(1)} ft</span>
      </div>
      <p className="note">
        {positionedTracks} positioned {positionedTracks === 1 ? 'track' : 'tracks'} ·{' '}
        {points.length} quarter-second dwell samples
        {missing > 0 && (
          <>
            {' · '}
            <span className="warn">
              {missing} of {sampleCount} raw box samples have no valid field position
            </span>
          </>
        )}
        . Calibration: {positionSources.length > 0 ? positionSources.join(', ') : 'unavailable'}.
        {sampleCount > 0 && positionedTracks === 0 && (
          <> Re-run this job to generate field positions from the AprilTag homography.</>
        )}
      </p>
    </div>
  );
}

function formatTime(seconds: number): string {
  const safeSeconds = Math.max(0, Math.floor(seconds));
  return `${Math.floor(safeSeconds / 60)}:${String(safeSeconds % 60).padStart(2, '0')}`;
}

/** Dark blue -> cyan -> amber -> white. Perceptually rising, readable on a dark field. */
function heatColour(v: number): string {
  const stops: Array<[number, [number, number, number]]> = [
    [0.0, [30, 60, 130]],
    [0.35, [40, 160, 180]],
    [0.65, [232, 185, 59]],
    [1.0, [255, 245, 235]],
  ];
  let i = 0;
  while (i < stops.length - 2 && v > stops[i + 1][0]) i++;
  const [t0, c0] = stops[i];
  const [t1, c1] = stops[i + 1];
  const u = (v - t0) / (t1 - t0 || 1);
  const c = c0.map((ch, k) => Math.round(ch + (c1[k] - ch) * u));
  return `rgba(${c[0]},${c[1]},${c[2]},${0.25 + 0.65 * v})`;
}
