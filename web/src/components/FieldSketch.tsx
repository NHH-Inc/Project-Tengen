import type { ReactNode } from 'react';

// A top-down field in the ALLIANCE FRAME (contracts/strategy/2026.json): the robot's own
// alliance wall on the left, x toward the opponent, y to the driver's left (up here). Every
// strategy position is in this frame, so one team's red and blue matches draw in the same place.

export const FIELD_LENGTH_FT = 54.268;
export const FIELD_WIDTH_FT = 26.473;
const OWN_HUB = { x: 15.176, y: 13.237 };
const OPPONENT_HUB = { x: 39.093, y: 13.237 };
const HUB_HALF = 1.98;

export interface FieldSketchProps {
  width?: number;
  children?: (toX: (ft: number) => number, toY: (ft: number) => number, scale: number) => ReactNode;
  caption?: string;
}

export function FieldSketch({ width = 520, children, caption }: FieldSketchProps) {
  const pad = 10;
  const scale = (width - 2 * pad) / FIELD_LENGTH_FT;
  const height = FIELD_WIDTH_FT * scale + 2 * pad;
  const toX = (ft: number) => pad + ft * scale;
  const toY = (ft: number) => pad + (FIELD_WIDTH_FT - ft) * scale;
  return (
    <figure className="field-sketch">
      <svg viewBox={`0 0 ${width} ${height}`} width="100%" role="img" aria-label={caption ?? 'field'}>
        <rect x={pad} y={pad} width={FIELD_LENGTH_FT * scale} height={FIELD_WIDTH_FT * scale}
              fill="#11161f" stroke="#3a4352" />
        <line x1={toX(FIELD_LENGTH_FT / 2)} x2={toX(FIELD_LENGTH_FT / 2)} y1={toY(0)} y2={toY(FIELD_WIDTH_FT)}
              stroke="#2a313d" strokeDasharray="4 4" />
        {[OWN_HUB, OPPONENT_HUB].map((hub, i) => (
          <rect key={i} x={toX(hub.x - HUB_HALF)} y={toY(hub.y + HUB_HALF)} width={2 * HUB_HALF * scale}
                height={2 * HUB_HALF * scale} rx={2} fill={i === 0 ? '#2f81f7' : '#6e7681'} opacity={0.8} />
        ))}
        <text x={toX(0.6)} y={toY(FIELD_WIDTH_FT) + 13} fontSize={11} fill="#8b949e">own wall</text>
        <text x={toX(FIELD_LENGTH_FT) - 4} y={toY(FIELD_WIDTH_FT) + 13} fontSize={11} fill="#8b949e"
              textAnchor="end">opponent wall</text>
        {children?.(toX, toY, scale)}
      </svg>
      {caption && <figcaption className="muted">{caption}</figcaption>}
    </figure>
  );
}

export const ROBOT_COLOURS = ['#f2cc60', '#79c0ff', '#ff7b72', '#7ee787', '#d2a8ff', '#ffa657'];
