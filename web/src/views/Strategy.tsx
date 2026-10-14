import { useEffect, useState } from 'react';
import { getApi } from '../api';
import type { MatchStrategy, RobotFeatures } from '../api/strategy';
import { FieldSketch, ROBOT_COLOURS } from '../components/FieldSketch';

// The strategy layer for one match (ingest/strategy, served by ingest/hosted.py): what each
// robot did. Positions are in each robot's own alliance frame. Anything the match could not
// measure -- no calibration, no team -- shows as a dash, never as zero.

export interface StrategyPanelProps {
  jobId: string;
  /** Bumped when the page's own corrections change, so team assignments re-flow in. */
  refreshKey?: number;
  onOpenTeam: (team: number) => void;
}

const dash = (value: number | null | undefined, digits = 1, suffix = '') =>
  value == null ? '—' : `${value.toFixed(digits)}${suffix}`;

export function StrategyPanel({ jobId, refreshKey, onOpenTeam }: StrategyPanelProps) {
  const [data, setData] = useState<MatchStrategy | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let live = true;
    setLoading(true);
    setError(null);
    getApi()
      .then((api) => api.getStrategy(jobId))
      .then((result) => live && setData(result))
      .catch((err) => live && setError(err instanceof Error ? err.message : String(err)))
      .finally(() => live && setLoading(false));
    return () => {
      live = false;
    };
  }, [jobId, refreshKey]);

  if (loading) return <div className="panel"><p className="empty">Loading strategy…</p></div>;
  if (error) return <div className="panel"><p className="error">{error}</p></div>;
  if (!data || data.robots.length === 0) {
    return (
      <div className="panel">
        <p className="empty">No strategy data for this match. It needs a server with the strategy layer and a
          finished analysis.</p>
      </div>
    );
  }

  const robots = [...data.robots].sort((a, b) => (a.alliance ?? 'z').localeCompare(b.alliance ?? 'z'));
  const positioned = robots.some((r) => r.movement);
  return (
    <div className="panel">
      <div className="panel-head">
        <h2>Strategy</h2>
        <span className="muted">
          {data.clock?.source === 'scoreboard_clock'
            ? `match clock read from the scoreboard (auto at ${dash(data.clock.auto_start_t)} s)`
            : 'match time assumed from the clip start'}
          {' · '}positions in each robot's own alliance frame
        </span>
      </div>
      <div className="table-scroll">
        <table className="stats">
          <thead>
            <tr>
              <th>Robot</th><th>Alliance</th><th>Seen</th><th>On field</th><th>Shots</th><th>Top spot</th>
              <th>Volley gap</th><th>Dwell</th><th>Near opp.</th><th>Start</th>
            </tr>
          </thead>
          <tbody>
            {robots.map((r) => (
              <RobotRow key={r.robot_key} robot={r} onOpenTeam={onOpenTeam} />
            ))}
          </tbody>
        </table>
      </div>
      {positioned ? (
        <FieldSketch caption="Shooting spots (circles, sized by shots) and starting positions (squares), in each robot's alliance frame.">
          {(toX, toY, scale) =>
            robots.map((r, i) => (
              <g key={r.robot_key}>
                {(r.shooting?.spots ?? []).slice(0, 3).map((s, j) => (
                  <circle key={j} cx={toX(s.ax)} cy={toY(s.ay)} r={Math.max(3, Math.sqrt(s.launches) * scale * 0.35)}
                          fill={ROBOT_COLOURS[i % ROBOT_COLOURS.length]} fillOpacity={0.35}
                          stroke={ROBOT_COLOURS[i % ROBOT_COLOURS.length]}>
                    <title>{`${label(r)}: ${s.launches} shots from (${s.ax.toFixed(0)}, ${s.ay.toFixed(0)}) ft`}</title>
                  </circle>
                ))}
                {r.start && (
                  <rect x={toX(r.start.ax) - 4} y={toY(r.start.ay) - 4} width={8} height={8}
                        fill={ROBOT_COLOURS[i % ROBOT_COLOURS.length]}>
                    <title>{`${label(r)} started here`}</title>
                  </rect>
                )}
              </g>
            ))
          }
        </FieldSketch>
      ) : (
        <p className="muted">No field positions for this match: the camera could not be calibrated, so where
          robots went is unknown. Shot timing above does not need positions.</p>
      )}
    </div>
  );
}

function label(r: RobotFeatures) {
  return r.team ? `Team ${r.team}` : r.robot_key.replace('track:', 'robot ');
}

function RobotRow({ robot: r, onOpenTeam }: { robot: RobotFeatures; onOpenTeam: (team: number) => void }) {
  const spot = r.shooting?.spots?.[0];
  return (
    <tr onClick={() => r.team && onOpenTeam(r.team)} title={r.team ? 'Open this team’s profile' : 'Assign a team to see a profile'}>
      <td>{label(r)}{r.identity_issues.length > 0 && <span className="error"> ⚠</span>}</td>
      <td className={r.alliance ?? ''}>{r.alliance ?? '—'}</td>
      <td>{r.coverage.observed_s.toFixed(0)} s</td>
      <td>{r.coverage.positioned_s ? `${r.coverage.positioned_s.toFixed(0)} s` : '—'}</td>
      <td>{r.launches.total}</td>
      <td>{spot ? `(${spot.ax.toFixed(0)}, ${spot.ay.toFixed(0)}) ×${spot.launches}` : '—'}</td>
      <td>{dash(r.shooting?.median_volley_interval_s, 1, ' s')}</td>
      <td>{dash(r.shooting?.median_dwell_s, 1, ' s')}</td>
      <td>{dash(r.defense?.near_opponent_s, 0, ' s')}</td>
      <td>{r.start?.lane ?? '—'}</td>
    </tr>
  );
}
