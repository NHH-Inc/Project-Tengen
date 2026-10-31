import { useEffect, useMemo, useRef, useState } from 'react';
import { getApi } from '../api';
import type { StrategyTeam, TeamProfile } from '../api/strategy';
import { useTeamProfiles } from '../state/useTeamProfiles';
import { DEFAULT_MAX_SPEED_FTPS, defaultParams, poseAt, reached, simulateMatch, type SimRobot } from '../lib/sim';
import fieldBackgroundUrl from '../assets/field-heatmap-background.png';

// The interactive field sim (PLAN-2026 §2; docs/SIM-DESIGN.md). Pick an attacker and a defender,
// and watch the match play out: the attacker drives to its best shooting spot, the defender shadows
// and blocks it. The physics (motion limits + robot-robot blocking) run client-side in lib/sim.ts,
// a TypeScript port of the Python sim core. Everything is in the alliance frame the profiles use.

// Strategy-geometry field (contracts/strategy/2026.json), the frame the profiles' coordinates live in.
const FIELD = { length: 54.268, width: 26.473 };
const HUB = { x: 39.093, y: 13.237 }; // opponent hub: the default target if a team has no hot spot
const DURATION_S = 10.0;

function attackerTarget(p: TeamProfile | undefined): [number, number] {
  const spot = p?.scoring_map?.hot_spot;
  return spot ? [spot.x, spot.y] : [HUB.x, HUB.y];
}

function startOf(p: TeamProfile | undefined, fallback: [number, number]): [number, number] {
  const s = p?.tendencies?.start;
  return s && s.mean_ax != null && s.mean_ay != null ? [s.mean_ax, s.mean_ay] : fallback;
}

function paramsOf(p: TeamProfile | undefined) {
  const top = p?.tendencies?.movement?.top_speed_ftps;
  return defaultParams({ maxSpeedFtps: top ?? DEFAULT_MAX_SPEED_FTPS });
}

export function Sim() {
  const [teams, setTeams] = useState<StrategyTeam[]>([]);
  const [attacker, setAttacker] = useState<number | null>(null);
  const [defender, setDefender] = useState<number | null>(null);
  const [defenderOn, setDefenderOn] = useState(true);
  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    getApi().then((api) => api.listStrategyTeams()).then(setTeams)
      .catch((err) => setError(err instanceof Error ? err.message : String(err)));
  }, []);

  const { profiles } = useTeamProfiles([attacker, defender].filter((x): x is number => x != null));
  const attProfile = attacker != null ? profiles.get(attacker) : undefined;
  const defProfile = defender != null ? profiles.get(defender) : undefined;

  const scenario = useMemo(() => {
    if (attacker == null || !attProfile) return null;
    const target = attackerTarget(attProfile);
    const att: SimRobot = {
      id: 'att', start: startOf(attProfile, [2, FIELD.width / 2]), params: paramsOf(attProfile),
      waypoints: [target], alliance: 'us',
    };
    const hasDef = defenderOn && defender != null && defProfile;
    const robots = [att];
    if (hasDef) {
      robots.push({ id: 'def', start: [Math.min(target[0] + 6, FIELD.length - 2), target[1]],
                    params: paramsOf(defProfile), shadow: 'att', alliance: 'them' });
    }
    const solo = simulateMatch([att], { durationS: DURATION_S });
    const tracks = simulateMatch(robots, { durationS: DURATION_S });
    return {
      target, tracks, robots,
      soloReach: reached(solo.get('att')!, target),
      defReach: reached(tracks.get('att')!, target),
    };
  }, [attacker, defender, defenderOn, attProfile, defProfile]);

  // Animate: advance t while playing, looping over the scenario duration.
  useEffect(() => {
    if (!playing) return;
    let raf = 0;
    let last = performance.now();
    const tick = (now: number) => {
      const dt = (now - last) / 1000; last = now;
      setT((prev) => (prev + dt > DURATION_S ? 0 : prev + dt));
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing]);

  // Draw the field, trails up to t, and robot markers at t.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    const cssW = canvas.clientWidth;
    const cssH = Math.round((cssW * FIELD.width) / FIELD.length);
    canvas.style.height = `${cssH}px`;
    canvas.width = Math.round(cssW * dpr);
    canvas.height = Math.round(cssH * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);
    const toX = (x: number) => (x / FIELD.length) * cssW;
    const toY = (y: number) => (y / FIELD.width) * cssH;

    if (scenario) {
      // Target ring.
      ctx.strokeStyle = 'rgba(242,204,96,0.9)'; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(toX(scenario.target[0]), toY(scenario.target[1]), 7, 0, Math.PI * 2); ctx.stroke();
      for (const robot of scenario.robots) {
        const path = scenario.tracks.get(robot.id)!;
        const colour = robot.alliance === 'them' ? '#ff6973' : '#69a5ff';
        // Trail up to t.
        ctx.strokeStyle = colour + '66'; ctx.lineWidth = 2; ctx.beginPath();
        let started = false;
        for (const p of path) {
          if (p.t > t) break;
          const cx = toX(p.x), cy = toY(p.y);
          if (!started) { ctx.moveTo(cx, cy); started = true; } else ctx.lineTo(cx, cy);
        }
        ctx.stroke();
        // Current marker.
        const now = poseAt(path, t);
        if (now) {
          ctx.fillStyle = colour; ctx.strokeStyle = '#0d0f14'; ctx.lineWidth = 2;
          ctx.beginPath(); ctx.arc(toX(now.x), toY(now.y), 7, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
          ctx.font = '600 10px ui-monospace, monospace'; ctx.fillStyle = '#f4f6fb';
          ctx.fillText(robot.alliance === 'them' ? 'D' : 'A', toX(now.x) - 3, toY(now.y) + 3);
        }
      }
    }
  }, [scenario, t]);

  const sideToggle = (team: number, which: 'att' | 'def') => {
    const [get, set] = which === 'att' ? [attacker, setAttacker] : [defender, setDefender] as const;
    set(get === team ? null : team);
    setT(0);
  };

  return (
    <div className="panel">
      <div className="panel-head">
        <h2>Field sim</h2>
        <span className="muted">attacker drives to its spot · defender shadows and blocks</span>
      </div>
      {error && <p className="error">{error}</p>}
      {teams.length === 0 && <p className="muted">No teams yet: assign team numbers to robots in a match.</p>}

      <div className="plan-pick">
        <div className="plan-row">
          <span className="plan-tag us">Attacker</span>
          <div className="team-picker">
            {teams.map(({ team }) => (
              <button key={team} type="button" className={attacker === team ? 'on us' : ''}
                      disabled={defender === team} onClick={() => sideToggle(team, 'att')}>{team}</button>
            ))}
          </div>
        </div>
        <div className="plan-row">
          <span className="plan-tag them">Defender</span>
          <div className="team-picker">
            {teams.map(({ team }) => (
              <button key={team} type="button" className={defender === team ? 'on them' : ''}
                      disabled={attacker === team} onClick={() => sideToggle(team, 'def')}>{team}</button>
            ))}
          </div>
          <label className="check" style={{ marginLeft: 8 }}>
            <input type="checkbox" checked={defenderOn} onChange={(e) => setDefenderOn(e.target.checked)} />
            defense on
          </label>
        </div>
      </div>

      {attacker == null ? (
        <p className="empty">Pick an attacker to run the sim.</p>
      ) : (
        <>
          <canvas ref={canvasRef} className="heatmap" style={{
            backgroundColor: '#101827', backgroundImage: `url(${fieldBackgroundUrl})`,
            backgroundPosition: 'center', backgroundRepeat: 'no-repeat', backgroundSize: '100% 100%',
          }} />
          <div className="sim-controls">
            <button type="button" className="chip" onClick={() => setPlaying((p) => !p)}>
              {playing ? '❚❚ Pause' : '▶ Play'}
            </button>
            <input type="range" min={0} max={DURATION_S} step={0.05} value={t}
                   onChange={(e) => { setPlaying(false); setT(Number(e.target.value)); }} style={{ flex: 1 }} />
            <span className="muted" style={{ fontFamily: 'var(--mono)' }}>{t.toFixed(1)}s</span>
          </div>
          {scenario && (
            <p className="note">
              <b>A</b> (attacker {attacker}) drives to its best spot ({scenario.target[0].toFixed(0)}, {scenario.target[1].toFixed(0)}) ft.{' '}
              {scenario.soloReach != null
                ? <>Unopposed it arrives in <b>{scenario.soloReach.toFixed(1)}s</b>. </>
                : <>Unopposed it does not reach the spot in {DURATION_S}s. </>}
              {defenderOn && defender != null && (
                scenario.defReach != null
                  ? <>With <b>D</b> (defender {defender}) it is delayed to <b>{scenario.defReach.toFixed(1)}s</b>.</>
                  : <>With <b>D</b> (defender {defender}) it is <b>denied</b> — it never reaches the spot in {DURATION_S}s.</>
              )}
            </p>
          )}
        </>
      )}
    </div>
  );
}
