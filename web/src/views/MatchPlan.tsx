import { useEffect, useMemo, useState } from 'react';
import { getApi } from '../api';
import type { StrategyTeam } from '../api/strategy';
import { buildMatchPlan } from '../lib/strategyEngine';
import { useTeamProfiles } from '../state/useTeamProfiles';

// Surfaces the match-plan engine (ingest/strategy/engine.py) for the drive team: pick our three and
// the opponents' three, and read the calls -- auto splits, opponent autos to contest, teleop deny
// zones, our strengths, and defenders to expect -- each with the evidence behind it. The numbers are
// computed client-side over the fetched profiles (lib/strategyEngine.ts).

const ALLIANCE_SIZE = 3;

export function MatchPlan() {
  const [teams, setTeams] = useState<StrategyTeam[]>([]);
  const [us, setUs] = useState<number[]>([]);
  const [them, setThem] = useState<number[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getApi().then((api) => api.listStrategyTeams()).then(setTeams)
      .catch((err) => setError(err instanceof Error ? err.message : String(err)));
  }, []);

  const { profiles, loading } = useTeamProfiles([...us, ...them]);
  const plan = useMemo(
    () => (us.length && them.length ? buildMatchPlan(us, them, profiles) : null),
    [us, them, profiles],
  );

  const toggle = (team: number, side: 'us' | 'them') => {
    const [get, set] = side === 'us' ? [us, setUs] : [them, setThem] as const;
    const other = side === 'us' ? them : us;
    if (get.includes(team)) { set(get.filter((t) => t !== team)); return; }
    if (other.includes(team) || get.length >= ALLIANCE_SIZE) return;
    set([...get, team]);
  };

  const sideOf = (team: number) => (us.includes(team) ? 'us' : them.includes(team) ? 'them' : null);

  return (
    <div className="panel">
      <div className="panel-head">
        <h2>Match plan</h2>
        <span className="muted">pick our three and theirs — the calls update live</span>
      </div>
      {error && <p className="error">{error}</p>}
      {teams.length === 0 && <p className="muted">No teams yet: assign team numbers to robots in a match.</p>}

      <div className="plan-pick">
        <div className="plan-row">
          <span className="plan-tag us">Us</span>
          <div className="team-picker">
            {teams.map(({ team }) => (
              <button key={team} type="button" className={sideOf(team) === 'us' ? 'on us' : ''}
                      disabled={sideOf(team) === 'them'} onClick={() => toggle(team, 'us')}>{team}</button>
            ))}
          </div>
        </div>
        <div className="plan-row">
          <span className="plan-tag them">Them</span>
          <div className="team-picker">
            {teams.map(({ team }) => (
              <button key={team} type="button" className={sideOf(team) === 'them' ? 'on them' : ''}
                      disabled={sideOf(team) === 'us'} onClick={() => toggle(team, 'them')}>{team}</button>
            ))}
          </div>
        </div>
      </div>

      {!plan && <p className="empty">Pick at least one team on each side to see the plan.</p>}
      {plan && (
        <>
          <h3>The calls</h3>
          {loading && profiles.size === 0 ? (
            <p className="empty">Loading profiles…</p>
          ) : plan.advice.length === 0 ? (
            <p className="muted">No clear call yet — these teams need more analysed matches.</p>
          ) : (
            <ul className="counters">
              {plan.advice.map((a, i) => (
                <li key={`${a.id}-${i}`}><b>{a.advice}</b><br /><span className="muted">{a.evidence}</span></li>
              ))}
            </ul>
          )}

          {plan.missing.length > 0 && (
            <p className="note">No profile yet for: {plan.missing.join(', ')}. They are left out of the plan.</p>
          )}

          <div className="plan-grid">
            <PlanList title="Deny (opponents score here)"
                      rows={plan.teleop.deny.map((z) => `${z.zone.replace(/_/g, ' ')} · ${z.lane} lane — weight ${z.weight} (${z.teams.join(', ')})`)} />
            <PlanList title="Our strengths (feed here)"
                      rows={plan.teleop.our_strengths.map((z) => `${z.zone.replace(/_/g, ' ')} · ${z.lane} lane — weight ${z.weight}`)} />
            <PlanList title="Auto conflicts (split these)"
                      rows={plan.auto.partner_conflicts.map((c) => `${c.teams.join(' & ')} both at (${c.x}, ${c.y}) ft`)} />
            <PlanList title="Contest their auto"
                      rows={plan.auto.contest_opponent_autos.map((a) => `team ${a.team} at (${a.x}, ${a.y}) ft — ${Math.round(a.share * 100)}% over ${a.matches}`)} />
          </div>
        </>
      )}
    </div>
  );
}

function PlanList({ title, rows }: { title: string; rows: string[] }) {
  return (
    <div className="plan-card">
      <h4>{title}</h4>
      {rows.length === 0 ? <p className="muted">—</p> : (
        <ul className="facts">{rows.map((r, i) => <li key={i}>{r}</li>)}</ul>
      )}
    </div>
  );
}
