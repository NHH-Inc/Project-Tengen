import { useEffect, useMemo, useState } from 'react';
import { getApi } from '../api';
import type { StrategyTeam } from '../api/strategy';
import { rankPartners } from '../lib/strategyEngine';
import { useTeamProfiles } from '../state/useTeamProfiles';

// Surfaces the alliance-pick engine (ingest/strategy/picklist.py): choose our team(s), and every
// other team is ranked as a partner by offensive power minus how much it would conflict with us.
// Built for alliance selection -- our own order, not the official rankings. Computed client-side
// over the fetched profiles (lib/strategyEngine.ts), so it recomputes instantly as picks come off
// the board (remove a team once it is taken).

export function PickList() {
  const [teams, setTeams] = useState<StrategyTeam[]>([]);
  const [us, setUs] = useState<number[]>([]);
  const [taken, setTaken] = useState<number[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getApi().then((api) => api.listStrategyTeams()).then(setTeams)
      .catch((err) => setError(err instanceof Error ? err.message : String(err)));
  }, []);

  const allTeams = useMemo(() => teams.map((t) => t.team), [teams]);
  const { profiles, loading } = useTeamProfiles(allTeams);

  const available = useMemo(
    () => allTeams.filter((t) => !us.includes(t) && !taken.includes(t)),
    [allTeams, us, taken],
  );
  const ranked = useMemo(
    () => (us.length ? rankPartners(us, available, profiles) : []),
    [us, available, profiles],
  );

  const toggleUs = (team: number) =>
    setUs((cur) => (cur.includes(team) ? cur.filter((t) => t !== team) : [...cur, team]));
  const toggleTaken = (team: number) =>
    setTaken((cur) => (cur.includes(team) ? cur.filter((t) => t !== team) : [...cur, team]));

  return (
    <div className="panel">
      <div className="panel-head">
        <h2>Pick list</h2>
        <span className="muted">our order, by power minus conflict — not the official ranking</span>
      </div>
      {error && <p className="error">{error}</p>}
      {teams.length === 0 && <p className="muted">No teams yet: assign team numbers to robots in a match.</p>}

      <div className="plan-row">
        <span className="plan-tag us">Our alliance</span>
        <div className="team-picker">
          {allTeams.map((team) => (
            <button key={team} type="button" className={us.includes(team) ? 'on us' : ''}
                    onClick={() => toggleUs(team)}>{team}</button>
          ))}
        </div>
      </div>

      {us.length === 0 ? (
        <p className="empty">Pick our team to rank everyone else as a partner.</p>
      ) : (
        <>
          <h3>Best available partners</h3>
          {loading && profiles.size <= us.length ? (
            <p className="empty">Loading profiles…</p>
          ) : ranked.length === 0 ? (
            <p className="muted">Nobody left to rank. Clear a taken team, or analyse more matches.</p>
          ) : (
            <table className="picklist">
              <thead>
                <tr><th>#</th><th>Team</th><th>Score</th><th>Power</th><th>Conflict</th><th></th></tr>
              </thead>
              <tbody>
                {ranked.map((r, i) => (
                  <tr key={r.team} className={i === 0 ? 'best' : ''}>
                    <td className="muted">{i + 1}</td>
                    <td className="team">{r.team}</td>
                    <td className="score">{r.score.toFixed(1)}</td>
                    <td>{r.power.toFixed(1)}</td>
                    <td className={r.conflict > 0 ? 'warn' : 'muted'}>{r.conflict.toFixed(2)}</td>
                    <td><button type="button" className="chip" onClick={() => toggleTaken(r.team)}
                                title="Mark this team as picked by another alliance">taken</button></td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          <p className="note">
            Power is shots per match; conflict is shared auto cells and scoring zones with us.
            Mark teams <b>taken</b> as they come off the board and the order updates.
          </p>
          {taken.length > 0 && (
            <p className="muted">Taken: {taken.join(', ')} ·{' '}
              <button type="button" className="linkish" onClick={() => setTaken([])}>reset</button></p>
          )}
        </>
      )}
    </div>
  );
}
