import { useEffect, useState } from 'react';
import { getApi } from '../api';
import type { StrategyTeam, TeamProfileResponse } from '../api/strategy';
import { FieldSketch, ROBOT_COLOURS } from '../components/FieldSketch';

// One team across every analysed match: its tendencies, a prediction for its next match, and
// counters, each with the evidence that produced it (ingest/strategy/profile.py). Every number
// says how many matches it rests on, because "always" from one match is a guess.

export interface TeamProfilesProps {
  team: number | null;
  onSelectTeam: (team: number) => void;
}

const fmt = (v: number | null | undefined, digits = 1, suffix = '') => (v == null ? '—' : `${v.toFixed(digits)}${suffix}`);
/** Negative means its last shot came after the endgame began: it keeps scoring. */
const endgameLine = (s: number | null) =>
  s == null ? 'Endgame timing unknown'
    : s > 0 ? `Stops scoring ${s.toFixed(0)} s before the endgame`
      : `Keeps scoring ${Math.abs(s).toFixed(0)} s into the endgame`;
const pct = (v: number | null | undefined) => (v == null ? '—' : `${Math.round(v * 100)}%`);

export function TeamProfiles({ team, onSelectTeam }: TeamProfilesProps) {
  const [teams, setTeams] = useState<StrategyTeam[]>([]);
  const [data, setData] = useState<TeamProfileResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    getApi().then((api) => api.listStrategyTeams()).then(setTeams)
      .catch((err) => setError(err instanceof Error ? err.message : String(err)));
  }, []);

  useEffect(() => {
    if (team == null) {
      setData(null);
      return;
    }
    let live = true;
    setLoading(true);
    getApi().then((api) => api.getTeamProfile(team))
      .then((result) => live && setData(result))
      .catch((err) => live && setError(err instanceof Error ? err.message : String(err)))
      .finally(() => live && setLoading(false));
    return () => {
      live = false;
    };
  }, [team]);

  const profile = data?.profile;
  const t = profile?.tendencies;
  return (
    <div className="panel">
      <div className="panel-head">
        <h2>Team profiles</h2>
        <span className="muted">every analysed match, across events</span>
      </div>
      {error && <p className="error">{error}</p>}
      <div className="team-picker">
        {teams.length === 0 && <span className="muted">No teams yet: assign team numbers to robots in a match.</span>}
        {teams.map((entry) => (
          <button key={entry.team} type="button" className={entry.team === team ? 'on' : ''}
                  onClick={() => onSelectTeam(entry.team)}
                  title={`${entry.matches.length} matches, ${entry.with_positions} with field positions`}>
            {entry.team} <span className="muted">×{entry.matches.length}</span>
          </button>
        ))}
      </div>
      {team != null && loading && <p className="empty">Loading team {team}…</p>}
      {profile && (
        <>
          <h3 className="profile-title">Team {profile.team}</h3>
          <p className="muted">
            {profile.n_matches} {profile.n_matches === 1 ? 'match' : 'matches'}, {profile.n_with_positions} with
            field positions · {fmt(profile.launches_per_match, 0)} shots per match
            {profile.timing?.median_volley_interval_s != null && ` · a volley every ${fmt(profile.timing.median_volley_interval_s)} s`}
          </p>

          <h3>Counters</h3>
          {profile.counters.length === 0 ? (
            <p className="muted">None yet. Counters need 3+ matches with field positions and a clear tendency.</p>
          ) : (
            <ul className="counters">
              {profile.counters.map((c) => (
                <li key={c.id}><b>{c.advice}</b><br /><span className="muted">{c.evidence}</span></li>
              ))}
            </ul>
          )}

          {t && profile.prediction && (
            <>
              <h3>Next match, predicted</h3>
              <ul className="facts">
                <li>Starts in the <b>{profile.prediction.start_lane ?? '?'}</b> lane
                  ({t.start.lane_matches}/{t.start.known} matches)</li>
                {profile.prediction.auto_routine_confidence != null && (
                  <li>Runs its usual auto routine (confidence {pct(profile.prediction.auto_routine_confidence)})</li>
                )}
                <li>About <b>{fmt(profile.prediction.launches, 0)}</b> shots</li>
                <li>{profile.prediction.plays_defense ? 'Likely to play defense' : 'Unlikely to play defense'}
                  {' '}({t.defense.matches_played}/{t.defense.of} matches)</li>
              </ul>

              <h3>Tendencies</h3>
              <ul className="facts">
                <li>Dwell at a spot {fmt(t.shooting.median_dwell_s, 1, ' s')} · shoots on the move {pct(t.shooting.on_the_move_share)}
                  {' '}· {fmt(t.shooting.median_distance_to_hub_ft, 1, ' ft')} from its hub</li>
                <li>Top speed {fmt(t.movement.top_speed_ftps, 1, ' ft/s')} · stopped {pct(t.movement.stationary_share)} of the time</li>
                <li>Lanes: {Object.entries(t.lanes).map(([k, v]) => `${k} ${pct(v)}`).join(' · ') || '—'}</li>
                <li>{endgameLine(t.endgame.median_stopped_scoring_before_endgame_s)}</li>
              </ul>

              <FieldSketch caption="Shooting spots across matches (circles) and auto routines (lines), in the team's alliance frame.">
                {(toX, toY, scale) => (
                  <>
                    {t.auto.routines.map((routine, i) => {
                      const points = routine.path.filter((p): p is [number, number] => p !== null);
                      return (
                        <polyline key={i} points={points.map(([x, y]) => `${toX(x)},${toY(y)}`).join(' ')}
                                  fill="none" stroke={ROBOT_COLOURS[i % ROBOT_COLOURS.length]} strokeWidth={2}>
                          <title>{`Auto routine ${i + 1}: ${routine.matches} matches`}</title>
                        </polyline>
                      );
                    })}
                    {t.shooting.spots.map((s, i) => (
                      <circle key={i} cx={toX(s.ax)} cy={toY(s.ay)} r={Math.max(4, Math.sqrt(s.launches) * scale * 0.3)}
                              fill="#f2cc60" fillOpacity={0.35} stroke="#f2cc60">
                        <title>{`${s.launches} shots, in ${s.matches} matches`}</title>
                      </circle>
                    ))}
                  </>
                )}
              </FieldSketch>
            </>
          )}
          {!t && <p className="muted">No match with field positions yet, so only timing is known.</p>}
        </>
      )}
    </div>
  );
}
