import { useMemo, useState } from 'react';
import { EVENT_TYPES, type EventType, type Job, type ScoutEvent, type Track } from '../contracts';
import type { SeasonConfig } from '../season';
import type { ViewEvent } from '../lib/corrections';
import { EVENT_LABEL, PHASE_LABEL, SOURCE_LABEL, fmtTime } from '../lib/format';
import { robotName } from '../lib/tracks';

// Doc 3: "Build this early. Users will find wrong calls, and a tool that cannot be corrected
// will not be trusted." And: "Minimum useful version: scrub to an event, see what the
// pipeline claimed, fix the team attribution or delete the event, add a missed one."
//
// Every action here writes a correction row. Nothing overwrites model output.

export interface EventInspectorProps {
  job: Job;
  events: ViewEvent[];
  deleted: ScoutEvent[];
  currentTime: number;
  confidenceThreshold: number;
  onConfidenceThreshold: (v: number) => void;
  selectedEventId: string | null;
  onSelectEvent: (id: string | null) => void;
  onSeek: (t: number) => void;
  onPatch: (eventId: string, fields: Partial<ScoutEvent>) => Promise<void>;
  onDelete: (eventId: string) => Promise<void>;
  onCreate: (event: Omit<ScoutEvent, 'eventId' | 'corrected' | 'correctionId'>) => Promise<void>;
  season: SeasonConfig;
  tracks: Track[];
  onPatchTrack: (trackId: number, team: number | null) => Promise<void>;
}

export function EventInspector(props: EventInspectorProps) {
  const {
    job,
    events,
    deleted,
    currentTime,
    confidenceThreshold,
    onConfidenceThreshold,
    selectedEventId,
    onSelectEvent,
    onSeek,
    onPatch,
    onDelete,
    onCreate,
    season,
    tracks,
    onPatchTrack,
  } = props;

  const [teamFilter, setTeamFilter] = useState<number | 'all' | 'none'>('all');
  const [typeFilter, setTypeFilter] = useState<EventType | 'all'>('all');
  const [onlyLow, setOnlyLow] = useState(false);
  const [onlyCorrected, setOnlyCorrected] = useState(false);
  const [busy, setBusy] = useState(false);

  // Suggestions, not a whitelist. TBA has no data for plenty of matches -- the first real match
  // analysed had alliances: null -- and when this list came only from there the dropdown was
  // empty, so nobody could attribute anything at all. Bumper OCR reads the roster off the
  // scoreboard in that case, so the teams it found are on the tracks even when TBA knew nothing.
  const allTeams = useMemo(() => {
    const seen = new Set<number>([
      ...(job.alliances?.red ?? []),
      ...(job.alliances?.blue ?? []),
    ]);
    tracks.forEach((t) => t.team != null && seen.add(t.team));
    events.forEach((e) => e.team != null && seen.add(e.team));
    return [...seen].sort((a, b) => a - b);
  }, [job.alliances, tracks, events]);

  const shown = useMemo(
    () =>
      events.filter((e) => {
        if (onlyLow && e.confidence >= confidenceThreshold) return false;
        if (onlyCorrected && !e.corrected) return false;
        if (typeFilter !== 'all' && e.eventType !== typeFilter) return false;
        if (teamFilter === 'none' && e.team != null) return false;
        if (typeof teamFilter === 'number' && e.team !== teamFilter) return false;
        return true;
      }),
    [events, onlyLow, onlyCorrected, typeFilter, teamFilter, confidenceThreshold]
  );

  const lowCount = events.filter((e) => e.confidence < confidenceThreshold).length;
  const selected = events.find((e) => e.eventId === selectedEventId) ?? null;

  const run = async (fn: () => Promise<void>) => {
    setBusy(true);
    try {
      await fn();
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="panel inspector">
      <div className="panel-head">
        <h2>Events &amp; corrections</h2>
        <span className="muted">
          {shown.length} of {events.length} · {lowCount} below threshold
        </span>
      </div>

      <TrackPanel
        tracks={tracks}
        teams={allTeams}
        busy={busy}
        onPatchTrack={(id, team) => run(() => onPatchTrack(id, team))}
      />

      {/* Doc 3: "Surface it, and let users filter the view by threshold." */}
      <div className="filters">
        <label className="filter conf">
          Confidence ≥ <strong>{confidenceThreshold.toFixed(2)}</strong>
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={confidenceThreshold}
            onChange={(e) => onConfidenceThreshold(Number(e.target.value))}
          />
        </label>
        <label className="filter">
          Team
          <select
            value={String(teamFilter)}
            onChange={(e) => {
              const v = e.target.value;
              setTeamFilter(v === 'all' || v === 'none' ? v : Number(v));
            }}
          >
            <option value="all">all</option>
            {allTeams.map((t) => (
              <option key={t} value={t}>{t}</option>
            ))}
            <option value="none">unattributed</option>
          </select>
        </label>
        <label className="filter">
          Type
          <select
            value={typeFilter}
            onChange={(e) => setTypeFilter(e.target.value as EventType | 'all')}
          >
            <option value="all">all</option>
            {EVENT_TYPES.map((t) => (
              <option key={t} value={t}>{EVENT_LABEL[t]}</option>
            ))}
          </select>
        </label>
        <label className="filter check">
          <input type="checkbox" checked={onlyLow} onChange={(e) => setOnlyLow(e.target.checked)} />
          Low confidence only
        </label>
        <label className="filter check">
          <input
            type="checkbox"
            checked={onlyCorrected}
            onChange={(e) => setOnlyCorrected(e.target.checked)}
          />
          Corrected only
        </label>
      </div>

      <AddEventRow
        job={job}
        season={season}
        currentTime={currentTime}
        teams={allTeams}
        busy={busy}
        onCreate={(e) => run(() => onCreate(e))}
      />

      <div className="event-list">
        {shown.length === 0 && <p className="empty">Nothing matches these filters.</p>}
        {shown.map((e) => (
          <EventRow
            key={e.eventId}
            e={e}
            tracks={tracks}
            low={e.confidence < confidenceThreshold}
            selected={e.eventId === selectedEventId}
            onClick={() => {
              onSelectEvent(e.eventId === selectedEventId ? null : e.eventId);
              onSeek(e.tSeconds);
            }}
          />
        ))}
      </div>

      {selected && (
        <EditPanel
          e={selected}
          tracks={tracks}
          season={season}
          teams={allTeams}
          busy={busy}
          onPatch={(fields) => run(() => onPatch(selected.eventId, fields))}
          onDelete={() => run(() => onDelete(selected.eventId))}
          onSeek={() => onSeek(selected.tSeconds)}
        />
      )}

      {deleted.length > 0 && (
        <details className="deleted">
          <summary>{deleted.length} deleted by a reviewer</summary>
          <p className="note">
            Still in the raw event table — a delete is a correction row, not a removal, so
            these remain available for model evaluation and training export.
          </p>
          <ul>
            {deleted.map((e) => (
              <li key={e.eventId}>
                <button type="button" className="linky" onClick={() => onSeek(e.tSeconds)}>
                  {fmtTime(e.tSeconds)}
                </button>{' '}
                {EVENT_LABEL[e.eventType]} · {e.team ?? 'unattributed'} · conf{' '}
                {e.confidence.toFixed(2)}
              </li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}

function EventRow({
  e,
  tracks,
  low,
  selected,
  onClick,
}: {
  e: ViewEvent;
  tracks: Track[];
  low: boolean;
  selected: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      className={`event-row ${low ? 'low' : ''} ${selected ? 'sel' : ''} ${e.corrected ? 'corrected' : ''}`}
      onClick={onClick}
    >
      <span className="t">{fmtTime(e.tSeconds)}</span>
      <span className={`team ${e.team == null ? 'none' : ''}`}>
        {e.team ?? (e.trackId != null
          ? (() => {
              const track = tracks.find((candidate) => candidate.trackId === e.trackId);
              return track ? robotName(track, tracks) : '--';
            })()
          : '--')}
      </span>
      <span className="type">{EVENT_LABEL[e.eventType]}</span>
      <span className="phase">{PHASE_LABEL[e.phase]}</span>
      <span className="conf" title={`confidence ${e.confidence}`}>
        <span className="conf-bar" style={{ width: `${e.confidence * 100}%` }} />
        <span className="conf-num">{e.confidence.toFixed(2)}</span>
      </span>
      {e.corrected && <span className="tag edited">corrected</span>}
      {e.source !== 'model' && !e.corrected && (
        <span className="tag src">{SOURCE_LABEL[e.source]}</span>
      )}
    </button>
  );
}

function EditPanel({
  e,
  tracks,
  season,
  teams,
  busy,
  onPatch,
  onDelete,
  onSeek,
}: {
  e: ViewEvent;
  tracks: Track[];
  season: SeasonConfig;
  teams: number[];
  busy: boolean;
  onPatch: (fields: Partial<ScoutEvent>) => void;
  onDelete: () => void;
  onSeek: () => void;
}) {
  return (
    <div className="edit-panel">
      <div className="edit-head">
        <strong>{EVENT_LABEL[e.eventType]}</strong>
        <button type="button" className="linky" onClick={onSeek}>
          {fmtTime(e.tSeconds)}
        </button>
        <code className="muted">{e.eventId}</code>
      </div>

      {e.original && (
        <p className="note was">
          Model originally said:{' '}
          <strong>{e.original.team ?? 'unattributed'}</strong> ·{' '}
          {EVENT_LABEL[e.original.eventType]} · conf {e.original.confidence.toFixed(2)}
        </p>
      )}

      <div className="edit-grid">
        <label>
          Team
          <select
            disabled={busy}
            value={e.team ?? ''}
            onChange={(ev) => onPatch({ team: ev.target.value ? Number(ev.target.value) : null })}
          >
            <option value="">unattributed</option>
            {teams.map((t) => (
              <option key={t} value={t}>{t}</option>
            ))}
          </select>
        </label>

        <label>
          Event type
          <select
            disabled={busy}
            value={e.eventType}
            onChange={(ev) => onPatch({ eventType: ev.target.value as EventType })}
          >
            {EVENT_TYPES.map((t) => (
              <option key={t} value={t}>{EVENT_LABEL[t]}</option>
            ))}
          </select>
        </label>

        {(e.eventType === 'shot_attempt' || e.eventType === 'shot_made') && (
          <label>
            Goal
            <select
              disabled={busy}
              value={e.goal ?? ''}
              onChange={(ev) => onPatch({ goal: ev.target.value || null })}
            >
              <option value="">unknown</option>
              {season.goals.map((g) => (
                <option key={g} value={g}>{g}</option>
              ))}
            </select>
          </label>
        )}

        <div className="edit-meta">
          <span>
            {(() => {
              const track = e.trackId == null
                ? null
                : tracks.find((candidate) => candidate.trackId === e.trackId);
              return track ? robotName(track, tracks) : 'robot --';
            })()}
          </span>
          <span>{PHASE_LABEL[e.phase]}</span>
          <span>{SOURCE_LABEL[e.source]}</span>
          <span>
            field{' '}
            {e.fieldX != null && e.fieldY != null
              ? `${e.fieldX.toFixed(1)}, ${e.fieldY.toFixed(1)} ft`
              : 'no homography'}
          </span>
        </div>
      </div>

      <div className="edit-actions">
        <button type="button" className="danger" disabled={busy} onClick={onDelete}>
          Delete event
        </button>
        <span className="note">
          Writes a correction row. The raw event stays in the table for evaluation.
        </span>
      </div>

      <p className="note warn-note">
        This edits one row. A misread bumper is a track-level problem — use “Re-attribute
        track” above to fix the track and every event on it in one action.
      </p>
    </div>
  );
}

function AddEventRow({
  job,
  season,
  currentTime,
  teams,
  busy,
  onCreate,
}: {
  job: Job;
  season: SeasonConfig;
  currentTime: number;
  teams: number[];
  busy: boolean;
  onCreate: (e: Omit<ScoutEvent, 'eventId' | 'corrected' | 'correctionId'>) => void;
}) {
  const [team, setTeam] = useState<number | ''>(teams[0] ?? '');
  const [type, setType] = useState<EventType>('shot_made');
  const [goal, setGoal] = useState<string>('');
  const isShot = type === 'shot_attempt' || type === 'shot_made';

  const phaseAt = (t: number) =>
    t < 15 ? ('auto' as const) : t < 130 ? ('teleop' as const) : ('endgame' as const);

  return (
    <div className="add-row">
      <span className="add-label">Add missed event at {fmtTime(currentTime)}</span>
      <select value={team} onChange={(e) => setTeam(e.target.value ? Number(e.target.value) : '')}>
        {teams.map((t) => (
          <option key={t} value={t}>{t}</option>
        ))}
      </select>
      <select value={type} onChange={(e) => setType(e.target.value as EventType)}>
        {EVENT_TYPES.map((t) => (
          <option key={t} value={t}>{EVENT_LABEL[t]}</option>
        ))}
      </select>
      {/* Legal goals come from the season config, never a hardcoded list. */}
      {isShot && (
        <select value={goal} onChange={(e) => setGoal(e.target.value)} title="Goal">
          <option value="">goal?</option>
          {season.goals.map((g) => (
            <option key={g} value={g}>{g}</option>
          ))}
        </select>
      )}
      <button
        type="button"
        disabled={busy || team === ''}
        onClick={() =>
          onCreate({
            jobId: job.jobId,
            matchId: job.matchId ?? '',
            team: team === '' ? null : team,
            trackId: null,
            tSeconds: Math.round(currentTime * 1000) / 1000,
            phase: phaseAt(currentTime),
            eventType: type,
            confidence: 1,
            fieldX: null,
            fieldY: null,
            goal: isShot && goal ? goal : null,
            source: 'manual',
          })
        }
      >
        Add
      </button>
    </div>
  );
}


/**
 * Doc 3: "The most common correction is a misread bumper, and it is a track-level fix, not an
 * event-level one. One bad OCR read mislabels forty-odd events and every box on that robot...
 * Build that path first; per-event editing is the exception."
 *
 * team_confidence is what makes this actionable: it says how sure the OCR was about the whole
 * track, so the tracks most likely to be wrong sort to the top.
 */
function TrackPanel({
  tracks,
  teams,
  busy,
  onPatchTrack,
}: {
  tracks: Track[];
  teams: number[];
  busy: boolean;
  onPatchTrack: (trackId: number, team: number | null) => void;
}) {
  const ordered = [...tracks].sort(
    (a, b) => (a.teamConfidence ?? -1) - (b.teamConfidence ?? -1)
  );
  if (ordered.length === 0) return null;

  // Blur fires on every focus change, so send only an actual change -- otherwise simply tabbing
  // through the list would file a correction per track and bury the real ones.
  const commitTeam = (track: Track, raw: string) => {
    const trimmed = raw.trim();
    if (trimmed === '') {
      if (track.team != null) onPatchTrack(track.trackId, null);
      return;
    }
    const team = Number(trimmed);
    if (!Number.isInteger(team) || team <= 0 || team === track.team) return;
    onPatchTrack(track.trackId, team);
  };

  return (
    <details className="track-panel" open>
      <summary>
        Re-attribute robot <span className="muted">{ordered.length} tracked robots</span>
      </summary>
      <p className="note">
        Fixes the track and every event on it in one action. Least-confident identifications
        first — those are the ones worth checking.
      </p>
      <div className="track-rows">
        {ordered.map((t) => {
          const shaky = t.teamConfidence != null && t.teamConfidence < 0.75;
          return (
            <div key={t.trackId} className={`track-row ${shaky ? 'shaky' : ''}`}>
              <span className={`chip ${t.alliance ?? ''}`}>
                {robotName(t, tracks)}{t.team != null ? ` · ${t.team}` : ''}
              </span>
              <span className="muted tconf">
                {t.teamConfidence != null ? `id ${t.teamConfidence.toFixed(2)}` : 'unidentified'}
              </span>
              {t.gaps.length > 0 && (
                <span className="muted" title={t.gaps.map((g) => g.reason).join(', ')}>
                  {t.gaps.length} gap{t.gaps.length === 1 ? '' : 's'}
                </span>
              )}
              {/* A list of suggestions with free entry, not a closed dropdown. The roster can
                  be wrong or incomplete -- OCR misreads, TBA has no data, a team plays under a
                  number nothing on screen showed -- and a human who can see the robot must never
                  be unable to say which one it is. The backend accepts any team number. */}
              <input
                type="number"
                min={1}
                step={1}
                list={`teams-${t.trackId}`}
                className="team-entry"
                placeholder="team"
                disabled={busy}
                defaultValue={t.team ?? ''}
                onBlur={(e) => commitTeam(t, e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') (e.target as HTMLInputElement).blur();
                }}
              />
              <datalist id={`teams-${t.trackId}`}>
                {teams.map((n) => (
                  <option key={n} value={n} />
                ))}
              </datalist>
            </div>
          );
        })}
      </div>
    </details>
  );
}
