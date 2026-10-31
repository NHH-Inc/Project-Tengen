import { useMemo, useState } from 'react';
import type { Job } from '../contracts';
import { STATUS_LABEL, fmtAgo, fmtSigned, parseMatchKey } from '../lib/format';

// The browsable library of scanned matches (docs/PLAN-2026.md section 3): every analysed match,
// grouped by event and sortable by match order or by when it was scanned. Clicking a match opens
// it into the normal analysis views. The data is the job list the queue already loads -- this is a
// second, match-first lens on it, not a new fetch. Multi-robot path overlays live on the Heat map
// view (its "Whole match" toggle); this is the index that gets you to a match in the first place.

export interface MatchLibraryProps {
  jobs: Job[];
  loading: boolean;
  selectedJobId: string | null;
  onSelectJob: (jobId: string) => void;
}

type Sort = 'match' | 'recent';

const UNASSIGNED = 'unassigned';

export function MatchLibrary({ jobs, loading, selectedJobId, onSelectJob }: MatchLibraryProps) {
  const [sort, setSort] = useState<Sort>('match');
  const [eventFilter, setEventFilter] = useState<string>('all');
  const [completeOnly, setCompleteOnly] = useState(false);

  // Every event present, most-recent activity first, so the newest event leads the filter.
  const events = useMemo(() => {
    const latest = new Map<string, number>();
    for (const job of jobs) {
      const key = parseMatchKey(job.matchId)?.event ?? UNASSIGNED;
      latest.set(key, Math.max(latest.get(key) ?? 0, Date.parse(job.createdAt) || 0));
    }
    return [...latest.entries()].sort((a, b) => b[1] - a[1]).map(([event]) => event);
  }, [jobs]);

  const groups = useMemo(() => {
    const shown = jobs.filter((job) => {
      if (completeOnly && job.status !== 'complete') return false;
      if (eventFilter === 'all') return true;
      return (parseMatchKey(job.matchId)?.event ?? UNASSIGNED) === eventFilter;
    });
    const byEvent = new Map<string, Job[]>();
    for (const job of shown) {
      const key = parseMatchKey(job.matchId)?.event ?? UNASSIGNED;
      (byEvent.get(key) ?? byEvent.set(key, []).get(key)!).push(job);
    }
    for (const list of byEvent.values()) list.sort(compare(sort));
    // Event sections follow the same order as the filter: most recent event first.
    return events.filter((e) => byEvent.has(e)).map((event) => ({ event, matches: byEvent.get(event)! }));
  }, [jobs, events, sort, eventFilter, completeOnly]);

  const total = groups.reduce((n, g) => n + g.matches.length, 0);

  return (
    <div className="panel library">
      <div className="panel-head">
        <h2>Match library</h2>
        <span className="muted">{total} {total === 1 ? 'match' : 'matches'} · click one to open it</span>
      </div>

      <div className="library-controls">
        <label>
          Event
          <select value={eventFilter} onChange={(e) => setEventFilter(e.target.value)}>
            <option value="all">All events</option>
            {events.map((event) => (
              <option key={event} value={event}>{event === UNASSIGNED ? 'Unassigned' : event}</option>
            ))}
          </select>
        </label>
        <div className="seg">
          <button type="button" className={sort === 'match' ? 'on' : ''} onClick={() => setSort('match')}
                  title="Qualification first, then playoffs, by match number">Match order</button>
          <button type="button" className={sort === 'recent' ? 'on' : ''} onClick={() => setSort('recent')}
                  title="Most recently scanned first">Most recent</button>
        </div>
        <label className="check">
          <input type="checkbox" checked={completeOnly} onChange={(e) => setCompleteOnly(e.target.checked)} />
          Complete only
        </label>
      </div>

      {loading && jobs.length === 0 && <p className="empty">Loading matches…</p>}
      {!loading && total === 0 && (
        <p className="empty">No scanned matches yet. Queue a video from the sidebar to add one.</p>
      )}

      {groups.map(({ event, matches }) => (
        <section key={event} className="library-event">
          <h3>{event === UNASSIGNED ? 'Unassigned' : event} <span className="muted">×{matches.length}</span></h3>
          <div className="library-rows">
            {matches.map((job) => (
              <MatchRow key={job.jobId} job={job} selected={job.jobId === selectedJobId}
                        onSelect={() => onSelectJob(job.jobId)} />
            ))}
          </div>
        </section>
      ))}
    </div>
  );
}

function compare(sort: Sort): (a: Job, b: Job) => number {
  if (sort === 'recent') {
    return (a, b) => (Date.parse(b.createdAt) || 0) - (Date.parse(a.createdAt) || 0);
  }
  return (a, b) => {
    const pa = parseMatchKey(a.matchId);
    const pb = parseMatchKey(b.matchId);
    // Unparseable keys sort to the end, then by recency so they are at least grouped sensibly.
    if (!pa && !pb) return (Date.parse(b.createdAt) || 0) - (Date.parse(a.createdAt) || 0);
    if (!pa) return 1;
    if (!pb) return -1;
    return pa.levelIndex - pb.levelIndex || pa.number - pb.number;
  };
}

function MatchRow({ job, selected, onSelect }: { job: Job; selected: boolean; onSelect: () => void }) {
  const parts = parseMatchKey(job.matchId);
  const label = parts?.label ?? job.matchId ?? 'unresolved match';
  const score = job.tbaScore;
  const margin = score ? score.red - score.blue : null;
  return (
    <button type="button" className={`library-row ${job.status} ${selected ? 'sel' : ''}`} onClick={onSelect}
            disabled={job.status !== 'complete'}
            title={job.status === 'complete' ? 'Open this match' : `${STATUS_LABEL[job.status]} — not ready to open`}>
      <span className={`dot ${job.status}`} />
      <span className="library-match">{label}</span>
      {job.alliances ? (
        <span className="library-teams">
          <span className="red">{job.alliances.red.join(' ') || '—'}</span>
          <span className="vs">vs</span>
          <span className="blue">{job.alliances.blue.join(' ') || '—'}</span>
        </span>
      ) : (
        <span className="library-teams muted">teams unknown</span>
      )}
      {score ? (
        <span className="library-score">
          <span className="red">{score.red}</span>–<span className="blue">{score.blue}</span>
          {margin != null && margin !== 0 && (
            <span className={`muted margin ${margin > 0 ? 'red' : 'blue'}`}> {fmtSigned(margin)}</span>
          )}
        </span>
      ) : (
        <span className="library-score muted">—</span>
      )}
      <span className="library-when muted">
        {job.captureMode === 'local' ? 'uploaded' : job.captureMode === 'live' ? 'live capture' : 'recording'}
        {' · '}{fmtAgo(job.createdAt)}
      </span>
      {job.status !== 'complete' && <span className="library-status muted">{STATUS_LABEL[job.status]}</span>}
    </button>
  );
}
