import { useState } from 'react';
import type { ContractViolation, Job } from '../contracts';
import { STATUS_LABEL, fmtPercent, isMatchKey, parseVideoId } from '../lib/format';

// Doc 3: "Sidebar for pasting YouTube links, viewing queue status, and browsing extracted
// results." A stored job can also be retried or deliberately re-run over its previous result.

export interface SidebarProps {
  jobs: Job[];
  loading: boolean;
  error: string | null;
  violations: ContractViolation[];
  selectedJobId: string | null;
  apiMode: 'http' | 'fixture';
  onSelectJob: (jobId: string) => void;
  onCreate: (input: { url: string; matchId?: string | null; liveCapture?: boolean }) => Promise<unknown>;
  onDelete: (jobId: string) => Promise<void>;
  onRetry: (job: Job) => Promise<unknown>;
}

export function Sidebar(props: SidebarProps) {
  const {
    jobs,
    loading,
    error,
    violations,
    selectedJobId,
    apiMode,
    onSelectJob,
    onCreate,
    onDelete,
    onRetry,
  } = props;

  return (
    <aside className="sidebar">
      <header className="brand">
        <h1>Project Tengen</h1>
        <span className={`mode ${apiMode}`}>
          {apiMode === 'fixture' ? 'fixture data · no backend' : 'ingest :8080'}
        </span>
      </header>

      <NewJobForm onCreate={onCreate} />

      <section className="queue">
        <h2>
          Queue <span className="muted">{jobs.length}</span>
        </h2>
        {loading && jobs.length === 0 && <p className="empty">Loading…</p>}
        {error && <p className="error">{error}</p>}
        {jobs.map((job) => (
          <JobCard
            key={job.jobId}
            job={job}
            selected={job.jobId === selectedJobId}
            onSelect={() => onSelectJob(job.jobId)}
            onDelete={() => onDelete(job.jobId)}
            onRetry={() => onRetry(job)}
          />
        ))}
      </section>

      {violations.length > 0 && <ViolationList violations={violations} />}
    </aside>
  );
}

function NewJobForm({
  onCreate,
}: {
  onCreate: (input: { url: string; matchId?: string | null; liveCapture?: boolean }) => Promise<unknown>;
}) {
  const [url, setUrl] = useState('');
  const [matchId, setMatchId] = useState('');
  const [liveCapture, setLiveCapture] = useState(false);
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);

  const videoId = parseVideoId(url);
  const matchOk = matchId.trim() === '' || isMatchKey(matchId);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setProblem(null);
    if (!videoId) {
      setProblem('That does not contain an 11-character YouTube video ID.');
      return;
    }
    if (!matchOk) {
      setProblem('Match key looks wrong. TBA keys are lowercase, like 2026casf_qm42.');
      return;
    }
    setBusy(true);
    try {
      // match_id is optional; component 2 resolves it from video metadata when omitted.
      await onCreate({ url: url.trim(), matchId: matchId.trim() || null, liveCapture });
      setUrl('');
      setMatchId('');
      setLiveCapture(false);
    } catch (err) {
      setProblem((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <form className="new-job" onSubmit={submit}>
      <label>
        YouTube link
        <input
          type="text"
          value={url}
          placeholder="https://youtube.com/watch?v=…"
          onChange={(e) => setUrl(e.target.value)}
        />
      </label>
      {url && (
        <p className={`hint ${videoId ? 'ok' : 'bad'}`}>
          {videoId ? `video ${videoId}` : 'no video ID found'}
        </p>
      )}
      <label>
        Match key <span className="muted">optional</span>
        <input
          type="text"
          value={matchId}
          placeholder="2026casf_qm42"
          onChange={(e) => setMatchId(e.target.value)}
        />
      </label>
      <label className="check">
        <input
          type="checkbox"
          checked={liveCapture}
          onChange={(e) => setLiveCapture(e.target.checked)}
        />
        Capture this live stream
      </label>
      {liveCapture && (
        <p className="hint">
          The link must be live now. This computer records until YouTube ends the stream, then
          analyzes the finished video. It does not create scouting data during the match.
        </p>
      )}
      {!matchOk && <p className="hint bad">lowercase TBA key, e.g. 2026casf_qm42</p>}
      <button type="submit" disabled={busy}>
        {busy ? 'Queueing…' : liveCapture ? 'Start live capture' : 'Queue video'}
      </button>
      {problem && <p className="error">{problem}</p>}
    </form>
  );
}

function JobCard({
  job,
  selected,
  onSelect,
  onDelete,
  onRetry,
}: {
  job: Job;
  selected: boolean;
  onSelect: () => void;
  onDelete: () => void;
  onRetry: () => void;
}) {
  const busy = job.status === 'downloading' || job.status === 'analyzing';
  const [rerunning, setRerunning] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);
  const canRunAgain = job.status === 'complete' || job.status === 'failed' || job.status === 'downloaded';

  const runAgain = async () => {
    setActionError(null);
    setRerunning(true);
    try {
      await onRetry();
    } catch (err) {
      setActionError((err as Error).message);
    } finally {
      setRerunning(false);
    }
  };

  return (
    <div className={`job ${job.status} ${selected ? 'sel' : ''}`}>
      <button type="button" className="job-main" onClick={onSelect}>
        <div className="job-top">
          <span className={`dot ${job.status}`} />
          <span className="job-match">{job.matchId ?? 'unresolved match'}</span>
          <span className="job-status">{STATUS_LABEL[job.status]}</span>
        </div>
        <div className="job-sub">
          <code>{job.videoId}</code>
          {job.startOffset > 0 && (
            <span className="muted" title="Segment clipped out of a longer stream">
              +{Math.round(job.startOffset)}s
            </span>
          )}
        </div>

        {busy && (
          // Contract A has no progress field, so an indeterminate bar unless component 2
          // happens to send one. See OPEN_QUESTIONS.md #4.
          <div className={`bar ${job.progress == null ? 'indet' : ''}`}>
            <span style={job.progress != null ? { width: `${job.progress * 100}%` } : undefined} />
          </div>
        )}
        {busy && job.stage && (
          <div className="job-stage muted">
            {job.captureMode === 'live' && job.status === 'downloading'
              ? 'capturing live stream'
              : job.stage}
            {job.progress != null ? ` · ${fmtPercent(job.progress, 1)}` : ''}
          </div>
        )}
        {job.status === 'failed' && job.error && <div className="job-error">{job.error}</div>}
      </button>

      <div className="job-actions">
        {canRunAgain && (
          <button
            type="button"
            onClick={() => void runAgain()}
            disabled={rerunning}
            title="Run the pipeline again and replace this job's previous result"
          >
            {rerunning ? 'Starting…' : job.status === 'complete' ? 'Run pipeline again' : 'Retry pipeline'}
          </button>
        )}
        <button type="button" className="danger" onClick={onDelete}>
          Remove
        </button>
      </div>
      {actionError && <div className="job-error">{actionError}</div>}
    </div>
  );
}

function ViolationList({ violations }: { violations: ContractViolation[] }) {
  // Doc 0: "Anything unrecognized is a bug, not a fallback." Rather than coercing an unknown
  // enum value into something renderable, the parsers drop the row and report it here.
  const grouped = new Map<string, number>();
  for (const v of violations) {
    const key = `${v.what}: ${v.detail}`;
    grouped.set(key, (grouped.get(key) ?? 0) + 1);
  }
  return (
    <section className="violations">
      <h2>Contract violations</h2>
      <p className="note">
        Data that did not match <code>/contracts/</code>. These rows were dropped, not coerced.
      </p>
      <ul>
        {[...grouped].map(([msg, n]) => (
          <li key={msg}>
            {msg}
            {n > 1 && <span className="muted"> ×{n}</span>}
          </li>
        ))}
      </ul>
    </section>
  );
}
