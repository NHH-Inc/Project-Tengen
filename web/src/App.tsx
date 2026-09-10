import { useCallback, useEffect, useMemo, useState } from 'react';
import { getApi } from './api';
import { isPlayable } from './contracts';
import { seasonConfig } from './season';
import { EventInspector } from './components/EventInspector';
import { ExportPanel } from './components/ExportPanel';
import { Sidebar } from './components/Sidebar';
import { VideoPlayer } from './player/VideoPlayer';
import { useJobTracks } from './state/useJobTracks';
import { useJobs } from './state/useJobs';
import { useMatch } from './state/useMatch';
import { useRunResult } from './state/useRunResult';
import { useShots } from './state/useShots';
import { AccuracyPanel } from './views/Accuracy';
import { AnalysisPanel } from './views/Analysis';
import { HeatMap } from './views/HeatMap';
import { TeamStats } from './views/TeamStats';
import { Timeline } from './views/Timeline';
import { robotName } from './lib/tracks';
import { fmtPercent } from './lib/format';

type Tab = 'timeline' | 'analysis' | 'teams' | 'heatmap' | 'accuracy' | 'export';

const TABS: Array<[Tab, string]> = [
  ['timeline', 'Timeline'],
  ['analysis', 'Analysis'],
  ['teams', 'Team stats'],
  ['heatmap', 'Heat map'],
  ['accuracy', 'Accuracy'],
  ['export', 'Export'],
];

export default function App() {
  const jobsState = useJobs();
  const { jobs } = jobsState;

  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [selectedEventId, setSelectedEventId] = useState<string | null>(null);
  const [selectedTeam, setSelectedTeam] = useState<number | null>(null);
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.5);
  const [tab, setTab] = useState<Tab>('timeline');
  const [currentTime, setCurrentTime] = useState(0);
  const [seekTo, setSeekTo] = useState<{ t: number; nonce: number } | null>(null);
  const [apiMode, setApiMode] = useState<'http' | 'fixture'>('fixture');
  const [streamVideoSrc, setStreamVideoSrc] = useState<string | null>(null);
  const [streamAudioSrc, setStreamAudioSrc] = useState<string | null>(null);

  useEffect(() => {
    void getApi().then((api) => setApiMode(api.mode));
  }, []);

  // Land on something watchable rather than an empty stage.
  useEffect(() => {
    if (selectedJobId || jobs.length === 0) return;
    setSelectedJobId((jobs.find((j) => j.status === 'complete') ?? jobs[0]).jobId);
  }, [jobs, selectedJobId]);

  const job = useMemo(
    () => jobs.find((j) => j.jobId === selectedJobId) ?? null,
    [jobs, selectedJobId]
  );

  useEffect(() => {
    if (!job) {
      setStreamVideoSrc(null);
      setStreamAudioSrc(null);
      return;
    }
    void getApi().then((api) => {
      // A downloaded/local job must use the range-capable local endpoint.  Stream-only jobs
      // still use the yt-dlp proxy and its separate DASH audio track.  Keeping this choice at
      // the API boundary lets the player stay agnostic about where the media came from while
      // keeping shot/ball overlay timestamps aligned to the analyzed segment.
      const localMedia = Boolean(job.localPath);
      setStreamVideoSrc(localMedia ? api.videoUrl(job) : api.streamVideoUrl(job));
      setStreamAudioSrc(localMedia ? null : api.streamAudioUrl(job));
    });
  }, [job]);

  // Events are written after analysis. Job-level YOLO tracks are also published incrementally
  // while a job is moving, so the player can show the live overlay before completion.
  const analysisComplete = job?.status === 'complete';
  const match = useMatch(job?.matchId ?? null, analysisComplete);
  const runResult = useRunResult(job?.jobId ?? null, Boolean(analysisComplete));
  const shotState = useShots(job?.jobId ?? null, Boolean(analysisComplete));

  // A complete job whose media metadata never arrived cannot drive a player -- rather than
  // defaulting a duration and drawing a wrong scrub bar, the stage says so.
  const playable = isPlayable(job) ? job : null;
  const jobTracks = useJobTracks(
    job?.jobId ?? null,
    Boolean(playable),
    job?.status === 'analyzing'
  );
  const overlayTracks = jobTracks.tracks.length > 0 ? jobTracks.tracks : match.tracks;
  const overlayRobotLabels = useMemo(
    () => new Set(overlayTracks.map((track) => robotName(track, overlayTracks))).size,
    [overlayTracks]
  );
  const overlaySampleRate = jobTracks.boxSampleRate > 0
    ? jobTracks.boxSampleRate
    : match.boxSampleRate;

  // Doc 0: the season config is selected by the job's `season` field, so old footage stays
  // analyzable after the game changes. An unknown season is a bug, not something to guess.
  const season = job ? seasonConfig(job.season) : null;

  const seek = useCallback((t: number) => {
    setSeekTo({ t, nonce: Date.now() });
  }, []);

  const matchIds = useMemo(
    () => [...new Set(jobs.map((j) => j.matchId).filter((m): m is string => m != null))],
    [jobs]
  );

  return (
    <div className="app">
      <Sidebar
        jobs={jobs}
        loading={jobsState.loading}
        error={jobsState.error}
        violations={[...jobsState.violations, ...match.violations]}
        selectedJobId={job?.jobId ?? null}
        apiMode={apiMode}
        onSelectJob={(id) => {
          setSelectedJobId(id);
          setSelectedEventId(null);
        }}
        onCreate={async (input) => {
          const created = await jobsState.createJob(input);
          setSelectedJobId(created.jobId);
          setSelectedEventId(null);
          return created;
        }}
        onDelete={jobsState.deleteJob}
        onRetry={jobsState.retryJob}
      />

      <main className="main">
        {!job && <div className="stage-empty">Queue a video, or pick one from the queue.</div>}

        {job && !playable && job.status !== 'complete' && (
          <div className="stage-empty">
            <p>
              <strong>{job.matchId ?? job.videoId}</strong> is {job.status}.
            </p>
            <p className="muted">
              {job.status === 'failed'
                ? 'Retry from the sidebar — the video ID is stored on the job, so there is nothing to re-paste.'
                : 'The player opens from the yt-dlp stream as soon as the media metadata is ready.'}
            </p>
          </div>
        )}

        {job && season == null && (
          <div className="stage-empty">
            <p>
              No season config for <strong>{job.season}</strong>.
            </p>
            <p className="muted">
              Add <code>contracts/seasons/{job.season}.json</code>. Phase boundaries and field
              dimensions both come from it, so the timeline and heat map cannot be drawn without it.
            </p>
          </div>
        )}

        {job && job.status === 'complete' && !playable && (
          <div className="stage-empty">
            <p>
              <strong>{job.matchId ?? job.videoId}</strong> is complete, but the job record has
              no media metadata.
            </p>
            <p className="muted">
              duration, fps, width and height are still null. The ingest service has to write
              stream metadata back to the job row before playback can start.
            </p>
          </div>
        )}

        {job && playable && season && streamVideoSrc && (
          <>
            {job.status !== 'complete' && (
              <div className={`media-status ${job.status === 'failed' ? 'failed' : ''}`}>
                {job.status === 'failed' ? (
                  <>
                    Pipeline failed, but the selected video source is still available.{' '}
                    <span className="muted">{job.error}</span>
                  </>
                ) : (
                  <>
                    Ad-free yt-dlp stream ready.{' '}
                    Pipeline is {job.status}
                    {job.stage ? ` · ${job.stage}` : ''}
                    {job.progress != null ? ` · ${fmtPercent(job.progress, 1)}` : ''}.
                  </>
                )}
              </div>
            )}
            <div className="video-source" aria-label="Video source">
              <div className="video-source-main">
                <span className="video-source-label">Video</span>
                <span className="video-source-note stream">
                  yt-dlp stream only · video and audio stay on localhost; no match file is downloaded.
                </span>
              </div>
            </div>

            <VideoPlayer
              job={playable}
              season={season}
              src={streamVideoSrc}
              audioSrc={streamAudioSrc ?? undefined}
              mediaStartSeconds={job.startOffset}
              tracks={overlayTracks}
              events={match.events}
              shots={shotState.shots}
              shotGoals={shotState.goals}
              confidenceThreshold={confidenceThreshold}
              boxSampleRate={overlaySampleRate || job.fps || 30}
              selectedEventId={selectedEventId}
              onSelectEvent={setSelectedEventId}
              seekTo={seekTo}
              onTimeChange={setCurrentTime}
            />

            {analysisComplete && (
              <nav className="tabs">
                {TABS.map(([id, label]) => (
                  <button
                    key={id}
                    type="button"
                    className={tab === id ? 'on' : ''}
                    onClick={() => setTab(id)}
                  >
                    {label}
                  </button>
                ))}
                <span className="tabs-meta muted">
                  {match.loading
                    ? 'loading…'
                    : `${match.events.length} events · ${shotState.statistics.attempted} shots (${shotState.statistics.unknown} unknown) · ${overlayRobotLabels} robot labels · ${overlayTracks.length} track fragments · boxes @ ${(overlaySampleRate || job.fps || 0).toFixed(0)} Hz`}
                </span>
              </nav>
            )}

            {analysisComplete && match.error && <p className="error">{match.error}</p>}

            {analysisComplete && tab === 'timeline' && (
              <Timeline
                job={playable}
                season={season}
                events={match.events}
                confidenceThreshold={confidenceThreshold}
                currentTime={currentTime}
                selectedEventId={selectedEventId}
                onSelectEvent={setSelectedEventId}
                onSeek={seek}
              />
            )}
            {analysisComplete && tab === 'analysis' && (
              <AnalysisPanel
                result={runResult.result}
                loading={runResult.loading}
                error={runResult.error}
              />
            )}
            {analysisComplete && tab === 'teams' && (
              <TeamStats
                job={job}
                events={match.events}
                selectedTeam={selectedTeam}
                onSelectTeam={setSelectedTeam}
              />
            )}
            {analysisComplete && tab === 'heatmap' && (
              <HeatMap
                season={season}
                tracks={overlayTracks}
                selectedTeam={selectedTeam}
                currentTime={currentTime}
              />
            )}
            {analysisComplete && tab === 'accuracy' && (
              <AccuracyPanel accuracy={match.accuracy} season={season} fromRaw />
            )}
            {analysisComplete && tab === 'export' && <ExportPanel matchIds={matchIds} />}
          </>
        )}
      </main>

      {job && job.status === 'complete' && season && (
        <EventInspector
          job={job}
          events={match.events}
          deleted={match.deleted}
          currentTime={currentTime}
          confidenceThreshold={confidenceThreshold}
          onConfidenceThreshold={setConfidenceThreshold}
          selectedEventId={selectedEventId}
          onSelectEvent={setSelectedEventId}
          onSeek={seek}
          onPatch={match.patchEvent}
          onDelete={match.removeEvent}
          onCreate={match.addEvent}
          season={season}
          tracks={match.tracks}
          onPatchTrack={(trackId, team) => match.patchTrack(job.jobId, trackId, team)}
        />
      )}
    </div>
  );
}
