import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { getApi } from './api';
import { scoringCountsAt } from './api/shots';
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
import { MatchLibrary } from './views/MatchLibrary';
import { StrategyPanel } from './views/Strategy';
import { TeamProfiles } from './views/TeamProfiles';
import { TeamStats } from './views/TeamStats';
import { Timeline } from './views/Timeline';
import { robotName } from './lib/tracks';
import { fmtPercent } from './lib/format';

type Tab = 'strategy' | 'profiles' | 'timeline' | 'analysis' | 'teams' | 'heatmap' | 'accuracy' | 'export';

const TABS: Array<[Tab, string]> = [
  ['strategy', 'Strategy'],
  ['profiles', 'Team profiles'],
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
  const [tab, setTab] = useState<Tab>('strategy');
  const [profileTeam, setProfileTeam] = useState<number | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [seekTo, setSeekTo] = useState<{ t: number; nonce: number } | null>(null);
  const [apiMode, setApiMode] = useState<'http' | 'fixture'>('fixture');
  const [streamVideoSrc, setStreamVideoSrc] = useState<string | null>(null);
  const [streamAudioSrc, setStreamAudioSrc] = useState<string | null>(null);

  useEffect(() => {
    void getApi().then((api) => setApiMode(api.mode));
  }, []);

  // On first load, land on a watchable match rather than an empty stage. Only once, though: after
  // that, a deliberate "back to library" (selectedJobId = null) must stick instead of snapping back
  // to a match.
  const autoSelectedOnce = useRef(false);
  useEffect(() => {
    if (autoSelectedOnce.current || selectedJobId || jobs.length === 0) return;
    autoSelectedOnce.current = true;
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
      // A published run's video copy is served like a local file (ingest/hosted.py).
      const localMedia = Boolean(job.localPath) || job.hostedVideo;
      void api.mediaToken(job.jobId).catch(() => null).then((token) => {
        setStreamVideoSrc(localMedia ? api.videoUrl(job, token) : api.streamVideoUrl(job, token));
        setStreamAudioSrc(localMedia ? null : api.streamAudioUrl(job, token));
      });
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
        onUpload={async (input) => {
          const created = await jobsState.uploadJob(input);
          setSelectedJobId(created.jobId);
          setSelectedEventId(null);
          return created;
        }}
        onDelete={jobsState.deleteJob}
        onRetry={jobsState.retryJob}
      />

      <main className="main">
        {!job && (
          <MatchLibrary
            jobs={jobs}
            loading={jobsState.loading}
            selectedJobId={selectedJobId}
            onSelectJob={(id) => {
              setSelectedJobId(id);
              setSelectedEventId(null);
            }}
          />
        )}

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
                  {job.localPath ? `Local match video · ${job.height}p · ${job.fps?.toFixed(2)} FPS`
                    : 'Live video stream'}
                  {shotState.goalCalibration?.source === 'apriltag_pose'
                    ? ` · automatic goals from ${shotState.goalCalibration.tagsUsed.length} AprilTags`
                    : shotState.goals.length > 0 ? ' · configured goal regions' : ''}
                </span>
              </div>
            </div>

            <VideoPlayer
              job={playable}
              season={season}
              src={streamVideoSrc}
              audioSrc={streamAudioSrc ?? undefined}
              // A hosted copy starts at the clip's t=0, not at its place in the YouTube video.
              mediaStartSeconds={job.hostedVideo ? 0 : job.startOffset}
              unavailableHint={
                apiMode === 'http' && !job.localPath && !job.hostedVideo
                  ? 'No video for this match on the server. YouTube blocks cloud servers, so the website ' +
                    'plays the copy uploaded when a match is published: republish it from the desktop app.'
                  : undefined
              }
              tracks={overlayTracks}
              events={match.events}
              shots={shotState.shots}
              shotGoals={shotState.goals}
              goalEntries={shotState.goalEntries}
              goalCameraGaps={shotState.goalCalibration?.cameraGaps}
              confidenceThreshold={confidenceThreshold}
              boxSampleRate={overlaySampleRate || job.fps || 30}
              selectedEventId={selectedEventId}
              onSelectEvent={setSelectedEventId}
              seekTo={seekTo}
              onTimeChange={setCurrentTime}
            />

            {analysisComplete && (
              <nav className="tabs">
                <button type="button" className="to-library"
                        onClick={() => setSelectedJobId(null)}
                        title="Back to the match library">← Library</button>
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
                    : `${match.events.length} events · ${shotState.statistics.attempted} shots (${shotState.statistics.unknown} unknown) · ${shotState.goals.length ? `${scoringCountsAt(shotState.shots, shotState.goalEntries).made} balls in` : 'goals not calibrated'} · ${overlayRobotLabels} robot labels · ${overlayTracks.length} track fragments · boxes @ ${(overlaySampleRate || job.fps || 0).toFixed(0)} Hz`}
                </span>
              </nav>
            )}

            {analysisComplete && match.error && <p className="error">{match.error}</p>}

            {analysisComplete && tab === 'strategy' && (
              <StrategyPanel
                jobId={job.jobId}
                refreshKey={match.corrections.length}
                onOpenTeam={(team) => {
                  setProfileTeam(team);
                  setTab('profiles');
                }}
              />
            )}
            {analysisComplete && tab === 'profiles' && (
              <TeamProfiles team={profileTeam} onSelectTeam={setProfileTeam} />
            )}
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
                onSeek={seek}
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
