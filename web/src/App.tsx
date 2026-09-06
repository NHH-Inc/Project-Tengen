import { useCallback, useEffect, useMemo, useState, type ChangeEvent } from 'react';
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
import { AccuracyPanel } from './views/Accuracy';
import { AnalysisPanel } from './views/Analysis';
import { HeatMap } from './views/HeatMap';
import { TeamStats } from './views/TeamStats';
import { Timeline } from './views/Timeline';

type Tab = 'timeline' | 'analysis' | 'teams' | 'heatmap' | 'accuracy' | 'export';
type VideoAlignment = 'segment' | 'original';
type VideoSource = 'job' | 'stream' | 'local';

interface LocalVideo {
  name: string;
  url: string;
}

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
  const [jobVideoSrc, setJobVideoSrc] = useState<string | null>(null);
  const [streamVideoSrc, setStreamVideoSrc] = useState<string | null>(null);
  const [streamAudioSrc, setStreamAudioSrc] = useState<string | null>(null);
  const [localVideo, setLocalVideo] = useState<LocalVideo | null>(null);
  const [videoAlignment, setVideoAlignment] = useState<VideoAlignment>('segment');
  const [videoSource, setVideoSource] = useState<VideoSource>('job');

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
      setJobVideoSrc(null);
      setStreamVideoSrc(null);
      setStreamAudioSrc(null);
      return;
    }
    void getApi().then((api) => {
      setJobVideoSrc(api.videoUrl(job));
      setStreamVideoSrc(api.streamVideoUrl(job));
      setStreamAudioSrc(api.streamAudioUrl(job));
    });
  }, [job]);

  // A browser object URL keeps a selected match recording entirely on this computer.
  // Revoke it when it is replaced so repeated review sessions do not leak memory.
  useEffect(() => {
    return () => {
      if (localVideo) URL.revokeObjectURL(localVideo.url);
    };
  }, [localVideo]);

  // A local recording selected for one job must never silently carry over to another.
  useEffect(() => {
    setLocalVideo(null);
    setVideoAlignment('segment');
    setVideoSource((source) => source === 'local' ? 'job' : source);
  }, [job?.jobId]);

  // A queued HTTP job has metadata before its merged MP4 exists. Start on the yt-dlp DASH
  // proxy so the supplied link is watchable immediately; keep that choice after download.
  useEffect(() => {
    if (apiMode === 'http' && job && !job.localPath) setVideoSource('stream');
  }, [apiMode, job?.jobId, job?.localPath]);

  const chooseLocalVideo = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setLocalVideo({ name: file.name, url: URL.createObjectURL(file) });
    setVideoAlignment('segment');
    setVideoSource('local');
    // Allow selecting the same file again after switching back to the job video.
    event.target.value = '';
  };

  // Events are written after analysis. Job-level YOLO tracks are also published incrementally
  // while a job is moving, so the player can show the live overlay before completion.
  const analysisComplete = job?.status === 'complete';
  const match = useMatch(job?.matchId ?? null, analysisComplete);
  const runResult = useRunResult(job?.jobId ?? null, Boolean(analysisComplete));

  // A complete job whose media metadata never arrived cannot drive a player -- rather than
  // defaulting a duration and drawing a wrong scrub bar, the stage says so.
  const playable = isPlayable(job) ? job : null;
  const jobTracks = useJobTracks(
    job?.jobId ?? null,
    Boolean(playable),
    job?.status === 'analyzing'
  );
  const overlayTracks = jobTracks.tracks.length > 0 ? jobTracks.tracks : match.tracks;
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
          setVideoSource(apiMode === 'http' ? 'stream' : 'job');
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
                : 'The player opens as soon as yt-dlp finishes the local download.'}
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
              them back to the job row once the download reports them — see
              contracts/job.schema.json, which requires them from status <code>downloaded</code>
              onward.
            </p>
          </div>
        )}

        {job && playable && season && jobVideoSrc && (
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
                    {videoSource === 'stream' ? 'Ad-free yt-dlp stream ready.' : 'Local video ready.'}{' '}
                    Pipeline is {job.status}
                    {job.stage ? ` · ${job.stage}` : ''}
                    {job.progress != null ? ` · ${Math.round(job.progress * 100)}%` : ''}.
                  </>
                )}
              </div>
            )}
            <div className="video-source" aria-label="Video source">
              <div className="video-source-main">
                <span className="video-source-label">Video</span>
                <button
                  type="button"
                  className={videoSource === 'job' ? 'on' : ''}
                  disabled={apiMode === 'http' && !job.localPath}
                  title={job.localPath ? 'Play the downloaded match segment' : 'Available after yt-dlp finishes the local download'}
                  onClick={() => {
                    setLocalVideo(null);
                    setVideoAlignment('segment');
                    setVideoSource('job');
                  }}
                >
                  Downloaded file
                </button>
                {apiMode === 'http' && (
                  <button
                    type="button"
                    className={videoSource === 'stream' ? 'on' : ''}
                    onClick={() => {
                      setLocalVideo(null);
                      setVideoSource('stream');
                    }}
                  >
                    yt-dlp stream
                  </button>
                )}
                <label className={`video-file-button ${videoSource === 'local' ? 'on' : ''}`}>
                  Choose matching video…
                  <input type="file" accept="video/*,.mp4,.mov,.webm,.m4v" onChange={chooseLocalVideo} />
                </label>
                {localVideo && <strong className="video-source-name" title={localVideo.name}>{localVideo.name}</strong>}
              </div>

              {videoSource === 'stream' && (
                <div className="video-source-alignment">
                  <span className="video-source-note stream">
                    Ad-free native stream resolved by local yt-dlp · video and audio stay on localhost.
                  </span>
                </div>
              )}

              {videoSource === 'local' && localVideo && (
                <div className="video-source-alignment">
                  <label>
                    Timing
                    <select
                      value={videoAlignment}
                      onChange={(event) => setVideoAlignment(event.target.value as VideoAlignment)}
                    >
                      <option value="segment">Clipped match segment (starts at 0:00)</option>
                      <option value="original">
                        Full original recording (match starts at {job.startOffset}s)
                      </option>
                    </select>
                  </label>
                  <span className="video-source-note">
                    Boxes use this job's tracks. Select the exact recording analyzed for this job so robots and timestamps line up.
                  </span>
                </div>
              )}
            </div>

            <VideoPlayer
              job={playable}
              season={season}
              src={
                videoSource === 'stream' && streamVideoSrc
                  ? streamVideoSrc
                  : videoSource === 'local' && localVideo
                    ? localVideo.url
                    : jobVideoSrc
              }
              audioSrc={videoSource === 'stream' ? streamAudioSrc ?? undefined : undefined}
              mediaStartSeconds={
                videoSource === 'stream' || (videoSource === 'local' && videoAlignment === 'original')
                  ? job.startOffset
                  : 0
              }
              tracks={overlayTracks}
              events={match.events}
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
                    : `${match.events.length} events · ${overlayTracks.length} tracks · boxes @ ${(overlaySampleRate || job.fps || 0).toFixed(0)} Hz`}
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
              <HeatMap season={season} events={match.events} selectedTeam={selectedTeam} />
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
