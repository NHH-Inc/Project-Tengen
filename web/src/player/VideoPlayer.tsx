import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { PlayableJob, Track } from '../contracts';
import { scoringCountsAt, type GoalEntry, type ShotGoal, type ShotRecord } from '../api/shots';
import type { ViewEvent } from '../lib/corrections';
import { EVENT_LABEL, fmtClock, fmtTime, youtubeUrlAt } from '../lib/format';
import { robotName, visibleBoxes } from '../lib/tracks';
import { phaseBounds, type SeasonConfig } from '../season';

// requestVideoFrameCallback is what makes the overlay frame-accurate instead of merely
// close. lib.dom declares it, but not every browser implements it (Firefox), so it is
// feature-detected rather than assumed.
interface FrameMeta {
  mediaTime: number;
  presentedFrames: number;
}
interface RVFCCapable {
  requestVideoFrameCallback(cb: (now: number, meta: FrameMeta) => void): number;
  cancelVideoFrameCallback(handle: number): void;
}

const RED = '#e0555f';
const BLUE = '#4c8cf0';
const GREY = '#8a8f9c';
const LOW_CONF = '#e8b93b';
// Same normalized crop passed to the YOLO worker: the upper broadcast panel marked in the
// supplied reference image. The stage is sized to this rectangle and the full video is shifted
// underneath it, so the live video and normalized model boxes stay aligned.
const VIDEO_CROP = { left: 0.02, top: 0.035, right: 0.98, bottom: 0.66 } as const;
const VIDEO_CROP_WIDTH = VIDEO_CROP.right - VIDEO_CROP.left;
const VIDEO_CROP_HEIGHT = VIDEO_CROP.bottom - VIDEO_CROP.top;
// Broadcast padding: the countdown before the match and the score card after it. Trimming these
// makes review faster, but the amount of padding is a property of whoever cut the upload, not a
// constant. Fixed values are dangerous here in one specific direction: an FRC match ends with
// endgame -- the last `endgame_seconds` of play, when robots climb -- so an over-long tail trim
// silently hides the highest-value part of the match, and nothing in the UI would say so.
//
// Defaults are therefore derived, not invented. The trim never exceeds the padding the clip
// actually has, which is `duration - matchLength` where matchLength comes from the season config.
const DEFAULT_LEAD_IN_SECONDS = 5;
const DEFAULT_TAIL_SECONDS = 10;

export interface PlaybackTrim {
  leadInSeconds?: number;
  tailSeconds?: number;
}

function getPlaybackWindow(
  duration: number,
  matchLengthSeconds: number,
  trim?: PlaybackTrim
) {
  const requestedLead = trim?.leadInSeconds ?? DEFAULT_LEAD_IN_SECONDS;
  const requestedTail = trim?.tailSeconds ?? DEFAULT_TAIL_SECONDS;

  // How much of this clip is definitely not match play. If the upload is barely longer than a
  // match, there is nothing safe to cut and we show all of it.
  const padding = Math.max(0, duration - matchLengthSeconds);
  if (padding <= 0) {
    return { start: 0, end: duration };
  }

  // Never trim more than the padding, and never let the two ends meet.
  const lead = Math.max(0, Math.min(requestedLead, padding));
  const tail = Math.max(0, Math.min(requestedTail, padding - lead));
  if (duration - lead - tail <= 0) {
    return { start: 0, end: duration };
  }
  return { start: lead, end: duration - tail };
}

export interface VideoPlayerProps {
  job: PlayableJob;
  season: SeasonConfig;
  src: string;
  /** Optional separate DASH audio track for an ad-free yt-dlp stream. */
  audioSrc?: string;
  /** Source-media timestamp corresponding to t=0 in the analyzed segment. */
  mediaStartSeconds?: number;
  tracks: Track[];
  events: ViewEvent[];
  shots: ShotRecord[];
  shotGoals: ShotGoal[];
  goalEntries: GoalEntry[];
  goalCameraGaps?: [number, number | null][];
  /** Events below this are drawn as suspect. Doc 3: low confidence must be visually distinct. */
  confidenceThreshold: number;
  boxSampleRate: number;
  /** Broadcast padding to skip. Omitted values fall back to conservative defaults that are
   *  capped by the clip's real padding, so endgame can never be trimmed away. */
  trim?: PlaybackTrim;
  selectedEventId: string | null;
  onSelectEvent: (eventId: string | null) => void;
  /** Set by the parent when the user scrubs to an event from elsewhere in the UI. */
  seekTo: { t: number; nonce: number } | null;
  onTimeChange?: (t: number) => void;
}

export function VideoPlayer({
  job,
  season,
  src,
  audioSrc,
  mediaStartSeconds = 0,
  tracks,
  events,
  shots,
  shotGoals,
  goalEntries,
  goalCameraGaps = [],
  confidenceThreshold,
  boxSampleRate,
  trim,
  selectedEventId,
  onSelectEvent,
  seekTo,
  onTimeChange,
}: VideoPlayerProps) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const stageRef = useRef<HTMLDivElement | null>(null);
  const [time, setTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [rate, setRate] = useState(1);
  const [volume, setVolume] = useState(1);
  const [muted, setMuted] = useState(false);
  const [showBoxes, setShowBoxes] = useState(true);
  const [showShots, setShowShots] = useState(true);
  const [ready, setReady] = useState(false);
  const [mediaError, setMediaError] = useState<string | null>(null);
  const [sourceSize, setSourceSize] = useState<{ width: number; height: number } | null>(null);

  // The overlay redraws from whatever these hold, so the frame callback never re-subscribes.
  const drawState = useRef({
    tracks, events, shots, shotGoals, goalEntries, goalCameraGaps,
    confidenceThreshold, showBoxes, showShots, boxSampleRate,
  });
  drawState.current = {
    tracks, events, shots, shotGoals, goalEntries, goalCameraGaps,
    confidenceThreshold, showBoxes, showShots, boxSampleRate,
  };

  const duration = job.duration;
  // 15s auto + 135s teleop + 20s endgame for 2026. Read from the season config so the trim
  // cannot outlive a rules change.
  const matchLengthSeconds =
    season.autoSeconds + season.teleopSeconds + season.endgameSeconds;
  const playbackWindow = useMemo(
    () => getPlaybackWindow(duration, matchLengthSeconds, trim),
    [duration, matchLengthSeconds, trim]
  );
  const { start: playbackStart, end: playbackEnd } = playbackWindow;
  const PHASE_BOUNDS = phaseBounds(season);

  useEffect(() => {
    setReady(false);
    setMediaError(null);
    setTime(playbackStart);
    setSourceSize(null);
  }, [audioSrc, playbackStart, src]);

  const segmentTime = useCallback(
    (mediaTime: number) => mediaTime - mediaStartSeconds,
    [mediaStartSeconds]
  );

  const boundedSegmentTime = useCallback(
    (mediaTime: number) => Math.max(playbackStart, Math.min(playbackEnd, segmentTime(mediaTime))),
    [playbackEnd, playbackStart, segmentTime]
  );

  // ---- drawing

  const draw = useCallback((t: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const cssW = canvas.clientWidth;
    const cssH = canvas.clientHeight;
    if (cssW === 0 || cssH === 0) return;
    if (canvas.width !== Math.round(cssW * dpr) || canvas.height !== Math.round(cssH * dpr)) {
      canvas.width = Math.round(cssW * dpr);
      canvas.height = Math.round(cssH * dpr);
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);

    const s = drawState.current;
    if (!s.showBoxes && !s.showShots) return;

    if (s.showShots) {
      ctx.save();
      ctx.lineJoin = 'round';
      ctx.lineCap = 'round';
      const goalsPaused = s.goalCameraGaps.some(([start, end]) => t >= start && (end === null || t < end));
      if (goalsPaused) {
        ctx.fillStyle = '#ffd078';
        ctx.font = '600 12px ui-monospace, SFMono-Regular, Menlo, monospace';
        ctx.fillText('Goal counting paused: AprilTags not verified', 12, 48);
      }
      for (const goal of goalsPaused ? [] : s.shotGoals) {
        if (goal.confirmationPolygon) {
          ctx.beginPath();
          goal.confirmationPolygon.forEach(([x, y], i) => {
            if (i === 0) ctx.moveTo(x * cssW, y * cssH);
            else ctx.lineTo(x * cssW, y * cssH);
          });
          ctx.closePath();
          ctx.fillStyle = 'rgba(80, 220, 114, 0.10)';
          ctx.fill();
        }
        const labelPoint = goal.madeBoundary?.line[0] ?? goal.polygon?.[0];
        if (labelPoint) {
          const made = s.goalEntries.filter((entry) => entry.regionId === goal.regionId && entry.tSeconds <= t).length;
          ctx.fillStyle = '#50dc72';
          ctx.font = '600 11px ui-monospace, SFMono-Regular, Menlo, monospace';
          ctx.fillText(`${goal.regionId.replaceAll('_', ' ')}: ${made} in`, labelPoint[0] * cssW, labelPoint[1] * cssH - 8);
        }
        if (goal.polygon && goal.polygon.length >= 3) {
          ctx.beginPath();
          ctx.moveTo(goal.polygon[0][0] * cssW, goal.polygon[0][1] * cssH);
          for (const point of goal.polygon.slice(1)) {
            ctx.lineTo(point[0] * cssW, point[1] * cssH);
          }
          ctx.closePath();
          ctx.strokeStyle = '#50dc72';
          ctx.lineWidth = 2;
          ctx.stroke();
        }
        if (goal.madeBoundary) {
          const [first, second] = goal.madeBoundary.line;
          ctx.beginPath();
          ctx.moveTo(first[0] * cssW, first[1] * cssH);
          ctx.lineTo(second[0] * cssW, second[1] * cssH);
          ctx.strokeStyle = '#50dc72';
          ctx.lineWidth = 3;
          ctx.stroke();
        }
        for (const boundary of goal.missBoundaries) {
          const [first, second] = boundary.line;
          ctx.beginPath();
          ctx.moveTo(first[0] * cssW, first[1] * cssH);
          ctx.lineTo(second[0] * cssW, second[1] * cssH);
          ctx.strokeStyle = '#ef5964';
          ctx.lineWidth = 2;
          ctx.stroke();
        }
      }

      for (const shot of s.shots) {
        const lastTime = shot.ballTrack.at(-1)?.tSeconds ?? shot.launchTSeconds;
        if (t < shot.launchTSeconds - 0.2 || t > lastTime + 0.20) continue;
        const path = shot.ballTrack.filter((point) => point.tSeconds <= t + 1e-3
          && point.tSeconds >= t - .35);
        if (path.length === 0) continue;
        const outcomeKnown = shot.outcomeTSeconds != null && t >= shot.outcomeTSeconds;
        const colour = outcomeKnown && shot.outcome === 'made'
          ? '#50dc72'
          : outcomeKnown && shot.outcome === 'missed'
            ? '#ef5964'
            : '#4de4ee';
        if (path.length >= 2) {
          ctx.beginPath();
          ctx.moveTo(path[0].x * cssW, path[0].y * cssH);
          for (let i = 1; i < path.length; i++) {
            const point = path[i];
            if (point.tSeconds - path[i - 1].tSeconds > .085) ctx.moveTo(point.x * cssW, point.y * cssH);
            else ctx.lineTo(point.x * cssW, point.y * cssH);
          }
          ctx.strokeStyle = colour;
          ctx.lineWidth = 2;
          ctx.stroke();
        }
        const current = path[path.length - 1];
        const radius = Math.max(3, current.radius * Math.hypot(cssW, cssH));
        ctx.beginPath();
        ctx.arc(current.x * cssW, current.y * cssH, radius, 0, Math.PI * 2);
        ctx.strokeStyle = colour;
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.font = '600 11px ui-monospace, SFMono-Regular, Menlo, monospace';
        ctx.fillStyle = colour;
        if (outcomeKnown && shot.outcome !== 'unknown') {
          ctx.fillText(shot.outcome === 'made' ? 'IN' : 'MISS',
            current.x * cssW + radius + 3, current.y * cssH - radius);
        }
      }

      for (const entry of s.goalEntries) {
        if (t < entry.tSeconds || t > entry.tSeconds + .25 || entry.ballTrack.length === 0) continue;
        ctx.beginPath();
        entry.ballTrack.forEach((point, i) => {
          if (i === 0 || point.tSeconds - entry.ballTrack[i - 1].tSeconds > .085)
            ctx.moveTo(point.x * cssW, point.y * cssH);
          else ctx.lineTo(point.x * cssW, point.y * cssH);
        });
        ctx.strokeStyle = '#50dc72';
        ctx.lineWidth = 3;
        ctx.stroke();
        const last = entry.ballTrack[entry.ballTrack.length - 1];
        ctx.fillStyle = '#50dc72';
        ctx.font = '700 12px ui-monospace, SFMono-Regular, Menlo, monospace';
        ctx.fillText(`IN${entry.robotTrackId == null ? '' : ` · R${entry.robotTrackId}`}`,
          last.x * cssW + 8, last.y * cssH);
      }

      const shotsDetected = s.shots.filter(
        (shot) => shot.launchTSeconds <= t + 1e-3
      ).length;
      const made = scoringCountsAt(s.shots, s.goalEntries, t);
      const rows = [`Shots detected: ${shotsDetected}`,
        s.shotGoals.length > 0 ? `Balls in: ${made.made} · source unknown: ${made.unassigned}`
          : 'Goal regions not configured'];
      ctx.font = '700 13px ui-monospace, SFMono-Regular, Menlo, monospace';
      const panelWidth = Math.max(...rows.map((row) => ctx.measureText(row).width)) + 18;
      const panelX = cssW - panelWidth - 8;
      ctx.fillStyle = 'rgba(10, 12, 16, 0.82)';
      ctx.fillRect(panelX, 8, panelWidth, 49);
      ctx.fillStyle = '#f2f3f5';
      rows.forEach((row, i) => ctx.fillText(row, panelX + 9, 27 + i * 20));
      ctx.restore();
    }

    if (!s.showBoxes) return;

    // Hold a box for one sample period past its last sample so it does not strobe at the
    // sample boundary; beyond that the track really is gone and must not draw.
    const hold = 1 / Math.max(1e-6, s.boxSampleRate);
    const shown = visibleBoxes(s.tracks, t, hold);

    for (const { track, box } of shown) {
      const x = box.x * cssW;
      const y = box.y * cssH;
      const w = box.w * cssW;
      const h = box.h * cssH;

      const colour = track.alliance === 'red' ? RED : track.alliance === 'blue' ? BLUE : GREY;
      const identified = track.team != null;

      ctx.lineWidth = 2;
      ctx.strokeStyle = colour;
      // An unidentified track is dashed: the box is real, the attribution is not.
      ctx.setLineDash(identified ? [] : [5, 4]);
      ctx.strokeRect(x, y, w, h);
      ctx.setLineDash([]);

      // Track ids are implementation details and can be large or change in older output. The
      // runner now persists robot1/robot2/...; robotName supplies a deterministic fallback for
      // already-saved jobs.
      const label = robotName(track, s.tracks);
      ctx.font = '600 12px ui-monospace, SFMono-Regular, Menlo, monospace';
      const tw = ctx.measureText(label).width;
      const lh = 16;
      const ly = y - lh < 0 ? y + h : y - lh;
      ctx.fillStyle = colour;
      ctx.fillRect(x, ly, tw + 10, lh);
      ctx.fillStyle = '#0d0f14';
      ctx.fillText(label, x + 5, ly + 12);

      // Any event for this track within a beat of now, so a shot is visible as it happens.
      const near = s.events.filter(
        (e) => e.trackId === track.trackId && Math.abs(e.tSeconds - t) < 0.6
      );
      if (near.length > 0) {
        const low = near.some((e) => e.confidence < s.confidenceThreshold);
        ctx.strokeStyle = low ? LOW_CONF : '#ffffff';
        ctx.lineWidth = low ? 2 : 3;
        ctx.setLineDash(low ? [4, 3] : []);
        ctx.strokeRect(x - 4, y - 4, w + 8, h + 8);
        ctx.setLineDash([]);
        ctx.font = '600 11px ui-monospace, SFMono-Regular, Menlo, monospace';
        ctx.fillStyle = low ? LOW_CONF : '#ffffff';
        ctx.fillText(EVENT_LABEL[near[0].eventType], x, y + h + 13);
      }
    }
  }, []);

  // ---- frame loop
  //
  // rVFC fires once per presented video frame with the exact mediaTime of that frame, which
  // is the whole reason for a self-hosted <video> over the YouTube iframe. Fall back to rAF
  // plus currentTime where it is unavailable (Firefox), which is visibly looser.

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    let handle = 0;
    let raf = 0;
    let cancelled = false;

    const rvfc = (video as Partial<RVFCCapable>).requestVideoFrameCallback;
    const cancelRvfc = (video as Partial<RVFCCapable>).cancelVideoFrameCallback;
    if (typeof rvfc === 'function') {
      const onFrame = (_now: number, meta: FrameMeta) => {
        if (cancelled) return;
        const segment = segmentTime(meta.mediaTime);
        const t = boundedSegmentTime(meta.mediaTime);
        setTime(t);
        draw(t);
        const audio = audioRef.current;
        if (audio && !video.paused && Math.abs(audio.currentTime - meta.mediaTime) > 0.12) {
          audio.currentTime = meta.mediaTime;
        }
        if (segment >= playbackEnd && !video.paused) {
          video.pause();
          audio?.pause();
        }
        handle = rvfc.call(video, onFrame);
      };
      handle = rvfc.call(video, onFrame);
      return () => {
        cancelled = true;
        cancelRvfc?.call(video, handle);
      };
    }

    const tick = () => {
      if (cancelled) return;
      const segment = segmentTime(video.currentTime);
      const t = boundedSegmentTime(video.currentTime);
      setTime(t);
      draw(t);
      const audio = audioRef.current;
      if (audio && !video.paused && Math.abs(audio.currentTime - video.currentTime) > 0.12) {
        audio.currentTime = video.currentTime;
      }
      if (segment >= playbackEnd && !video.paused) {
        video.pause();
        audio?.pause();
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => {
      cancelled = true;
      cancelAnimationFrame(raf);
    };
  }, [boundedSegmentTime, draw, playbackEnd, ready, segmentTime]);

  // Paused frames still need redrawing when the caller changes filters or the box toggle.
  useEffect(() => {
    if (!playing) draw(time);
  }, [
    draw, playing, tracks, events, shots, shotGoals, goalEntries, goalCameraGaps,
    confidenceThreshold, showBoxes, showShots, time,
  ]);

  // draw() bails when the canvas has no layout yet, and on first load the track data can
  // arrive before that happens -- leaving the overlay blank until the user hits play or
  // touches a filter. Redrawing on resize covers both that first sizing and any later one.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || typeof ResizeObserver === 'undefined') return;
    const ro = new ResizeObserver(() => draw(time));
    ro.observe(canvas);
    return () => ro.disconnect();
  }, [draw, time]);

  useEffect(() => {
    onTimeChange?.(time);
  }, [time, onTimeChange]);

  // ---- transport

  const seek = useCallback((t: number) => {
    const video = videoRef.current;
    if (!video) return;
    const clamped = Math.max(playbackStart, Math.min(playbackEnd, t));
    video.currentTime = clamped + mediaStartSeconds;
    if (audioRef.current) audioRef.current.currentTime = clamped + mediaStartSeconds;
    setTime(clamped);
    draw(clamped);
  }, [draw, mediaStartSeconds, playbackEnd, playbackStart]);

  // Changing between a clipped segment and a full recording does not change the file URL,
  // so metadata will not fire again. Re-anchor playback explicitly when the mode changes.
  useEffect(() => {
    const video = videoRef.current;
    if (!video || !ready) return;
    if (Number.isFinite(video.duration) && mediaStartSeconds >= video.duration) {
      setMediaError(
        `The match offset (${mediaStartSeconds}s) is past the end of this ${fmtClock(video.duration)} video.`
      );
      return;
    }
    setMediaError(null);
    video.currentTime = mediaStartSeconds + playbackStart;
    if (audioRef.current) audioRef.current.currentTime = mediaStartSeconds + playbackStart;
    setTime(playbackStart);
    draw(playbackStart);
  }, [draw, mediaStartSeconds, playbackStart, ready]);

  useEffect(() => {
    if (seekTo) seek(seekTo.t);
    // nonce lets the parent request the same timestamp twice in a row
  }, [seekTo?.nonce, seekTo?.t, seek]);

  const togglePlay = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;
    if (video.paused) {
      if (time < playbackStart || time >= playbackEnd) seek(playbackStart);
      void video.play();
      const audio = audioRef.current;
      if (audio) {
        audio.currentTime = video.currentTime;
        void audio.play().catch(() => {
          setMediaError('The yt-dlp audio stream could not start. Try Play again.');
        });
      }
    }
    else {
      video.pause();
      audioRef.current?.pause();
    }
  }, [playbackEnd, playbackStart, seek, time]);

  const step = useCallback(
    (frames: number) => {
      const video = videoRef.current;
      if (!video) return;
      video.pause();
      audioRef.current?.pause();
      seek(time + frames / job.fps);
    },
    [job.fps, seek, time]
  );

  // Keyboard transport, the way anyone reviewing footage expects it.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (target && /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName)) return;
      if (e.key === ' ') {
        e.preventDefault();
        togglePlay();
      } else if (e.key === 'ArrowLeft') {
        e.preventDefault();
        step(e.shiftKey ? -job.fps : -1);
      } else if (e.key === 'ArrowRight') {
        e.preventDefault();
        step(e.shiftKey ? job.fps : 1);
      } else if (e.key === 'b') {
        setShowBoxes((v) => !v);
      } else if (e.key === 's') {
        setShowShots((v) => !v);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [togglePlay, step, job.fps]);

  // ---- scrub bar markers

  // The range input covers the trimmed playback window, not the entire source duration. Using
  // `seconds / duration` here made the white playhead drift from the native range thumb by the
  // lead-in and tail trims, and could place event markers outside the visible bar.
  const scrubSpan = playbackEnd - playbackStart;
  const positionPct = (seconds: number) => (
    scrubSpan > 0
      ? Math.min(100, Math.max(0, ((seconds - playbackStart) / scrubSpan) * 100))
      : 0
  );

  const markers = useMemo(
    () =>
      events
        .filter((e) => e.eventType === 'shot_made' || e.eventType === 'shot_attempt' || e.eventType === 'foul')
        .map((e) => ({
          id: e.eventId,
          tSeconds: e.tSeconds,
          left: positionPct(e.tSeconds),
          low: e.confidence < confidenceThreshold,
          type: e.eventType,
        })),
    [events, playbackStart, playbackEnd, scrubSpan, confidenceThreshold]
  );

  const selected = events.find((e) => e.eventId === selectedEventId) ?? null;
  const phasePct = positionPct;

  const toggleFullscreen = async () => {
    try {
      if (document.fullscreenElement) await document.exitFullscreen();
      else await stageRef.current?.requestFullscreen();
    } catch {
      setMediaError('This browser did not allow fullscreen playback.');
    }
  };

  return (
    <section className="player">
      <div
        ref={stageRef}
        className="player-stage"
        style={{
          aspectRatio: sourceSize
            ? `${sourceSize.width * VIDEO_CROP_WIDTH} / ${sourceSize.height * VIDEO_CROP_HEIGHT}`
            : `${job.width * VIDEO_CROP_WIDTH} / ${job.height * VIDEO_CROP_HEIGHT}`,
        }}
      >
        <video
          ref={videoRef}
          className="player-video"
          src={src}
          preload="auto"
          playsInline
          style={{
            position: 'absolute',
            width: `${(100 / VIDEO_CROP_WIDTH).toFixed(4)}%`,
            height: 'auto',
            left: `${(-100 * VIDEO_CROP.left / VIDEO_CROP_WIDTH).toFixed(4)}%`,
            top: `${(-100 * VIDEO_CROP.top / VIDEO_CROP_HEIGHT).toFixed(4)}%`,
            maxWidth: 'none',
          }}
          onPlay={() => {
            setPlaying(true);
            const audio = audioRef.current;
            if (audio && audio.paused) {
              audio.currentTime = videoRef.current?.currentTime ?? mediaStartSeconds;
              void audio.play().catch(() => undefined);
            }
          }}
          onPause={() => {
            setPlaying(false);
            audioRef.current?.pause();
          }}
          onEnded={() => {
            setPlaying(false);
            audioRef.current?.pause();
          }}
          onRateChange={(e) => {
            const nextRate = (e.target as HTMLVideoElement).playbackRate;
            setRate(nextRate);
            if (audioRef.current) audioRef.current.playbackRate = nextRate;
          }}
          onVolumeChange={(e) => {
            if (audioSrc) return;
            const video = e.target as HTMLVideoElement;
            setVolume(video.volume);
            setMuted(video.muted);
          }}
          onLoadedMetadata={(e) => {
            const video = e.target as HTMLVideoElement;
            // Stream audio is played by the synchronized hidden audio element. This also avoids
            // double audio when yt-dlp falls back to one legacy file containing both tracks.
            if (audioSrc) video.muted = true;
            if (video.videoWidth > 0 && video.videoHeight > 0) {
              setSourceSize({ width: video.videoWidth, height: video.videoHeight });
            }
            if (Number.isFinite(video.duration) && mediaStartSeconds >= video.duration) {
              setMediaError(
                `The match offset (${mediaStartSeconds}s) is past the end of this ${fmtClock(video.duration)} video.`
              );
              return;
            }
            video.currentTime = mediaStartSeconds + playbackStart;
            setTime(playbackStart);
            draw(playbackStart);
          }}
          onLoadedData={() => {
            setReady(true);
            setMediaError(null);
          }}
          onError={(e) => {
            const video = e.target as HTMLVideoElement;
            setReady(false);
            setMediaError(
              video.error?.message ||
                'The downloaded file could not be played. Check that it uses a browser-supported MP4 codec.'
            );
          }}
        />
        {audioSrc && (
          <audio
            ref={audioRef}
            src={audioSrc}
            preload="auto"
            onLoadedMetadata={(e) => {
              const audio = e.target as HTMLAudioElement;
              audio.currentTime = videoRef.current?.currentTime ?? mediaStartSeconds;
              audio.playbackRate = rate;
            }}
            onVolumeChange={(e) => {
              const audio = e.target as HTMLAudioElement;
              setVolume(audio.volume);
              setMuted(audio.muted);
            }}
            onError={() => {
              setMediaError('The yt-dlp audio stream could not be loaded.');
            }}
          />
        )}
        <canvas ref={canvasRef} className="player-overlay" />
        {!ready && !mediaError && <div className="player-loading">loading video…</div>}
        {mediaError && <div className="player-loading player-media-error">{mediaError}</div>}
      </div>

      <div className="scrub">
        <div className="scrub-track">
          {/* Phase bands, so auto/teleop/endgame are readable at a glance. */}
          <div
            className="scrub-phase auto"
            style={{ left: 0, width: `${phasePct(PHASE_BOUNDS.autoEnd)}%` }}
            title="Auto"
          />
          <div
            className="scrub-phase teleop"
            style={{
              left: `${phasePct(PHASE_BOUNDS.autoEnd)}%`,
              width: `${phasePct(PHASE_BOUNDS.teleopEnd) - phasePct(PHASE_BOUNDS.autoEnd)}%`,
            }}
            title="Teleop"
          />
          <div
            className="scrub-phase endgame"
            style={{
              left: `${phasePct(PHASE_BOUNDS.teleopEnd)}%`,
              width: `${phasePct(PHASE_BOUNDS.matchEnd) - phasePct(PHASE_BOUNDS.teleopEnd)}%`,
            }}
            title="Endgame"
          />
          {markers.map((m) => (
            <button
              key={m.id}
              type="button"
              className={`scrub-marker ${m.type} ${m.low ? 'low' : ''} ${m.id === selectedEventId ? 'on' : ''}`}
              style={{ left: `clamp(1px, ${m.left}%, calc(100% - 2px))` }}
              title={`${EVENT_LABEL[m.type as keyof typeof EVENT_LABEL]} @ ${fmtTime(m.tSeconds)}`}
              onClick={() => onSelectEvent(m.id)}
            />
          ))}
          <div
            className="scrub-playhead"
            style={{ left: `clamp(1px, ${positionPct(time)}%, calc(100% - 1px))` }}
          />
          <input
            className="scrub-input"
            type="range"
            min={playbackStart}
            max={playbackEnd}
            step={1 / job.fps}
            value={time}
            onChange={(e) => seek(Number(e.target.value))}
            aria-label="Seek"
          />
        </div>
      </div>

      <div className="transport">
        <button type="button" onClick={() => step(-job.fps)} title="Back 1s (Shift+Left)">«</button>
        <button type="button" onClick={() => step(-1)} title="Previous frame (Left)">‹</button>
        <button type="button" className="primary" onClick={togglePlay} title="Play/pause (Space)">
          {playing ? 'Pause' : 'Play'}
        </button>
        <button type="button" onClick={() => step(1)} title="Next frame (Right)">›</button>
        <button type="button" onClick={() => step(job.fps)} title="Forward 1s (Shift+Right)">»</button>

        <span className="transport-time" title="Segment time, and position in the original video">
          <strong>{fmtTime(time)}</strong>
          <span className="muted"> / {fmtClock(duration)} seg</span>
        </span>

        <label className="transport-rate">
          Speed
          <select
            value={rate}
            onChange={(e) => {
              const v = Number(e.target.value);
              if (videoRef.current) videoRef.current.playbackRate = v;
              if (audioRef.current) audioRef.current.playbackRate = v;
              setRate(v);
            }}
          >
            {[0.25, 0.5, 1, 1.5, 2].map((r) => (
              <option key={r} value={r}>{r}×</option>
            ))}
          </select>
        </label>

        <div className="transport-volume">
          <button
            type="button"
            onClick={() => {
              const media = audioRef.current ?? videoRef.current;
              if (media) media.muted = !media.muted;
            }}
            aria-label={muted || volume === 0 ? 'Unmute' : 'Mute'}
            title={muted || volume === 0 ? 'Unmute' : 'Mute'}
          >
            {muted || volume === 0 ? 'Muted' : 'Sound'}
          </button>
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={muted ? 0 : volume}
            aria-label="Volume"
            onChange={(e) => {
              const next = Number(e.target.value);
              const media = audioRef.current ?? videoRef.current;
              if (media) {
                media.volume = next;
                media.muted = next === 0;
              }
              setVolume(next);
              setMuted(next === 0);
            }}
          />
        </div>

        <label className="transport-toggle" title="Toggle overlay (B)">
          <input type="checkbox" checked={showBoxes} onChange={(e) => setShowBoxes(e.target.checked)} />
          Boxes
        </label>

        <label className="transport-toggle" title="Toggle shot tracks and goal geometry (S)">
          <input type="checkbox" checked={showShots} onChange={(e) => setShowShots(e.target.checked)} />
          Shots
        </label>

        <button type="button" onClick={() => void toggleFullscreen()} title="Fullscreen">
          Fullscreen
        </button>

        {/* The one place component 3 adds start_offset -- doc 0 says nothing else ever should. */}
        <a
          className="yt-link"
          href={youtubeUrlAt(job.videoId, time, job.startOffset)}
          target="_blank"
          rel="noreferrer"
          title={`Original video at ${fmtClock(time + job.startOffset)} (segment ${fmtClock(time)} + ${job.startOffset}s offset)`}
        >
          Open on YouTube ↗
        </a>
      </div>

      {selected && (
        <div className={`player-selected ${selected.confidence < confidenceThreshold ? 'low' : ''}`}>
          <strong>{EVENT_LABEL[selected.eventType]}</strong>
          <span>{selected.team != null ? `team ${selected.team}` : 'unattributed'}</span>
          <span className="muted">{fmtTime(selected.tSeconds)}</span>
          <span className="muted">conf {selected.confidence.toFixed(2)}</span>
          <button type="button" onClick={() => onSelectEvent(null)}>clear</button>
        </div>
      )}
    </section>
  );
}
