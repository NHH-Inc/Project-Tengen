"""Small, testable yt-dlp wrapper for local media acquisition."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, urlparse

import yt_dlp
from yt_dlp.utils import DownloadError, download_range_func


ProgressCallback = Callable[[float | None, str], None]


def _timestamp_seconds(value: str | None) -> float:
    """Parse YouTube's ``t=90`` and ``t=1h2m3s`` timestamp forms."""
    if not value:
        return 0.0
    value = value.strip().lower()
    try:
        return max(0.0, float(value))
    except ValueError:
        pass

    match = re.fullmatch(
        r"(?:(?P<hours>\d+(?:\.\d+)?)h)?"
        r"(?:(?P<minutes>\d+(?:\.\d+)?)m)?"
        r"(?:(?P<seconds>\d+(?:\.\d+)?)s)?",
        value,
    )
    if not match or not any(match.groupdict().values()):
        return 0.0
    return (
        float(match.group("hours") or 0) * 3600
        + float(match.group("minutes") or 0) * 60
        + float(match.group("seconds") or 0)
    )


def start_time_from_url(url: str) -> float:
    """Return a timestamp carried by a YouTube URL, or zero when it has none."""
    try:
        parsed = urlparse(url)
    except ValueError:
        return 0.0
    query = parse_qs(parsed.query)
    fragment = parse_qs(parsed.fragment)
    for key in ("t", "start", "time_continue"):
        values = query.get(key) or fragment.get(key)
        if values:
            return _timestamp_seconds(values[0])
    return 0.0


# Doc 2: "Set a retention policy early or disk usage will get out of hand fast, since a single
# event's footage is tens of gigabytes." These are the guards that make that concrete.
#
# The numbers are deliberately conservative because the team's machines have ~40-107 GB free,
# and we have already filled 9.2 GB by accident with one unclipped VOD.

#: Refuse to start a download when the disk is below this. A partial file on a full disk is the
#: worst outcome: it fails anyway AND leaves nothing else room to work.
MIN_FREE_GB = float(os.environ.get("FRC_MIN_FREE_GB", "10"))

#: Space to reserve before starting a live capture, expressed as hours of footage. This does NOT
#: truncate the recording -- a live capture must run to the end of the stream, because a
#: capped file looks complete but is not, and that is worse than failing. It only refuses to
#: START a capture the disk plainly cannot hold. An FRC event stream is 8-12 hours, and
#: live_from_start means joining late still pulls everything from the beginning.
LIVE_RESERVE_HOURS = float(os.environ.get("FRC_LIVE_RESERVE_HOURS", "6"))


def free_gb(path: "Path | str") -> float:
    """Free space on the filesystem holding `path`, in GB."""
    target = Path(path)
    while not target.exists() and target != target.parent:
        target = target.parent
    return shutil.disk_usage(target).free / (1024 ** 3)


def require_free_space(path: "Path | str", need_gb: float = 0.0) -> None:
    """Raise before writing anything if the disk cannot take it.

    Failing here is much kinder than failing halfway through a 9 GB write, because the disk is
    still usable afterwards.
    """
    available = free_gb(path)
    required = max(MIN_FREE_GB, need_gb + 1.0)
    if available < required:
        raise RuntimeError(
            f"Only {available:.1f} GB free on {Path(path)}; need at least {required:.1f} GB. "
            "Free space or delete old segments (see docs/RUNNING.md, retention)."
        )



def apply_cookies(ydl_opts: dict) -> dict:
    """Attach whichever cookie source is configured, if any.

    YouTube intermittently answers "Sign in to confirm you're not a bot". It is not a permanent
    block -- six consecutive requests succeeded right after one failed -- but it recurs, and it
    recurs hardest when fetching many videos in a row, which is exactly what a collection run does.

    Two sources, because the obvious one does not work here. `cookiesfrombrowser` cannot read
    Chrome on Windows any more: Chrome encrypts its cookie store with app-bound DPAPI and yt-dlp
    fails with "Failed to decrypt with DPAPI" (yt-dlp issue 10927). Firefox still works, and an
    exported cookies.txt always works, so the file option is the one to reach for on Windows.

    A cookie file makes requests as whoever exported it. Use a throwaway Google account rather
    than a personal one.
    """
    cookie_file = os.environ.get("YTDLP_COOKIES_FILE")
    if cookie_file:
        ydl_opts["cookiefile"] = cookie_file
        return ydl_opts
    browser = os.environ.get("YTDLP_COOKIES_FROM_BROWSER")
    if browser:
        ydl_opts["cookiesfrombrowser"] = (browser,)
    return ydl_opts

class VideoDownloader:
    """Downloads browser-playable local MP4 files, one at a time."""

    # YouTube throttles concurrent download fleets. A waiting worker re-checks the cache
    # after taking this process-wide lock, so duplicate jobs still fetch only once.
    _download_lock = threading.Lock()

    def __init__(self, download_dir: str = "/data/segments"):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self._stream_cache: dict[str, tuple[float, dict]] = {}
        self._stream_lock = threading.Lock()

    @property
    def has_ffmpeg(self) -> bool:
        return shutil.which("ffmpeg") is not None

    @property
    def format_selector(self) -> str:
        # Prefer H.264/AAC MP4 for broad <video> support, while retaining fallbacks for
        # uploads where YouTube does not expose those exact codecs.
        return (
            "bv[height<=1080][ext=mp4][vcodec^=avc1]+ba[ext=m4a]/"
            "bv[height<=1080][ext=mp4]+ba[ext=m4a]/"
            "b[height<=1080][ext=mp4]/b[height<=1080]"
        )

    def get_video_info(self, url: str) -> dict:
        """Validate a link and return metadata without fetching media."""
        ydl_opts: dict = {
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "skip_download": True,
            "format": self.format_selector,
        }
        apply_cookies(ydl_opts)
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                return ydl.extract_info(url, download=False)
        except DownloadError as exc:
            raise RuntimeError(f"yt-dlp could not read that video: {exc}") from exc

    def resolve_stream(self, video_id: str) -> dict:
        """Resolve browser-playable video and audio URLs without YouTube's web player.

        Signed media URLs expire, so cache them briefly. The ingest API proxies the URL and
        forwards byte ranges; the browser never loads an iframe, ads, or YouTube controls.
        """
        if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
            raise ValueError("Invalid YouTube video ID")

        now = time.monotonic()
        cached = self._stream_cache.get(video_id)
        if cached and cached[0] > now:
            return cached[1]

        # YouTube commonly exposes browser-compatible video and audio as separate DASH files.
        # Resolve both and let the review player keep two native media elements synchronized.
        # The fallback `b` handles older uploads that still have one combined file.
        options: dict = {
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "skip_download": True,
            "format": (
                "bv[ext=mp4][vcodec^=avc1][height<=1080]+ba[ext=m4a]/"
                "bv[ext=mp4][height<=1080]+ba[ext=m4a]/"
                "bv[height<=1080]+ba/b[height<=1080]/best"
            ),
        }
        apply_cookies(options)

        url = f"https://www.youtube.com/watch?v={video_id}"
        try:
            with self._stream_lock:
                cached = self._stream_cache.get(video_id)
                if cached and cached[0] > time.monotonic():
                    return cached[1]
                with yt_dlp.YoutubeDL(options) as ydl:
                    info = ydl.extract_info(url, download=False)
        except DownloadError as exc:
            raise RuntimeError(f"yt-dlp stream resolution failed: {exc}") from exc

        formats = info.get("requested_formats") or [info]

        def descriptor(format_info: dict, kind: str) -> dict:
            media_url = format_info.get("url")
            protocol = str(format_info.get("protocol") or "")
            if not isinstance(media_url, str) or not media_url.startswith(("https://", "http://")):
                raise RuntimeError(f"yt-dlp did not return a directly streamable {kind} URL")
            if protocol and not protocol.startswith("http"):
                raise RuntimeError(f"yt-dlp selected unsupported {kind} protocol {protocol}")
            extension = str(format_info.get("ext") or "").lower()
            if extension == "webm":
                content_type = f"{kind}/webm"
            else:
                # m4a is an MP4 container, and yt-dlp often omits ext in test/custom extractors.
                content_type = f"{kind}/mp4"
            return {
                "url": media_url,
                "headers": dict(format_info.get("http_headers") or info.get("http_headers") or {}),
                "content_type": content_type,
                "format_id": format_info.get("format_id"),
            }

        video_format = next(
            (item for item in formats if item.get("vcodec") not in (None, "none")), None
        )
        audio_format = next(
            (
                item for item in formats
                if item.get("acodec") not in (None, "none")
                and item.get("vcodec") in (None, "none")
            ),
            None,
        )
        if video_format is None:
            raise RuntimeError("yt-dlp did not select a video stream")

        resolved = {
            "video": descriptor(video_format, "video"),
            # For a combined legacy format, let the hidden audio element decode the same file.
            # The visible video element is muted whenever an audio source exists.
            "audio": descriptor(audio_format or video_format, "audio"),
        }
        self._stream_cache[video_id] = (time.monotonic() + 15 * 60, resolved)
        return resolved

    def download_segment(
        self,
        video_id: str,
        start_time: float,
        duration: float,
        job_id: str,
        *,
        full_video: bool = False,
        on_progress: ProgressCallback | None = None,
    ) -> str:
        """Download a whole video or a bounded section and return its local MP4 path."""
        del job_id  # Cache by stable video/window tuple, not a transient job id.
        if duration <= 0:
            raise ValueError("Video duration must be greater than zero")
        # Roughly 4 GB per hour at the 1080p cap, plus headroom for the ffmpeg merge. Refusing
        # up front beats failing halfway through a 9 GB write and leaving a full disk behind.
        require_free_space(self.download_dir, need_gb=(duration / 3600.0) * 4.0)
        if start_time < 0:
            raise ValueError("Video start time cannot be negative")
        if not self.has_ffmpeg:
            raise RuntimeError(
                "ffmpeg is required to merge YouTube video and audio into a local MP4. "
                "Install ffmpeg and retry this job."
            )

        end_time = start_time + duration
        output_path = self.download_dir / (
            f"{video_id}_{int(start_time):05d}_{int(end_time):05d}.mp4"
        )

        def cached() -> str | None:
            if output_path.exists() and output_path.stat().st_size > 0:
                if on_progress:
                    on_progress(1.0, "cached")
                return str(output_path)
            return None

        existing = cached()
        if existing:
            return existing

        def progress_hook(data: dict):
            if not on_progress:
                return
            status = data.get("status")
            if status == "finished":
                on_progress(1.0, "finalizing")
                return
            if status != "downloading":
                return
            total = data.get("total_bytes") or data.get("total_bytes_estimate")
            downloaded = data.get("downloaded_bytes")
            fraction = downloaded / total if total and downloaded is not None else None
            on_progress(
                min(1.0, max(0.0, fraction)) if fraction is not None else None,
                "yt-dlp",
            )

        ydl_opts: dict = {
            "format": self.format_selector,
            "outtmpl": str(output_path),
            "noplaylist": True,
            "quiet": True,
            "noprogress": True,
            "no_warnings": True,
            "retries": 3,
            "fragment_retries": 3,
            "concurrent_fragment_downloads": 1,
            "progress_hooks": [progress_hook],
        }
        ydl_opts["merge_output_format"] = "mp4"
        if not full_video:
            ydl_opts["download_ranges"] = download_range_func([], [(start_time, end_time)])
            ydl_opts["force_keyframes_at_cuts"] = True
        apply_cookies(ydl_opts)

        url = f"https://www.youtube.com/watch?v={video_id}"
        try:
            with self._download_lock:
                existing = cached()
                if existing:
                    return existing
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
        except DownloadError as exc:
            raise RuntimeError(f"yt-dlp download failed: {exc}") from exc

        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError(f"yt-dlp reported success but {output_path.name} is missing")
        if on_progress:
            on_progress(1.0, "downloaded")
        return str(output_path)

    def capture_live(
        self,
        video_id: str,
        job_id: str,
        *,
        on_progress: ProgressCallback | None = None,
    ) -> str:
        """Record a live source until it finishes, then return one immutable MP4.

        This is capture-then-analyse, not real-time inference. The resulting file deliberately
        enters the same safe, repeatable pipeline as an ordinary downloaded segment.
        """
        if not self.has_ffmpeg:
            raise RuntimeError(
                "ffmpeg is required to record and merge a live YouTube stream. "
                "Install ffmpeg and retry this job."
            )

        # Do not cache: two captures of the same stream may start at different times.
        output_path = self.download_dir / f"{video_id}_live_{job_id}.mp4"

        # A live capture has no known end. Without a cap, one event stream can fill the disk --
        # `live_from_start` means joining late still pulls everything from the beginning.
        require_free_space(self.download_dir, need_gb=LIVE_RESERVE_HOURS * 4.0)

        def progress_hook(data: dict):
            if not on_progress:
                return
            status = data.get("status")
            if status == "finished":
                on_progress(1.0, "finalizing")
            elif status == "downloading":
                # A live feed has no reliable final byte count.
                on_progress(None, "capturing_live")

        ydl_opts: dict = {
            "format": self.format_selector,
            "outtmpl": str(output_path),
            "noplaylist": True,
            "quiet": True,
            "noprogress": True,
            "no_warnings": True,
            "retries": 3,
            "fragment_retries": 3,
            "concurrent_fragment_downloads": 1,
            "progress_hooks": [progress_hook],
            # Fail clearly if YouTube cannot provide the stream from its beginning rather
            # than silently producing a partial recording that looks like a full match.
            "live_from_start": True,
            "merge_output_format": "mp4",
        }

        apply_cookies(ydl_opts)

        url = f"https://www.youtube.com/watch?v={video_id}"
        try:
            with self._download_lock:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
        except DownloadError as exc:
            raise RuntimeError(f"yt-dlp live capture failed: {exc}") from exc

        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError(f"yt-dlp reported success but {output_path.name} is missing")
        if on_progress:
            on_progress(1.0, "captured")
        return str(output_path)

    def probe_media(self, media_path: str) -> dict[str, float | int]:
        """Get authoritative metadata from the completed recording, not a live listing."""
        ffprobe = shutil.which("ffprobe")
        if not ffprobe:
            raise RuntimeError("ffprobe is required after a live capture; install the full ffmpeg package")
        try:
            completed = subprocess.run(
                [
                    ffprobe, "-v", "error", "-show_entries",
                    "format=duration:stream=codec_type,width,height,avg_frame_rate",
                    "-of", "json", media_path,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            info = json.loads(completed.stdout)
        except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Could not read captured video metadata: {exc}") from exc

        video = next(
            (stream for stream in info.get("streams", []) if stream.get("codec_type") == "video"),
            None,
        )
        duration = float((info.get("format") or {}).get("duration") or 0)
        if not video or duration <= 0:
            raise RuntimeError("Captured live stream has no readable video track")
        fps_text = str(video.get("avg_frame_rate") or "0/1")
        try:
            numerator, denominator = fps_text.split("/", 1)
            fps = float(numerator) / float(denominator)
        except (ValueError, ZeroDivisionError):
            fps = 0.0
        width, height = int(video.get("width") or 0), int(video.get("height") or 0)
        if fps <= 0 or width <= 0 or height <= 0:
            raise RuntimeError("Captured live stream has incomplete video metadata")
        return {"duration": duration, "fps": fps, "width": width, "height": height}

    def dependency_status(self) -> dict:
        return {
            "yt_dlp": getattr(yt_dlp.version, "__version__", "unknown"),
            "ffmpeg": self.has_ffmpeg,
        }
