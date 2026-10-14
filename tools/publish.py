"""Publish an analysed match to the hosted Tengen, so the team can see it from anywhere.

    python -m tools.publish data/desktop/runs/<id>
    python -m tools.publish data/desktop/runs/<id> --youtube refM5LLkuJ8 --offset 0
    python -m tools.publish --setup https://tengen-api.onrender.com     # once per machine

The server address and the team's write key live in data/tengen-server.json, which is
git-ignored: `--setup` asks for the key without echoing it. Team numbers assigned in the desktop
app are applied to the tracks before upload, so the website shows the same teams. Only the
results go up, never the video: a run from a YouTube match (or `--youtube`) streams from YouTube
on the website, and a local recording stays on the laptop that made it.
"""

from __future__ import annotations

import argparse
import getpass
import json
import re
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "data" / "tengen-server.json"
FILES = ("tracks.jsonl", "events.jsonl", "shots.jsonl", "result.json", "match_clock.json",
         "goal_entries.jsonl", "ball_scouting.config.json")
YOUTUBE_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")


def load_config() -> dict:
    if not CONFIG.is_file():
        raise SystemExit("No server set up. Run: python -m tools.publish --setup <server URL>")
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def setup(url: str) -> None:
    key = getpass.getpass("Team write key (not shown): ").strip()
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(json.dumps({"url": url.rstrip("/"), "key": key}, indent=1) + "\n", encoding="utf-8")
    print(f"Saved to {CONFIG.relative_to(ROOT)} (git-ignored).")


def corrected_tracks(run: Path) -> bytes:
    """tracks.jsonl with the desktop app's team assignments applied."""
    tracks = [json.loads(l) for l in (run / "tracks.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    corrections = run / "desktop-corrections.json"
    teams = {}
    if corrections.is_file():
        for row in json.loads(corrections.read_text(encoding="utf-8")):
            if row.get("scope") == "track" and "team" in (row.get("fields") or {}):
                teams[int(row["target"])] = row["fields"]["team"]
    for track in tracks:
        if track["track_id"] in teams:
            track["team"] = teams[track["track_id"]]
    return "".join(json.dumps(t) + "\n" for t in tracks).encode()


def job_record(run: Path, youtube: str | None, offset: float | None) -> dict:
    job = json.loads((run / "job.json").read_text(encoding="utf-8"))
    guess = job.get("video_id") or ""
    youtube = youtube or (guess if YOUTUBE_ID.match(guess) else None)
    return {
        "job_id": job["job_id"], "match_id": job.get("match_id"), "season": job.get("season", 2026),
        "youtube_id": youtube, "start_offset": offset if offset is not None else job.get("start_offset", 0.0),
        "duration": job.get("duration"), "fps": job.get("fps"), "width": job.get("width"),
        "height": job.get("height"), "alliances": job.get("alliances"),
    }


def multipart(fields: dict[str, str], files: list[tuple[str, bytes]]) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    parts = []
    for name, value in fields.items():
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode())
    for filename, data in files:
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"files\"; filename=\"{filename}\"\r\n"
                     f"Content-Type: application/octet-stream\r\n\r\n".encode() + data + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def publish(run: Path, youtube: str | None = None, offset: float | None = None) -> dict:
    config = load_config()
    if not (run / "tracks.jsonl").is_file():
        raise SystemExit(f"{run} has no tracks.jsonl: analyse it first")
    files = [("tracks.jsonl", corrected_tracks(run))]
    files += [(name, (run / name).read_bytes()) for name in FILES[1:] if (run / name).is_file()]
    body, content_type = multipart({"job": json.dumps(job_record(run, youtube, offset))}, files)
    request = urllib.request.Request(f"{config['url']}/api/publish", data=body, method="POST", headers={
        "Content-Type": content_type, "X-Tengen-Key": config["key"], "User-Agent": "tengen-publish"})
    try:
        # A free server asleep takes up to a minute to wake.
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        raise SystemExit(f"Publish failed: {error.code} {error.read().decode(errors='replace')[:300]}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tools.publish", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", nargs="?", type=Path)
    parser.add_argument("--setup", metavar="URL", help="save the server address and team key")
    parser.add_argument("--youtube", help="YouTube id the run's video came from, so the website can play it")
    parser.add_argument("--offset", type=float, help="seconds into that YouTube video where the clip starts")
    args = parser.parse_args(argv)
    if args.setup:
        setup(args.setup)
        return 0
    if not args.run:
        parser.error("give a run folder, or --setup URL")
    summary = publish(args.run, args.youtube, args.offset)
    config = load_config()
    print(f"Published {summary['match_id'] or summary['job_id']}: {summary['boxes']} boxes, "
          f"{len(summary['artifacts'])} files. View it at {config.get('site', config['url'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
