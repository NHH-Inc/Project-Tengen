"""A small, dependency-free client for the Discord REST API, using the bot token in .env.

bot.py builds the server from a config and needs discord.py's gateway. Everything else the
project asks of the bot -- reading a channel, auditing permissions, posting model results --
is a handful of REST calls, so it goes through here with only the standard library.

The token is read from DISCORD_TOKEN or the git-ignored .env beside this file and is never
printed or logged.
"""

from __future__ import annotations

import json
import mimetypes
import os
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
API = "https://discord.com/api/v10"
USER_AGENT = "TengenBot (https://github.com/NHH-Inc/Project-Tengen, 1)"


def token() -> str:
    value = os.environ.get("DISCORD_TOKEN", "")
    env = HERE / ".env"
    if not value and env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("DISCORD_TOKEN="):
                value = line.split("=", 1)[1].strip()
    if not value:
        raise SystemExit("No DISCORD_TOKEN in the environment or tools/discord-bot/.env")
    return value


def request(method: str, path: str, body: dict | None = None, files: list[Path] | None = None,
            reason: str | None = None):
    """One REST call. Waits out rate limits rather than failing on them."""
    headers = {"Authorization": f"Bot {token()}", "User-Agent": USER_AGENT}
    if reason:
        headers["X-Audit-Log-Reason"] = reason
    data = None
    if files:
        boundary = uuid.uuid4().hex
        parts = [f"--{boundary}\r\nContent-Disposition: form-data; name=\"payload_json\"\r\n"
                 f"Content-Type: application/json\r\n\r\n{json.dumps(body or {})}\r\n".encode()]
        for index, upload in enumerate(files):
            kind = mimetypes.guess_type(upload.name)[0] or "application/octet-stream"
            parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"files[{index}]\"; "
                         f"filename=\"{upload.name}\"\r\nContent-Type: {kind}\r\n\r\n".encode()
                         + upload.read_bytes() + b"\r\n")
        parts.append(f"--{boundary}--\r\n".encode())
        data = b"".join(parts)
        headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
    elif body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    for _ in range(5):
        req = urllib.request.Request(API + path, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req) as response:
                raw = response.read()
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as error:
            if error.code == 429:
                time.sleep(float(json.loads(error.read() or b"{}").get("retry_after", 1.0)) + 0.1)
                continue
            raise RuntimeError(f"{method} {path} -> {error.code}: {error.read().decode(errors='replace')[:300]}")
    raise RuntimeError(f"{method} {path}: still rate limited")


def guild() -> dict:
    guilds = request("GET", "/users/@me/guilds")
    if len(guilds) != 1:
        raise SystemExit(f"The bot is in {len(guilds)} servers; pass the one you mean.")
    return guilds[0]


def channels(guild_id: str) -> list[dict]:
    return request("GET", f"/guilds/{guild_id}/channels")


def channel_named(guild_id: str, name: str) -> dict | None:
    return next((c for c in channels(guild_id) if c["name"] == name), None)


def messages(channel_id: str, limit: int = 500) -> list[dict]:
    """Newest first, paging back until `limit` or the start of the channel."""
    out, before = [], None
    while len(out) < limit:
        path = f"/channels/{channel_id}/messages?limit=100" + (f"&before={before}" if before else "")
        page = request("GET", path)
        if not page:
            break
        out.extend(page)
        before = page[-1]["id"]
    return out[:limit]
