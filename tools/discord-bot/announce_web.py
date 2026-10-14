"""One-off: announce the web version of Tengen (2026-10-14).

    python tools/discord-bot/announce_web.py          # preview
    python tools/discord-bot/announce_web.py --send   # post

Public #announcements gets what it is and how to get in; private #dev-general gets how it works
and how to publish. Pings use the opt-in Ping roles, never @everyone. No key is ever posted:
people get them from a maintainer directly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import discord_api as api  # noqa: E402

APP = "https://nhh-inc.github.io/Project-Tengen/app/"
SITE = "https://nhh-inc.github.io/Project-Tengen/"
REPO = "https://github.com/NHH-Inc/Project-Tengen"
GREEN, BLUE = 0x2EA043, 0x2F81F7

PUBLIC = {
    "role": "Ping: Releases",
    "content": "<@&{role}> **Tengen is on the web.** 🎉",
    "embeds": [{
        "title": "Open the Tengen app",
        "url": APP,
        "color": GREEN,
        "description": (
            "Tengen watches FRC match video and works out **how each team plays, and how to beat it**. "
            "Analysed matches can now be viewed from anywhere, on any device, in a browser.\n\n"
            f"**[Open the app]({APP})** · [Website]({SITE}) · [GitHub]({REPO})"),
        "fields": [
            {"name": "Strategy", "inline": False,
             "value": "For each match: every robot's shots, shooting spots, how long it sits there, volley "
                      "timing, defense and starting position, drawn on the field."},
            {"name": "Team profiles", "inline": False,
             "value": "Each team across all its matches: tendencies, a prediction for its next match, and "
                      "counters, each with the evidence behind it."},
            {"name": "Getting in", "inline": False,
             "value": "Scouting data is private to the team. Ask a maintainer for the **team key**, then paste "
                      "it under *Team key* in the app's sidebar. Never post the key in a channel."},
            {"name": "Good to know", "inline": False,
             "value": "The server sleeps when nobody has used it for 15 minutes, so the first visit after that "
                      "takes about a minute to load. Problems? Ask in #support."},
        ],
        "image": {"url": "attachment://status.png"},
        "footer": {"text": "Model and feature progress: " + SITE + "progress.html"},
    }],
    "files": [ROOT / "docs" / "status" / "status.png"],
}

DEV = {
    "role": "Ping: Dev Updates",
    "content": "<@&{role}> **The web version is live, and it changes how results get to the team.**",
    "embeds": [{
        "title": "How it works now",
        "color": BLUE,
        "description": (
            "Analysis stays on a laptop or desktop (YOLO needs real compute). When a match is done, you "
            "**publish** its results, and the hosted server shows them to everyone, with every team PC "
            "switched off. No PC of ours is a server.\n\n"
            "```\nlaptop / desktop ──publish──▶ API (Render) ──▶ database (Supabase)\n"
            "                                   ▲\n                  web app (GitHub Pages)\n```"),
        "fields": [
            {"name": "Publish a match", "inline": False,
             "value": "Desktop app: open the run, click **Publish to web**.\nCommand line:\n"
                      "```\npython -m tools.publish --setup https://tengen-api.onrender.com   (once)\n"
                      "python -m tools.publish data/desktop/runs/<id>\n```"
                      "Only results go up, never video. Team numbers you set in the desktop app go with it."},
            {"name": "Keys", "inline": False,
             "value": "**Read key** to view, **write key** to edit or publish. Ask Justin for them, and keep them "
                      "out of channels, commits and screenshots. The server refuses everything if its keys "
                      "are missing, rather than opening up."},
            {"name": "Limits", "inline": False,
             "value": "Free tier: ~1 min wake-up after 15 min idle; room for ~150 matches; YouTube matches play "
                      "in the browser, our own recordings show data without video. The server only shows "
                      "published matches: it cannot analyse."},
            {"name": "Needs eyes", "inline": False,
             "value": "• **Robert**: review the identity fix in `training/track_yolo.py` (fe374d1). Six robots "
                      "tracked 68% of a match now, was 25%.\n"
                      "• **Robert + Nathaniel**: confirm Doc 0's 2026-10-13 revision and the 2026 timing "
                      "(20 / 140 / 30, endgame inside teleop)."},
            {"name": "Read more", "inline": False,
             "value": f"[STRATEGY-PLAN.md]({REPO}/blob/main/docs/STRATEGY-PLAN.md) · "
                      f"[IMPROVING-THE-MODEL.md]({REPO}/blob/main/docs/IMPROVING-THE-MODEL.md) · "
                      f"[Doc 0]({REPO}/blob/main/docs/frc-scouting-0-contract.md) · "
                      f"[Progress]({SITE}progress.html)"},
        ],
    }],
    "files": [],
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--send", action="store_true")
    args = parser.parse_args(argv)
    guild = api.guild()["id"]
    roles = {r["name"]: r["id"] for r in api.request("GET", f"/guilds/{guild}/roles")}
    for channel_name, post in (("announcements", PUBLIC), ("dev-general", DEV)):
        channel = api.channel_named(guild, channel_name)
        role = roles[post["role"]]
        body = {"content": post["content"].format(role=role), "embeds": post["embeds"],
                # Only the named opt-in role may be pinged; nothing else in the text can mention anyone.
                "allowed_mentions": {"parse": [], "roles": [role]}}
        if not args.send:
            sys.stdout.reconfigure(encoding="utf-8")
            print(f"#{channel_name}: {body['content']}")
            for embed in post["embeds"]:
                print(f"  [{embed['title']}] {len(embed['description'])} chars, "
                      f"{len(embed.get('fields', []))} fields")
            continue
        api.request("POST", f"/channels/{channel['id']}/messages", body, files=post["files"] or None)
        print(f"posted to #{channel_name}")
    if not args.send:
        print("Preview only; --send to post.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
