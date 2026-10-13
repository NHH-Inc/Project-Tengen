"""Post the model registry and status graphics to Discord.

    python tools/discord-bot/post_models.py          # preview: prints what would be posted
    python tools/discord-bot/post_models.py --send   # post it

Two places, because #model-training is private to the team:

  #model-scoreboard  public and read-only. Holds ONE scoreboard message: the table and the
                     status picture (treemaps and history). The bot deletes its own previous scoreboard
                     before posting, so the channel is always exactly the current state.
  #model-training    private. A fuller update for the team: every model's facts and the
                     newest model's training curves. Appended, never deleted, as a record.

The channel is created on first use (read-only for @everyone, under THE PROJECT) if it does not
exist. Everything posted is generated from training/registry; run `python -m tools.models
evaluate` and `python -m tools.status` first so the pictures and numbers are current.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import discord_api as api  # noqa: E402
from tools.models import load, models_with_scores  # noqa: E402

SCOREBOARD = "model-scoreboard"
TRAINING = "model-training"
STATUS = ROOT / "docs" / "status"
REPO = "https://github.com/NHH-Inc/Project-Tengen"
VIEW, SEND, ADD_REACTIONS, THREADS = 1 << 10, 1 << 11, 1 << 6, (1 << 35) | (1 << 36)
BLUE = 0x2F81F7


def pct(value) -> str:
    return "—" if value is None else f"{value:.1%}"


def scoreboard_embed(models: list[dict]) -> dict:
    current = next(m for m in models if m["status"] == "current")
    lines = ["```", f"{'model':8}{'all 6 found':>12}{'mean':>7}{'blind':>7}  trained", "-" * 46]
    for m in reversed(models):
        h = m["heldout"] or {}
        star = "*" if m["status"] == "current" else " "
        lines.append(f"{star}{m['id']:7}{pct(h.get('full_share')):>12}"
                     f"{(format(h['mean_robots'], '.2f') if h else '—'):>7}"
                     f"{(str(h['blind_frames']) if h else '—'):>7}  {m['trained_on']}")
    lines.append("```")
    tracking = (current.get("pipeline") or {})
    fields = [
        {"name": "Current model", "value": f"**{current['id']}** · {current['architecture']} · "
                                           f"by {current['trained_by']}", "inline": True},
        {"name": "Held-out: all six found", "value": pct(current["heldout"]["full_share"]), "inline": True},
    ]
    if tracking:
        fields.append({"name": "Real match: all six tracked",
                       "value": f"{tracking['all_six_tracked_share']:.0%} ({tracking['match']})", "inline": True})
    return {
        "title": "Tengen model scoreboard",
        "url": f"{REPO}/blob/main/docs/MODELS.md",
        "color": BLUE,
        "description": ("Every robot detector, scored on the same 400 held-out frames of real 2026 "
                        "broadcasts that no model trained on. **all 6 found** is the share of frames "
                        "where the model finds every robot.\n" + "\n".join(lines)),
        "fields": fields,
        "image": {"url": "attachment://status.png"},
        "footer": {"text": "Generated from training/registry · python -m tools.status"},
    }


def training_update(models: list[dict]) -> tuple[str, list[dict], list[Path]]:
    current = next(m for m in models if m["status"] == "current")
    best_val = max(models, key=lambda m: m["val"]["map50_95"])
    content = ("**Model registry update.** Every model is now numbered and scored on the same held-out "
               f"pack; the public scoreboard is in #{SCOREBOARD}. Details: {REPO}/blob/main/docs/MODELS.md")
    embeds = []
    for m in reversed(models):
        h = m["heldout"] or {}
        v = m["val"]
        description = (f"{m['architecture']}, {m['epochs']} epochs at {m['train_size_px']} px, trained "
                       f"{m['trained_on']} by {m['trained_by']}.\n**Data** {m['dataset']}.\n"
                       f"**Held-out** all six found {pct(h.get('full_share'))}, mean {h.get('mean_robots', '—')}, "
                       f"{h.get('blind_frames', '—')} blind.\n"
                       f"**Own val** P {v['precision']:.3f} · R {v['recall']:.3f} · mAP50 {v['map50']:.3f} · "
                       f"mAP50-95 {v['map50_95']:.3f}\n{m['notes']}")
        embeds.append({"title": f"{m['id']} — {m['name']}" + (" (current)" if m is current else ""),
                       "description": description[:4000],
                       "color": BLUE if m is current else 0x6E7681})
    embeds.append({
        "title": "Worth knowing",
        "color": 0xD29922,
        "description": (f"Val mAP does not compare across versions: each model has its own val split. "
                        f"{best_val['id']} has the best val mAP50-95 ({best_val['val']['map50_95']:.3f}) but "
                        f"not the best held-out score. And v2's 1,017 machine-labelled frames took it from "
                        f"v1's 41.0% to 13.8%: unreviewed labels made the model worse. Next: the loop in "
                        f"{REPO}/blob/main/docs/IMPROVING-THE-MODEL.md"),
        "image": {"url": "attachment://status.png"},
    })
    files = [STATUS / "status.png"]
    run = current.get("run")
    if run and (ROOT / run).is_file():
        files.append(ROOT / run)
    return content, embeds[:10], files


def ensure_scoreboard(guild_id: str, send: bool) -> dict | None:
    existing = api.channel_named(guild_id, SCOREBOARD)
    if existing or not send:
        return existing
    project = api.channel_named(guild_id, "THE PROJECT")
    created = api.request("POST", f"/guilds/{guild_id}/channels", {
        "name": SCOREBOARD,
        "type": 0,
        "parent_id": project["id"] if project else None,
        "topic": "Every robot detector, scored on the same held-out frames. Updated by the bot.",
        # Read-only for everyone, like #releases and #roadmap. The bot posts with Administrator.
        "permission_overwrites": [{"id": guild_id, "type": 0, "allow": str(VIEW),
                                   "deny": str(SEND | ADD_REACTIONS | THREADS)}],
    }, reason="Model scoreboard: public, read-only")
    print(f"created #{SCOREBOARD}")
    return created


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--send", action="store_true", help="actually post (default: preview only)")
    parser.add_argument("--new-team-post", action="store_true",
                        help="post a new team update instead of editing the last one (e.g. for a new model)")
    args = parser.parse_args(argv)

    models = models_with_scores()
    missing = [m["id"] for m in models if not m["heldout"]]
    if missing:
        print(f"note: no held-out score yet for {', '.join(missing)}")
    for name in ("status.png",):
        if not (STATUS / name).is_file():
            raise SystemExit(f"docs/status/{name} is missing: run python -m tools.status first")

    board = scoreboard_embed(models)
    content, embeds, files = training_update(models)
    if not args.send:
        sys.stdout.reconfigure(encoding="utf-8")
        print(f"#{SCOREBOARD}: {board['title']}\n{board['description']}\n" +
              "\n".join(f"  {f['name']}: {f['value']}" for f in board["fields"]) +
              "\n  attachments: status.png")
        print(f"\n#{TRAINING}: {content}\n" + "\n".join(f"  [{e['title']}]" for e in embeds) +
              f"\n  attachments: {', '.join(f.name for f in files)}")
        print("\nPreview only. Re-run with --send to post.")
        return 0

    guild_id = api.guild()["id"]
    me = api.request("GET", "/users/@me")["id"]
    channel = ensure_scoreboard(guild_id, send=True)
    for message in api.messages(channel["id"], 50):
        if message["author"]["id"] == me:
            api.request("DELETE", f"/channels/{channel['id']}/messages/{message['id']}")
    api.request("POST", f"/channels/{channel['id']}/messages", {"embeds": [board]},
                files=[STATUS / "status.png"])
    print(f"posted the scoreboard to #{SCOREBOARD}")

    training = api.channel_named(guild_id, TRAINING)
    if training is None:
        print(f"#{TRAINING} not found; skipped the team update")
        return 0
    payload = {"content": content, "embeds": embeds}
    previous = next((m for m in api.messages(training["id"], 50)
                     if m["author"]["id"] == me and m["content"].startswith("**Model registry update.**")), None)
    if previous and not args.new_team_post:
        # Edit the bot's last update in place: the team channel keeps people's own messages, and a
        # second near-identical post every time the numbers move would bury them. `attachments: []`
        # drops the old files; the new ones are attached.
        api.request("PATCH", f"/channels/{training['id']}/messages/{previous['id']}",
                    {**payload, "attachments": []}, files=files)
        print(f"updated the team post in #{TRAINING}")
    else:
        api.request("POST", f"/channels/{training['id']}/messages", payload, files=files)
        print(f"posted the team update to #{TRAINING}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
