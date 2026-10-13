# Discord server setup bot

Describes the server in a JSON file and makes Discord match it. Faster and less error-prone than
clicking through twenty channels, and re-runnable when the structure changes.

## Two rules, because this runs with admin on a real server

**It plans before it acts.** The default is a dry run — it prints exactly what it would do and
changes nothing. You need `--apply` to make it real.

**It never deletes.** Anything on the server but not in the config is listed as left alone. A
config file is a poor reason to destroy a channel with history in it.

Re-running is safe and boring: whatever already matches is skipped.

## Setup

Only you can do steps 1–3; they need your Discord account.

**1. Create the bot.** [discord.com/developers/applications](https://discord.com/developers/applications)
→ New Application → name it → **Bot** → **Reset Token** → copy it.

**2. Save the token** in `tools/discord-bot/.env` — git-ignored, never commit it:

```
DISCORD_TOKEN=paste-it-here
```

**3. Invite it.** OAuth2 → URL Generator → scope **bot** → permission **Administrator** → open the
generated URL and add it to your server.

Administrator is heavy-handed but simplest for a temporary bot that creates roles and sets
permissions. That is also why you should kick it afterwards.

**4. Get the server id.** Discord Settings → Advanced → Developer Mode on, then right-click the
server → Copy Server ID.

**5. Run it.**

```bash
pip install discord.py
cp server.example.json server.json      # then edit server.json
python bot.py --guild 123456789012345678
```

That prints the plan. Read it. Then:

```bash
python bot.py --guild 123456789012345678 --apply
```

**6. Kick the bot** when you are happy. Nothing here needs it to stay, and a dormant bot with
Administrator is a standing risk for no benefit. Re-invite it later if you want to restructure —
the config file is the thing worth keeping, not the bot.

## What the config can express

| key | meaning |
|---|---|
| `roles` | name, colour, whether it shows separately in the member list, mentionable, permissions |
| `categories` | a group of channels |
| `categories[].private_to` | hidden from `@everyone`, visible only to the named roles |
| `channels[].type` | `text` or `voice` |
| `channels[].topic` | the description under the channel name |
| `channels[].read_only_for` | those roles can read and react but not post — for announcements |

Roles are created before categories, so `private_to` can name a role the same file creates.

## Notes

**Existing things are not modified.** A role or channel that already exists is left exactly as it
is, even if the config disagrees. That is deliberate: silently rewriting permissions on a channel
people are using is worse than doing nothing and saying so. Delete it and re-run if you want it
rebuilt from the config.

**Order in the file is not order in Discord.** Discord sorts by its own position values; drag
things where you want them once, and re-runs will not move them.

## Webhooks and invites (done 2026-09-03)

**GitHub → Discord** is live on `NHH-Inc/Project-Tengen`:

| hook | events | channel |
|---|---|---|
| `674260503` | push, pull_request, issues, issue_comment, create, delete | `#github` |
| `674260505` | release | `#releases` |

Both point at a Discord webhook URL with `/github` appended — Discord parses GitHub's payload
format natively at that path, so no relay service is involved.

A Discord webhook URL is a credential: anyone holding it can post to that channel as anything.
They are not in this repository and were never printed. To rotate one, delete the webhook in the
channel's settings and re-register with `gh api repos/OWNER/REPO/hooks`.

**Invites.** The server has exactly one, permanent, and `@everyone` can no longer create more —
every member-generated link is another URL that can leak or outlive the person who made it. The
link is displayed as a voice channel at the top of START HERE that nobody can connect to: its name
*is* the link, so it sits in the sidebar and can be read and copied without being joinable.

## Channels with history win

The setup bot created `#general` and `#announcements` without knowing the server already had
them, with messages in. The fix was to keep the originals, move them into the new categories, and
delete the empty duplicates — never the other way round. Anything holding even one message is
left alone.

## Community mode and safety (done 2026-09-03)

Community is on, which is the gate for the server description, discovery and welcome screen.
Discord requires all of these together, so they were set in one edit:

| setting | value |
|---|---|
| rules channel | `#rules` |
| Discord's own notices | `#mod-log` |
| verification | Medium |
| explicit media filter | all members |

AutoMod runs three rules: **Blocked words** (Discord's maintained profanity / sexual-content /
slurs presets), **Invite links**, and **Mention spam** (limit 6). Maintainer and Developer are
exempt.

The presets are deliberate. A hand-written word list is stale the day it is committed and turns
moderation into an argument about edge cases; Discord maintains theirs.

## What still needs a person

**Emojis, stickers and soundboard sounds** can be uploaded through the API, but they need actual
files — art and audio this repository does not have. Drop images in and they can be added in one
pass; inventing placeholder art would just be clutter someone has to delete.

**Inviting other bots** cannot be scripted: adding a bot is an OAuth consent screen in a browser,
under your account, by design.

**Discovery traits** need the server to meet Discord's discovery thresholds (member count and
activity) before the fields become editable at all.

## Onboarding instead of reaction roles

Role selection is Discord's native onboarding, not a reaction-role bot. Reaction roles need a
process running forever to watch for reactions; kick the bot or let it crash and nobody can pick
a role again. Onboarding is served by Discord itself, keeps working with no bot present, and stays
editable in Server Settings — which is what makes it safe to remove this bot when setup is done.

Two prompts: **What brings you here?** (reveals relevant channels) and **What should we ping you
about?** (assigns the four `Ping: …` roles). Those roles carry **no permissions** — a notification
preference must never quietly grant access to anything.

`#frc-announcements` exists as a destination. Following has to be done from the *source* side:
open the FRC server's announcement channel and use Follow, then pick this channel. A bot cannot do
it, because the permission being exercised belongs to a member of that server, not to us.

## Posting model results

```powershell
.venv-desktop\Scripts\python tools\discord-bot\post_models.py          # preview
.venv-desktop\Scripts\python tools\discord-bot\post_models.py --send   # post
```

Posts the model scoreboard to the public, read-only `#model-scoreboard` (created on first use;
the bot replaces its own previous scoreboard so the channel is always the current state), and a
fuller update to the private `#model-training`. Run `python -m tools.models evaluate` and
`python -m tools.status` first. `discord_api.py` is the small REST client both use; it reads the
token from `.env` and never prints it.
