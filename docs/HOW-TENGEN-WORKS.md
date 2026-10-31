# How Tengen Works — the whole thing, for the devs

You've been told "the AI wrote all of it and nobody knows how it works." This doc fixes that.
It's the plain-English tour of everything Tengen does today: what each piece is, how footage
turns into strategy, where it lives in the repo, and the decisions that are already baked in so
you don't re-argue them from scratch. Read it top to bottom once; after that it's a map.

> One-line version: **Tengen watches FRC match video, finds and follows all six robots, puts
> them on the field in real feet, reads the match clock, measures how each team plays, and turns
> that into profiles and counters.** It runs offline on a laptop, and there's a web version too.

---

## 1. The shape of it: three front-ends, one engine

There is **one analysis engine** and **three ways to use it**. This is the single most important
thing to understand, because every "where does X happen" question resolves to it.

```
                         ┌──────────────────────────────┐
                         │      ANALYSIS ENGINE           │
   video in ───────────► │  detect → track → locate →     │ ───► data out
   (broadcast, phone,    │  read clock → measure → profile │      (tracks, events,
    webcam, YouTube)     └──────────────────────────────┘       shots, strategy)
                                   ▲        ▲        ▲
                 ┌─────────────────┘        │        └─────────────────┐
         ┌───────────────┐          ┌───────────────┐          ┌───────────────┐
         │  DESKTOP APP   │          │   WEB APP      │          │  THE BOT /     │
         │  (PyQt, local) │          │  (browser)     │          │  SCRIPTS       │
         │  primary tool  │          │  view anywhere │          │  Discord, CI   │
         └───────────────┘          └───────────────┘          └───────────────┘
```

- **Desktop app** (`desktop/`, launched by `tengen.py` or `Launch Tengen.cmd`) — the primary
  tool. Runs the whole engine locally on your machine, GPU and all. Open a video, analyse it,
  scrub the timeline, fix mistakes, see the strategy, publish to the web.
- **Web app** (`web/`, a React/TypeScript site) — the same views in a browser, for people who
  aren't at a dev machine. It does **not** run the heavy model itself; it reads results that were
  already produced and published. Live at the GitHub Pages `/app/` link.
- **The API** (`ingest/`, a FastAPI service) — the server both the web app and the hosted copy
  talk to. It stores jobs, serves tracks/events/shots, and can run analysis itself on a machine
  that has the model.

The engine is a **pipeline of stages**. Everything below is those stages in order.

---

## 2. The pipeline, stage by stage

This is what happens to a video from the moment you hit "analyse."

### Stage 1 — Detection: *"where are the robots in this frame?"*
A trained **YOLO** neural network looks at each frame and draws a box around every robot it sees.
- Lives in `training/track_yolo.py` (the tracker) and is driven through
  `ingest/yolo_orchestrator.py`. The native/fast path is an RF-DETR/ONNX backend in `analysis/`.
- The model only looks at **part of the frame** — the `MODEL_CROP`. Broadcasts stack two camera
  panels and a score bar; feeding the whole frame to the model wastes it on the scoreboard and
  the second panel. Different events have different layouts, so a job can override the crop
  (`ABOVE_SCORE_BAR` for Space Coast's layout, `FULL_FRAME` for a phone recording with no
  overlay). This is why "it worked at Champs but not at the off-season event" happens — it's the
  crop, and it's a per-job setting now.

### Stage 2 — Tracking + identity: *"which box is which robot, over time?"*
Boxes in single frames become **tracks** — "robot #3 is this path through the whole match." The
hard part is **keeping identity through occlusion**: when a robot drives behind another or off
the panel, the tracker must re-attach the same ID when it reappears instead of inventing a new
robot.
- This is the thing GRFD flagged, and it's the known weak point. There are already two fixes in
  `track_yolo.py`: a **size gate** (a tiny false-positive box can't steal a real robot's ID) and
  **stale-identity recovery** (a robot that vanishes for a few seconds gets its old ID back when
  it returns). On the Champs Curie qm10 test, that took "all six robots tracked" from 25% of the
  match to ~68%.
- It's better, not solved. More labelled 2026 footage is the lever — see §6. Details in
  `docs/TRACKING-IDENTITY.md`.

### Stage 3 — Localisation: *"where on the real field is that, in feet?"*
A box in the video is in pixels. Strategy needs **field coordinates in feet**. Tengen reads the
field's **AprilTags** (the fiducial markers around the field) to compute a homography — the
math that maps "pixel here" to "15.2 ft, 8.1 ft on the field." Then it rotates the red alliance
180° onto blue, so a team's matches line up no matter which side they played. That's why the
top-down view in the hero image exists, and why heatmaps are even possible.
- Field geometry comes from the WPILib AprilTag layout, stored season-agnostically in
  `contracts/fields/` and `contracts/strategy/2026.json`.
- Caveat: a camera angle with **no readable tags** (some wide off-season shots) can't be
  calibrated, so that clip has tracks but no field positions. Not a bug — there's nothing to
  solve the math from.

### Stage 4 — The match clock: *"when is auto, teleop, endgame?"*
Tengen reads the **scoreboard clock** with OCR so phases are where the *referees* had them, not
where your clip happens to start. It votes across readings, handles the frozen clock between
periods, and knows the 2026 structure (auto 20s, teleop 140s including a 30s endgame).
- `ingest/scoreboard.py` (reads the bar) and `ingest/strategy/clock.py` (works out the timing).
- Why it matters: "shoots a lot in the first 15 seconds" is only meaningful if you know where
  second 0 actually is. Without the clock, every time-based stat drifts by however long the clip's
  lead-in was.

### Stage 5 — Measurement: *"what did each robot actually do?"*
Now the behaviour. From the located, time-aligned tracks, Tengen extracts **events** and
**features**: shooting spots and how long a robot dwells there, volley timing (shots per trip),
start position and auto path, which zones and lanes it uses, defense and who it marked, when it
breaks for endgame. There are ~29 catalogued features.
- `ingest/strategy/features.py`, `ingest/strategy/timeline.py`, `ingest/action_extraction.py`,
  `ingest/extract_events.py`, shot stats in `ingest/shot_stats.py`.
- Every feature is defined once in `ingest/strategy/catalogue.py` with a label and a date it was
  added, which is what the "features built" status badge counts.

### Stage 6 — Profiles + counters: *"how does this team play, and how do we beat it?"*
Features from many matches become a **team profile**, and profiles become **counters**. The rule
that keeps this honest: **every claim carries how many matches it rests on, and links back to the
video.** "Team 254 shoots from the left corner" means a lot more with "across 9 matches, here are
the clips" attached.
- `ingest/strategy/profile.py`. You can drive the whole layer from the command line:
  `python -m ingest.strategy clock|timeline|features|profile|evaluate <RUN>`.

That's the engine. Output is four kinds of data: **tracks** (who was where), **events** (what
happened when), **shots** (launches and outcomes), and **strategy** (profiles/counters).

---

## 3. The data model — the four files, and the contract

Every analysed match produces the same files. If you learn these four names you can read any run
folder under `data/`:

| File | What's in it |
|---|---|
| `tracks.jsonl` | one line per robot: its box path over time, team number, alliance |
| `events.jsonl` | one line per event: match_start, a shot, a phase change, etc. |
| `shots.jsonl` | launches with timing, which robot, outcome |
| `result.json` | run-level summary: frame rate, frames processed, etc. |

These shapes are **frozen by a contract** so the desktop app, the web app, and the API never
disagree about what a "track" is. That contract is **Doc 0** (`docs/frc-scouting-0-contract.md`),
and the current schema is **version 3**. When you see `schema_version: 3` in the data, that's why.
Change a shape without bumping the contract and something three layers away breaks silently — the
contract exists specifically so that doesn't happen. There are machine-readable copies in
`contracts/`.

The engine **thins** the data when publishing (positions sampled to 10 Hz, not every frame),
because the web doesn't need 30 fps of coordinates to draw a path. Storage math in §5.

---

## 4. The model — what "the model" actually is, and the registry

"The model" = the YOLO network from Stage 1. It has versions, and they are **numbered and
scored**, not vibes.
- Every version is in `training/registry/models.json` with its scores; `docs/MODELS.md` is the
  generated human-readable table; `tools/models.py` regenerates it.
- The headline score is **held-out recall**: against 400 frames from 25 events *no model trained
  on*, what share of robots does it find? v1 was 41%, the current v3 is ~64.5%. The whole status
  page (`tools/status.py` → GitHub Pages) tracks this over time, AnyPS5-style.
- Training code is in `training/` (`train_yolo.py`, plus RF-DETR and TensorFlow variants for
  different GPUs). The honest bottleneck is **labelled 2026 footage** — the detector barely sees
  this season yet. How to improve it is its own doc, `docs/IMPROVING-THE-MODEL.md`.

---

## 5. Hosting — where the web version lives, and the storage truth

This is the part the chat was confused about, so here it is straight.

**What's deployed right now:**
- **The API** runs on **Render** (free tier) at `tengen-api.onrender.com`.
- **The database** is **Supabase Postgres** — it stores the *data* (jobs, tracks, events, shots).
- **Video** goes to a **Supabase Storage** bucket, private, served by short-lived signed links.
- **The web app** is static files on **GitHub Pages** (`/Project-Tengen/app/`).

**"Are we uploading annotated videos to Supabase?" — yes, but understand what and why.**
- The *primary* thing stored is **position data** (the tracks), not video. GRFD is right that the
  client can redraw the match from positions — that's already the model, and it's cheap.
- Video is the **heavy optional extra**. The web can't stream the original from YouTube (YouTube
  refuses cloud servers — that "sign in to confirm you're not a bot" wall), and a local recording
  would have no video online at all. So when you publish, `tools/publish.py` encodes a **small
  480p copy** (~15–20 MB for a match) and uploads *that*. It's the "watch the clip with the boxes
  on it in the browser" feature, nothing more.
- **The ~100-match ceiling** everyone's worried about is purely the video copies filling the free
  Supabase bucket (50 MB-per-file limit, limited total). The position data is tiny by comparison.
  So the storage decision in the plan is really "do we keep paying the video tax, and if so
  where" — the data itself is not the problem. (See the plan, §Storage.)

**Security — what's actually in place** (pistol's "weak sites get owned in 24h" is fair, so here's
the current posture, in `ingest/hosted.py`):
- Reading and writing both require a **key** sent in a header (`X-Tengen-Key`), never in the URL
  (URLs end up in server logs — that exact leak happened once and got fixed).
- Separate **read** vs **write** keys; the read key can't publish.
- A **publish-only** mode so the hosted server will serve results but refuse to run new analysis.
- Video links are **signed, expiring tokens** scoped to one match, not the key.
- Secrets live in Render/`.env`, never in the repo. The repo's `.gitignore` keeps weights, video,
  `.env`, and service-account JSON out.
- This is a sane baseline, not a finished audit. Hardening is a named line in the plan.

Full detail: `docs/HOSTING.md`, `docs/WEB-AND-API.md`, `docs/media-streaming.md`.

---

## 6. The supporting cast

- **Discord bot** (`tools/discord-bot/`) — dependency-free REST client (`discord_api.py`). Posts
  the model scoreboard and status graphics, announces releases, manages the server. `backup.py`
  takes a full server backup. The bot has Administrator; that's power to respect.
- **Corrections** — when you fix a team number or delete a bad event in either app, that's stored
  as a **correction** (`ingest/corrections.py`) layered over the raw analysis, so the model's
  mistakes get overridden without re-running anything and the fix reaches the strategy layer.
- **Field/season contracts** (`contracts/`) — all the season-specific numbers (field size, tag
  positions, phase lengths, zones) live here as data, so a new season is a new JSON, not a code
  rewrite. This is what "season-agnostic" means in practice.
- **Fixtures + tests** (`fixtures/`, `*/tests/`) — there's a synthetic match used to test the
  whole chain end to end, plus hundreds of unit tests. The design rule the project keeps proving:
  **the real failures are between components, not inside them** — so the tests that matter most are
  the ones across the seams (desktop↔API↔web agreeing on a contract).

---

## 7. Where to look first (cheat sheet)

| You want to… | Start here |
|---|---|
| Run the app | `tengen.py`, `docs/DESKTOP.md` |
| Understand the engine's stages | `ingest/yolo_orchestrator.py`, `ingest/orchestrator.py` |
| See what the API offers | routes in `ingest/main.py` + `ingest/hosted.py` |
| Touch detection/tracking | `training/track_yolo.py`, `docs/TRACKING-IDENTITY.md` |
| Touch strategy | `ingest/strategy/`, `docs/STRATEGY-PLAN.md` |
| Understand a data shape | `docs/frc-scouting-0-contract.md` (Doc 0), `contracts/` |
| Improve the model | `docs/IMPROVING-THE-MODEL.md`, `training/registry/` |
| Change the web app | `web/src/` (`App.tsx`, `views/`, `api/`) |
| Hosting / deploy | `docs/HOSTING.md`, `ingest/hosted.py` |

If one sentence has to survive: **footage → six tracked robots → real-field positions →
clock-aligned behaviour → profiles and counters, one engine behind three front-ends, with every
data shape pinned by Doc 0.**
