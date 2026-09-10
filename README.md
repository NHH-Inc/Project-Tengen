# Project Tengen

Watches recorded FRC match video and produces per-robot scouting data.

> "FRC" throughout these docs means the competition — FRC matches, FRC robots, FRC fields.
> The project itself is Tengen. Environment variables keep their `FRC_` prefix on purpose:
> renaming them would break every teammate's `.env` for no benefit.

```
[3] web  ──HTTP──▶  [2] ingest  ──exec──▶  [1] analysis
                         │                       │
                         ├── yt-dlp              └── events.jsonl, tracks.jsonl
                         ├── TBA API
                         └── Postgres
```

Three components, one repo, **no cross-imports**. Component 3 only ever talks to component 2 over
HTTP; component 2 runs component 1 as a subprocess. `docs/frc-scouting-0-contract.md` is the only
shared surface, and nothing in `/contracts/` changes without all three of us agreeing.

| Dir | Component | Language | Owns |
|---|---|---|---|
| `analysis/` | 1 | C++ | Detection, tracking, OCR, event extraction |
| `ingest/` | 2 | Python | yt-dlp, TBA/YouTube APIs, job queue, HTTP API, the database |
| `web/` | 3 | TypeScript | UI, player, overlay, corrections, Sheets export |
| `contracts/` | — | — | Shared schemas + per-year season configs. **Owned by everyone.** |
| `fixtures/` | — | — | Golden test data. **Owned by everyone.** |
| `training/` | — | Python | RF-DETR detector training. Runs on an NVIDIA GPU only. |
| `docs/` | — | — | Everything below |

## Which doc do I want?

| I want to… | Read |
|---|---|
| **Understand the whole project** | [docs/HANDOFF.md](docs/HANDOFF.md) |
| **Know what to do next** | [docs/PLAN.md](docs/PLAN.md) |
| **Just run it** | [docs/RUNNING.md](docs/RUNNING.md) |
| **Understand or do model training** | [docs/TRAINING.md](docs/TRAINING.md) |
| **Get frames labelled by people** | [docs/LABELLING.md](docs/LABELLING.md) |
| **Host it / know where videos and data live** | [docs/HOSTING.md](docs/HOSTING.md) |
| **Set up the one shared instance** | [docs/CENTRAL-SETUP.md](docs/CENTRAL-SETUP.md) |
| **Know why something is the way it is** | [docs/DECISIONS.md](docs/DECISIONS.md) |
| **Write code against the data formats** | [docs/frc-scouting-0-contract.md](docs/frc-scouting-0-contract.md), then [contracts/README.md](contracts/README.md) |
| **Understand the player / overlay timing rules** | [docs/media-streaming.md](docs/media-streaming.md) |

## Fastest possible start

Nothing installed, no backend, no database — the whole UI against the golden fixtures:

```powershell
cd web
npm install
npm run dev
```

Open `http://localhost:5173`. You get a real 152-second clip with real tracks and events. This is
the right way to work on the web app, and the right way to see what the project does.

## Review robot boxes in competition images

The standalone Python labeler opens local image/JSON pairs, overlays the JSON boxes, and autosaves
red/blue corrections back to the matching JSON file:

```powershell
python -m pip install -r requirements-labeler.txt
python robot_box_labeler.py "C:\path\to\competition-images"
```

Full input-format and editing details: [docs/ROBOT-BOX-LABELER.md](docs/ROBOT-BOX-LABELER.md).

## The real thing

```powershell
.\run.ps1 setup
.\run.ps1 full
```

`run.ps1` is the single entry point. `setup` installs everything; `full` starts ingest and the web
app together. Ports are the doc 0 defaults — web `5173`, ingest `8080`, Postgres `5432`.

| Command | Does |
|---|---|
| `.\run.ps1 setup` | One-time install of every dependency |
| `.\run.ps1 doctor` | Tells you what is missing and how to fix it |
| `.\run.ps1 web` | Web app only, fixture data, no backend |
| `.\run.ps1 api` | Ingest service only |
| `.\run.ps1 full` | Both, wired together |
| `.\run.ps1 check` | Every test and contract check |
| `.\run.ps1 clean` | Reclaim disk from cached segments |

Full walkthrough, prerequisites, and troubleshooting: [docs/RUNNING.md](docs/RUNNING.md).

## The rules that keep getting rediscovered

These are in doc 0, but they are the ones people break:

- **Corrections never overwrite model output.** `?raw=true` must always return exactly what the
  model said. A correction composes on read.
- **`phase` is derived** from `contracts/seasons/<year>.json`. Never hardcode 15/135/20.
- **Tracks carry a required `gaps` array.** Never interpolate across a gap, and never split a
  track at one — a gap means "the camera cut away," not "a new robot."
- **Aggregates are queried, never stored.**
- **Anything unrecognised is a bug, not a fallback.** Drop the row and report it; do not coerce it.
- **`goal` is season-scoped** — legal values come from the season config's `goals` array, not from
  a doc 0 enum.
- Point values stay zero until the 2026 game is public.

## Configuration

Component 2 reads these from the environment. All are optional — the service degrades rather than
fails.

| Variable | Effect when unset |
|---|---|
| `DATABASE_URL` | SQLite at `./frc_scouting.db`. Set to Postgres in production. |
| `TBA_API_KEY` | `alliances` and `tba_score` stay null; no scores to check accuracy against. |
| `SHEETS_SPREADSHEET_ID` + `GOOGLE_APPLICATION_CREDENTIALS` | `POST /api/export/sheets` returns 503 instead of silently not writing. |
| `FRC_DATA_DIR` | `./data` — segments in `data/segments/`, job output in `data/jobs/`. |
| `ANALYSIS_BINARY` | `./analysis/build/bin/analysis` |
| `FRC_ANALYSIS_BACKEND` | `auto` — selects YOLO + ByteTrack when configured, otherwise native |
| `YOLO_PYTHON` | unset — dedicated Python executable containing Ultralytics, so native is used unless set |
| `YOLO_MODEL_PATH` | trained YOLO `.pt` path; defaults to the local FRC robot checkpoint when present |
| `FRC_YOLO_TRACKER` | `bytetrack` |
| `FRC_YOLO_CONFIDENCE` / `FRC_YOLO_IMAGE_SIZE` / `FRC_YOLO_FRAME_STRIDE` / `FRC_YOLO_DEVICE` | `0.25` / `960` / `1` / `0`; every source frame by default, enforced when ball scouting is configured |
| `FRC_YOLO_REID_MEMORY_SECONDS` | `5.0` — appearance-backed identity memory after a robot disappears |
| `FRC_YOLO_REID_APPEARANCE_THRESHOLD` / `FRC_YOLO_REID_SCORE_MARGIN` | `0.60` / `0.08` — merge evidence and ambiguity gates |
| `FRC_YOLO_REID_MAX_DISTANCE` / `FRC_YOLO_REID_MAX_SPEED` | `0.60` / `0.75` — normalized image-space motion gates when field calibration is unavailable |
| `FRC_YOLO_REID_EDGE_THRESHOLD` | `0.10` — boundary band for edge-consistent re-entry |
| `FRC_YOLO_REID_TEMPLATE_GALLERY_SIZE` / `FRC_YOLO_REID_TEMPLATE_CONFIRMATION_FRAMES` | `5` / `3` — protected appearance gallery settings |
| `FRC_YOLO_REID_TEMPLATE_MIN_CONFIDENCE` | `0.50` — minimum detector confidence for a new gallery exemplar |
| `FRC_YOLO_REID_ALLIANCE_LOCK_SECONDS` / `FRC_YOLO_REID_ALLIANCE_LOCK_MARGIN_SECONDS` | `5.0` / `2.0` — trusted observation time and evidence lead required before an identity's alliance becomes permanent |
| `FRC_YOLO_STARTUP_POSITION_SECONDS` / `FRC_YOLO_STARTUP_SPLIT_X` | `2.0` / `0.50` — during the opening window, image-left authoritatively locks blue and image-right locks red; set seconds to `0` to disable |
| `FRC_BALL_SCOUTING_CONFIG` | unset disables the separate HSV ball/shot pipeline; the checked-in `.env.example` enables it with the starter config, which can be replaced by a camera-specific file; see [docs/BALL-SCOUTING.md](docs/BALL-SCOUTING.md) |
| `FRC_AUTO_HOMOGRAPHY` | `1` — fit a carpet mapping from steady field AprilTags for each job |
| `FRC_HOMOGRAPHY_CONFIG` | optional fixed camera calibration JSON; takes precedence over automatic calibration |
| `FRC_DEFAULT_SEASON` | `2026` |
| `FRC_MIN_FREE_GB` | `10` — refuses to start a download below this. |
| `FRC_SEGMENT_GRACE_DAYS` | `7` — how long a completed job's video survives before `clean` reclaims it. |

Never commit `ingest/.env`, `data/`, ONNX weights, service-account JSON, or Hugging Face tokens.

## Checks

```powershell
.\run.ps1 check
```

Which runs: web typecheck + build + fixture validation, 71 Contract E smoke checks, and 30 ingest
unit tests.
