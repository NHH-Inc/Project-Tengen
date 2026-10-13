# The web app and the ingest service

The desktop app ([DESKTOP.md](DESKTOP.md)) is Tengen's primary app. The web app and the ingest
service it talks to remain for viewing, sharing between our own machines, YouTube and live-stream
jobs, and Google Sheets export. This page is what used to be most of the README.

> "FRC" throughout these docs means the competition: FRC matches, FRC robots, FRC fields. The
> project itself is Tengen. Environment variables keep their `FRC_` prefix on purpose: renaming
> them would break every teammate's `.env` for no benefit.

```
[3] web  ──HTTP──▶  [2] ingest  ──exec──▶  analysis engine (Python; C++ frozen)
                         │                       │
                         ├── yt-dlp              └── events.jsonl, tracks.jsonl, shots.jsonl
                         ├── TBA API
                         └── Postgres
```

The web app only ever talks to ingest over HTTP; ingest runs the analysis engine as a subprocess.
[Doc 0](frc-scouting-0-contract.md) is the shared contract, and nothing in `/contracts/` changes
without all three of us agreeing.

| Dir | Component | Language | Owns |
|---|---|---|---|
| `analysis/` | 1 | C++ | Detection, tracking, OCR, events. Frozen as reference. |
| `ingest/` | 2 | Python | yt-dlp, TBA/YouTube APIs, job queue, HTTP API, database, strategy layer |
| `web/` | 3 | TypeScript | UI, player, overlay, corrections, Sheets export |
| `desktop/` | 4 | Python | The primary app |
| `training/` | — | Python | The analysis engine (`track_yolo.py`), training, the model registry |
| `contracts/` | — | — | Shared schemas and per-year season configs. **Owned by everyone.** |
| `fixtures/` | — | — | Golden test data. **Owned by everyone.** |

## Fastest possible start

Nothing installed, no backend, no database: the whole UI against the golden fixtures.

```powershell
cd web
npm install
npm run dev
```

Open `http://localhost:5173`. You get a synthetic 162-second clip with known tracks and events.

## The real thing

```powershell
.\run.ps1 setup
.\run.ps1 full
```

`run.ps1` is the single entry point. `setup` installs everything; `full` starts ingest and the web
app together. Ports are the Doc 0 defaults: web `5173`, ingest `8080`, Postgres `5432`.

| Command | Does |
|---|---|
| `.\run.ps1 setup` | One-time install of every dependency |
| `.\run.ps1 doctor` | Tells you what is missing and how to fix it |
| `.\run.ps1 web` | Web app only, fixture data, no backend |
| `.\run.ps1 api` | Ingest service only |
| `.\run.ps1 full` | Both, wired together |
| `.\run.ps1 check` | Every test and contract check |
| `.\run.ps1 clean` | Reclaim disk from cached segments |

Full walkthrough, prerequisites and troubleshooting: [RUNNING.md](RUNNING.md).

## The rules that keep getting rediscovered

These are in Doc 0, but they are the ones people break:

- **Corrections never overwrite model output.** `?raw=true` must always return exactly what the
  model said. A correction composes on read.
- **`phase` is derived** from `contracts/seasons/<year>.json`, with `teleop_seconds` including the
  endgame. Never hardcode 20/140/30.
- **Tracks carry a required `gaps` array.** Never interpolate across a gap, and never split a track
  at one: a gap means "the camera cut away", not "a new robot".
- **Aggregates are queried, never stored.**
- **Anything unrecognised is a bug, not a fallback.** Drop the row and report it; do not coerce it.
- **`goal` is season-scoped**: legal values come from the season config's `goals` array.
- **Unknown is not zero.** A position, a team or an outcome that was not observed stays null.

## Configuration

The ingest service reads these from the environment. All are optional: the service degrades
rather than fails.

| Variable | Effect when unset |
|---|---|
| `DATABASE_URL` | SQLite at `./frc_scouting.db`. Set to Postgres in production. |
| `TBA_API_KEY` | `alliances` and `tba_score` stay null; no scores to check accuracy against. |
| `SHEETS_SPREADSHEET_ID` + `GOOGLE_APPLICATION_CREDENTIALS` | `POST /api/export/sheets` returns 503 instead of silently not writing. |
| `FRC_DATA_DIR` | `./data`: segments in `data/segments/`, job output in `data/jobs/`. |
| `ANALYSIS_BINARY` | `./analysis/build/bin/analysis` |
| `FRC_ANALYSIS_BACKEND` | `auto`: YOLO + ByteTrack when configured, otherwise native |
| `YOLO_PYTHON` | unset: a dedicated Python with Ultralytics; native is used unless set |
| `YOLO_MODEL_PATH` | the trained `.pt`; defaults to the local v3 checkpoint when present |
| `FRC_YOLO_TRACKER` | `bytetrack` |
| `FRC_YOLO_CONFIDENCE` / `FRC_YOLO_IMAGE_SIZE` / `FRC_YOLO_FRAME_STRIDE` / `FRC_YOLO_DEVICE` | `0.25` / `960` / `1` / `0`; every source frame by default, enforced when ball scouting is configured |
| `FRC_YOLO_REID_MEMORY_SECONDS` | `5.0`: appearance-backed identity memory after a robot disappears |
| `FRC_YOLO_REID_APPEARANCE_THRESHOLD` / `FRC_YOLO_REID_SCORE_MARGIN` | `0.60` / `0.08`: merge evidence and ambiguity gates |
| `FRC_YOLO_REID_MAX_DISTANCE` / `FRC_YOLO_REID_MAX_SPEED` | `0.60` / `0.75`: image-space motion gates when field calibration is unavailable |
| `FRC_YOLO_REID_EDGE_THRESHOLD` | `0.10`: boundary band for edge-consistent re-entry |
| `FRC_YOLO_REID_TEMPLATE_GALLERY_SIZE` / `FRC_YOLO_REID_TEMPLATE_CONFIRMATION_FRAMES` | `5` / `3`: protected appearance gallery |
| `FRC_YOLO_REID_TEMPLATE_MIN_CONFIDENCE` | `0.50`: minimum detector confidence for a new gallery exemplar |
| `FRC_YOLO_REID_ALLIANCE_LOCK_SECONDS` / `FRC_YOLO_REID_ALLIANCE_LOCK_MARGIN_SECONDS` | `5.0` / `2.0`: evidence needed before an identity's alliance becomes permanent |
| `FRC_YOLO_STARTUP_POSITION_SECONDS` / `FRC_YOLO_STARTUP_SPLIT_X` | `2.0` / `0.50`: image-left locks blue and image-right red at the start; `0` disables |
| `FRC_BALL_SCOUTING_CONFIG` | unset disables ball and shot scouting; see [BALL-SCOUTING.md](BALL-SCOUTING.md) |
| `FRC_AUTO_HOMOGRAPHY` | `1`: fit a carpet mapping from steady field AprilTags for each job |
| `FRC_HOMOGRAPHY_CONFIG` | optional fixed camera calibration JSON; overrides automatic calibration |
| `FRC_DEFAULT_SEASON` | `2026` |
| `FRC_MIN_FREE_GB` | `10`: refuses to start a download below this |
| `FRC_SEGMENT_GRACE_DAYS` | `7`: how long a completed job's video survives before `clean` reclaims it |

Never commit `ingest/.env`, `data/`, ONNX weights, service-account JSON, or Hugging Face tokens.

## Checks

```powershell
.\run.ps1 check
```

Web typecheck, build and fixture validation, the Contract E smoke checks, and the ingest unit tests.
