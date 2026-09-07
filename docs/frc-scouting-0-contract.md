# Project Tengen — 0. Shared Contract

**SCHEMA_VERSION = 3**

Read this before any of the other three documents. Every person and every AI session working on this project reads this file first, regardless of which component they are building.

The other three documents describe *what* each component does. This one defines *how they connect*. If three people build against these definitions independently, the pieces fit together on the first try. If anyone deviates, they do not.

Version 2 resolved thirteen open questions raised by component 3. Version 3 adds the optional, season-configured `goal` field to Contract B after all three component owners agreed. The rulings are listed in the changelog at the bottom. Where this document and documents 1, 2 or 3 disagree, **this document wins.**

## The rule

**This document is the only shared surface. Nothing in it changes without all three people agreeing.**

Everything else is private to its component. How the C++ backend structures its classes is nobody else's business. What the event JSON looks like is everybody's business.

If you need something added to a contract here, that is a conversation, not a commit. Bump `SCHEMA_VERSION` and tell the other two.

## Components and ownership

| # | Component | Language | Owns | Document |
|---|-----------|----------|------|----------|
| 1 | Analysis backend | C++ | Detection, tracking, OCR, event extraction | `frc-scouting-1-ai.md` |
| 2 | Ingest service | Python | yt-dlp, TBA/YouTube APIs, job queue, HTTP API | `frc-scouting-2-youtube.md` |
| 3 | Web app | TypeScript | UI, player, overlay, corrections, Sheets export | `frc-scouting-3-webapp.md` |

Component 2 is also the orchestrator. It owns the database and the HTTP API. Component 1 is a command-line binary it invokes. Component 3 only ever talks to component 2 over HTTP. Component 1 and component 3 never touch each other.

```
[3] Web app  ──HTTP──▶  [2] Ingest service  ──exec──▶  [1] Analysis binary
                              │                              │
                              ├── yt-dlp                     └── writes events.jsonl
                              ├── TBA API                                tracks.jsonl
                              └── Postgres                               result.json
```

## Repository layout

Monorepo. One repo, three top-level directories, no cross-imports.

```
/analysis/        C++ — component 1
/ingest/          Python — component 2
/web/             TypeScript — component 3
/contracts/       shared schemas, owned by everyone
  events.schema.json
  tracks.schema.json
  job.schema.json
  result.schema.json
  correction.schema.json
  enums.md
  seasons/
    2025.json
    2026.json
/fixtures/        golden test data, owned by everyone
/docs/            these four documents
```

`/contracts/` and `/fixtures/` are the only directories more than one person edits.

## Shared vocabulary

Use these exact terms in code, comments, and column names. Do not invent synonyms.

- **match** — one FRC match. Identified by TBA match key.
- **video** — a YouTube video. Identified by its 11-character ID.
- **segment** — the portion of a video containing one match. May be the whole video.
- **job** — one segment queued for analysis.
- **track** — one robot followed across a match. Survives occlusion and re-identification, so a track may contain gaps. Has a `track_id`, arbitrary and job-local.
- **team** — an FRC team number. Integer. Resolved from a track by bumper OCR.
- **event** — a single timestamped thing that happened.
- **detection** — one box in one frame. Internal to component 1. Never crosses a boundary.
- **cycle** — the interval between one `reload` event and the next `reload` event for the same **team**. Acquire to acquire, not acquire to score, so a missed shot still costs a cycle. Computed per team, never per track, because track ids are job-local and a re-identified robot may span several. An unterminated final cycle is discarded, not counted.

## Identifier formats

| Thing | Format | Example |
|-------|--------|---------|
| `match_id` | TBA match key, lowercase | `2026casf_qm42` |
| `video_id` | YouTube ID, 11 chars, case-sensitive | `dQw4w9WgXcQ` |
| `job_id` | UUIDv4 | `f81d4fae-7dec-11d0-a765-00a0c91e6bf6` |
| `event_id` | UUIDv4 | `9c1e0b22-4a3f-4b18-9f0e-2d7a1c5e8b44` |
| `correction_id` | UUIDv4 | `3fa85f64-5717-4562-b3fc-2c963f66afa6` |
| `team` | integer, no leading zeros, no `frc` prefix | `254` |
| `track_id` | integer, **unique within a job only** | `7` |

Team numbers are integers everywhere. TBA returns them as `frc254`; strip the prefix at the ingest boundary so nothing downstream ever sees it.

Because `track_id` is job-local, any endpoint addressing a track is scoped by job. There is no global track address.

## Units and coordinate systems

Getting these wrong is the most likely source of silent breakage, so they are fixed here.

**Time.** Float seconds, three decimal places. `t_seconds` is always relative to the start of the *segment*, not the original video. To get a position in the original video, add `start_offset` from the job record. Component 3 does that conversion when linking to YouTube; nothing else ever should.

**Image space.** Normalized floats, `0.0` to `1.0`. Origin at top-left. `x` right, `y` down. Never pixels, so overlays work at any player size and boxes survive a resolution change.

**Field space.** Feet, float. Origin at the center of the field. `+x` toward the blue alliance wall, `+y` toward the scoring table side. Nominal field dimensions live in the season config, not in code.

**Rates.** Hertz, float. `box_sample_rate` is samples per second, not a frame interval.

**Confidence.** Float `0.0` to `1.0`. Never a percentage, never a string.

**Booleans.** Actual booleans in JSON. Not `0`/`1`, not `"true"`.

## Enums

Closed sets. Adding a value is a contract change. Anything unrecognized is a bug, not a fallback.

```
phase            = auto | teleop | endgame | unknown
alliance         = red | blue
job_status       = queued | downloading | downloaded | analyzing | complete | failed
stage            = downloading | decoding | detecting | tracking | ocr | events
error_code       = video_unavailable | download_failed | rate_limited
                 | no_match_data | analysis_failed | timeout | internal
event_type       = match_start | match_end | phase_change
                 | shot_attempt | shot_made
                 | reload
                 | defense_start | defense_end
                 | immobile_start | immobile_end
                 | foul
source           = model | scoreboard_ocr | tba | manual
gap_reason       = shot_change | occlusion | out_of_frame | detection_lost
correction_scope = event | track
correction_action= edit | delete | create
```

`source` records where an event came from. `corrected` (see Contract E) records whether a human has since touched it. They are independent: a model event that a human fixed keeps `source: "model"` and gains `corrected: true`.

## Phase is derived, not detected

`phase` is a pure function of match-relative time and the season config. Both component 1 and component 3 compute it with the same function from the same file, so they cannot disagree.

```
t_match = t_seconds - t(match_start)

t_match <  0                                  → unknown
t_match <  auto_seconds                       → auto
t_match <  auto_seconds + teleop_seconds
            - endgame_seconds                 → teleop
t_match <= auto_seconds + teleop_seconds      → endgame
otherwise                                     → unknown
```

Nobody hardcodes 15, 135 or 20. Component 1 reads the season config and stamps the result on each event; component 3 reads the same config to draw timeline bands. If a season changes the structure, one file changes.

## Season config

`/contracts/seasons/<year>.json`. Selected by the `season` field on the job record, so old footage stays analyzable after the game changes.

```json
{
  "season": 2026,
  "field_length_ft": 54.0,
  "field_width_ft": 26.6,
  "auto_seconds": 15,
  "teleop_seconds": 135,
  "endgame_seconds": 20,
  "game_pieces": ["ball"],
  "goals": ["high", "low"],
  "point_values": {
    "auto":    { "shot_made_high": 0, "shot_made_low": 0 },
    "teleop":  { "shot_made_high": 0, "shot_made_low": 0 },
    "endgame": {}
  }
}
```

Point values are zero placeholders until the 2026 game is public. Score reconstruction is not meaningful until they are filled in, and that is expected. Do not invent values to make a test pass.

## Contract A — Job record

Produced by component 2, consumed by components 1 and 3.

```json
{
  "schema_version": 3,
  "job_id": "f81d4fae-7dec-11d0-a765-00a0c91e6bf6",
  "match_id": "2026casf_qm42",
  "season": 2026,
  "video_id": "dQw4w9WgXcQ",
  "local_path": "/data/segments/dQw4w9WgXcQ_00120_00272.mp4",
  "start_offset": 120.0,
  "duration": 152.0,
  "fps": 30.0,
  "width": 1920,
  "height": 1080,
  "status": "downloaded",
  "stage": null,
  "progress": null,
  "error_code": null,
  "error": null,
  "attempt": 1,
  "created_at": "2026-08-28T14:20:00Z",
  "updated_at": "2026-08-28T14:22:31Z",
  "alliances": {
    "red":  [254, 1678, 971],
    "blue": [118, 148, 2056]
  },
  "tba_score": { "red": 91, "blue": 84 }
}
```

**Nullability is conditional on status.** A queued job knows none of its media metadata, because that comes out of the download.

- `local_path`, `duration`, `fps`, `width`, `height` are nullable, and **must** be non-null when `status` is `downloaded`, `analyzing` or `complete`.
- `error_code` and `error` are nullable, and **must** be non-null when `status` is `failed`.
- `progress` is a float `0.0`–`1.0` or null. `stage` is a `stage` enum value or null. Both are populated during `downloading` and `analyzing`, null otherwise. Component 2 populates them by reading component 1's stdout.
- `match_id` is nullable when resolution from video metadata fails.
- `alliances` and `tba_score` are nullable when TBA has no data for the match. The job is still valid; component 1 falls back to raw OCR without elimination.

`alliances` is what component 1 uses to narrow bumper OCR candidates to three per side.

## Contract B — Event record

**This is the authoritative event shape.** Documents 1 and 3 previously listed an eight-field subset. That subset is wrong and has been corrected. If you are building component 1 from document 1, emit what is here.

```json
{
  "schema_version": 3,
  "job_id": "f81d4fae-7dec-11d0-a765-00a0c91e6bf6",
  "match_id": "2026casf_qm42",
  "event_id": "9c1e0b22-4a3f-4b18-9f0e-2d7a1c5e8b44",
  "team": 254,
  "track_id": 7,
  "t_seconds": 42.300,
  "phase": "teleop",
  "event_type": "shot_made",
  "confidence": 0.81,
  "field_x": -12.4,
  "field_y": 3.1,
  "goal": "high",
  "source": "model"
}
```

Nullable: `team`, `track_id`, `field_x`, `field_y`, `goal`. Everything else is required.

- `team` is null when a track was never identified.
- `track_id` is null for match-level events (`match_start`, `match_end`, `phase_change`), which belong to the match rather than to any robot. Those events also carry `team: null` and null field coordinates. They stay in the same stream rather than getting their own record type, because a separate type would double the plumbing for three event types.
- `field_x` / `field_y` are null when homography failed for that frame range.
- `goal` is the season-configured goal for `shot_attempt` or `shot_made`, and null when it is unknown or the event is not a shot. It is deliberately **not** a closed enum: valid values are the selected season config's `goals` array.

`event_id` is generated by component 1, not by the database. Corrections reference it, so it has to exist before anything is stored.

Component 1 writes these as JSON Lines to `events.jsonl`, one object per line, ascending by `t_seconds`.

## Contract C — Track record

Produced by component 1 alongside events. This is what draws the boxes.

```json
{
  "schema_version": 3,
  "track_id": 7,
  "team": 254,
  "alliance": "red",
  "team_confidence": 0.93,
  "boxes": [
    { "t": 42.300, "x": 0.412, "y": 0.556, "w": 0.061, "h": 0.078 }
  ],
  "gaps": [
    { "start": 61.200, "end": 65.400, "reason": "shot_change" }
  ]
}
```

`x` and `y` are the top-left corner in normalized image space.

**`gaps` is required, possibly empty.** A gap is an interval where the robot was not observed: a skipped shot-change segment, an occlusion, a robot out of frame, or lost detection. Consumers must not interpolate across a listed gap. Without this, a four-second hole is indistinguishable from a low sample rate, and the overlay draws a robot gliding through footage nobody analyzed.

Tracks are **not** split at gaps. Re-identification exists specifically to stitch a robot's fragments into one logical track, and splitting would undo that. One robot, one track, holes marked explicitly.

`team_confidence` is separate from per-event confidence. It reflects how sure the bumper OCR is about the whole track's identity, which is what the UI needs to flag likely misattributions.

Box sample rate does not need to match video frame rate. It is stated in `result.json` and served to component 3 in the tracks response.

## Contract D — Analysis binary invocation

Component 2 calls component 1 as a subprocess. This is the entire interface between them.

```
analysis --job <path/to/job.json> --season <path/to/season.json> --out <output/dir>
```

On success: exit 0, and `<out>/events.jsonl`, `<out>/tracks.jsonl`, `<out>/result.json` exist.

`result.json`, with the key names now pinned:

```json
{
  "schema_version": 3,
  "job_id": "f81d4fae-7dec-11d0-a765-00a0c91e6bf6",
  "model_version": "rfdetr-2026.1",
  "box_sample_rate": 10.0,
  "homography_ok": true,
  "frames_total": 4560,
  "frames_analyzed": 4210,
  "frames_skipped_shot_change": 350,
  "tracks_emitted": 6,
  "events_emitted": 214,
  "reconstructed_score": { "red": 88, "blue": 84 },
  "started_at": "2026-08-28T14:22:31Z",
  "finished_at": "2026-08-28T14:26:04Z"
}
```

`reconstructed_score` is nullable while season point values are placeholders.

On failure: nonzero exit, human-readable reason on stderr, and an `error_code` enum value as the last line of stderr so component 2 can classify it without parsing prose.

Progress goes to stdout as one JSON object per line:

```json
{"progress": 0.42, "stage": "tracking"}
```

`stage` must be a value from the `stage` enum. Component 2 copies both onto the job record verbatim.

Component 1 does not write to the database, does not make network calls, and does not know that YouTube exists. It reads files and writes files.

## Contract E — HTTP API

Component 2 serves, component 3 consumes. JSON in, JSON out. Errors use standard status codes with `{"error_code": "...", "error": "message"}`.

```
POST   /api/jobs                      { url, match_id?, season? }        → job
GET    /api/jobs                      → { jobs: [job] }
GET    /api/jobs/:job_id              → job
POST   /api/jobs/:job_id/retry        → job (same job_id, attempt + 1)
DELETE /api/jobs/:job_id
GET    /api/jobs/:job_id/result       → result.json contents
GET    /api/video/:job_id             → serves the local segment file

GET    /api/matches/:match_id/events?min_confidence=&raw=   → { events: [event] }
GET    /api/matches/:match_id/tracks?raw=                   → { box_sample_rate, tracks: [track] }
GET    /api/matches/:match_id/corrections                   → { corrections: [correction] }
GET    /api/matches/:match_id/accuracy                      → accuracy

POST   /api/events                    → create a manual event (source: "manual")
PATCH  /api/events/:event_id          → correct one event
DELETE /api/events/:event_id
PATCH  /api/jobs/:job_id/tracks/:track_id   { team }  → re-attribute a whole track
DELETE /api/corrections/:correction_id      → undo a correction

GET    /api/teams/:team/stats?event_key=&min_confidence=    → team_stats
POST   /api/export/sheets             { match_ids, mode }   → export_result
```

**Retry/re-run reuses the job id.** It resets `status` to `queued`, clears `error_code` and `error`, increments `attempt`, and replaces the prior raw analyzer result when the new run succeeds. Creating a new job would orphan the selected video's history.

**`raw=true` is honoured on `/events` and `/tracks` only.** Default is corrected. Raw is what the accuracy comparison and the training-data export use.

**Collection endpoints return an object, never a bare array.** That is what let `box_sample_rate` land on the tracks response without a breaking change, and it leaves room for pagination later.

### Read-only annotations on returned events

API responses add two fields not present in the stored Contract B record:

```json
{ "...": "contract B fields", "corrected": true, "correction_id": "3fa85f64-..." }
```

`corrected` is false and `correction_id` null for untouched events. This is what lets the UI mark human-touched rows and offer undo without a second request and a local diff.

### Response shapes

```json
// accuracy
{
  "match_id": "2026casf_qm42",
  "tba_available": true,
  "reconstructed": { "red": 88, "blue": 84 },
  "tba":           { "red": 91, "blue": 84 },
  "delta":         { "red": -3, "blue": 0 }
}
```

`tba` and `delta` are null when `tba_available` is false.

```json
// team_stats
{
  "team": 254,
  "event_key": "2026casf",
  "min_confidence": 0.5,
  "matches_played": 12,
  "cycles": 87,
  "avg_cycle_seconds": 11.4,
  "shot_attempts": 143,
  "shots_made": 118,
  "shot_accuracy": 0.825,
  "avg_shot_interval_seconds": 2.1,
  "reloads": 99,
  "defense_seconds": 214.5,
  "immobile_seconds": 31.0,
  "fouls": 3,
  "low_confidence_events": 22
}
```

Fields may be added to this object additively. None may be renamed or removed.

```json
// export_result
{
  "spreadsheet_id": "1AbC...",
  "spreadsheet_url": "https://docs.google.com/spreadsheets/d/1AbC.../edit",
  "mode": "aggregate",
  "rows_written": 72,
  "rows_skipped": 0
}
```

The URL is required. Without it the UI cannot link the user to the sheet it just wrote.

## Contract F — Correction record

Corrections never overwrite model output. A correction is a new row referencing what it changes.

```json
{
  "schema_version": 3,
  "correction_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "scope": "track",
  "job_id": "f81d4fae-7dec-11d0-a765-00a0c91e6bf6",
  "target_id": "7",
  "action": "edit",
  "fields": { "team": 1678 },
  "created_at": "2026-08-28T14:31:00Z",
  "created_by": "justin"
}
```

`scope: "event"` means `target_id` is an `event_id`. `scope: "track"` means `target_id` is a `track_id` and `job_id` is required to disambiguate it.

**A track-scoped correction re-attributes the track and every event on it, as one action.** Fixing a misread bumper is the most common correction in the product, and it affects forty-odd rows in a single match. Doing it one event at a time would also leave the boxes mislabelled, since the overlay reads `team` from the track, not from events.

Reads apply corrections on top of raw records. Deleting a correction undoes it.

## Database

Component 2 owns the schema and is the only thing that writes to it. Postgres in production, SQLite acceptable locally.

Tables: `jobs`, `events`, `tracks`, `corrections`, `matches`.

`events` and `tracks` store raw model output only. Aggregates are never stored, only queried. If a stat is needed often enough to hurt, add a materialized view, not a column.

## Versioning

`SCHEMA_VERSION` is an integer in `/contracts/`. Every record carries it.

Additive changes (a new optional field, a new enum value) bump the version and stay backward compatible. Consumers ignore fields they do not recognize.

Breaking changes (renaming, removing, changing a type or a unit) require all three components to update together. Avoid them.

## Golden fixtures

`/fixtures/` contains one fully worked example: a short match segment, its job record, a hand-verified `events.jsonl`, a `tracks.jsonl`, a `result.json`, and a snapshot of the TBA response.

This exists so all three people can work alone. Component 3 builds the whole UI against fixture data with no backend running. Component 2 tests the pipeline with a stub binary that copies the fixture output. Component 1 validates against hand-verified ground truth.

**If your component works against the fixtures, it will work against the others.** Adding a feature means adding a fixture case for it first.

Fixture coverage must include the awkward cases, not just the happy path: a track with a `shot_change` gap, an unidentified track with `team: null`, a match-level event with `track_id: null`, a failed job with an `error_code`, and a match with `alliances: null`.

## Working in parallel

- Build against the contracts and the fixtures, not against the other person's actual code.
- Do not wait for someone else's component. If it does not exist, stub it from a fixture.
- Do not reach across a boundary because it is faster. Component 3 wanting to shell out to yt-dlp, or component 1 wanting to hit the TBA API, is how this ends up unmergeable.
- Anything that feels like it needs a new shared field is a contract change. Raise it instead of adding it locally.
- If an AI session suggests changing a schema, a unit, or a coordinate convention to make its part easier, say no. The convention is not there because it is optimal, it is there because it is agreed.

## Defaults

These are arbitrary and can be changed once, together, at the start. After that, leave them.

- Ports: ingest API `8080`, web dev server `5173`, Postgres `5432`
- Segment storage: `/data/segments/`, job output: `/data/jobs/<job_id>/`
- Naming: `snake_case` in JSON, SQL, C++, and Python. `camelCase` only inside TypeScript, converted at the API boundary.
- Timestamps in metadata: ISO 8601, UTC, `Z` suffix. Timestamps within video: float seconds.

## Changelog

### v3 — optional `goal` on Contract B

All three component owners agreed to add `goal`, a nullable string selected from the active season config's `goals` array. It identifies the goal for a shot without baking game-specific goal names into a permanent enum. Existing v2 readers can ignore this additive field; v3 producers and API responses always carry `schema_version: 3`.

### v2 — resolves the thirteen open questions

1. **`track_id` on match-level events** — nullable, same as `team`. Match-level events stay in the one event stream with `team`, `track_id`, `field_x`, `field_y` all null. No separate record type.
2. **`box_sample_rate` unreachable by component 3** — the tracks endpoint now returns `{ box_sample_rate, tracks }`. All collection endpoints return objects rather than bare arrays. `result.json` key names are pinned in Contract D, and `GET /api/jobs/:job_id/result` exposes the whole thing.
3. **No endpoint reads corrections** — added `GET /api/matches/:match_id/corrections` and `DELETE /api/corrections/:correction_id`. API event responses now carry `corrected` and `correction_id` so no client-side diff is needed. `raw=true` is honoured on `/events` and `/tracks`.
4. **No progress on the job record** — added `progress` and `stage`, both nullable. `stage` is a closed enum so component 1's stdout strings and component 3's labels cannot drift.
5. **No retry, no timestamps** — added `POST /api/jobs/:job_id/retry`, which reuses the job id and increments `attempt`. Added `created_at`, `updated_at`, `attempt`.
6. **Unspecified response shapes** — the three assumed shapes were close and are now written out, with `tba_available` added to accuracy for the no-TBA-data case, `rows_skipped` added to export, and the stats fields enumerated.
7. **Season config location** — confirmed at `/contracts/seasons/<year>.json`, and the job record now carries `season` so old footage stays analyzable after the game changes. Point values stay at zero until the game is public.
8. **Unmarked gaps in tracks** — added a required `gaps` array with a `gap_reason` enum. Tracks are not split, because re-identification exists to stitch fragments together. The threshold heuristic in `web/src/lib/tracks.ts` can be deleted.
9. **Docs 1 and 3 contradict Contract B** — Contract B is authoritative and documents 1 and 3 have been corrected to point at it instead of restating a stale eight-field subset.
10. **Track-level team correction** — added `PATCH /api/jobs/:job_id/tracks/:track_id` and a `scope` field on corrections. One action re-attributes the track and all its events. Scoped by job because `track_id` is job-local.
11. **Cycle time defined twice** — settled as acquire-to-acquire, computed per team, not per track. Now in the shared vocabulary. Document 1 corrected.
12. **Nobody owns the endgame boundary** — `phase` is a derived pure function of match-relative time and the season config, specified above. Nobody hardcodes durations.
13. **Media metadata unknowable before download** — the conditional nullability is accepted as written, plus `error_code` as a closed enum so the UI knows whether a retry is worth offering.
