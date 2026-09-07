# /contracts

The only shared surface, per `docs/frc-scouting-0-contract.md`. Owned by everyone.
Nothing here changes without all three people agreeing. Bump `SCHEMA_VERSION` and tell
the other two.

| File | Contract | Status |
|---|---|---|
| `SCHEMA_VERSION` | — | `3` |
| `enums.md` | closed sets, identifiers, units | transcribed from doc 0 |
| `job.schema.json` | A — job record | v2: season, attempt, error_code, progress/stage, timestamps |
| `events.schema.json` | B — event record | v2: UUID event_id, nullable track_id, corrected/correction_id |
| `raw-tracklets.schema.json` | Immutable raw tracker observations | v1: raw/recycled tracker ids plus nullable stable assignment |
| `tracks.schema.json` | C — track record | v2: **required `gaps`**, `team_confidence` |
| `correction.schema.json` | F — corrections | v2: `scope` (event/track), `target_id` |
| `result.schema.json` | D — `result.json` metadata | key names now pinned by doc 0 |
| `seasons/<year>.json` | field, periods, point values | per year, selected by `job.season` |
| `OPEN_QUESTIONS.md` | — | all resolved through v3 |

Doc 0 names `events.schema.json`, `job.schema.json` and `enums.md`. The other files are
transcriptions of contracts that doc 0 states normatively in prose (Contract C, the
corrections layer, `result.json`, and the season config) but does not give a file for. No
fields were invented; where doc 0 left something genuinely undefined it is in
`OPEN_QUESTIONS.md` rather than settled here.

Validate the fixtures against these schemas with:

    cd web && npm run validate:fixtures

## SCHEMA_VERSION 3

All thirteen questions raised against v1 are resolved; the rulings are in doc 0's changelog.
Two provisional stances from v1 were overturned, and both would have caused real damage:

- **Splitting tracks at gaps** would have undone re-identification, whose whole purpose is
  stitching a robot's fragments into one logical track. v2 keeps one track with an explicit
  `gaps` array. The interpolation threshold that stood in for it is deleted.
- **A single current-season config** breaks the first time someone loads 2025 footage. Season
  configs are per year and selected by `job.season`.

Point values stay at zero until a game is public. Score reconstruction is not meaningful
until they are filled in, and the UI says so rather than showing a confident delta.
