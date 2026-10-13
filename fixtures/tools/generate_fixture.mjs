// Regenerates /fixtures/ -- the worked examples doc 0 asks for, at SCHEMA_VERSION 3.
//
// Deterministic: same seed in, byte-identical JSON out (UUIDs included). Run with:
//     node fixtures/tools/generate_fixture.mjs
// Requires ffmpeg on PATH for segment.mp4 (pass --no-video to skip).
//
// The synthetic match is authored here rather than hand-typed so the video, the tracks and
// the events are guaranteed to agree: robot positions come from one motion model, the video
// renders that model at 30 fps, tracks.jsonl samples it, and events.jsonl comes off the same
// schedule. A box that does not sit on the robot under it is a real bug, not fixture drift.
//
// Doc 0 requires the awkward cases, not just the happy path. All five are covered:
//   - a track with a `shot_change` gap            -> every robot track, 61.2-65.4
//   - an unidentified track with `team: null`     -> track 14
//   - a match-level event with `track_id: null`   -> match_start / phase_change / match_end
//   - a failed job with an `error_code`           -> fixtures/failed_download/
//   - a match with `alliances: null`              -> fixtures/2026casf_qm43_no_tba/

import { writeFileSync, mkdirSync, readFileSync } from 'node:fs';
import { spawn } from 'node:child_process';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, '..', '..');
const FIXTURES = resolve(ROOT, 'fixtures');
const OUT = resolve(FIXTURES, '2026casf_qm42');
const SEASON_YEAR = 2026;
const SEASON = JSON.parse(
  readFileSync(resolve(ROOT, 'contracts', 'seasons', `${SEASON_YEAR}.json`), 'utf8')
);

// ---------------------------------------------------------------- constants

const JOB_ID = 'f81d4fae-7dec-11d0-a765-00a0c91e6bf6';
const MATCH_ID = '2026casf_qm42';
const VIDEO_ID = 'dQw4w9WgXcQ';
const START_OFFSET = 120.0;
const FPS = 30;
const W = 640;
const H = 360;
const BOX_HZ = 5; // deliberately != FPS so component 3 must interpolate
// Legal goal names are whatever this season declares. Never hardcode them.
const GOALS = SEASON.goals;

const FIELD_L = SEASON.field_length_ft;
const FIELD_W = SEASON.field_width_ft;
const AUTO = SEASON.auto_seconds;
const TELEOP = SEASON.teleop_seconds;
const ENDGAME = SEASON.endgame_seconds;
const T_MATCH_END = AUTO + TELEOP;
// The clip is the match plus two seconds, from the season file rather than a constant: the
// fixed 152 s was 2026's old 150 s match, and when the season was corrected to 160 s the
// generator placed the last events past the end of the clip.
const DURATION = T_MATCH_END + 2.0;
const CLIP_SECONDS = String(Math.round(DURATION)).padStart(5, '0');

// mulberry32, so the fixture is reproducible
function makeRng(seed) {
  let s = seed >>> 0;
  return function next() {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const rnd = makeRng(0x5f3a71c9);
const jitter = (a) => (rnd() * 2 - 1) * a;
const r3 = (n) => Math.round(n * 1000) / 1000;
const r2 = (n) => Math.round(n * 100) / 100;
const smooth = (u) => u * u * (3 - 2 * u);
const clamp01 = (u) => (u < 0 ? 0 : u > 1 ? 1 : u);

/** Deterministic UUIDv4 from the seeded RNG, so regeneration stays byte-identical. */
function uuid() {
  const hex = [];
  for (let i = 0; i < 32; i++) hex.push(Math.floor(rnd() * 16).toString(16));
  hex[12] = '4';
  hex[16] = ((parseInt(hex[16], 16) & 0x3) | 0x8).toString(16);
  const s = hex.join('');
  return `${s.slice(0, 8)}-${s.slice(8, 12)}-${s.slice(12, 16)}-${s.slice(16, 20)}-${s.slice(20)}`;
}

// Doc 0: phase is a pure function of match-relative time and the season config.
function phaseAt(t) {
  if (t < 0) return 'unknown';
  if (t < AUTO) return 'auto';
  if (t < AUTO + TELEOP - ENDGAME) return 'teleop';
  if (t <= AUTO + TELEOP) return 'endgame';
  return 'unknown';
}

// +x is toward the BLUE alliance wall, so red robots load near x = -22 and score near x = +15.
const ROBOTS = [
  { team: 254, track_id: 7, alliance: 'red', cycle: 8.5, acc: 0.86, lane: -7.5, tconf: 0.96 },
  { team: 1678, track_id: 3, alliance: 'red', cycle: 10.2, acc: 0.77, lane: 0.0, tconf: 0.91 },
  { team: 971, track_id: 11, alliance: 'red', cycle: 12.6, acc: 0.61, lane: 7.5, tconf: 0.58 },
  { team: 118, track_id: 5, alliance: 'blue', cycle: 9.1, acc: 0.81, lane: -4.0, tconf: 0.94 },
  { team: 148, track_id: 9, alliance: 'blue', cycle: 11.3, acc: 0.7, lane: 3.5, tconf: 0.88 },
  { team: 2056, track_id: 2, alliance: 'blue', cycle: 13.4, acc: 0.65, lane: 10.5, tconf: 0.83 },
];
const UNKNOWN_TRACK = { team: null, track_id: 14, from: 88.0, to: 96.0 };

// A broadcast cut to a replay: nobody is observed, on any track.
const SHOT_CHANGE = { start: 61.2, end: 65.4, reason: 'shot_change' };
// 971 disappears behind traffic in the scoring zone.
const OCCLUSION = { team: 971, start: 100.0, end: 103.5, reason: 'occlusion' };

const IMMOBILE = { team: 971, from: 72.4, to: 95.1 };
const DEFENSE = { team: 2056, from: 40.2, to: 70.6, target: 254 };
const HOMOGRAPHY_GAP = { from: 100.0, to: 110.0 }; // field_x/field_y null, still observed
const FOUL = { team: 148, t: 118.35 };

const inGap = (t, g) => t >= g.start && t <= g.end;
function gapsFor(team) {
  const gaps = [{ ...SHOT_CHANGE }];
  if (team === OCCLUSION.team) {
    gaps.push({ start: OCCLUSION.start, end: OCCLUSION.end, reason: OCCLUSION.reason });
  }
  return gaps.sort((a, b) => a.start - b.start);
}
const unobserved = (team, t) => gapsFor(team).some((g) => inGap(t, g));

// ---------------------------------------------------------------- motion

function buildSchedule(r) {
  const sign = r.alliance === 'red' ? -1 : 1;
  const load = { x: sign * 22.5, y: r.lane * 0.55 };
  const score = { x: -sign * 15.0, y: r.lane };
  const legs = [];
  let t = 0;
  let leg = 0;
  while (t < DURATION) {
    const c = Math.max(4.5, r.cycle + jitter(r.cycle * 0.16));
    const travel = c * 0.4;
    const dwell = c * 0.1;
    const outbound = leg % 2 === 0;
    legs.push({
      t0: t, t1: t + travel,
      from: outbound ? load : score, to: outbound ? score : load,
      kind: 'travel',
    });
    t += travel;
    const at = outbound ? score : load;
    legs.push({
      t0: t, t1: t + dwell, from: at, to: at,
      kind: 'dwell', action: outbound ? 'shoot' : 'reload',
    });
    t += dwell;
    leg++;
  }
  return legs;
}
const SCHED = new Map(ROBOTS.map((r) => [r.team, buildSchedule(r)]));

function basePos(team, t) {
  const legs = SCHED.get(team);
  let lo = 0;
  let hi = legs.length - 1;
  while (lo < hi) {
    const m = (lo + hi) >> 1;
    if (legs[m].t1 <= t) lo = m + 1;
    else hi = m;
  }
  const l = legs[lo];
  const u = l.kind === 'dwell' ? 1 : smooth(clamp01((t - l.t0) / (l.t1 - l.t0)));
  return { x: l.from.x + (l.to.x - l.from.x) * u, y: l.from.y + (l.to.y - l.from.y) * u };
}

function pos(team, t) {
  if (team === IMMOBILE.team && t >= IMMOBILE.from && t <= IMMOBILE.to) {
    return basePos(team, IMMOBILE.from);
  }
  if (team === DEFENSE.team && t >= DEFENSE.from && t <= DEFENSE.to) {
    const p = pos(DEFENSE.target, t);
    return { x: p.x + 3.1, y: p.y + 2.4 };
  }
  return basePos(team, t);
}
function unknownPos(t) {
  const u = clamp01((t - UNKNOWN_TRACK.from) / (UNKNOWN_TRACK.to - UNKNOWN_TRACK.from));
  return { x: -24.5 + u * 6.0, y: 11.6 - u * 1.2 };
}

// Field space -> normalized image space. Camera on the scoring-table side (+y), so larger y
// is nearer: lower on screen and a slightly larger box.
function project(p) {
  const u = 0.5 + (p.x / FIELD_L) * 0.86;
  const v = 0.52 + (p.y / FIELD_W) * 0.62;
  const s = 1 + 0.35 * (p.y / (FIELD_W / 2));
  const w = 0.055 * s;
  const h = 0.075 * s;
  return { x: u - w / 2, y: v - h / 2, w, h };
}

// ---------------------------------------------------------------- events

const events = [];
const byTeam = (n) => ROBOTS.find((r) => r.team === n);

function mk(t, type, team, trackId, conf, opts) {
  const o = opts || {};
  const isShot = type === 'shot_attempt' || type === 'shot_made';
  const matchLevel = type === 'match_start' || type === 'match_end' || type === 'phase_change';
  const p = matchLevel || team == null ? o.pos || null : pos(team, t);
  const blind = t >= HOMOGRAPHY_GAP.from && t <= HOMOGRAPHY_GAP.to;
  return {
    schema_version: 3,
    job_id: JOB_ID,
    match_id: MATCH_ID,
    event_id: uuid(),
    // Match-level events belong to the match, not a robot: team, track_id and field
    // coordinates are all null.
    team: matchLevel ? null : team,
    track_id: matchLevel ? null : trackId,
    t_seconds: r3(t),
    phase: phaseAt(t),
    event_type: type,
    confidence: r2(Math.min(0.99, Math.max(0.05, conf))),
    field_x: matchLevel || blind || !p ? null : r2(p.x),
    field_y: matchLevel || blind || !p ? null : r2(p.y),
    // v3: which goal the shot went into. Null on anything that is not a shot, and null on
    // shots the model could not place -- absent goal is legal, a wrong one is not.
    goal: isShot ? (o.goal ?? null) : null,
    source: o.source || 'model',
  };
}

events.push(mk(0.0, 'match_start', null, null, 0.98, { source: 'scoreboard_ocr' }));
events.push(mk(AUTO, 'phase_change', null, null, 0.97, { source: 'scoreboard_ocr' }));
events.push(mk(AUTO + TELEOP - ENDGAME, 'phase_change', null, null, 0.95, { source: 'scoreboard_ocr' }));
events.push(mk(T_MATCH_END, 'match_end', null, null, 0.98, { source: 'scoreboard_ocr' }));

const suppressed = (team, t) =>
  (team === IMMOBILE.team && t >= IMMOBILE.from && t <= IMMOBILE.to) ||
  (team === DEFENSE.team && t >= DEFENSE.from && t <= DEFENSE.to) ||
  unobserved(team, t); // nothing is detected inside a gap

for (const r of ROBOTS) {
  for (const leg of SCHED.get(r.team)) {
    if (leg.kind !== 'dwell') continue;
    const t = leg.t0 + (leg.t1 - leg.t0) * 0.5;
    if (t > T_MATCH_END || suppressed(r.team, t)) continue;
    if (leg.action === 'reload') {
      events.push(mk(t, 'reload', r.team, r.track_id, 0.74 + jitter(0.14)));
    } else {
      // Most shots go high; a minority go low. One in twelve is unplaced, so consumers
      // have to handle a shot whose goal the model could not determine.
      const roll = rnd();
      const goal = roll < 0.08 ? null : roll < 0.75 ? GOALS[0] : GOALS[1];
      events.push(mk(t, 'shot_attempt', r.team, r.track_id, 0.8 + jitter(0.16), { goal }));
      if (rnd() < r.acc) {
        events.push(mk(t + 0.28, 'shot_made', r.team, r.track_id, 0.7 + jitter(0.22), { goal }));
      }
    }
  }
}

events.push(mk(IMMOBILE.from, 'immobile_start', 971, byTeam(971).track_id, 0.66));
events.push(mk(IMMOBILE.to, 'immobile_end', 971, byTeam(971).track_id, 0.59));
events.push(mk(DEFENSE.from, 'defense_start', 2056, byTeam(2056).track_id, 0.52));
events.push(mk(DEFENSE.to, 'defense_end', 2056, byTeam(2056).track_id, 0.47));
events.push(mk(FOUL.t, 'foul', 148, byTeam(148).track_id, 0.41, { source: 'scoreboard_ocr' }));
events.push(
  mk(91.2, 'shot_attempt', null, UNKNOWN_TRACK.track_id, 0.29, {
    pos: unknownPos(91.2),
    goal: null,
  })
);

events.sort((a, b) => a.t_seconds - b.t_seconds || a.event_type.localeCompare(b.event_type));

// ---------------------------------------------------------------- tracks

function sampleBoxes(fn, from, to, gaps) {
  const out = [];
  const step = 1 / BOX_HZ;
  for (let i = 0; ; i++) {
    const t = from + i * step;
    if (t > to + 1e-9) break;
    // No observation inside a gap, which is exactly why gaps must be declared.
    if (gaps.some((g) => inGap(t, g))) continue;
    const b = project(fn(t));
    out.push({ t: r3(t), x: r3(b.x), y: r3(b.y), w: r3(b.w), h: r3(b.h) });
  }
  return out;
}

const tracks = ROBOTS.map((r) => ({
  schema_version: 3,
  track_id: r.track_id,
  team: r.team,
  alliance: r.alliance,
  team_confidence: r.tconf,
  boxes: sampleBoxes((t) => pos(r.team, t), 0, DURATION, gapsFor(r.team)),
  gaps: gapsFor(r.team),
}));
tracks.push({
  schema_version: 3,
  track_id: UNKNOWN_TRACK.track_id,
  team: null,
  alliance: null,
  team_confidence: null,
  boxes: sampleBoxes(unknownPos, UNKNOWN_TRACK.from, UNKNOWN_TRACK.to, []),
  gaps: [],
});
tracks.sort((a, b) => a.track_id - b.track_id);

// ---------------------------------------------------------------- write

mkdirSync(OUT, { recursive: true });
const jsonl = (rows) => rows.map((r) => JSON.stringify(r)).join('\n') + '\n';
const json = (o) => JSON.stringify(o, null, 2) + '\n';
const teamsOf = (a) => ROBOTS.filter((r) => r.alliance === a).map((r) => r.team);

// Doc 0's Contract A example score. Score reconstruction is NOT meaningful yet -- the season
// point values are zero placeholders, so reconstructed_score is null rather than invented.
const TBA = { red: 91, blue: 84 };

writeFileSync(
  resolve(OUT, 'job.json'),
  json({
    schema_version: 3,
    job_id: JOB_ID,
    match_id: MATCH_ID,
    season: SEASON_YEAR,
    video_id: VIDEO_ID,
    local_path: `/data/segments/${VIDEO_ID}_${String(Math.round(START_OFFSET)).padStart(5, "0")}_${String(Math.round(START_OFFSET + DURATION)).padStart(5, "0")}.mp4`,
    start_offset: START_OFFSET,
    duration: DURATION,
    fps: FPS,
    width: W,
    height: H,
    status: 'complete',
    stage: null,
    progress: null,
    error_code: null,
    error: null,
    attempt: 1,
    created_at: '2026-08-28T14:20:00Z',
    updated_at: '2026-08-28T14:26:04Z',
    alliances: { red: teamsOf('red'), blue: teamsOf('blue') },
    tba_score: TBA,
  })
);
writeFileSync(resolve(OUT, 'events.jsonl'), jsonl(events));
writeFileSync(resolve(OUT, 'tracks.jsonl'), jsonl(tracks));

const framesTotal = Math.round(DURATION * FPS);
const framesSkipped = Math.round((SHOT_CHANGE.end - SHOT_CHANGE.start) * FPS);
writeFileSync(
  resolve(OUT, 'result.json'),
  json({
    schema_version: 3,
    job_id: JOB_ID,
    model_version: 'fixture-synthetic-0.3.0',
    box_sample_rate: BOX_HZ,
    homography_ok: true,
    frames_total: framesTotal,
    frames_analyzed: framesTotal - framesSkipped,
    frames_skipped_shot_change: framesSkipped,
    tracks_emitted: tracks.length,
    events_emitted: events.length,
    // Null while the season's point values are placeholders.
    reconstructed_score: null,
    started_at: '2026-08-28T14:22:31Z',
    finished_at: '2026-08-28T14:26:04Z',
  })
);

writeFileSync(
  resolve(OUT, 'tba_match.json'),
  json({
    key: MATCH_ID,
    comp_level: 'qm',
    set_number: 1,
    match_number: 42,
    event_key: '2026casf',
    winning_alliance: TBA.red > TBA.blue ? 'red' : 'blue',
    alliances: {
      red: { score: TBA.red, team_keys: teamsOf('red').map((t) => 'frc' + t), surrogate_team_keys: [], dq_team_keys: [] },
      blue: { score: TBA.blue, team_keys: teamsOf('blue').map((t) => 'frc' + t), surrogate_team_keys: [], dq_team_keys: [] },
    },
    time: 1774028400,
    actual_time: 1774028733,
    predicted_time: 1774028400,
    post_result_time: 1774028901,
    score_breakdown: null,
    videos: [{ type: 'youtube', key: VIDEO_ID + '?t=' + Math.round(START_OFFSET) }],
  })
);

// Contract F. Includes a TRACK-scoped correction, which doc 3 now says is the primary path:
// one bad OCR read mislabels forty-odd events and every box on that robot.
const lowConf = events.find((e) => e.team === null && e.event_type === 'shot_attempt');
const falsePos = events.find((e) => e.event_type === 'shot_made' && e.confidence < 0.55);
writeFileSync(
  resolve(OUT, 'corrections.jsonl'),
  jsonl([
    {
      schema_version: 3,
      correction_id: uuid(),
      scope: 'track',
      job_id: JOB_ID,
      target_id: String(byTeam(971).track_id),
      action: 'edit',
      fields: { team: 971 },
      created_at: '2026-08-28T14:31:00Z',
      created_by: 'justin',
    },
    {
      schema_version: 3,
      correction_id: uuid(),
      scope: 'event',
      job_id: JOB_ID,
      target_id: lowConf.event_id,
      action: 'edit',
      fields: { team: 971 },
      created_at: '2026-08-28T14:32:10Z',
      created_by: 'justin',
    },
    {
      schema_version: 3,
      correction_id: uuid(),
      scope: 'event',
      job_id: JOB_ID,
      target_id: falsePos.event_id,
      action: 'delete',
      fields: null,
      created_at: '2026-08-28T14:33:12Z',
      created_by: 'justin',
    },
  ])
);

// ---- awkward case: a match TBA has no data for (alliances and tba_score both null)

const NO_TBA = resolve(FIXTURES, '2026casf_qm43_no_tba');
mkdirSync(NO_TBA, { recursive: true });
const NO_TBA_JOB = '4b7c2e19-3d5a-4f81-9c06-8e1b2a7d5f30';
const NO_TBA_MATCH = '2026casf_qm43';
writeFileSync(
  resolve(NO_TBA, 'job.json'),
  json({
    schema_version: 3,
    job_id: NO_TBA_JOB,
    match_id: NO_TBA_MATCH,
    season: SEASON_YEAR,
    video_id: 'kJQP7kiw5Fk',
    local_path: `/data/segments/kJQP7kiw5Fk_00000_${CLIP_SECONDS}.mp4`,
    start_offset: 0.0,
    duration: DURATION,
    fps: 30.0,
    width: 640,
    height: 360,
    status: 'complete',
    stage: null,
    progress: null,
    error_code: null,
    error: null,
    attempt: 1,
    created_at: '2026-08-28T15:02:00Z',
    updated_at: '2026-08-28T15:07:44Z',
    // TBA has nothing for this match. The job is still valid; component 1 falls back to raw
    // OCR without elimination, so tracks stay unidentified.
    alliances: null,
    tba_score: null,
  })
);
const noTbaTracks = [1, 4].map((id, i) => ({
  schema_version: 3,
  track_id: id,
  team: null,
  alliance: i === 0 ? 'red' : 'blue',
  team_confidence: null,
  boxes: sampleBoxes((t) => pos(i === 0 ? 254 : 118, t), 0, 40, []),
  gaps: [],
}));
const noTbaEvents = [
  {
    schema_version: 3, job_id: NO_TBA_JOB, match_id: NO_TBA_MATCH, event_id: uuid(),
    team: null, track_id: null, t_seconds: 0.0, phase: 'auto', event_type: 'match_start',
    confidence: 0.96, field_x: null, field_y: null, goal: null, source: 'scoreboard_ocr',
  },
  {
    schema_version: 3, job_id: NO_TBA_JOB, match_id: NO_TBA_MATCH, event_id: uuid(),
    team: null, track_id: 1, t_seconds: 12.4, phase: 'auto', event_type: 'shot_made',
    confidence: 0.55, field_x: -8.2, field_y: 1.4, goal: GOALS[0], source: 'model',
  },
];
writeFileSync(resolve(NO_TBA, 'events.jsonl'), jsonl(noTbaEvents));
writeFileSync(resolve(NO_TBA, 'tracks.jsonl'), jsonl(noTbaTracks));
writeFileSync(
  resolve(NO_TBA, 'result.json'),
  json({
    schema_version: 3,
    job_id: NO_TBA_JOB,
    model_version: 'fixture-synthetic-0.3.0',
    box_sample_rate: BOX_HZ,
    homography_ok: true,
    frames_total: 4560,
    frames_analyzed: 4560,
    frames_skipped_shot_change: 0,
    tracks_emitted: noTbaTracks.length,
    events_emitted: noTbaEvents.length,
    reconstructed_score: null,
    started_at: '2026-08-28T15:03:10Z',
    finished_at: '2026-08-28T15:07:44Z',
  })
);

// ---- awkward case: a failed job carrying an error_code

const FAILED = resolve(FIXTURES, 'failed_download');
mkdirSync(FAILED, { recursive: true });
writeFileSync(
  resolve(FAILED, 'job.json'),
  json({
    schema_version: 3,
    job_id: 'b3f0a71c-9d24-4e6a-8c15-0f7b2e9d4a83',
    match_id: '2026casf_qm44',
    season: SEASON_YEAR,
    video_id: 'M7lc1UVf-VE',
    local_path: null,
    start_offset: 0.0,
    duration: null,
    fps: null,
    width: null,
    height: null,
    status: 'failed',
    stage: null,
    progress: null,
    // rate_limited is worth retrying; video_unavailable is not. That distinction is the
    // whole reason error_code is a closed enum.
    error_code: 'rate_limited',
    error: 'yt-dlp: HTTP Error 429: Too Many Requests (backing off)',
    attempt: 2,
    created_at: '2026-08-28T15:40:00Z',
    updated_at: '2026-08-28T15:41:12Z',
    alliances: null,
    tba_score: null,
  })
);

console.log(
  `events=${events.length} tracks=${tracks.length} gaps=${tracks.reduce((n, t) => n + t.gaps.length, 0)} ` +
  `+ no_tba fixture + failed_download fixture`
);

// ---------------------------------------------------------------- video

if (process.argv.includes('--no-video')) {
  console.log('skipped segment.mp4 (--no-video)');
  process.exit(0);
}

const GLYPHS = {
  0: [7, 5, 5, 5, 7], 1: [2, 6, 2, 2, 7], 2: [7, 1, 7, 4, 7], 3: [7, 1, 7, 1, 7],
  4: [5, 5, 7, 1, 1], 5: [7, 4, 7, 1, 7], 6: [7, 4, 7, 5, 7], 7: [7, 1, 1, 1, 1],
  8: [7, 5, 7, 5, 7], 9: [7, 5, 7, 1, 7],
};

const frame = Buffer.alloc(W * H * 3);
function px(x, y, r, g, b) {
  if (x < 0 || y < 0 || x >= W || y >= H) return;
  const i = (y * W + x) * 3;
  frame[i] = r;
  frame[i + 1] = g;
  frame[i + 2] = b;
}
function rect(x0, y0, w, h, r, g, b) {
  const xs = Math.max(0, Math.round(x0));
  const ys = Math.max(0, Math.round(y0));
  const xe = Math.min(W, Math.round(x0 + w));
  const ye = Math.min(H, Math.round(y0 + h));
  for (let y = ys; y < ye; y++) for (let x = xs; x < xe; x++) px(x, y, r, g, b);
}
function digits(text, x0, y0, scale, r, g, b) {
  let cx = x0;
  for (const ch of text) {
    const rows = GLYPHS[ch];
    if (rows) {
      for (let ry = 0; ry < 5; ry++) {
        for (let rx = 0; rx < 3; rx++) {
          if ((rows[ry] >> (2 - rx)) & 1) rect(cx + rx * scale, y0 + ry * scale, scale, scale, r, g, b);
        }
      }
    }
    cx += 4 * scale;
  }
}

function drawFrame(t) {
  // Inside the shot-change window the broadcast has cut away, so the field is not on screen.
  // This is what makes the gap real rather than asserted.
  if (t >= SHOT_CHANGE.start && t <= SHOT_CHANGE.end) {
    rect(0, 0, W, H, 12, 12, 16);
    digits('0', W / 2 - 40, H / 2 - 20, 4, 70, 70, 80);
    digits(String(Math.round(t)), W / 2 + 8, H / 2 - 20, 4, 70, 70, 80);
    return;
  }

  rect(0, 0, W, H, 26, 29, 36);
  const topLeft = project({ x: -FIELD_L / 2, y: -FIELD_W / 2 });
  const botRight = project({ x: FIELD_L / 2, y: FIELD_W / 2 });
  const fx0 = topLeft.x * W;
  const fy0 = topLeft.y * H;
  const fx1 = botRight.x * W;
  const fy1 = botRight.y * H;
  rect(fx0, fy0, fx1 - fx0, fy1 - fy0, 38, 42, 52);
  rect(fx0, fy0, (fx1 - fx0) * 0.16, fy1 - fy0, 62, 32, 38);
  rect(fx1 - (fx1 - fx0) * 0.16, fy0, (fx1 - fx0) * 0.16, fy1 - fy0, 32, 44, 68);
  rect((fx0 + fx1) / 2 - 1, fy0, 2, fy1 - fy0, 70, 76, 90);

  const drawn = [];
  for (const r of ROBOTS) {
    if (unobserved(r.team, t)) continue;
    drawn.push({ b: project(pos(r.team, t)), team: r.team, alliance: r.alliance });
  }
  if (t >= UNKNOWN_TRACK.from && t <= UNKNOWN_TRACK.to) {
    drawn.push({ b: project(unknownPos(t)), team: null, alliance: null });
  }
  drawn.sort((a, b) => a.b.y - b.b.y);
  for (const d of drawn) {
    const x = d.b.x * W;
    const y = d.b.y * H;
    const w = d.b.w * W;
    const h = d.b.h * H;
    const col = d.alliance === 'red' ? [196, 54, 62] : d.alliance === 'blue' ? [48, 104, 200] : [120, 124, 134];
    rect(x, y, w, h, col[0], col[1], col[2]);
    rect(x + 2, y + 2, w - 4, h - 4, 24, 26, 32);
    if (d.team != null) {
      const s = Math.max(1, Math.round(w / 16));
      const label = String(d.team);
      digits(label, x + (w - label.length * 4 * s) / 2, y + (h - 5 * s) / 2, s, 236, 238, 242);
    }
  }
}

const nFrames = Math.round(DURATION * FPS);
const ff = spawn(
  'ffmpeg',
  ['-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', W + 'x' + H, '-r', String(FPS), '-i', 'pipe:0',
   '-an', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '30', '-pix_fmt', 'yuv420p',
   '-movflags', '+faststart', resolve(OUT, 'segment.mp4')],
  { stdio: ['pipe', 'ignore', 'pipe'] }
);
let ffErr = '';
ff.stderr.on('data', (d) => { ffErr += d.toString(); });

const write = (buf) =>
  new Promise((res, rej) => {
    if (ff.stdin.write(buf)) res();
    else ff.stdin.once('drain', res);
    ff.stdin.once('error', rej);
  });

console.log('rendering ' + nFrames + ' frames at ' + W + 'x' + H + '...');
for (let i = 0; i < nFrames; i++) {
  drawFrame(i / FPS);
  await write(frame);
}
ff.stdin.end();
const code = await new Promise((res) => ff.on('close', res));
if (code !== 0) {
  console.error(ffErr.split('\n').slice(-15).join('\n'));
  process.exit(code);
}
console.log('wrote segment.mp4');
