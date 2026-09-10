import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import Ajv from 'ajv/dist/2020.js';
import { parseShotResponse, scoringCountsAt } from '../src/api/shots.ts';

const point = (frame) => ({ frame_index: frame, t_seconds: frame / 60, x: .5, y: .5, radius: .01 });
const wireShot = { shot_id: 's1', launch_frame: 0, launch_t_seconds: 0, robot_track_id: 3,
  ball_track_id: 1, confidence: .9, attribution_confidence: .9, outcome: 'made', goal: 'high',
  outcome_frame: 60, outcome_t_seconds: 1, ball_track: [point(0), point(60)] };
const entry = { entry_id: 'e1', region_id: 'blue_hub', goal: 'high', frame_index: 60, t_seconds: 1,
  robot_track_id: 3, shot_id: 's1', confidence: .9, association: 'same_track',
  ball_track: [point(59), point(60)] };
const response = parseShotResponse({ shots: [wireShot], goals: [], goal_entries: [entry,
  { ...entry, entry_id: 'e2', shot_id: null, robot_track_id: null, t_seconds: 2 }] });
assert.deepEqual(scoringCountsAt(response.shots, response.goalEntries, .9), { made: 0, unassigned: 0 });
assert.deepEqual(scoringCountsAt(response.shots, response.goalEntries, 1), { made: 1, unassigned: 0 });
assert.deepEqual(scoringCountsAt(response.shots, response.goalEntries, 2), { made: 2, unassigned: 1 });
// Scrubbing backwards reconstructs totals rather than retaining future counts.
assert.deepEqual(scoringCountsAt(response.shots, response.goalEntries, .5), { made: 0, unassigned: 0 });
const legacy = parseShotResponse({ shots: [wireShot], goals: [] });
assert.deepEqual(scoringCountsAt(legacy.shots, legacy.goalEntries), { made: 1, unassigned: 0 });
assert.equal(parseShotResponse({ shots: [], goals: [], goal_entries: [{ ...entry, t_seconds: NaN }] }).goalEntries.length, 0);
const schema = JSON.parse(readFileSync(new URL('../../contracts/goal-entries.schema.json', import.meta.url)));
const validate = new Ajv({ strict: false, validateFormats: false }).compile(schema);
const record = { ...entry, schema_version: 1, ball_track_id: 1, crossing_frame: 59,
  crossing_t_seconds: 59 / 60, association_confidence: .9, coordinate_space: 'model_crop_normalized',
  ball_track: entry.ball_track.map((p) => ({ ...p, observed: true })) };
assert.ok(validate(record), JSON.stringify(validate.errors));
assert.ok(validate({ ...record, shot_id: null, robot_track_id: null, association: 'unassigned' }));
assert.ok(!validate({ ...record, coordinate_space: 'full_frame' }));
assert.ok(!validate({ ...record, confidence: 1.1 }));
if (process.argv[2]) {
  for (const line of readFileSync(process.argv[2], 'utf8').split('\n').filter(Boolean)) {
    assert.ok(validate(JSON.parse(line)), JSON.stringify(validate.errors));
  }
}
console.log('10 scoring parser, playback and contract checks passed');
