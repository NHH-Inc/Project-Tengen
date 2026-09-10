# Goal-entry validation — 2026-09-10

The scoring system observes entries independently of robot launches. It requires the configured
approach, direction, interior and depth evidence; unknown sources do not acquire invented robots
or shot attempts. Configuration and output details are in [BALL-SCOUTING.md](BALL-SCOUTING.md).

## Controlled checks

- **474 Python tests passed**, including 20 added scoring/API cases. Cases cover ordered goal
  approach, scoring depth, rim reversal, upward flights, side exits/reentry, disappearing balls,
  observation gaps, duplicate broadcast images, scene resets, two balls scoring together,
  duplicate track evidence, same-track attribution, ballistic relinking, competing sources,
  competing entries, old API outputs and malformed sidecars.
- **73 API smoke checks passed**, with external exports and network workers disabled.
- The production web build and fixture validation passed. **10 scoring parser, playback and
  contract checks passed**, including scrubbing backwards, avoiding linked-entry double counts,
  legacy made shots, unknown sources, malformed timestamps and the goal-entry JSON contract.
- The configured YOLO Python runtime processed the existing 24-frame smoke clip on CPU, with
  `--frame-stride 4` deliberately requested. It overrode stride to 1, wrote robot observations at
  all 24 source timestamps and wrote one entry to `goal_entries.jsonl`.

## Real video replay

Input: `data/validation/rapid-fire-review/opening.mp4`, using cached robot boxes from
`data/validation/rapid-fire-review/strict/tracks.jsonl` and
`analysis/config/ball_scouting.einstein-wide.json`. The profile was measured against the supplied
fixed wide camera view and default model crop; it is not valid for arbitrary videos.

All **961 frames at 60 FPS** were analyzed. The replay counted **10 observed entries**:
**five in the blue hub and five in the red hub**, including two separate blue entries confirmed
on source frame 796. Processing took about 72 seconds including annotated video output, without
rerunning YOLO. The existing launch detector still reported 40 attempts.

All 10 entries remained **source unknown**: the available trajectories did not establish which
launch produced each goal entry. The broadcast graphic occludes substantial portions of the
flight paths. Accordingly, the 40 launch records retain unknown outcomes and their robot make
counts remain zero. Independent basket totals and robot-attributed shot outcomes are different
measurements; combining them would produce unsupported per-robot accuracy figures.

The ten detected entries were visually inspected in short source-frame sequences. This is a
development check, not independent ground truth. Recall and precision remain unmeasured until
a human labels every entry and its source where visible. Occluded entries, unresolved touching
balls, and perspective overlap may still be missed or mistaken; a physical entry is also not an
official points total or evidence that a hub was active.

Local outputs under `data/validation/goal-review/final-review/`:

- `annotated.mp4`: full-frame-rate trajectories, calibrated regions, green IN trails and counts.
- `goal_entries.jsonl`: observed crossing and confirmation evidence with source association.
- `shots.jsonl`, `shot_methods.json`, `ball_scouting.config.json`, `report.json`: launch evidence,
  configuration snapshot and measured runtime/counts.

The separate `data/validation/goal-review/runner-smoke/verification.json` records the live
vision-runtime stride/sidecar checks. Large videos and local validation outputs are ignored by
Git; the algorithm, configuration, tests, contracts and this report are committed.
