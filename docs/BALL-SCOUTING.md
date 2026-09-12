# Rapid-fire ball and shot scouting

The ball pipeline runs beside the YOLO robot detector in `training/track_yolo.py`. It uses
the same decoded model-crop frame and the same stable robot IDs, but ball detection itself is an
independent OpenCV component in `training/ball_scouting.py`. A future learned ball detector only
needs to supply the same centre/radius detections; launch, attribution, goal, and output logic can
remain unchanged. Launch counting combines short observed departures and robot-relative paired
photogates in `training/launch_signals.py`. It does not require a complete flight track.

Every source frame is analyzed by default. When `--ball-config` / `FRC_BALL_SCOUTING_CONFIG` is
configured, stride is forced to **1**, including when an older environment requests a larger
stride. Robot-only runs can still explicitly opt into sampling. Expect higher processing cost;
the offline pipeline consumes all frames rather than dropping them to maintain playback speed.

## How rapid-fire counting works

1. The detector preserves native-resolution yellow cores and adds weaker yellow only where
   there is temporal image change. Moving elongated blobs can survive the shape filter, while
   a static yellow bar cannot use that exception. Distance-transform peaks separate touching
   balls; a large uniform blob is never converted into an estimated number of shots.
2. The ball tracker solves a global one-to-one assignment, with explicit unmatched options,
   size consistency, local ballistic prediction and short occlusion coasting. New balls at the
   shooter get a wider initial association gate. Flying tracks cannot use that gate to acquire
   the next ball at the muzzle after an occlusion.
3. A short departure can count after two observed frames, independently of the general track
   confirmation setting. It must start within two ball radii of the robot edge, leave above the
   configured floor-contact cutoff, and move fast and outward relative to the robot. The source
   box is taken from the ball's birth frame. Newly occupied yellow pixels and bidirectional
   pyramidal optical flow verify correspondence; low-novelty touching balls need longer motion
   evidence. The old weaker edge heuristic cannot bypass these checks.
4. A verified launch seeds two thin strips outside that robot's exit edge. The strips follow the
   robot box and measure yellow occupancy at native resolution. Adaptive peak/valley hysteresis
   can separate pulses even when the signal does not fall to zero between balls. A pulse must
   travel from the inner gate to the outer gate at a plausible speed before it becomes a shot.
   The gate expires without recent launch evidence and resets across missing/jumping boxes.
5. Track and gate evidence are fused by crossing time and overlapping observed positions. There
   is no robot-wide firing cooldown. Two matching observations are required to merge evidence
   across separately learned gates; simultaneous balls and single-frame path intersections stay
   distinct. Long tracks are used for visible scoring-boundary crossings when available.

Camera cuts, image-size changes and timestamp gaps clear temporal associations. Unknown outcomes
remain unknown. `confidence` values are heuristic evidence scores, not calibrated probabilities.

## Enable it

Copy `analysis/config/ball_scouting.example.json` to a camera-specific file. Then either set:

```powershell
$env:FRC_BALL_SCOUTING_CONFIG = "analysis\config\ball_scouting.my-camera.json"
$env:FRC_YOLO_SAVE_ANNOTATED = "1"
```

or run the vision process directly:

```powershell
python -m training.track_yolo `
  --model data\models\YOUR_MODEL\weights\best.pt `
  --video data\segments\MATCH.mp4 `
  --output data\jobs\BALL_TEST\tracks.jsonl `
  --auto-homography `
  --ball-config analysis\config\ball_scouting.my-camera.json `
  --shots-output data\jobs\BALL_TEST\shots.jsonl `
  --annotated-output data\jobs\BALL_TEST\annotated.mp4
```

All configured geometry is normalized to the cropped frame passed to YOLO. The current default
crop is `(left=.02, top=.035, right=.98, bottom=.66)`. It is not normalized to the complete
broadcast image.

## Tune in this order

1. Set `auto_goals` to `false` while tuning detection alone. Tune `hsv_lower`/`hsv_upper`, pixel size, circularity, fill, and aspect
   limits until the yellow outlines in `annotated.mp4` cover real balls without covering field
   graphics or robot decorations. Temporarily set `debug.show_mask` to `true` to see the binary
   HSV mask in the lower-right corner.
2. Tune blob linking. Orange trails and `b<ID>` labels should stay on one physical ball through
   ordinary motion. `maximum_missed_frames` permits a short occlusion; the tracker does not emit
   synthetic observations while coasting.
3. Tune launch thresholds and inspect the paired gate overlays (magenta inner, cyan outer).
   `launch_signals.short_track_minimum_hits` defaults to 2; increasing it requires more evidence
   at a recall cost. `maximum_observation_gap_seconds` limits short-track bridging.
   `pulse_minimum_prominence` is an absolute occupancy change and `pulse_relative_prominence`
   controls how deep a valley must be to separate adjacent peaks. Gate spacing is measured in
   ball radii. The carried-ball state machine remains available for longer observed launches.
4. Restore automatic goals and run `--auto-homography`, or supply explicit camera-specific
   geometry. Inspect visible crossings. Green boundaries are makes and red boundaries are explicit misses.

Thresholds ending in `_ratio_per_second` are fractions of the processed-frame diagonal per
second, so they remain meaningful across frame rates and nearby resolutions. Size filters are in
pixels because apparent ball size changes strongly with a camera's position; keep a separate
configuration per camera/view.

## Goal geometry

With `--auto-homography` (or a trusted `--homography` pose), an empty `goals` list now projects
the 2026 hub funnels automatically. Calibration retains all four AprilTag corners and fits
horizontal and vertical focal scales separately to handle squeezed broadcast panels. The carpet
homography locates robots; the associated 3-D camera pose locates the elevated openings.
Explicit goal regions take precedence, and `auto_goals: false` disables generation.

The effective `ball_scouting.config.json` records generated boundaries, tag IDs, calibration
residual and camera validity intervals. Two decoded tags must remain near their calibrated
positions. A recently verified pose survives up to 3 seconds of unreadable tags, with retries
every 0.1 seconds instead of every second. Detected cuts or timestamp gaps immediately discard
that grace period; sustained tag loss also pauses counting. Returning to the calibrated view
resumes it. The player hides stale goal regions during those intervals. See
[automatic-goal validation](AUTO-GOALS-VALIDATION.md) for the real-match test and commands.

Goal counting runs independently of launch counting in `training/goal_scoring.py`. It can
count a visible basket entry even when the launch was hidden or its ball track was lost.
Each entry has its own evidence and a physical `region_id`, such as `blue_hub` or `red_hub`.
An entry is credited to a shot and robot only through the same track or a unique short
ballistic continuation. Relinking uses observed positions, fit residual, direction and a
maximum 250 ms gap; competing matches stay unassigned. Predicted positions never count as
goal observations. A shot is no longer closed at its apex when goals are configured.

For an overhead basket, use an approach line, a deeper scoring line and an interior polygon:

```json
{
  "id": "high",
  "region_id": "blue_hub",
  "entry_direction": [0, 1],
  "approach_boundary": {"line": [[0.30, 0.40], [0.70, 0.40]], "direction": [0, 1]},
  "made_boundary": {"line": [[0.32, 0.60], [0.68, 0.60]], "direction": [0, 1]},
  "confirmation_polygon": [[0.30, 0.40], [0.70, 0.40], [0.68, 0.90], [0.32, 0.90]],
  "confirmation_frames": 2,
  "minimum_depth_radii": 0.5,
  "maximum_gap_seconds": 0.085,
  "maximum_confirmation_seconds": 0.25,
  "miss_boundaries": []
}
```

The ball must cross the approach line in order, move inward in the interior on at least two
observations, cross the scoring line and reach the required depth below it. Put the scoring
line inside the actual basket, below the rim where a visible crossing establishes an entry.
Reversal, leaving the interior, stale evidence and long observation gaps cancel confirmation.
Duplicate broadcast images preserve state but supply no additional inward-motion evidence.
There is no goal-wide cooldown: different balls can score in the same frame. Overlapping
evidence from duplicate tracks is fused. Confidence values are heuristic, not probabilities.

`analysis/config/ball_scouting.einstein-wide.json` includes measured regions for the supplied
Einstein Final 1 wide view with the default crop. Select it with `--ball-config` (or `--config`
for replay), or set `FRC_BALL_SCOUTING_CONFIG` to that path and restart the ingest service for
new runs of that view. It is **not a universal camera calibration**. The general example keeps
goals empty for automatic generation; runs without a trusted 3-D pose display an explicit notice.
Recalibrate after changing camera angle, crop or zoom. The scoreboard hides part of this
sample's flight paths, so visible entries can be counted while their source remains unknown.

For backward compatibility, simple line-only goals default to one inward observation and
zero depth. Use the stronger configuration above for baskets; a simple line cannot reject
every rim graze. A physical entry count is not an official point total or proof that a hub
was active under the season rules.

A made outcome may use a finite directed line:

```json
{
  "id": "high",
  "entry_direction": [1, 0],
  "made_boundary": {
    "line": [[0.89, 0.25], [0.89, 0.48]],
    "direction": [1, 0]
  },
  "miss_boundaries": [
    {
      "line": [[0.92, 0.50], [0.92, 0.70]],
      "direction": [1, 0]
    }
  ]
}
```

or a polygon, in which case a make is an outside-to-inside transition in `entry_direction`:

```json
{
  "id": "low",
  "entry_direction": [-1, 0],
  "polygon": [[0.04, 0.46], [0.11, 0.46], [0.11, 0.64], [0.04, 0.64]],
  "miss_boundaries": []
}
```

Image `x` increases rightward and `y` increases downward. A line is finite: passing beside its
endpoints is not a crossing. Multiple geometry entries may share an `id` (for example, the two
physical high goals), but the id must be legal in the selected season contract.

A miss is only recorded after crossing one of the optional directed `miss_boundaries`. A ball
that disappears, times out, is occluded, or merely passes near a goal stays `unknown`.

## Outputs and review

`shots.jsonl` is validated by `contracts/shots.schema.json`. Each row has a UUID, launch frame and
time, stable robot track ID or null, ball track ID, event and attribution confidences, outcome,
goal, outcome frame/time, and the normalized observed trajectory. It is the evidence artifact to
use for later review and retraining.

For jobs run through the service, `GET /api/jobs/{job_id}/shots` returns both the individual rows
and counts for attempted, made, missed, and unknown shots, split by stable robot ID plus an
`unassigned` bucket. These statistics are calculated from the shot rows on demand.

`goal_entries.jsonl` (validated by `contracts/goal-entries.schema.json`) records independent
confirmed entries, crossing and confirmation timestamps, physical region, observed ball path,
source association method, and optional shot/robot IDs. The same API returns `goal_entries`
and `goal_statistics` with made, attributed, unassigned, per-region and per-robot totals. These
totals are separate from attempted-shot statistics: an unseen launch does not become a fake
attempt. Old runs without the sidecar remain readable; reanalysis is required to add entries.

The video player displays cumulative **Balls in**, source-unknown totals and per-goal counts
at the current playback time. Confirmed paths flash green with an **IN** label. Counts rewind
correctly when scrubbing, linked shots and entries count once, and long missing path sections
are not drawn as observed flight. Robot-attributed makes also update existing shot statistics.

The regular `events.jsonl` remains Contract B compatible: every confirmed launch adds one
`shot_attempt`; a confirmed goal entry linked to that shot adds `shot_made`. Unknown outcomes never become
misses, and unassigned shots keep `track_id: null`.

## Replay and measure accuracy

Cached robot boxes let you tune the ball pipeline without rerunning YOLO:

```powershell
python -m tools.replay_ball_scouting `
  --video data\segments\MATCH.mp4 `
  --robot-tracks data\jobs\BALL_TEST\tracks.jsonl `
  --config analysis\config\ball_scouting.my-camera.json `
  --output-dir data\validation\ball-replay-001 `
  --annotated
```

Use the exact original video and crop that produced the cached boxes. Only short gaps between
cached robot observations are interpolated. Outputs include a full-rate annotated video,
`shots.jsonl`, `shot_methods.json`, and `report.json` with frame counts and processing speed.
For precision/recall, supply `--reference-shots labels.json`: a JSON array of independently
labelled `{ "launch_t_seconds": 12.3, "robot_track_id": 1 }` records. Matching is one-to-one,
checks robot attribution, and defaults to a 75 ms tolerance. Without labels, the report explicitly
marks accuracy as unmeasured. A higher shot count by itself is not evidence of higher accuracy.

Regression tests: `python -m pytest ingest/tests/test_ball_scouting.py ingest/tests/test_rapid_fire.py -q`.
The synthetic suite includes same-lane bursts, touching balls, two-frame visibility, brief
occlusions, duplicate 30 FPS images in a 60 FPS stream, camera cuts, moving carried balls, inbound
throws, rejected optical flow, gate-only counting, and cross-method duplicate suppression.

## Limits and conservative behavior

- If two robot boxes contain the ball during the carry window, a later launch may be counted but
  remains unassigned.
- Several visible balls inside one robot do not count merely because they are yellow or moving.
  A local departure or a verified outward pair of gate pulses is required.
- A temporarily merged/occluded blob may coast for a few frames. No guessed coordinates are
  written to the evidence path.
- Off-screen/inbound balls have no observed robot launch and are excluded from robot shot counts.
  The system does not claim to identify an unseen human player.
- Height, relative motion, pixel novelty and optical flow reduce floor-contact/decoration errors;
  perspective and occlusion can still make the source ambiguous.
- Completely hidden balls or an unresolved continuous yellow stream cannot be counted exactly
  from pixels alone. A constant occupied gate is not expanded into an assumed firing cadence.
  Better camera coverage, exposure, resolution or a trained ball detector may still be needed.
- Paired gates learn their location from an observed departure, so a fully obscured burst onset
  can be missed. A single gate lane can also merge simultaneous side-by-side launches; distinct
  blob tracks can resolve them when visible. Camera-specific labelled validation is necessary.

The global assignment uses [SciPy's linear assignment solver](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html).
Blob separation uses [OpenCV distance-transform peaks](https://docs.opencv.org/4.x/d2/dbd/tutorial_distance_transform.html).
