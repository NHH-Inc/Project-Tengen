# Colour-based ball and shot scouting

The first ball pipeline runs beside the YOLO robot detector in `training/track_yolo.py`. It uses
the same decoded model-crop frame and the same stable robot IDs, but ball detection itself is an
independent OpenCV component in `training/ball_scouting.py`. A future learned ball detector only
needs to supply the same centre/radius detections; launch, attribution, goal, and output logic can
remain unchanged.

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
  --ball-config analysis\config\ball_scouting.my-camera.json `
  --shots-output data\jobs\BALL_TEST\shots.jsonl `
  --annotated-output data\jobs\BALL_TEST\annotated.mp4
```

All configured geometry is normalized to the cropped frame passed to YOLO. The current default
crop is `(left=.02, top=.035, right=.98, bottom=.66)`. It is not normalized to the complete
broadcast image.

## Tune in this order

1. Leave `goals` empty. Tune `hsv_lower`/`hsv_upper`, pixel size, circularity, fill, and aspect
   limits until the yellow outlines in `annotated.mp4` cover real balls without covering field
   graphics or robot decorations. Temporarily set `debug.show_mask` to `true` to see the binary
   HSV mask in the lower-right corner.
2. Tune blob linking. Orange trails and `b<ID>` labels should stay on one physical ball through
   ordinary motion. `maximum_missed_frames` permits a short occlusion; the tracker does not emit
   synthetic observations while coasting.
3. Tune launch thresholds. A ball must first move with one robot for
   `carry_confirmation_frames`, leave its padded box, move fast relative to that robot, increase
   its distance from the robot, and maintain a sufficiently direct path over multiple frames.
   Yellow that merely remains visible inside a moving robot cannot satisfy the exit condition.
4. Add goal geometry and tune it on visible crossings. Green boundaries are makes and red
   boundaries are explicit misses.

Thresholds ending in `_ratio_per_second` are fractions of the processed-frame diagonal per
second, so they remain meaningful across frame rates and nearby resolutions. Size filters are in
pixels because apparent ball size changes strongly with a camera's position; keep a separate
configuration per camera/view.

## Goal geometry

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

The regular `events.jsonl` remains Contract B compatible: every confirmed launch adds one
`shot_attempt`; only a made-boundary crossing adds `shot_made`. Unknown outcomes never become
misses, and unassigned shots keep `track_id: null`.

## Deliberately conservative behavior

- If two robot boxes contain the ball during the carry window, a later launch may be counted but
  remains unassigned.
- Several visible balls inside one robot are separate blob tracks; none count until their own
  track leaves and passes the relative-motion confirmation.
- A temporarily merged/occluded blob may coast for a few frames. No guessed coordinates are
  written to the evidence path.
- Yellow field objects and stationary floor balls can be detected and tracked, but cannot become
  shots without an observed robot-carry-to-departure transition.
- If colour segmentation loses a fast or motion-blurred ball before multi-frame confirmation, the
  system abstains. Lowering confirmation gates increases recall at the cost of false shot events.
