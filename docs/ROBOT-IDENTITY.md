# Robot identity through occlusion

YOLO detects robots, ByteTrack supplies short-lived tracklet IDs, and
`AppearanceTrackMemory` in `training/track_yolo.py` assigns those tracklets to at most six persistent
robot identities. A hidden robot may return with a new ByteTrack ID, or ByteTrack may swap two
similar robots' IDs. Raw ID continuity is evidence, not proof of identity.

## Motion prediction

For each robot, fit its recent visible positions to a velocity vector: speed plus direction.
Least squares over a short history reduces box jitter and avoids the old estimator's bias toward
zero speed during its first few observations. Field positions in feet take priority when both
observations are calibrated. Otherwise use normalized image coordinates; supplied camera offsets
are subtracted from the motion history as well as the candidate displacement.

When a robot reappears, compare its real detection with the predicted position. The matching
radius starts with positional slack and expands with the hidden time and possible acceleration.
After one second, stop extending the old heading and widen uncertainty instead. A physical
maximum-speed gate still rejects impossible travel, regardless of appearance.

The association score combines this motion evidence with protected appearance templates, alliance,
edge/reentry constraints, and raw-tracklet continuity. Assignment is one-to-one over the complete
frame. If competing assignments are too similar, leave the detection unidentified instead of
forcing a swap. A longer disappearance resets the velocity history on reacquisition, since an
average displacement through a hidden turn is not the robot's current speed.

Predictions are internal association hints. They never create detections, advance `last_seen`,
insert positions during an occlusion, or fill the public track's explicit gaps. Saved runs must be
reanalyzed to apply the updated tracker. The YOLO weights are unchanged; this is a tracking change,
not retraining.

## Defaults

The injectable `ReIDConfig` holds the motion parameters. These defaults apply automatically to
the desktop, file analyzer, and streaming analyzer; no new dependencies or flags are required.

| Parameter | Default | Meaning |
|---|---:|---|
| `motion_history_seconds` | 0.5 s | Recent visible observations used to fit velocity; also capped at 64 samples. |
| `motion_min_span_seconds` | 0.08 s | Minimum observed time span before trusting speed/direction. |
| `motion_reset_gap_seconds` | 0.5 s | A longer gap starts a fresh velocity history when the robot returns. |
| `prediction_horizon_seconds` | 1.0 s | Maximum travel time extrapolated along the old heading. |
| `normalized_acceleration` | 0.75 frame units/s² | Image-space uncertainty growth for acceleration/turns. |
| `field_acceleration_ftps2` | 20 ft/s² | Calibrated uncertainty growth for acceleration/turns. |

The existing maximum speed, motion slack, appearance threshold, memory window and global score
margin remain in force. Untrained velocity estimates use the broad physical travel gate.
Losing field calibration clears its separate velocity history until it can be measured again.

## Verification

```sh
.venv-desktop/bin/python -m pytest -q ingest/tests/test_robot_identity_motion.py ingest/tests/test_yolo_orchestrator.py
```

The motion tests include hidden crossings with identical appearance, new and swapped raw IDs,
different speeds in the same direction, calibrated positions, supplied camera motion, jitter,
hidden turns, long absences, lost calibration, impossible travel, ambiguity and bounded storage.
Existing tests retain the appearance/alliance, stale recovery, six-robot cap and gap guarantees.

Passing those cases establishes the specified behavior, not a general real-match accuracy score.
Evaluate complete matches with labeled identities before changing the model registry's tracking
benchmark. Field calibration is important: the normal image-space fallback assumes a steady view
unless a caller supplies camera motion, so a broadcast camera cut can still require identity review.
