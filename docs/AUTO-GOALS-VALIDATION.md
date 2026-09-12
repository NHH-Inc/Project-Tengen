# Automatic hub detection: September 11, 2026

The default empty goal configuration now generates both 2026 hub regions from a trusted
AprilTag camera pose. No camera-specific goal pixels were supplied for these runs.

## Geometry and counting changes

- Retain decoded tag corners and the surveyed tag quaternions. Fit camera pose plus separate
  horizontal and vertical focal scales, with whole-tag outlier rejection and spatial coverage
  checks. This handles the vertically resized wide camera panel in the tested broadcast.
- Derive the carpet homography from that pose for robot locations. Project the elevated hub
  funnel in 3-D for goal counting, respecting the actual source resolution and rounded model crop.
- Let fast balls first visible near a goal keep a track even without a visible robot launch.
  A short, observed inward approach can establish an entry after an occluded rim crossing.
- Require inward depth and consecutive observations; inset boundaries by apparent ball radius;
  reject tiny fragments, repeated-image aliases, and ambiguous re-arming after a bounce.
- Check two decoded AprilTags near their expected positions every second and on detected cuts.
  Pause goal counting while the view is invalid and resume when it returns. Save these intervals
  for playback so the player also hides stale goal regions.
- Keep visible goal entries separate from robot-attributed makes. Occluded flight paths do not
  justify assigning an entry to a guessed shooter.

Geometry comes from the vendored WPILib field layout, the
[FIRST 2026 manual, sections 5.4, 5.10 and 5.11](https://firstfrc.blob.core.windows.net/frc2026/Manual/HTML/2026GameManual.htm),
and the funnel drawing in the
[hub assembly instructions](https://firstfrc.blob.core.windows.net/frc2026/FieldAssets/TE-26300-build-instructions.pdf).
The rim is 72 inches high and 41.7 inches across flats; FUEL is 5.91 inches in diameter.

## Actual-video checks

All paths below are relative to the repository. The default crop was
`left=.02, top=.035, right=.98, bottom=.66`. Both sources are 1920 × 1080.

| Check | Full match | Opening comparison clip |
| --- | --- | --- |
| Source | `data/segments/j8wz5vw5XfE_00000_00215.mp4` | `data/validation/rapid-fire-review/opening.mp4` |
| Source frames | 12,834 | 961 |
| Source frame rate | 59.822 fps | 60 fps |
| Retained AprilTags | 5, 6, 8, 13, 14, 17, 18, 27 | 5, 8, 13, 14, 17, 18, 27 |
| Corner correspondences | 32 | 28 |
| Maximum fitted corner residual | 4.581 pixels | 3.599 pixels |
| Automatically projected hubs | 2 | 2 |

The complete **Einstein Final Tiebreaker** replay analyzed all 12,834 frames and reports
**561 observed entry candidates: 160 blue, 401 red**. Of these, 24 are linked to a shooter
and 537 are unassigned. It detected 533 launch attempts. This is an offline run, averaging
12.44 processed frames per second on this machine; playback runs at the source frame rate.
During match play, several brief tag-visibility failures pause counting for one or two seconds;
the exact intervals are in `full-verified/report.json`. These pauses and crowded/occluded balls
remain sources of missed entries. Counting stops on the final camera change at 177.56 seconds.

The opening replay reports **39 entries: 23 blue, 16 red**, compared with **10: 5 blue,
5 red** in `data/validation/goal-review/final-review/report.json`. Only one new entry has enough
trajectory evidence to identify its shooter; 38 remain unassigned. Counting was paused during
the opening graphic, from 0 to 4.05 seconds. Candidate contact sheets and excluded edge/bounce
examples were visually inspected. The conservative final checks still miss some crowded entries.

These are detector output counts, not measured recall, precision, official score, or proof
that a hub was active. There is no human-labelled full-match reference. The method is currently
specific to the surveyed 2026 field and a stationary, sufficiently clear wide view; new venues,
camera angles, strong lens distortion, different field variants, and sustained occlusion need
separate validation. Reprojection residual measures fit consistency, not independent goal accuracy.

The final machine-readable outputs are in `data/validation/auto-goals/full-verified/` and
`data/validation/auto-goals/opening-delivery/`: `report.json`, `goal_entries.jsonl`, `shots.jsonl`,
`ball_scouting.config.json`, and `annotated.mp4`. The app plays the native source with synchronized
canvas annotations; the diagnostic MP4 also burns in ball tracking details.
`full-verified/annotated.browser.mp4` is an H.264 copy of that diagnostic video with the original
match audio, suitable for standalone playback.

## Reproduce

Run in the vision environment with the repository as working directory. A new end-to-end run:

```powershell
python -m training.track_yolo `
  --model data/models/robot-yolo11n-reviewed-aug-20260905-640-100ep/weights/best.pt `
  --video data/segments/j8wz5vw5XfE_00000_00215.mp4 `
  --output data/jobs/auto-goals-new/tracks.jsonl `
  --auto-homography --ball-config analysis/config/ball_scouting.example.json `
  --shots-output data/jobs/auto-goals-new/shots.jsonl `
  --annotated-output data/jobs/auto-goals-new/annotated.mp4 `
  --image-size 960 --device 0
```

To repeat only ball analysis using the saved robot tracks:

```powershell
python -m tools.replay_ball_scouting `
  --video data/segments/j8wz5vw5XfE_00000_00215.mp4 `
  --robot-tracks data/validation/auto-goals/full-match/tracks.jsonl `
  --config analysis/config/ball_scouting.example.json --auto-homography `
  --output-dir data/validation/auto-goals/replay-new --annotated
```

Independent CLI calibration on the actual full video sampled 24 frames and reproduced the
32-corner solution in `data/validation/auto-goals/full-cli-calibration.json`. No hand-marked
extra points or fixed camera goal polygons were used. Existing manual goal configs still override
generation; `auto_goals: false` explicitly disables it.

Validation: **484 Python tests pass**, including broadcast aspect recovery, inconsistent tag
rejection, resolution/crop transforms, marker-based camera movement detection and recovery,
fast arrivals with no visible robot, partial approaches and adjacent duplicate frames.
The web production build and 10 existing scoring/parser/playback checks pass.

The local review uses `data/validation/auto-goals/review.db` and `review-data/`, separate from the
normal application database. Vite serves port 5173 and the ingest API serves port 8080.
To restart this saved review on this machine, run
`.\data\validation\auto-goals\start-review.ps1` in PowerShell from the
repository. The script starts only missing listeners; it does not replace services already using
those ports. The first completed job is loaded automatically when the page opens.

## Marker continuity follow-up

A single unsuccessful tag decode previously invalidated the goal regions until the next
one-second check. Frame inspection of all 13 short interruptions found readable tags again
after 0.05–1.354 seconds. The failure was temporary visibility, not a new goal position.

The analyzer now retains a recently verified pose for up to three seconds and retries every
0.1 seconds while tags are unreadable. Detected scene changes and timestamp discontinuities
discard the grace period immediately; sustained tag loss still expires the pose. Startup
requires a successful verification. Goal geometry is unchanged.

The updated full replay is in `data/validation/auto-goals/full-stable-markers-final/` and is
loaded into the same localhost review job. Earlier counts and videos above remain historical
artifacts. All 486 Python tests pass, including prolonged occlusion followed by recovery,
fast retries, sustained failure, and immediate invalidation on a detected cut.

The final replay analyzed all 12,834 frames with **zero in-match camera gaps**, eliminating
all 13 earlier interruptions. Its only invalid intervals are startup (0–4.246 seconds)
and the post-match view (179.165 seconds onward). Goal polygons match the earlier replay
exactly. Continuous counting produces 644 candidate goal entries (27 attributed and
617 unassigned); these are detector results, not human-verified scoring accuracy.
