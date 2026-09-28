# Desktop verification — September 25, 2026

The corrected desktop player was verified on this Apple Silicon Mac with the original
high-frame-rate match recording, native Qt video presentation, the broadcast crop, robot
and shot overlays, and the heat map visible.

## Playback measurements

Each measurement ran for 10 seconds with six robot tracks, 435 shot records, two goal regions,
and 2,389 downsampled heat-map observations loaded. Received-frame and uniquely painted-frame
counts were measured against wall time, not inferred from the video's metadata.

| Source | Measured painted FPS | Playback speed | 95th-percentile frame interval |
|---|---:|---:|---:|
| 30 fps H.264 | 29.70 | 0.987× | 35.59 ms |
| 60 fps H.264 | 60.06 | 1.001× | 17.76 ms |

Both checks passed with no GUI errors. Results and screenshots are saved in
`data/desktop/verification/playback-30fps.json`, `playback-60fps.json`, and their matching PNGs.

## Full match analysis

Run: `data/desktop/runs/c6bd0d9a-bd6f-485b-a76e-f2e1c594ce99/`.

- Original recording: `https://www.youtube.com/watch?v=refM5LLkuJ8`.
- Full 214.5-second video analyzed at 30 fps: all 6,435 frames processed.
- Six robot track fragments, 435 shot records, 130 goal-entry observations, and 441 events.
- Trustworthy carpet calibration from AprilTags 5, 8, 17, and 18; two automatically projected
  goal regions and 18,113 robot samples with field positions.
- The run's optional `playback_path` selects the synchronized 60 fps original for review.
  Its `local_path` and options still identify the actual 30 fps analysis input. Both encodes
  begin at the same original timestamp; there is no time-stretching or optical-flow synthesis.
- Counts above are detector output, not human-validated scouting accuracy.

## Regression checks

**89 tests passed**, covering native playback/seeking, crop coordinates and full-frame toggling,
shot timing and count deduplication, heat-map time cutoff/filtering, calibration/shot export,
correction persistence/undo, cancellation and failure handling, plus existing YOLO, statistics,
automatic-goal, and goal-scoring regressions. `git diff --check` also passed.

The initial conversion had used an old 2 fps annotated export and omitted the web player's
crop, shot renderer, and density heat map. Those limitations are corrected. Existing 2 fps files
remain 2 fps and now display a warning; use the current high-frame-rate run for smooth playback.
