# Rapid-fire validation — 2026-09-10

The replacement pipeline was checked against the original implementation from Git HEAD,
synthetic rendered videos with known launch counts, cached robot tracks on local match footage,
and the configured YOLO runtime. No model retraining was performed.

## Synthetic comparison

Both implementations received the same 30 FPS frames, robot boxes and test thresholds.
Each case contains twelve physical launches from the same robot through the same lane.

| Case | Original implementation | Updated implementation |
| --- | ---: | ---: |
| Ball visible for only two frames, one launch every two frames | 0 / 12 | 12 / 12 |
| Separated balls visible for seven frames | 12 / 12 | 12 / 12 |
| Touching balls, eight-pixel centre spacing and five-pixel radius | 0 / 12 | 12 / 12 |

The 26 additional regressions also cover gate-only counting with all contour detections
removed, missing observations, moving robots, duplicated broadcast images, static decorations,
motion blur, floor contacts, inbound throws, optical-flow rejection, camera cuts, timestamp
validation, track coasting, event fusion, and reference-label matching. These are controlled
regressions, not an estimate of match-video precision or recall.

## Local footage

The replay used `data/validation/rapid-fire-review/opening.mp4` and cached robot boxes from
`data/validation/rapid-fire-review/strict/tracks.jsonl`, with the same model crop. It analyzed
all **961 source frames at 60 FPS**, including repeated broadcast images. The final replay
emitted **40 attempts**: 18 from longer trajectories, 13 from short launches and 9 from paired
gate pulses. Outcomes remain unknown because the example configuration has no scoring geometry.

Output: `data/validation/rapid-fire-review/final-review/` contains the annotated video,
shot evidence, detection-method mapping and timing report. Runtime was about 59 seconds on
the local CPU for the ball replay including annotation/video writing, without rerunning YOLO.
This is an offline analysis cost, not a real-time throughput claim.

The prior saved stride-two run contained 25 attempts. That count is contextual only: a larger
count does not establish better accuracy. Visual review caught yellow-patch jumps, duplicate
gate events and coasting tracks acquiring subsequent balls during development; fixes for those
failures are covered by the regression suite. Full-match precision and recall remain unmeasured
without independently labelled launch timestamps and source robots.

## Integration checks

- **454 Python tests passed** in the project's ingest environment.
- **73 API contract smoke checks passed**.
- The real configured YOLO runtime processed a 24-frame clip with `--frame-stride 4` and ball
  scouting enabled. It reported the override to 1 and wrote observations at every source-frame
  timestamp (60 FPS), confirming that sampling cannot silently remain enabled for shots.
- Python compilation and `git diff --check` passed.

The replay tool accepts a human-labelled reference and reports one-to-one precision/recall
with robot attribution. See [BALL-SCOUTING.md](BALL-SCOUTING.md) for configuration, commands and
remaining visibility limits. Perfect recall cannot be inferred from these tests: completely
occluded balls and an unresolved continuous yellow stream still contain insufficient evidence
for an exact count.
