# Experimental C++ fuel tracker

The supplied `fuel_tracker.cpp` is integrated as `analysis/src/fuel_tracker.cpp`,
with its own CMake executable, `analysis/build/bin/fuel_tracker`. It detects robots
with a YOLO ONNX model, finds small yellow blobs near one hub, tracks them, and
counts tracks that disappear inside the hub. Counts are experimental observations,
not verified baskets or official points. `R1`, `R2`, etc. are temporary tracking
IDs, not team numbers.

This is a standalone evaluation tool. The existing `analysis` executable and
Python/desktop scouting pipeline do not automatically run it or consume its counts.
Its manually selected pixel polygons are independent of `Homography.cpp`.

## Build and test

Requires a C++17 compiler, CMake, OpenCV with core, imgproc, videoio, dnn, highgui,
and calib3d (OpenCV 4) or geometry (OpenCV 5). The existing CMake JSON dependency
is reused. OpenCV's module split is described in the
[upstream migration guide](https://github.com/opencv/opencv/wiki/OpenCV-4-to-5-migration).

```sh
cmake -S analysis -B analysis/build -DCMAKE_BUILD_TYPE=Release
cmake --build analysis/build -j 4
ctest --test-dir analysis/build --output-on-failure
```

On this Mac, OpenCV is installed under Homebrew but is not globally linked. Configure with:

```sh
cmake -S analysis -B analysis/build -DCMAKE_BUILD_TYPE=Release \
  -DOpenCV_DIR=/opt/homebrew/opt/opencv/lib/cmake/opencv5 \
  -DCMAKE_PREFIX_PATH=/opt/homebrew
```

Set `-DFRC_BUILD_FUEL_TRACKER=OFF` to build the existing backend without the
additional dnn/highgui requirement. ONNX Runtime is not required by this tool;
it uses OpenCV DNN on CPU.

## Interactive run

From the repository root, using the local model and footage:

```sh
analysis/build/bin/fuel_tracker \
  --video data/desktop/media/match-30fps.mp4 \
  --model data/models/yolo/best.onnx --imgsz 960 \
  --save-config data/test-output/fuel-tracker/selected-hub.json \
  --report data/test-output/fuel-tracker/interactive-report.json
```

Choose a frame with A/D (15 frames), J/L (60 frames), then Space/Enter. Draw the
field polygon, one hub polygon, and optionally the human-player zone. Left-click
vertices, U undoes, Enter finishes, Esc cancels. Canceling the human zone skips it.
Use Q/Esc to stop playback. Saved configurations make repeat runs reproducible.

`--imgsz` must match the ONNX export. The local `best.onnx` requires **960**;
the imported file's original 640-pixel assumption fails on that model. Supported
outputs are raw Ultralytics-style `[1, 4 + classes, boxes]`, without objectness or
embedded NMS. `--robot-class` defaults to 0. Unsupported output layouts fail with
an error. The class count comes from the output tensor.

## Headless run and outputs

```sh
analysis/build/bin/fuel_tracker \
  --video data/desktop/media/match-30fps.mp4 \
  --model data/models/yolo/best.onnx --imgsz 960 \
  --config data/test-output/fuel-tracker/selected-hub.json --headless \
  --start-frame 2700 --max-frames 300 \
  --output data/test-output/fuel-tracker/preview.mp4 \
  --report data/test-output/fuel-tracker/report.json
```

Coordinates in the config are **original-video pixels**, not normalized points
or resized display coordinates. Required keys: `image_width`, `image_height`,
`field`, `hub`; optional `human`. Each polygon is an array of at least three
`[x,y]` pairs, with nonzero area and coordinates within the image. Dimensions
must match the video. Example for a synthetic 200×200 frame:

```json
{
  "image_width": 200,
  "image_height": 200,
  "field": [[0, 0], [199, 0], [199, 199], [0, 199]],
  "hub": [[80, 80], [120, 80], [120, 120], [80, 120]]
}
```

The MP4 includes polygons, robot IDs, fuel trails and inferred counts. The JSON
report records processed frames, processing FPS, robot/fuel observations, created
and pending fuel tracks, and inferred counts. Observation totals count repeated
detections across frames; they are not distinct robot or ball counts. Processing
FPS includes decoding, tracking, drawing and video encoding when enabled, but
excludes setup/model loading. Pending tracks are not force-counted at EOF or at a
frame limit; more frames are needed to observe disappearance. `--start-frame`
applies when loading a config; interactive mode uses the chosen setup frame.

## Changes from the supplied file

- Registered executable and tests with CMake/CTest; CI runs the C++ tests.
- Added portable CLI inputs, configurable model size/class, reusable polygon
  config, headless execution, annotated video and JSON reports.
- Use CPU explicitly, instead of relying on CUDA availability/fallback.
- Validate ONNX output layout and apply NMS only to the selected robot class.
- Retain missing robot tracks internally but do not use stale boxes as current
  robot observations for fuel attribution.
- Recognize fuel first seen inside the hub; require its last observation to still
  be inside the hub before counting a disappearance.
- Keep unknown scorers unattributed rather than labeling them human players.

## Accuracy limitations

- Greedy IoU robot tracking has no motion prediction. Fast motion, crossings and
  occlusion can split or exchange IDs. A synthetic robot moving 35 pixels/frame
  with a 40-pixel-wide box produces **9 IDs across 9 frames**.
- A disappearance inside the hub after six missed frames may be a basket,
  occlusion, color-detection failure, or camera cut. A synthetic occlusion produces
  a false inferred count. There is no trajectory-based basket confirmation.
- Owner assignment happens at track birth, when there is only one observation:
  the original `estimatedOrigin()` cannot yet estimate launch motion. Assigning
  the nearest robot within 140 pixels is weak evidence of the actual shooter.
- Detection examines only the hub bounding box plus 35 pixels. It cannot follow
  most of a shot's flight or reliably establish its origin.
- HSV, blob area (10–350 pixels), association distance and missed-frame limits are
  fixed in pixel/frame units. Resolution, lighting, motion blur, touching balls,
  and frame rate affect results. The detector has no circularity filter.
- Polygons are fixed to one camera view. Camera cuts/pans require a new setup and
  separate run. There is no automatic camera-cut reset in this imported tracker.

The regression checks establish implementation behavior, not match accuracy.
Labeled shots and shooter identities are needed to measure counting precision,
recall and attribution accuracy. See the local evaluation report under
`data/test-output/fuel-tracker/` for the recorded trial.

## Local evaluation — September 26, 2026

Built all targets on this Mac with AppleClang and OpenCV 5.0.0. All **4 CTest
entries passed**, including **22 fuel regression checks**. A generated 12-frame
video, processed through video decoding and the real ONNX model, correctly
produced one fuel track and one unattributed count. Invalid CLI inputs were
rejected. The existing backend also passed a fixture smoke run: 4,560 video
frames, two boundary events, and no fabricated robot tracks when its separate
ONNX Runtime detector was unconfigured. Interactive polygon selection was
compiled but not exercised during this headless evaluation.

Real-footage trials used `data/desktop/media/match-30fps.mp4`, frames 2700–2999
(90–100 seconds), and `data/models/yolo/best.onnx` at 960 pixels on CPU. Both
processed all 300 frames and wrote verified 300-frame, 30-FPS annotated videos.

| Measurement | Saved thin hub polygon | Broader hub enclosure |
|---|---:|---:|
| Processing FPS, including annotated output | 18.84 | 19.16 |
| Fuel observations across frames | 222 | 327 |
| Fuel tracks created | 23 | 28 |
| Inferred counts | 0 | 12 |
| Counts attributed to a robot | 0 | 0 |

The thin polygon came from the existing desktop run's red-hub calibration,
converted from normalized crop coordinates to full-image pixels. The broader
polygon was manually approximated from the inspected 90-second frame to include
the hub enclosure; it is not ground-truth calibration. The broader run created
**70 robot IDs in 10 seconds** and left all 12 counts unattributed. This confirms
substantial identity fragmentation and sensitivity to the chosen hub region.
The 30-FPS input was processed slower than real time.

There are no labeled shot/shooter annotations for this trial, so no accuracy
percentage, precision or recall is claimed. The implementation runs, but these
results do not support using it for reliable per-team scoring yet.

Local artifacts (ignored by Git):

- `data/test-output/fuel-tracker/evaluation.json`: combined measurements and provenance.
- `data/test-output/fuel-tracker/red-hub-enclosure-90-100s.mp4`: broader-region preview.
- `data/test-output/fuel-tracker/red-hub-90-100s.mp4`: thin-region preview.
- `data/test-output/fuel-tracker/enclosure-preview-299.jpg`: final annotated frame.
- `data/test-output/fuel-tracker/red_hub_enclosure.json`: broader-region config.
- `data/test-output/fuel-tracker/red_hub.json`: thin-region config.
- `data/test-output/fuel-tracker/synthetic-report.json`: end-to-end synthetic result.

To reproduce the broader-region trial, use the headless command above with
`--config data/test-output/fuel-tracker/red_hub_enclosure.json`.
