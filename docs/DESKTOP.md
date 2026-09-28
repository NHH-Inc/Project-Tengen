# Tengen Python desktop

Tengen's native PySide6 application uses Qt for video/audio playback and calls the existing
Python YOLO/ByteTrack analyzer through a background subprocess. The video is presented by
Qt’s native graphics video item, without a Python RGB conversion/copy for every frame. It does not start FastAPI,
Uvicorn, Vite, a browser, or an HTTP listener. Local runs are JSON/JSONL files, with no database
server required. The previous web/API implementation and its data remain available separately.

## Start on this computer

Open `~/Applications/Tengen.app` in Finder, or double-click `Launch Tengen.command` in the
project folder. Both launch the isolated `.venv-desktop` Python 3.11 environment. The launcher
is a small application bundle that refers to this checkout; keep the folder in place. If you
move it, recreate the virtual environment and reinstall the launcher at the new location.

From Terminal:

```sh
.venv-desktop/bin/python tengen.py
.venv-desktop/bin/python tengen.py --demo
.venv-desktop/bin/python tengen.py /path/to/match.mp4
.venv-desktop/bin/python tengen.py --run data/desktop/runs/<run-id>
```

For a fresh Mac, install Python 3.11 (`brew install python@3.11`) and run `./setup-desktop.sh`.
Then run `.venv-desktop/bin/python -m desktop.install_macos` for the Finder launcher.
**Playback frame rate follows the source video.** The application does not throttle it to an
analysis/snapshot sample rate. The old `yolo-match-annotated-2fps.mp4` really contains only two
frames per second; opening it still shows a low-frame-rate warning. Use the original recording
for smooth movement. On this Mac, native 30 fps and 60 fps H.264 copies of the full match are
available in `data/desktop/media/match-30fps.mp4` and `match-60fps.mp4`. They were made from the
original high-frame-rate recording, not by upsampling the old 2 fps export.

`requirements-desktop.txt` specifies the desktop/inference dependencies;
`requirements-desktop-lock.txt` records the exact verified environment on this Apple Silicon Mac.
The existing `.venv` and `.robot-venv` are left intact.

On other systems, create a Python 3.11 virtual environment, install
`requirements-desktop.txt`, and run `python tengen.py`. Only macOS was verified here.

## Review a video

1. **Open video…** selects a local MP4, MOV, MKV, AVI, or WebM file.
2. **Analyze video…** selects a trained `.pt` robot model and season. CPU is the tested default;
   `mps` requests the Apple GPU and `0` requests an NVIDIA GPU. Analysis processes every source
   frame. The default model is the existing checkpoint at
   `data/models/yolo-v3-960-20260906/weights/best.pt`.
3. Watch progress while the native player stays responsive. **Cancel analysis** stops the
   worker and its detector child. Completed runs appear in the left-hand list.
4. Play, pause, seek, change speed, mute audio, or toggle boxes/shots. **Crop broadcast**
   defaults to the same upper-camera crop as the web player. Robot boxes and ball trajectories
   share the crop coordinates; turning the crop off transforms those overlays into the full frame. Double-click an event to seek
   to its timestamp. Boxes use the decoded frame's timestamp and never interpolate over an
   explicit camera-cut/occlusion gap.
   **Shots** shows launch trails, outcome labels, calibrated goal regions, and counts through the
   current time. Double-click a row in the Shots tab to jump to its launch. The **Heat map** tab
   uses the web field background, dwell density, robot trails, and current robot markers; its team
   filter and time cutoff prevent showing future movement. Density and path geometry are cached
   at 4 Hz while lightweight current markers follow video frames.
5. Select a robot in **Robots**, then **Assign team**, or double-click its row. The correction
   also updates that track's events. Use **Add event**, **Edit event**, **Delete event**, and
   **Undo** for event review. Event times are relative to the selected clip; phases use the
   checked-in season configuration.
6. **Export…** saves a full JSON package or an event CSV. **Raw model output** shows/exports
   unchanged model results and disables corrections until unchecked.

**Load demo** is explicitly synthetic: it uses the included 152-second video, seven tracks,
and known event stream. It is useful for exercising review controls without running inference.
Demo edits are stored outside the fixture folder.

## Local files

- `data/desktop/runs/<uuid>/job.json`: source path and run status.
- `options.json`: detector and calibration settings for this run.
- `tracks.jsonl`, `tracks.raw.jsonl`, `events.jsonl`, `result.json`: analyzer output.
- `shots.jsonl`: detailed shot evidence, when ball scouting is configured.
- `annotated.mp4`: optional video with the detector's annotations (generated without audio).
- `desktop-corrections.json`: reversible review changes; raw analyzer files are untouched.
- `data/desktop/demo-corrections.json`: changes made while reviewing the synthetic demo.
- `data/desktop/robot_image_exports/`: periodic robot snapshots from the analyzer.

Source videos stay in their original location. Keep those files to play saved analyses later.
The desktop run library is separate from the legacy `frc_scouting.db` and API job library.
YouTube/live stream ingestion and Google Sheets transport remain in the legacy API; the desktop
workflow accepts local recordings and exports files that can be imported into Sheets.

## Analysis limits

The desktop conversion does not retrain the detector. It still uses the existing broadcast crop
(top portion of the image); camera views outside that format can require analyzer configuration
changes. Track fragments are not a reliable count of unique robots, and team numbers require
review when no identification is present.

Ball scouting now defaults to the same starter JSON as the web/API setup, and automatic
AprilTag calibration defaults on. A trustworthy pose generates field positions and goal regions;
an explicit camera-specific calibration/config can override it. If calibration fails, the UI
reports unavailable positions/goals instead of inventing them. Clear the ball config to disable
shot extraction. Older runs made without these settings must be re-analyzed to acquire shots and
field positions; adding a renderer cannot recover observations that were never analyzed.
An arbitrary local clip is assumed to begin at match time zero for phase/event purposes; trim
pre-match footage before analysis. Startup position alliance locking is disabled for arbitrary
clips because a video can start mid-match.

The current checked-in season file has placeholder scoring values. The desktop therefore says
score comparison is unavailable instead of displaying a fabricated accuracy result. Verify and
update season configuration before relying on reconstructed scores.

## Verification

```sh
.venv-desktop/bin/python -m pip install pytest
.venv-desktop/bin/python -m pytest -q desktop/tests ingest/tests/test_yolo_orchestrator.py ingest/tests/test_stats.py
.venv-desktop/bin/python -m desktop.verify --video data/desktop/media/match-30fps.mp4
```

The second command covers gap-aware interpolation, persistent corrections, raw data preservation,
exports, failure handling, actual native video playback/seeking, and the event editor. The final
command opens the real GUI, starts its actual analysis worker, verifies playback and seeking of
the completed run, checks export, and checks that the application has no listening server socket.
It saves `data/desktop/verification/report.json`, `desktop-match.png`, `scouting.json`, and
`events.csv`. This is an execution test, not a detector-accuracy benchmark. The current verification uses
the original high-frame-rate match recording, with ball scouting and automatic calibration
enabled. Human-labeled ground truth is still required to measure shot-counting accuracy.

To measure playback with robot, shot, and heat-map layers enabled:

```sh
.venv-desktop/bin/python -m desktop.benchmark --run data/desktop/runs/<run-id>
.venv-desktop/bin/python -m desktop.benchmark --run data/desktop/runs/<run-id> --video data/desktop/media/match-60fps.mp4
```

The report includes received and uniquely painted frame counts, wall-clock frame rates, playback
speed, frame timing, and a screenshot. It requires at least 90% of the native frame rate.
Qt video integration follows the official [QGraphicsVideoItem documentation](https://doc.qt.io/qtforpython-6/PySide6/QtMultimediaWidgets/QGraphicsVideoItem.html).
