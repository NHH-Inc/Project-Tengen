# Running it

PowerShell on Windows, because that is what all three of us use.

**Every command block below starts with the folder you must be in.** Getting this wrong is the
most common mistake — running the ingest setup from inside `web\` creates a junk venv at
`web\ingest\.venv` and fails with *"Could not open requirements file"*.

Two PowerShell rules that catch people out:

- A relative program needs `.\` in front. `ingest\.venv\Scripts\python` alone gives
  *"not recognized as the name of a cmdlet"*. Write `.\ingest\.venv\Scripts\python`.
- Environment variables are `$env:NAME = 'value'`. There is no `NAME=value command` form.

Throughout, **REPO** means wherever you cloned Project Tengen — the folder containing
`analysis\`, `ingest\`, `web\`. It is not the same path on everyone's machine, so substitute
your own; the commands below use `REPO` rather than pretending otherwise.

---

## The short version

There is one script. It works from any folder, because it resolves paths relative to itself
rather than to your shell.

```bash
# in: REPO
.\run.ps1 setup     # first time only, safe to re-run
.\run.ps1 doctor    # what is installed, what is missing, what is misconfigured
.\run.ps1 web       # the UI on fixture data, no backend
.\run.ps1 full      # ingest service + UI, wired together
.\run.ps1 serve     # build the UI and serve everything on one port
.\run.ps1 check     # every test CI runs
```

`setup` creates the venv, installs both dependency sets, copies `.env.example`, and deletes the
stray `web\ingest\` folder if a previous attempt made one. `full` writes `web\.env.local` for
you — that file is not in the repo, which is why you did not have it.

Start the Windows C++ dependency setup, then open a second PowerShell in REPO to see a live
progress bar (for example, `6/40` packages):

```bash
.\run.ps1 native-setup
.\run.ps1 native-progress
```

The rest of this document is what those commands do, for when something goes wrong.

---

## What exists, and what does not

Be clear on this before you go looking for a command that is not there.

| Step | Status | How to run it |
|---|---|---|
| 1. Pull video off YouTube | **works** | Part 3 |
| 2. Extract frames from a segment | **works** | Part 4 |
| 3. Label frames with the 3 local models | **works** | Part 4 |
| 4. Human review of those labels | **external setup needed** | Roboflow project/review queue, which needs an account owner |
| 5. Train a detector | **ready when labels exist** | Part 5 and `training\\run_rfdetr.ps1` on Robert's CUDA machine |
| 6. Analyse a match (the C++ backend) | **baseline ready** | decodes real MP4s, and runs RF-DETR ONNX + IoU tracking when a local model is configured |
| 7. Review results in the web app | **works** | Part 2 |

Steps 1–3 produce reviewable proposals. Step 4 still needs the Roboflow account/project, but a
temporary explicit auto-label baseline can now be materialized and trained in Part 5. Step 6
emits no robot tracks until a trained local model is configured, rather than pretending a
hand-placed box is a detection.

---

## Field coordinates: calibrating a camera

Without this, positions are pixels, `homography_ok` is false, and speed and distance have no
units. With it they are feet, which is the only form comparable between venues, camera positions
and zoom levels.

The field's AprilTags are surveyed and published, and OpenCV reads their family. The default pose
calibration uses tags at multiple heights, recovers the camera pose, and derives a carpet mapping:

```powershell
python -m ingest.collection.calibrate `
  --video data\segments\<clip>.mp4 `
  --out analysis\config\homography.<venue>.json `
  --region 0.0 0.68 `
  --method pose `
  --hfov-deg 70
```

`--hfov-deg` is the horizontal field of view of the source camera. Use the camera specification or
measure it when possible; the default is only an explicit broadcast-camera approximation. The
calibration stores the assumption and its pixel reprojection error in the JSON. If pose recovery
is impossible, add hand-read carpet points with `--extra-points` (the template explains the
format). `--method plane` remains available as a legacy fallback, but its coordinates are on the
AprilTag plane, not the carpet.

`--region` restricts the search to a band of the frame, given as fractions of its height. **Use it
whenever a broadcast stacks two camera views**, which the 2026 ones do: the same physical tag
appears in both views, and a fit across that boundary mixes two cameras and reports wrong
positions forever without complaining.

Then point the analyzer at the result:

```powershell
$env:FRC_HOMOGRAPHY_CONFIG = (Resolve-Path "analysis\config\homography.<venue>.json").Path
```

### When it cannot finish, and what to do

It refuses rather than guessing, and hands over what you need to finish by hand: a **reference
frame** with a pixel grid and the tags marked, and an **extra-points template**. Read pixel
coordinates for field features you can identify — the corners of the carpet are the usual choice —
pair them with their position in feet, and re-run with `--extra-points`.

Two refusals you should expect:

- **"the points are degenerate"** — the tags lie on a line. This happened on the real İstanbul
  footage: both goal structures carry their 3.68 ft tags at the same height, and a camera looking
  down the field sees all four within **0.1 px of one image row**. Four collinear points cannot
  define a plane mapping, and tags alone cannot calibrate that angle.
- **"camera is not static"** — a tag's pixel position wandered across the clip. A homography
  belongs to one camera pose; pooling detections from a panning camera invents points that were
  never there. Calibrate from a still passage, or per camera position.

### Two things to know about the result

**Four points cannot be checked.** Any four fit a homography exactly, so the reprojection error is
zero by construction and is not evidence. Five or more is where it starts to mean something, which
is what `--extra-points` is for. The tool says which case you are in rather than printing a
reassuring number.

**Pose mode maps to the carpet.** Robot coordinates use the bottom-centre of each detection box,
which approximates the bumper contact point. The output records `mapping_source` as
`carpet_pose`; the legacy fallback records `tag_plane` and its `plane_height_ft`. Neither mode
recovers robot body height or body orientation from a 2D box.

## Team identification needs Tesseract

Optional, and everything else works without it: with no Tesseract, every track simply stays
unattributed, which is the same honest degradation as a missing TBA key or a missing model.

```powershell
winget install --id UB-Mannheim.TesseractOCR
```

The Python side finds it on PATH, or set `TESSERACT_CMD` to the binary:

```powershell
$env:TESSERACT_CMD = "C:\Program Files\Tesseract-OCR\tesseract.exe"
```

Then, on a run that already has `tracks.jsonl`:

```powershell
python -m ingest.attribute_tracks --job data\jobs\<id>\job.json --tracks data\jobs\<id>\tracks.jsonl --out data\jobs\<id>\tracks.attributed.jsonl
```

It reads the roster off the broadcast scoreboard when the job has no TBA alliances, reads the
bumpers, and votes across each track. The input file is never modified.

**This lives in Python, and doc 0 says component 1 owns OCR.** That is a deliberate deviation, not
an oversight: it was built where it could be tuned in minutes rather than rebuild cycles, and
where the accuracy could be measured before committing the team to a vcpkg Tesseract dependency
on everyone's machine. Port it to C++ once the accuracy is worth freezing — the algorithm is the
part that was expensive, and it is written down.

## Part 1 — one-time setup

Do this once per machine. Three separate steps, each in a **different folder**.

### 1a. Web app — needs Node 20+

```bash
# in: REPO\web
cd REPO\web
npm install
```

### 1b. Ingest service — needs Python 3.12+ and ffmpeg on PATH

```bash
# in: REPO   <- the ROOT, not web\
cd REPO
python -m venv ingest\.venv
.\ingest\.venv\Scripts\python -m pip install -r ingest\requirements.txt
```

If you see `Could not open requirements file`, you are in the wrong folder. `cd` to REPO and
retry. If you already made `web\ingest\`, delete it — it is junk:

```bash
# in: REPO
Remove-Item -Recurse -Force web\ingest
```

### 1c. Config

```bash
# in: REPO
copy ingest\.env.example ingest\.env
```

Then open `ingest\.env` and fill in what you have. Nothing is required — every setting degrades
rather than breaking — but without `TBA_API_KEY` you get no alliance data and no accuracy check.
The file is gitignored. Keep it that way, and never paste a key into chat.

---

## Part 2 — the web app on its own

**No backend, no config, no Python.** This is the fastest way to see the whole UI and the right
starting point if you just want to look at it.

```bash
# in: REPO\web
npm run dev
```

Open <http://localhost:5173>.

You get the golden fixture match: a real 152-second video with boxes drawn over it, 224 events,
working corrections, timeline, team stats, heat map, export panel. Three jobs in the queue,
including a deliberately failed one so the retry path is reachable.

Nothing is faked — it serves the real fixture files through the same interface the HTTP client
uses, so anything that renders here renders against the real backend.

To reset corrections you have made: clear site data for `localhost:5173` in your browser.

---

## Part 3 — pulling video off YouTube

This needs the ingest service running. **Two terminals.**

### Terminal 1 — the service

```bash
# in: REPO
cd REPO
.\ingest\.venv\Scripts\python -m uvicorn ingest.main:app --reload --port 8080
```

Check it is alive: <http://localhost:8080/api/health> should say
`{"status":"ok","schema_version":3,...}`. Browsable API docs: <http://localhost:8080/docs>.

### Terminal 2 — the web app, pointed at it

First create the file `web\.env.local` containing one line:

```bash
VITE_API_MODE=http
```

A file, not `$env:`, because Vite only reads env vars at startup and you would lose it every time
you close the terminal. Then:

```bash
# in: REPO\web
npm run dev
```

Now paste a YouTube link into the sidebar. The job walks
`queued → downloading → downloaded → analyzing → complete`, and **the player opens as soon as the
download finishes** — you do not wait for analysis.

### Using the trained YOLO detector with ByteTrack

The ingest service can run the trained Ultralytics checkpoint in a separate vision environment,
so the lighter service environment does not need CUDA or Ultralytics installed. In
`ingest\.env`, set the dedicated Python executable and select the tracker:

```powershell
# in: REPO
YOLO_PYTHON=C:\yolo11-venv\Scripts\python.exe
YOLO_MODEL_PATH=.\data\models\yolo11n-frc-robots-20260901\weights\best.pt
FRC_ANALYSIS_BACKEND=yolo
FRC_YOLO_TRACKER=bytetrack
```

The API then invokes `training\track_yolo.py`, which writes Contract D `tracks.jsonl` and the
database receives those persistent track IDs. Use `FRC_ANALYSIS_BACKEND=auto` to prefer this
path when it is available, or `native` to force the C++ RF-DETR path. Annotated MP4 output is off
by default because the web app draws the stored tracks and an annotated 1080p60 copy is large;
set `FRC_YOLO_SAVE_ANNOTATED=1` when a rendered video is needed.

For a configured homography, point the same runner at the calibration JSON:

```powershell
C:\yolo11-venv\Scripts\python.exe -m training.track_yolo `
  --model data\models\robot-yolo11n-reviewed-aug-20260905-640-100ep\weights\best.pt `
  --video data\segments\unseen-match.mp4 `
  --homography analysis\config\homography.<venue>.json `
  --output data\jobs\JOB_ID\tracks.jsonl `
  --partial-output data\jobs\JOB_ID\tracks.partial.jsonl
```

Each box can then contain `field_x`, `field_y`, `velocity_x_ftps`, `velocity_y_ftps`,
`speed_ftps`, and `motion_heading_rad`. The first valid sample, samples after a tracking gap, and
off-field mappings leave velocity null. Motion heading is the direction the robot travelled; it is
not the robot's body orientation.

The local YOLO runner reads an MP4 lazily: it opens the file, crops each frame as the detector
requests it, and begins inference on the first decoded frame. It no longer writes a complete
temporary cropped video first. The `--stream-url` path does the same through yt-dlp, FFmpeg, and
PyAV, so an unseen video does not need a preprocessing or training pass. ByteTrack associates
detections online from the current frame history.

This is real-time capable, not automatically real-time guaranteed. It is real-time only when
detector/tracker throughput on the selected GPU and `--image-size` meets the source frame rate;
local files run as fast as the machine can process them, while a network stream can be limited by
download or decode speed. The model can detect a new viewpoint immediately, but accuracy depends
on how well its training data covers that venue, camera angle, blur, and occlusion.

Videos that need a login:

```bash
# in: REPO
$env:YTDLP_COOKIES_FROM_BROWSER = 'chrome'
.\ingest\.venv\Scripts\python -m uvicorn ingest.main:app --port 8080
```

Downloaded segments land in `data\segments\`. Note the filename — Part 4 needs it.

### Capture an active livestream

This is useful when you start the ingest service before an FRC broadcast. In the sidebar, paste
the active YouTube link, check **Capture this live stream**, and press **Start live capture**.
The ingest machine records the source locally until YouTube ends the stream. Only then does it
probe the completed MP4 and run normal analysis.

This is **not live scouting**: it will not show robot boxes, teams, or events during a match.
It is simply an automatic way to obtain a usable recording without waiting for YouTube to finish
processing the VOD. Start it only when you have enough free disk for the broadcast, and run it on
your home ingest machine rather than school hardware. If the source is not currently live, the
queue tells you to use ordinary video mode instead.

---

## Part 4 — labelling frames with the local models

### 4a. Install Ollama and the models (~13 GB)

```bash
# anywhere
winget install Ollama.Ollama
ollama pull qwen3-vl:4b
ollama pull qwen2.5vl:7b
ollama pull gemma3:4b
ollama list
```

Ollama serves on `http://127.0.0.1:11434`, which is what the config expects.

**On AMD:** Ollama supports RDNA3 on Windows, so an RX 7800 XT runs these on the GPU. PyTorch
does *not* support AMD on Windows, which is why training happens on the NVIDIA machine.

### 4b. Extract frames from a segment

Use a file from `data\segments\` that Part 3 downloaded.

```bash
# in: REPO
.\ingest\.venv\Scripts\python -m ingest.collection.cli extract `
  --segment data\segments\<the-file>.mp4 `
  --match-id 2026casf_qm42 `
  --video-id dQw4w9WgXcQ `
  --start-offset 120 `
  --config configs\data_collection.example.yaml
```

The backtick `` ` `` is PowerShell's line continuation. Put it all on one line if you prefer.

It prints a **collection id**. Copy it.

### 4c. Run the three models over those frames

```bash
# in: REPO
.\ingest\.venv\Scripts\python -m ingest.collection.cli annotate `
  --collection <the-collection-id> `
  --config configs\data_collection.example.yaml
```

Models load one at a time so they fit in memory. Output goes to
`data\collections\<collection-id>\`: the frames, a manifest, each model's raw proposals kept
separately, and the IoU comparison report.

**These are proposals, not labels.** Keeping each model's raw output is what lets you see where
they disagreed. Feeding unreviewed consensus straight into training teaches the next model to
copy this one's mistakes — that is why step 4 in the table above exists.

---

## Part 5 — training

Training is deliberately separate from ingest: it belongs on Robert's NVIDIA/CUDA machine, not
Justin's AMD Windows machine. The script creates its own environment and trains only a one-class
robot detector.

```powershell
# Materialize one or more annotated collections into RF-DETR's required COCO layout.
# --allow-unreviewed is a visible, temporary v1 quality compromise. Replace it with reviewed
# Roboflow labels as soon as the team project exists.
.\ingest\.venv\Scripts\python -m ingest.collection.cli export-coco `
  --collection data\collections\<collection-id> `
  --config configs\data_collection.example.yaml `
  --output data\datasets\robot-v1 `
  --allow-unreviewed

# On Robert's RTX 3060 machine only.
.\training\run_rfdetr.ps1 -Dataset data\datasets\robot-v1 -Output data\models\robot-v1
```

It uses RF-DETR Small, `640px`, 100 epochs, and `batch-size auto`. The exported model is
`data\models\robot-v1\onnx\inference_model.onnx`. Full directions are in `training\README.md`.

The remaining human setup is still real work: create the Roboflow project/review queue, review
the proposals, and export those reviewed labels. Detector labels are `(image, box)`; Nathaniel's
separate team-ID model is a **classifier** over human-confirmed `(crop, team)` examples from web
track corrections. It cannot be trained honestly until those corrections exist.

---

## Verifying nothing is broken

Run before pushing. `run.ps1 check` runs the same web, ingest, fixture, and C++ checks as CI.

```bash
# in: REPO\web
npm run typecheck
npm run build
npm run validate:fixtures
```

```bash
# in: REPO
.\ingest\.venv\Scripts\python -m pytest ingest\tests -q
.\ingest\.venv\Scripts\python -m ingest.smoke_test
```

| Command | What it actually checks |
|---|---|
| `validate:fixtures` | Every fixture record against `contracts\*.schema.json`, cross-file invariants, and the five required awkward cases |
| `smoke_test` | 71 checks driving every Contract E endpoint against the fixtures — no network, no yt-dlp, no analysis binary |
| `pytest` | Downloader, collection and API unit tests |
| `analysis` | Opens the fixture MP4 with OpenCV, counts its decodable frames, and checks the contract-valid unconfigured-detector output (two match-boundary events and zero invented tracks) |

**The C++ pipeline proof** is part of `run.ps1 check`.

You probably do not have to install CMake. Visual Studio Build Tools ships its own copy and vcpkg
downloads another, but neither puts it on `PATH` — so `cmake` can be missing from your shell on a
machine that already has two working copies. `run.ps1 check` now looks in both places (and at
`C:\vcpkg`) before telling you anything is missing, so on a machine with the C++ workload
installed it just works with nothing set.

You only need the steps below if `check` reports `cmake not installed`. Install Visual Studio
Build Tools with the **Desktop development with C++** workload, then set up vcpkg:

```bash
# Clone and bootstrap vcpkg (skip the clone if C:\vcpkg already exists).
git clone https://github.com/microsoft/vcpkg C:\vcpkg
C:\vcpkg\bootstrap-vcpkg.bat

# Only needed if vcpkg is somewhere other than C:\vcpkg -- that path is found automatically.
$env:VCPKG_ROOT = 'C:\vcpkg'

# in: REPO. First install OpenCV+FFmpeg and ONNX Runtime; use native-progress in a second window.
.\run.ps1 native-setup
.\run.ps1 native-progress

# Then configure and build. This derived-package directory has no spaces: it avoids a current
# ONNX Runtime/vcpkg Windows quoting bug when the repository path itself contains spaces.
cmake -S analysis -B analysis\build -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT\scripts\buildsystems\vcpkg.cmake" -DVCPKG_INSTALLED_DIR="$env:VCPKG_ROOT\frc-analysis-installed"
cmake --build analysis\build --config Release
```

OpenCV and ONNX Runtime are linked on a Windows inference machine. The detector stays disabled
until `FRC_DETECTOR_CONFIG` points to a trained local ONNX file; the reproducible smoke test
deliberately clears that variable so it always checks plumbing rather than local weights.

Both detector families are supported and the family is read from the model rather than declared:
one output is a **YOLO** export (letterboxed, `/255`, suppressed after decoding), two are
**RF-DETR** (`dets` + `labels`, ImageNet-normalised, stretched to square). Getting this wrong
does not fail loudly -- it produces boxes that look plausible and sit nowhere near a robot -- so
`analysis\config\detector.yolo.example.json` is the one to copy for `robot-v1`/`robot-v2`.
`run.ps1 check` finds this toolchain on its own and runs the same build.

**Regenerating fixtures.** Deterministic, so a clean regeneration changes nothing and CI fails
if it does:

```bash
# in: REPO
node fixtures\tools\generate_fixture.mjs --no-video   # data only, fast
node fixtures\tools\generate_fixture.mjs              # also re-renders the video, needs ffmpeg
```

---

## Hosting it

**There is nothing to deploy, and you should not put it on the public internet.**

The web app builds to static files, and the ingest service serves them, so one process on one
port gives you both:

```bash
# in: REPO
.\run.ps1 serve
```

It prints your LAN addresses. At a competition, run that on one laptop and everyone else opens
`http://<that-laptop-ip>:8080` on the venue wifi. If they cannot connect, Windows Firewall is
blocking it — allow `python.exe` on private networks.

Why not a real host: the service needs the database, the downloaded video files, yt-dlp and the
analysis binary, so it is not a static site. Video egress from a cloud host costs money for
files nobody outside the team needs. And doc 2 already notes that bulk downloading is against
YouTube's terms — running that as a public service turns a tolerated local tradeoff into
something with your name on it.

If you later want scouting numbers visible off the network, the Sheets export already does that
job: it is a shareable link with the aggregates and no video.

For access from outside the venue, Tailscale or ZeroTier gives the team a private network
without exposing anything publicly.

---

## When something goes wrong

**`Could not open requirements file`** — you are in `web\`. `cd` to REPO. Delete any
`web\ingest\` folder you created.

**`... is not recognized as the name of a cmdlet`** — missing `.\` on a relative program.

**Port already in use:**

```bash
Get-NetTCPConnection -LocalPort 5173 -State Listen | Select-Object -ExpandProperty OwningProcess | ForEach-Object { Stop-Process -Id $_ -Force }
```

**Jobs appear but no events, and the browser console says a module "does not provide an
export"** — Vite is holding a stale module graph, usually after a `git pull` changed files under
it. **Restart the dev server**; refreshing the page will not fix it. This looks exactly like a
code bug and is not one.

**Export returns 503** — correct when `SHEETS_SPREADSHEET_ID` and
`GOOGLE_APPLICATION_CREDENTIALS` are not both set. It refuses rather than reporting a write that
never happened.

**`alliances` and `tba_score` are null** — no `TBA_API_KEY`, or TBA has no data for that match.
Both are legal.

**A job failed** — read `error_code`, not just the message. It is a closed set and tells you
whether retrying helps: `rate_limited` yes, `video_unavailable` no.

---

## When a broadcast shows the field twice

Many venues composite a second camera view under the main one. Both panels show the same six
robots, so the detector finds each robot twice, the tracker makes two tracks of it, and — because
team attribution is **per track** — a human is asked for the same team number twice while every
shot on that robot is counted twice. A duplicate is worse for the numbers than a miss.

**39 of the 60 sources in `data\segments` do this.** Their seams sit between 0.42 and 0.78 of
frame height, median 0.70, so no single crop line serves them.

### Calibrate once per source

The seam cannot be read off a single frame. The obvious signal — the row where consecutive rows
stop resembling each other — was measured across the 400-frame viewpoint pack and does **not**
separate the two cases: stacked frames score a median 4.55 and single-view frames 4.49. A
scoreboard edge, a spectator rail and a carpet boundary are all just as sharp as a panel join.

What does separate them is where robots are *found*: one view puts them in a single horizontal
band, two views put the same robots in two bands with a gap between. That needs the detector, so
it is a calibration pass:

```powershell
python -m ingest.collection.calibrate_region --model data\robot-v3.onnx --segments data\segments --out analysis\config\regions.json
```

Point the detector config at the result, and the analyzer looks up each video by its filename stem:

```json
{ "regions_path": "../config/regions.json" }
```

An unlisted source keeps the whole frame, which is the safe default.

**Re-run this whenever the model changes materially**, because a model that cannot see the lower
panel cannot find the seam. The same corpus reports 35 stacked sources at a 640px export and 39
at 960, and a source wrongly called single-view goes on double-counting silently. The seams
themselves are stable — only 2 of the 33 sources found by both moved by more than 0.03.

### Filter, do not crop

Two ways to act on a region, and neither argument for the appealing one survived measurement.
Cropping before inference should spend the model's whole input on the panel that matters rather
than two thirds of it, and should be cheaper for handing over a smaller image. Across the stacked
segments of the viewpoint pack:

| export | filter | crop | cropping ahead in |
|---|---|---|---|
| 640px | **4.22** | 4.05 | 7 of 28 segments |
| 960px | 4.50 | 4.52 | 15 of 38 segments |

At 640 filtering wins outright — the model was trained on whole frames, and moving the aspect
ratio away from that costs more than the extra pixels return. At 960 they tie, which fits that
reading: the resolution the crop was buying stops mattering once there is enough of it.

The speed argument is simply false. Per frame at 960: **68.7 ms** whole-frame, **70.0 ms**
filtering, **70.8 ms** cropping. Both letterbox into the same square, so the model does identical
work either way.

So `"mode": "filter"` is the default. `"mode": "crop"` is kept because it is a few lines and
already tested, and because a model *trained* on cropped frames would change the comparison — not
because it helps today. **Re-measure before preferring it.**
