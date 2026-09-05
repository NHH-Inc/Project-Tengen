# Training the robot detector

This one document replaces the four that used to cover this (`Training.MD`, `ROBOFLOW.md`,
`SAM3.md`, `data-collection.md`). Read part 1 to understand what you are doing. Read part 2 when
you are ready to type commands.

---

# Part 1 — What this actually is

## The problem in one sentence

The C++ analysis program has to look at a frame of FRC match video and say *"there is a robot
here, here, and here"* — and no ordinary program can do that, because "robot" isn't a colour or a
shape you can write an `if` statement for.

So instead of writing rules, we show a neural network a few thousand examples of *"this rectangle
contains a robot"* until it learns the pattern itself. That process is **training**. The output is
a **model** — a file full of numbers that, given a frame, returns boxes.

## What we are training it to do — and what we are not

We are training **one** thing: draw a tight box around each robot. One class, literally named
`robot`.

We are deliberately **not** training it to:

- read team numbers off bumpers,
- recognise shots, pickups, or climbs,
- tell red alliance from blue,
- do anything live during a match.

Those are separate problems and each one gets easier once boxes work. Trying to learn "robot" and
"team 254's robot" at the same time makes both worse, because the network splits its limited
capacity across 3,000 team-number classes it will see two examples of each.

## The three different AI models, and why there are three

This is the part that confuses everyone, because three unrelated model families show up in the
pipeline and they all "find robots."

| | What it is | What it's for here | Runs where |
|---|---|---|---|
| **Ollama VLMs** (`qwen3-vl:4b`, `qwen2.5vl:7b`, `gemma3:4b`) | General "look at an image and describe it" models | **Guessing** first-draft boxes so a human isn't starting from a blank frame | Any of our PCs, locally |
| **SAM 3.1** (optional) | Meta's segmentation model — "outline the thing I described" | A *second opinion* on those first-draft boxes | Robert's NVIDIA PC only |
| **RF-DETR** | A real object detector | **The actual product.** This is the model that ships and that C++ runs | Trained on Robert's NVIDIA PC |

The key thing: **only RF-DETR is trained, and only RF-DETR is deployed.** The other two never run
in production and never touch the database. They exist purely to save human labelling time — they
produce guesses, a human fixes the guesses, and the fixed version becomes RF-DETR's homework.

Why not just deploy the Ollama models directly? They're too slow (seconds per frame, and we need
5 frames/second across hours of video), they're not box detectors by design, and they disagree
with each other. Why not deploy SAM? Same speed problem, plus a licence that isn't Apache 2.0.
RF-DETR is Apache 2.0, fast, and purpose-built for exactly this. (See `DECISIONS.md` — we
specifically avoided Ultralytics YOLO because AGPL-3.0 would infect the whole project.)

## The pipeline, as a story

```
1. EXTRACT     match video  ──►  a few hundred still frames
2. PROPOSE     frames       ──►  guessed boxes        (Ollama, optionally SAM)
3. REVIEW      guesses      ──►  correct boxes        ← A HUMAN. This is the real work.
4. TRAIN       correct boxes──►  a model file         (RF-DETR, on the GPU)
5. EXPORT      model file   ──►  inference_model.onnx (a portable format C++ can load)
6. PLUG IN     onnx         ──►  the C++ analyzer finds robots
```

Steps 1, 2, 4, 5 are commands you run. Step 3 is a person clicking in Roboflow for a few hours.
Step 3 is the bottleneck and the whole reason the rest exists.

## Five rules, and the reason for each

**1. At least three different matches. Not one match with more frames.**

Frames from one match look nearly identical — same field, same lighting, same camera operator,
same robots. If frame 400 lands in the training set and frame 401 lands in the validation set, the
model has effectively seen the answer key. It scores 95% and then falls apart on real footage.
That's called *leakage*. We prevent it by keeping every frame from a match together in one split,
which means you need several matches to have splits at all.

**2. A human must review the boxes before training.**

If you train on the Ollama guesses directly, you are teaching RF-DETR to imitate the guesses —
including the mistakes. The ceiling on your model's accuracy is the accuracy of its labels. There
is no way around this and no clever trick that skips it.

**3. One tight box per robot, bumper included.**

"Tight" means the box touches the robot on all four sides. Consistency matters more than
philosophy here: if half the labels include the bumper and half don't, the model learns that the
boundary is ambiguous and gets sloppy at both.

**4. Throw away bad frames instead of labelling them badly.**

Replay close-ups, camera cuts mid-blur, the score overlay, crowd shots — exclude them. A frame
labelled wrong is worse than no frame at all, because it actively teaches a mistake.

**5. Never overwrite a model or the raw output.**

Train into a new folder (`robot-v2`, not on top of `robot-v1`). Same principle as the
corrections system in the web app: raw model output is never destroyed, so you can always tell
what the model actually said versus what a human changed.

## What "good" looks like

After training you get a number called **mAP** (mean average precision, 0–1). Don't read too much
into it early. What actually matters:

- Run the model on a match it has **never seen**.
- Watch the overlay in the web app.
- Do boxes sit on robots and stay there? Do they disappear at camera cuts instead of sliding
  across the screen?

If yes, it works. If the mAP is 0.92 but the boxes drift, the mAP is lying to you — almost always
because of leakage (rule 1).

---

# Part 2 — Doing it

## Where each step runs

| Step | Machine | Why |
|---|---|---|
| Extract, propose (Ollama) | Any of ours | CPU/modest GPU is fine |
| SAM proposals *(optional)* | Robert's RTX 3060 | Needs CUDA 12.6+ |
| Review | Any — it's a website | Roboflow runs in a browser |
| **Train** | **Robert's RTX 3060** | Needs a real NVIDIA GPU |

Training does **not** work on Justin's AMD PC or on a Mac. It does not touch Supabase, Google
Sheets, or `ingest\.venv`.

## Before anything

```powershell
cd <wherever you cloned Project Tengen>
git pull
.\run.ps1 check
```

`check` must print **All checks passed** before you change anything.

## Step 1 — Extract frames

Run once per match, against a downloaded segment in `data\segments\`.

```powershell
.\ingest\.venv\Scripts\python.exe -m ingest.collection.cli extract `
  --segment data\segments\<match-file>.mp4 `
  --match-id <tba-key-like-2026casf_qm42> `
  --video-id <11-char-youtube-id> `
  --source-url "https://www.youtube.com/watch?v=<11-char-youtube-id>" `
  --start-offset 0 `
  --config configs\data_collection.example.yaml
```

It prints a collection path like `data\collections\2026casf_qm42-...`. **Save every one of these
paths** — you need them all in step 4.

Do this for **three or more different matches** (rule 1).

## Step 2 — Get proposed boxes

### One-time: install Ollama and the three models

Windows:

```powershell
winget install Ollama.Ollama
ollama pull qwen3-vl:4b
ollama pull qwen2.5vl:7b
ollama pull gemma3:4b
```

macOS:

```bash
brew install ffmpeg ollama
brew services start ollama
ollama pull qwen3-vl:4b qwen2.5vl:7b gemma3:4b
```

Check it's up — `auto-label` talks to this and nothing else:

```powershell
curl http://127.0.0.1:11434/api/version
ollama list
```

Ollama serves on `127.0.0.1:11434` and needs no API key. **Frames never leave your machine.** The
three models run one at a time and are unloaded between calls, so their weights don't compete for
memory — the whole set fits comfortably in 16 GB.

Connection and model names come from the `ollama` block in
`configs\data_collection.example.yaml`:

```yaml
ollama:
  url: http://127.0.0.1:11434
  models:
    - qwen3-vl:4b
    - qwen2.5vl:7b
    - gemma3:4b
  iou_threshold: 0.40
```

`iou_threshold: 0.40` is the agreement bar: two models "agree" when their boxes overlap by 40% or
more. Raise it for stricter consensus and fewer, cleaner proposals; lower it to surface more
candidates for review.

### Run it

Once per collection:

```powershell
.\ingest\.venv\Scripts\python.exe -m ingest.collection.cli auto-label `
  --collection data\collections\<collection-id> `
  --config configs\data_collection.example.yaml
```

Three local vision models each propose boxes; agreement between them (measured by IoU overlap)
becomes the consensus draft. These are **guesses marked `human_review_required: true`**, not
labels.

To see whether the three models are actually disagreeing usefully — or whether one is just
echoing another and adding nothing:

```powershell
.\ingest\.venv\Scripts\python.exe -m ingest.collection.cli compare-models `
  --collection data\collections\<collection-id>
```

### Optional: SAM — and why SAM 2 is not a substitute for SAM 3

**Do not start here.** SAM is optional, it is not blocking anything, and the review in step 3 is
what the project is actually waiting on. Read this only when you have review capacity to spare.

**SAM 2 cannot do what this integration needs.** The `sam3-propose` command works by handing the
model a *text* prompt — "FRC competition robot" — and getting boxes back. That capability is
called promptable concept segmentation and **SAM 3 introduced it**. SAM 2 accepts only points,
boxes, or masks: you have to tell it where the robot already is, which is the entire problem we
wanted the model to solve. Pointing `sam3-propose` at SAM 2 would leave it with nothing to call.

So if a machine can run SAM 2 but not SAM 3.1, the answer is not "use SAM 2 instead here". It is
either skip SAM, or build a different thing.

#### The different thing, which is probably worth more

SAM 2's real strength is **video object tracking**: give it an object on one frame and it follows
that object through the video with a streaming memory. We have video, and our bottleneck is
per-frame labelling — so the interesting use is not proposals at all:

```text
draw 6 boxes on ONE frame of a match
  -> SAM 2 propagates those objects through the clip
  -> sample the propagated boxes at our extracted-frame timestamps
  -> a whole match labelled from one frame of human work
```

That would turn "review 554 frames" into "draw ~10 frames, then verify". Nobody has built it, and
it is not free:

- **Our frames are 4 seconds apart.** Tracking needs contiguous video, so this runs against the
  source MP4s in `data\segments\`, not against the extracted frames, and samples at our
  timestamps afterwards.
- **Broadcast camera cuts break tracking.** Every cut needs a re-seed. The analyzer already
  detects shot changes for the `gaps` logic, so there is machinery to align with.
- **Robots occlude each other constantly**, and identity swap is SAM 2's classic failure — two
  robots cross, and the tracker comes out following the wrong one.
- **Video memory grows with clip length.** A 3.5-minute clip at 30 fps is ~6,300 frames; expect to
  chunk it or subsample.

Two things do get better with SAM 2: it is **Apache 2.0 with no Hugging Face gating** (SAM 3.1
checkpoints need an approved HF account and a token), and it runs comfortably on a 12 GB card.

#### If you still want SAM 3.1 text proposals

Requires Python 3.12+, PyTorch 2.7+, CUDA 12.6+, and an approved Hugging Face account. Its own
environment — **never install SAM into `ingest\.venv`**.

```powershell
mkdir C:\FRC-SAM3
cd C:\FRC-SAM3
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
git clone https://github.com/facebookresearch/sam3.git
.\.venv\Scripts\python -m pip install -e .\sam3
.\.venv\Scripts\python -m pip install huggingface_hub
.\.venv\Scripts\hf auth login
```

That last command asks for your own Hugging Face token. It stays on your PC — never in Git,
Discord, or chat.

Then test on ten frames before committing to a full run:

```powershell
C:\FRC-SAM3\.venv\Scripts\python.exe -m ingest.collection.cli sam3-propose `
  --collection data\collections\<collection-id> --limit 10
```

If the boxes are genuinely on robots, rerun without `--limit`. If they are not, drop SAM and move
on — reviewed boxes are the goal, not a tour of foundation models. SAM writes its own
`sam3-proposals.jsonl` and never overwrites the Ollama consensus, so the comparison stays honest.

## Step 3 — Review (the actual work)

First, package the proposals into COCO, the format Roboflow reads. Pass **all** your collection
paths at once so the splits are built across matches:

```powershell
.\ingest\.venv\Scripts\python.exe -m ingest.collection.cli export-coco `
  --collection data\collections\<match-1> data\collections\<match-2> data\collections\<match-3> `
  --config configs\data_collection.example.yaml `
  --output data\datasets\robot-v1 `
  --allow-unreviewed
```

`--allow-unreviewed` is the flag that says "I know these are guesses." It is required here and
**only** here — its whole purpose is to make the compromise visible in the command itself.

Then in Roboflow:

1. Create a **private Object Detection** project — `FRC Robot Boxes`.
2. Create exactly one class: `robot`. No team numbers, no goals, no events.
3. Upload the three folders from `data\datasets\robot-v1` — `train`, `valid`, `test` — preserving
   those split names. Choose **COCO JSON** if asked for a format.
4. Go frame by frame:
   - **Delete** boxes on people, field elements, the score overlay, robots shown on a screen.
   - **Add** every robot the models missed.
   - **Tighten** each box to one whole robot, bumper included.
   - **Exclude** replay close-ups and camera-cut blur rather than labelling them badly.
5. Export in **COCO JSON** to a new folder, e.g. `data\datasets\frc-robots-v2-coco`.

Verify before moving on:

```text
data\datasets\frc-robots-v2-coco\train\_annotations.coco.json
data\datasets\frc-robots-v2-coco\valid\_annotations.coco.json
```

Never overwrite the reviewed folder with a fresh unreviewed export. That silently throws away
hours of human work.

## Step 4 — Train

### Portable TensorFlow CPU baseline

Any team member can train a lightweight robot detector on a Ryzen CPU without CUDA, NVIDIA drivers, or an accelerator. It uses the same reviewed COCO dataset:

    .\training\run_tf_cpu.ps1 -Dataset data/datasets/frc-robots-v2-coco -Output data/models/robot-v1-tf

Start with `-Resolution 320 -BatchSize 2 -Epochs 10` to check the pipeline, then use a new output directory for a real candidate. `best.keras` is a TensorFlow model, not yet a drop-in replacement for the analyzer's RF-DETR ONNX format.

### AMD GPU (ROCm)

For a supported AMD GPU, run the TensorFlow path in Linux or WSL after installing the matching ROCm
driver. The script verifies that TensorFlow detects the GPU before it trains:

    ./training/run_tf_amd_rocm.sh --dataset data/datasets/frc-robots-v2-coco --output data/models/robot-v1-amd

If the default `tensorflow-rocm` package does not match your ROCm release, pass AMD's matching wheel
URL as `--tensorflow-package`. This route contains no CUDA or NVIDIA dependency.

### AMD GPU on native Windows (DirectML)

Linux is not required if the training computer runs Windows with a DirectX 12-compatible AMD GPU.
Install 64-bit Python 3.10, then run:

    .\training\run_tf_amd_directml.ps1 -Dataset data/datasets/frc-robots-v2-coco -Output data/models/robot-v1-amd

This uses the TensorFlow DirectML plugin, with no ROCm, CUDA, or NVIDIA dependency. The plugin is
paused and limited to TensorFlow 2.10, so it is isolated in `training/.venv-tf-amd-directml` and is
best-effort rather than the default CPU path.

On the NVIDIA machine, from the repo root:

```powershell
.\training\run_rfdetr.ps1 -Dataset data\datasets\frc-robots-v2-coco -Output data\models\robot-v1
```

The script creates `training\.venv`, installs RF-DETR and CUDA dependencies, then trains
**RF-DETR Small** for 100 epochs at 640 px with an automatically chosen batch size, and exports
ONNX at the end.

Augmentation is deliberately conservative — horizontal flips and mild brightness/contrast, plus
RF-DETR's normal resize jitter. No vertical flips or heavy rotation, because upside-down FRC
footage doesn't exist and training on it wastes capacity on a case that never occurs.

Smoke-test the installation first if you like — but never ship a 3-epoch model:

```powershell
.\training\run_rfdetr.ps1 -Dataset data\datasets\frc-robots-v2-coco -Output data\models\robot-smoke -Epochs 3
```

## Step 5 — Confirm the export

```text
data\models\robot-v1\onnx\inference_model.onnx     ← the model
data\models\robot-v1\training-config.json          ← what settings produced it
```

The `.onnx` is model data. **Never commit it.**

## Step 6 — Plug it into the analyzer

**Copy the example that matches your model.** `robot-v1` and `robot-v2` are YOLO:

```powershell
Copy-Item analysis\config\detector.yolo.example.json analysis\config\detector.local.json
```

For an RF-DETR export, copy `detector.example.json` instead. Point `model_path` at your file:

```json
"model_path": "../../data/robot-v2.onnx"
```

Then:

```powershell
$env:FRC_DETECTOR_CONFIG = (Resolve-Path "analysis\config\detector.local.json").Path
.\run.ps1 full
```

Queue a match the model has never seen. The analysis view should report a configured detector and
non-zero tracks, and the overlay boxes should track robots with gaps at broadcast cuts.

### What the C++ side does with the two families

They agree on nothing except taking an image and returning boxes, so the analyzer reads the
**model's own outputs** to decide which one it is holding rather than trusting the config:

| | YOLO (`robot-v1`, `robot-v2`) | RF-DETR |
|---|---|---|
| outputs | one, `(1, 4+classes, anchors)` or its transpose | two, `dets` and `labels` |
| input | letterboxed into the square, grey padding | stretched to fill the square |
| pixels | `/255` | ImageNet mean and standard deviation |
| scores | already probabilities | logits, so sigmoid first |
| duplicates | one prediction per anchor, so suppression is required | one query per object, so none is |

Every one of those differences is silent when you get it wrong. Feed a letterboxed model a
stretched image and it still returns boxes -- plausible-looking ones, sitting on nothing. Skip
suppression and one robot arrives as seven, inflating every count downstream. So
`analysis/tests/detector_test.cpp` pins the arithmetic directly, and the port was checked
box-for-box against `detect_runner.py` on real frames before it was trusted.

Resolution comes from the model too. If you set `input_width`/`input_height` and they contradict
the export, the analyzer refuses to run rather than quietly mis-scaling every box.

## Getting frames labelled by people

The detector cannot start this one. On 2026 footage it returns one to five robots per frame where
six are playing, and in the second of the two stacked camera views it returns nothing at all --
not at a lower threshold, not tiled, not at 3x magnification. A machine labelling loop can only
propose what the detector already sees, so a gap this shape needs people.

```powershell
python -m ingest.collection.label_pack --segments data\segments --out data\label-packs\v3 --frames 400 --model data\robot-v2.onnx
```

That writes a YOLO folder -- `images/`, `labels/`, `data.yaml`, `manifest.json` and a `README.md`
of instructions for whoever is labelling. Roboflow, CVAT and LabelImg all import it directly.

Frames are drawn round-robin across every segment, so a pack covers every venue before it takes a
second frame from any of them. Venue count is what buys generalisation: thresholds tuned on ten
venues here did not survive twenty-five.

**Tell your labellers the one rule.** Every robot in the frame, or skip the frame. A frame with
four robots boxed and two missed does not merely fail to teach -- unlabelled pixels are treated as
background, so it actively teaches that robots are background. Skipping costs nothing; guessing
costs the model. It is in the pack's README too, but say it out loud.

When labels come back, merge and retrain into a **new** folder. A pack whose labels are
still empty is dropped rather than merged -- an unlabelled image is not a negative example,
and asserting "no robots here" about a frame nobody looked at is how a dataset poisons a
model:

```powershell
python -m ingest.collection.dataset_merge --into data\datasets\frc-robots-v3 data\datasets\frc-robots-v2 data\label-packs\v3-viewpoint
```

## When it goes wrong

| Symptom | What to do |
|---|---|
| `CUDA was not detected` | You're on the wrong machine, or the NVIDIA driver needs updating. |
| Missing `_annotations.coco.json` | Re-export from Roboflow. One match is not a dataset. |
| Great mAP, bad overlay | Leakage. Check that each match sits entirely in one split. |
| No tracks at all | `FRC_DETECTOR_CONFIG` must be set in the *same* PowerShell window as `run.ps1 full`. Check the ONNX path resolves. |
| Boxes drift across camera cuts | Detector is fine — that's the gap logic. Never interpolate across a gap. |
| Boxes are just wrong | Review more matches, train `robot-v2` into a **new** folder. Never overwrite v1 or any raw output. |
| Boxes plausible but on nothing | Wrong family for the model. A one-output export is YOLO. |
| `model_version` says `detector-unconfigured` | `FRC_DETECTOR_CONFIG` was not set, or was set in a different window. |
| `Detector config says AxB but the model declares CxD` | Working as intended. Delete `input_width`/`input_height` and let the model say. |

## Never commit

```text
data\                                 videos, frames, datasets, weights
analysis\config\detector.local.json
ingest\.env                           Supabase URL and API keys
training\.venv\  ·  C:\FRC-SAM3\.venv\
SAM checkpoints, Hugging Face tokens, Google service-account JSON
```

Finish with:

```powershell
.\run.ps1 check
git status
```

Only source and docs get committed. Datasets, video, models, and secrets stay local.

---

Sources: [RF-DETR](https://github.com/roboflow/rf-detr) (Apache 2.0),
[SAM 3](https://github.com/facebookresearch/sam3) and its
[licence](https://github.com/facebookresearch/sam3/blob/main/LICENSE).

---

## Scoring a model where it actually fails

A validation split drawn from the training corpus measures accuracy on images that look like the
training images. That is not what broke. On 2026 footage v2 recognised one to five robots in a
six-robot match and nothing at all in the lower of two stacked camera views, while posting
perfectly respectable val numbers. **A model can report excellent mAP and still be blind here,
because the failure is distribution, not accuracy.**

So before believing a new model is better, run it on the held-out viewpoint pack — 400 frames
across 60 segments of exactly the footage that fails:

```powershell
python -m ingest.collection.viewpoint_eval --model data\robot-v3.onnx
```

The baseline to beat, measured on `robot-v2.onnx` at threshold 0.25:

| | v2 |
|---|---|
| mean detections per frame | **3.47** (a six-robot match has 6) |
| frames finding nothing | 13 / 400 (3.2%) |
| frames finding all six | 55 / 400 (13.8%) |
| weakest segment | 0.43 detections/frame |

Read the **per-segment** breakdown, not just the mean. Three-everywhere and six-at-half-the-venues
average the same and are not the same problem — only the first is fixed by more labels of the
same kind. v2's spread runs from 0.43 to well above six, so its gap is venue-shaped.

No ground truth is needed for this to be useful: a six-robot match contains six robots. Note that
the pack's own `labels/` are **the detector's own proposals**, not truth — scoring against them
measures self-agreement, and v2 scores 0.986 against boxes v2 drew. The tool refuses to report
recall from them. Once a pack comes back from a labeller, point at it:

```powershell
python -m ingest.collection.viewpoint_eval --model data\robot-v3.onnx --truth data\returned\labels
```

### mAP@50-95 is not cosmetic here

The spread between mAP@50 and mAP@50-95 measures how tightly boxes fit, and two things downstream
depend on it more than on whether a robot was found at all:

- **Tracking is IoU-based.** Loose or jittery boxes lower frame-to-frame overlap, association
  fails, and one robot becomes several track IDs. Team attribution is *per track*, so every extra
  fragment is another number a human has to type.
- **Bumper OCR reads inside the box.** The ring test asks whether a digit is ringed by bumper
  colour; a box padded with floor and truss dilutes exactly that signal.

A model with better mAP@50 and worse mAP@50-95 can therefore cost more human time than the one it
replaced.
