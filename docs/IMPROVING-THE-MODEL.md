# Improving the model, step by step

One loop, repeated: **analyse → mine the mistakes → label → train → score → ship only if better.**
Each pass takes an evening once you have done it once. Every model gets the next number (`v4`,
`v5`, ...) and its scores go in [the registry](MODELS.md).

```
 1 analyse ──▶ 2 mine ──▶ 3 label ──▶ 4 merge ──▶ 5 train ──▶ 6 export ──▶ 7 score ──▶ 8 decide
   matches     mistakes    (people)    datasets    (GPU)       ONNX         held-out    ship / keep
      ▲                                                                                    │
      └────────────────────────────────── next week ──────────────────────────────────────┘
```

Commands are PowerShell from the repository root. `py` means the desktop environment
(`.venv-desktop\Scripts\python`) and `ipy` the ingest environment (`ingest\.venv\Scripts\python`).

---

## The two rules that make this work

**1. Never train on a held-out frame.** `data/label-packs/v3-viewpoint` is the yardstick every
model is scored on. The moment any of its frames is in a training set, every later score is
inflated and the history chart is meaningless. If you label it, keep the labels as *truth* for
scoring (`--truth`), never as training data. The same goes for any new held-out pack.

**2. Every robot boxed, or the frame skipped.** An unlabelled robot teaches the model that robots
are background. A half-labelled frame is worse than no frame.

---

## Step 1: Analyse some matches

Anything goes, but variety beats volume: different venues, different broadcast layouts, our own
recordings. In the desktop app: **Open video… → Analyze video…**, and pick the right **Video
layout** (Championship broadcast, other broadcast, or our own camera). Each run lands in
`data/desktop/runs/<id>/`.

## Step 2: Mine the mistakes

```powershell
ipy -m ingest.collection.hard_examples data\desktop\runs --out data\label-packs\hard-2026-10-20 --per-match 25 --model data\robot-v3.onnx
```

This finds the frames where tracking shows the detector failed, not a random sample:

| Reason | What it means | Example |
|---|---|---|
| `hole` | a robot was missed for a stretch | robots half-hidden behind a hub |
| `vanished` | a track starts or ends mid-field | a robot lost in a crowd |
| `count_drop` | several robots missed at once | referees, spectators, pile-ups |
| `undersized` | a tiny box held a robot's slot | a wall fixture at a fifth of a robot's width |
| `static` | a box that never moves | roof trusses (Space Coast) |
| `flicker` | a box that lasts under a second | hands, reflections, bumpers in the crowd |

`hard_examples.json` in the pack says why each frame is there. Boxes from the current model are
pre-filled so labellers correct rather than draw.

## Step 3: Get it labelled

Split the pack into hand-out zips and send them out as in [LABELLING.md](LABELLING.md):

```powershell
ipy -m ingest.collection.split_pack --pack data\label-packs\hard-2026-10-20 --out data\label-packs\hard-2026-10-20-zips --chunks 4
```

The pack is YOLO format, which Roboflow, CVAT and LabelImg open directly. When labels come back,
drop the returned `labels\` folders over the pack. **Frames with no robots at all** (rafters,
crowds) are valuable: their label file is empty on purpose. Note which ones a person actually
checked, because an empty file the labeller never opened looks the same.

## Step 4: Merge into a new dataset

```powershell
ipy -m ingest.collection.dataset_merge --into data\datasets\frc-robots-v4 data\datasets\robot-yolo11n-reviewed-aug data\label-packs\hard-2026-10-20
```

- `robot-yolo11n-reviewed-aug` is v3's training set and lives on Robert's machine; get a copy
  first (or start from `frc-robots-merged`, which is here).
- Always a **new** folder; the tool refuses to overwrite one.
- Splits are by source image, so near-duplicates never straddle train and val.
- Add `--keep-empty` **only** if every frame in every source was reviewed. Otherwise empty label
  files are dropped, which loses the negatives but never teaches "no robots here" about a frame
  nobody looked at.
- **Never list a held-out pack here.**

## Step 5: Train

On the RTX 5070 Ti (CUDA PyTorch in `.venv-desktop`):

```powershell
py -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision   # once
yolo detect train data=data\datasets\frc-robots-v4\data.yaml model=yolo11n.pt epochs=100 imgsz=960 batch=16 device=0 project=data\models name=robot-v4
```

Things worth trying, one at a time so you know which helped:
- `imgsz=1280` (robots in wide views are 40-60 px across);
- `model=yolo11s.pt` (bigger network, slower: check it still keeps up with near-live);
- more epochs only if the val curves in `results.png` are still improving at the end.

Without the GPU, Robert's machine or CPU training (slow) works the same way; see
[TRAINING.md](TRAINING.md).

## Step 6: Export

```powershell
yolo export model=data\models\robot-v4\weights\best.pt format=onnx imgsz=960
copy data\models\robot-v4\weights\best.onnx data\robot-v4.onnx
copy data\models\robot-v4\weights\best.pt data\robot-v4.pt
```

Export at the size you will run at. v3 was trained at 640 and exported at 960, and the 960
export finds all six robots in 64.5% of held-out frames against 51.7% for the 640 export.

## Step 7: Register and score it

Add an entry to [`training/registry/models.json`](../training/registry/models.json), copying the
v3 entry: the next id, `"status": "candidate"`, the facts from training (date, who, network,
`imgsz`, epochs, dataset) and its own val numbers from the end of training. Copy
`data\models\robot-v4\results.png` to `training\registry\runs\v4\results.png`. Then:

```powershell
ipy -m tools.models evaluate --only v4
py -m tools.status
py -m tools.models markdown
py -m tools.models list
```

For the tracking number, analyse the reference match (Championship Curie qm10) with the new model
in the desktop app and count the share of the match with all six robots tracked.

## Step 8: Decide

A candidate becomes `current` only if it is **no worse on every held-out pack** and no worse on
the reference match. A higher val mAP is not enough: v1's val mAP@50-95 (0.602) is higher than
v3's (0.544), and v3 finds all six robots in 64.5% of held-out frames against v1's 41.0%. And v2
is the warning: 1,017 machine-labelled frames took it from v1's 41.0% down to 13.8%.

If it wins: set it `current`, the old one `retired`, point the desktop and API defaults at it,
commit the registry, the scores and the regenerated `docs/status` and `docs/MODELS.md`, and post it:

```powershell
py tools\discord-bot\post_models.py           # preview
py tools\discord-bot\post_models.py --send    # post to #model-scoreboard and #model-training
```

If it loses: keep it as `retired` in the registry anyway. A model that did not work, with its
numbers, stops someone from trying the same thing again.

---

## Beyond the detector

The detector is no longer the weakest link for strategy; identity is. On Space Coast Showdown the
boxes were mostly right, but same-alliance robots swapped identities, so per-team numbers were
wrong while per-alliance numbers matched the official score. The same loop builds two more models:

- **A bumper-number reader.** Every analysed match yields bumper crops; the match roster (TBA or
  the scoreboard) narrows each read to three candidates per alliance.
- **A robot appearance model** for re-identification, trained on crops whose team is known, to
  replace the colour histogram that cannot tell two robots of one alliance apart in a wide shot.
