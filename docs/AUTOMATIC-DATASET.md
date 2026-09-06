# Automatic FRC robot dataset pipeline

This is the implemented version of the Grounding DINO -> FRC filtering -> YOLO11 plan. It keeps
full competition frames, preserves empty frames, assigns splits by event, records why every box
passed or failed, and never presents an automatic proposal as ground truth.

The existing RF-DETR path remains supported. This YOLO path is optional because Ultralytics is
AGPL-3.0 unless you obtain a different license; decide whether that is acceptable before shipping
it in the product. The dataset-generation modules themselves do not change shared contracts.

## What is implemented

1. Deterministic timestamped full-frame extraction at a configurable rate.
2. Grounding DINO Swin-T prompt/threshold trials with JSONL proposals and optional previews.
3. FRC filters: confidence, size, aspect ratio, optional field ROI, NMS, and optional red/blue
   bumper-color support. Color is a confidence bonus, never a required condition.
4. Temporal association that supports repeat detections and short gaps. High-confidence isolated
   boxes survive; weak one-frame guesses are rejected with a recorded reason.
5. YOLO directory/label export with explicit negative labels and event/match leakage checks.
6. YOLO11n training, plus a switch to another YOLO11 checkpoint such as `yolo11s.pt`.
7. YOLO self-labeling for additional collections, using the same filter/export loop.
8. A label evaluator for a small human-reviewed benchmark.
9. ByteTrack or BoT-SORT production tracking that writes Contract C v3 `tracks.jsonl` records.

## 1. Create the dedicated Windows environment

The YOLO11 launcher uses Python 3.13 on this machine and keeps its environment separate from
`ingest/.venv` and the RF-DETR environment. The short path avoids Windows path-length failures
while installing Torch.

```powershell
py -3.13 -m venv --system-site-packages C:\yolo11-venv
C:\yolo11-venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

Install CUDA-enabled PyTorch with the current Windows/Pip/CUDA command from the PyTorch selector.
Then install the rest:

```powershell
pip install -r training\requirements-yolo.txt
git clone https://github.com/IDEA-Research/GroundingDINO.git
pip install -e .\GroundingDINO
```

Download `groundingdino_swint_ogc.pth` from the official Grounding DINO release and use the
repository's `groundingdino/config/GroundingDINO_SwinT_OGC.py`. Do not substitute Swin-B for the
first pass.

Verify the machine:

```powershell
python -m training.vision_doctor `
  --model-config .\GroundingDINO\groundingdino\config\GroundingDINO_SwinT_OGC.py `
  --checkpoint .\models\groundingdino_swint_ogc.pth
```

The report should show Python 3.13, CUDA PyTorch, the RTX 3060, OpenCV, Ultralytics, Grounding DINO,
and both model files.

## 2. Extract full-field frames

The extractor stores timestamps, hashes, match/event provenance, and one split for the entire
event. A canonical TBA match key supplies its event automatically; otherwise pass `--event-id`.

```powershell
python -m ingest.collection.cli extract `
  --segment data\segments\2026casf_qm42.mp4 `
  --match-id 2026casf_qm42 `
  --event-id 2026casf `
  --video-id VIDEO_ID `
  --config configs\automatic_dataset.example.yaml
```

At `sampling.fps: 1.0`, one frame is kept per second. Use `0.5` for one every two seconds or `2.0`
for one every half second. Full frames are always kept; the extractor never creates robot crops.

## 3. Test Grounding DINO prompts and thresholds

Start on a small batch and request previews:

```powershell
python -m ingest.collection.cli grounding-dino-propose `
  --collection data\collections\COLLECTION_ID `
  --config configs\automatic_dataset.example.yaml `
  --model-config .\GroundingDINO\groundingdino\config\GroundingDINO_SwinT_OGC.py `
  --checkpoint .\models\groundingdino_swint_ogc.pth `
  --limit 25 `
  --preview-dir data\previews\dino-trial
```

Tune `grounding_dino.prompt`, `box_threshold`, and `text_threshold` in a copied config. Add
`--force` to replace records for the selected frames. Each frame is saved immediately, so an
interruption does not lose completed work.

## 4. Apply FRC and temporal filters

```powershell
python -m ingest.collection.cli filter-detections `
  --collection data\collections\COLLECTION_ID `
  --config configs\automatic_dataset.example.yaml
```

The command reads `grounding-dino-proposals.jsonl` and writes `filtered-proposals.jsonl` plus
`filter-report.json`. Each accepted box includes confidence, bumper-color fractions, temporal
hits, a temporary track ID, and `filter_reasons`. Rejected boxes remain in `rejected_boxes` for
diagnosis. The filter never assumes six visible robots and does not reject blur, occlusion, robot
contact, or distance by category.

For YOLO self-labels, use `--input yolo-proposals.jsonl`.

## 5. Export the YOLO dataset

Pass every collection in one command. The exporter refuses a split group appearing in more than
one split and refuses unreviewed proposals unless the temporary-bootstrap override is explicit.

```powershell
python -m ingest.collection.cli export-yolo `
  --collection data\collections\COLLECTION_A data\collections\COLLECTION_B `
  --config configs\automatic_dataset.example.yaml `
  --output data\datasets\robot-yolo-bootstrap-v1 `
  --allow-unreviewed
```

For a real model, remove `--allow-unreviewed` and set each label row's `review_status` to
`reviewed`, `accepted`, or `approved` after human review. Every confidently visible robot in a kept
frame needs a box. Keep negative frames and realistic difficult cases.

The output is:

```text
dataset.yaml
images/{train,val,test}/
labels/{train,val,test}/
dataset.json
```

## 6. Measure the automatic labels

Keep a small human-reviewed benchmark outside the automatic label folder, with matching YOLO
label filenames. Then run:

```powershell
python -m training.evaluate_yolo_labels `
  --predictions data\datasets\robot-yolo-bootstrap-v1\labels\test `
  --ground-truth data\benchmarks\robot-v1\labels `
  --iou 0.5
```

The report includes precision, recall, and F1. This is the quality gate for filter changes and
self-training rounds; model confidence alone is not evidence that labels improved.

## 7. Train YOLO11n

```powershell
.\training\run_yolo.ps1 `
  -Dataset data\datasets\robot-yolo-v1-reviewed\dataset.yaml `
  -Output data\models\robot-yolo11n-v1
```

Defaults are 100 epochs, 960 px, CUDA device 0, and `yolo11n.pt`. Reduce batch/image size if the RTX
3060 runs out of VRAM. Once the pipeline works, compare `-Model yolo11s.pt` and/or 1280 px in a new
output directory. The script refuses to overwrite a non-empty model directory.

## 8. Self-label more matches

```powershell
python -m ingest.collection.cli yolo-propose `
  --collection data\collections\NEW_COLLECTION `
  --model data\models\robot-yolo11n-v1\weights\best.pt `
  --confidence 0.25 `
  --image-size 960 `
  --preview-dir data\previews\yolo-v1

python -m ingest.collection.cli filter-detections `
  --collection data\collections\NEW_COLLECTION `
  --config configs\automatic_dataset.example.yaml `
  --input yolo-proposals.jsonl
```

Review, export into a new dataset version, evaluate against the unchanged benchmark, and retrain
into a new model directory. Never overwrite raw proposals, reviewed labels, datasets, or models.

## 9. Production tracking

ByteTrack is the fast baseline; BoT-SORT is useful when broadcast camera motion causes ID swaps.

```powershell
C:\yolo11-venv\Scripts\python.exe -m training.track_yolo `
  --model data\models\robot-yolo11n-v1\weights\best.pt `
  --video data\segments\unseen-match.mp4 `
  --tracker bytetrack `
  --homography analysis\config\homography.<venue>.json `
  --output data\jobs\JOB_ID\tracks.jsonl `
  --partial-output data\jobs\JOB_ID\tracks.partial.jsonl `
  --annotated-output data\previews\unseen-match-tracked.mp4
```

The JSONL has one Contract C v3 record per persistent track, normalized boxes, null team/alliance
until OCR exists, and `detection_lost` gaps when a track disappears and later returns. With a valid
homography, boxes also carry carpet `field_x`/`field_y`, velocity components, scalar speed in
feet/second, and travel heading in radians. The heading is motion direction, not robot body
orientation. The partial file is atomically refreshed during processing for a live UI overlay.

The local-video path is lazy: it starts YOLO on the first decoded frame instead of first producing
an entire cropped copy. `--stream-url` is also lazy and decodes a yt-dlp/FFmpeg pipe. Therefore an
unseen video does not need to be processed or trained on beforehand; the trained detector and
ByteTrack begin from frame one. Whether it keeps up with live video depends on GPU inference,
image size, decoding, network speed, and the source FPS. A file is processed as quickly as
possible rather than wall-clock paced.

## Success gate

Use matches from completely unseen events. Pass only when the detector consistently finds small,
moving, blurred, touching, and partially occluded robots; the benchmark improves; and the tracked
overlay remains attached to robots through ordinary detector misses. Automatic labels are a way
to reduce drawing, not a substitute for validation.
