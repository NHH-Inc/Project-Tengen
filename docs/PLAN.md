# The plan

Last updated 2026-09-04.

## You are here

**The pipeline has run end to end on a real match, with a person reading the output.** That had
never happened before 2026-09-04, and it immediately found two things no unit test could.

The analyzer could only load RF-DETR. Both trained models are YOLO -- one output instead of two,
`/255` instead of ImageNet statistics, letterboxed instead of stretched, and thousands of
overlapping candidates needing suppression. So every real run analysed **zero frames of every
frame it decoded** and wrote that plainly in `result.json`, where nobody looked. Each component
was tested; the seam between two of them was not.

The tracker then bridged gaps of four to thirteen seconds, because its speed gate was a rate with
no ceiling: across thirteen seconds it permitted thirteen frame widths, which is no constraint at
all. All seven tracks in the match jumped between 0.4 and 0.7 frame widths mid-track -- one
robot's identity handed to another, which for a scouting system means one team's actions credited
to a different team.

Both are fixed, and the fixes are pinned by tests that fail without them. YOLO is the detector;
time is the second opinion.

```
[1] analysis  C++   ██████████  runs YOLO on real broadcast, tracks, homography wired in
[2] ingest    Py    ██████████  174 tests, 71 contract checks green
[3] web       TS    ██████████  player, overlay, corrections, Sheets export
    detection Py    ██████████  v2 trained on the AMD GPU; C++ agrees box-for-box
    DATA          ██████████  2,429 human labels + 50 matches densely auto-labelled
```

## The decision that simplified everything: YOLO alone, corroborated by time

RF-DETR was meant to be the second detector the confidence system needs. Robert trained one and it
came back undertrained -- 19% mAP, sprawling low-confidence boxes that never overlapped YOLO's, so
fusion found **0% agreement**. It is kept for later (a 300+ epoch retrain could make it a real
third voice) but it is off the critical path.

The second opinion that works is **time**. We have video, and a robot that persists across frames
is corroborated by physics -- an independent witness no single model can be. Fusing YOLO's raw
detections with their temporally-confirmed subset gives **95% agreement** where RF-DETR gave 0%,
at no cost of a second model, training run or GPU.

So the shipped pipeline is: YOLO detects, temporal consistency confirms, fusion produces a
confidence that actually reflects corroboration. That is the system that was asked for, reached by
a cheaper and more honest route than a second network.

## What exists now

| | |
|---|---|
| `data/datasets/frc-robots-merged/` (+ `-coco`) | 2,429 human-labelled images, YOLO and COCO |
| `data/robot-v1.onnx` | trained YOLO detector, verified on real footage |
| `data/dense-labels/` | 50 matches auto-labelled at 5 Hz, corroborated by time |
| `frame_quality` | rejects broadcast graphics, on disk or in memory |
| `box_fusion` | confidence from agreement, weights learned from corroboration |
| `temporal_consistency` | persistence as the second opinion |
| `dense_label` | the runner that generates next-gen training data |
| `homography` (Py + C++) | image-to-field feet, real 2026 AprilTag layout vendored |
| `dead_reckoning` | coasts a track through occlusion, never across a cut |

## The two things learned that should not be relearned

**Both source datasets leaked across their splits.** Roboflow assigns augmented copies
independently; WorBots' published 97.6% mAP is inflated by it. We regroup by source image.

**Thresholds tuned on ten venues did not survive twenty-five.** The frame filter rejected 94% of
one match -- all real gameplay -- before recalibration. Tune on the widest sample and check the
extremes by eye.

---

## Phase 1 — Get to a working detector

This is the only phase that matters right now. Everything is sequential; each step needs the one
before it.

### 1.1 — Match footage · **Justin** · DONE

50 qualification matches from **25 different 2026 events**, 4.6 GB. Two per venue, spaced within
each event so the pair does not share lighting and field state. One match is one indivisible split
group, so venue count is what buys generalisation.

### 1.2 — Frames, filtered · **Justin** · DONE

3,300 frames at 0.25 fps, of which **3,077 are usable**. The rest are FIRST logos, sponsor stings,
"ALLIANCE WINS" cards and score screens, rejected by `frame_quality.py` before any model sees them.

### 1.3 — Training data · **Justin** · DONE

Not our own guesses, in the end. Two CC BY 4.0 datasets from Roboflow Universe merged down to a
single `robot` class: **2,429 human-labelled images**, 5,815 boxes, at
`data/datasets/frc-robots-merged/`. Their splits leaked and were rebuilt; see the notes above.

### 1.4 — Train a labeller · **Robert, then Justin** · DONE

```powershell
yolo detect train data=<path>/frc-robots-merged/data.yaml model=yolo11s.pt epochs=100 imgsz=640
yolo export model=runs/detect/train/weights/best.pt format=onnx
```

Send back the **ONNX** — about 20 MB. Send the model, never the frames: our frame set is several
GB and the model is not.

**Done when:** a `.onnx` file exists and its boxes land on robots in a match it never saw.

### 1.5 — Label our footage · **Justin** · ~30 min once weights arrive

Run the ONNX over the 3,077 usable frames and fuse. `detect_runner.py` does both; the fusion
recomputes each box's confidence from how much detector weight backs it, how tightly the backers
agree, and how reliable each detector has proven across the collection.

With one detector there is no agreement signal, so a second source is what makes the confidence
mean anything. RF-DETR trained on the same merged dataset is the obvious second.

**Done when:** `detector-consensus.jsonl` exists per collection, and the top-ranked boxes are on
robots rather than on the score bar.

### 1.6 — Plug it into the analyzer and watch · **anyone** · ~15 min

Point `detector.local.json` at the ONNX, set `FRC_DETECTOR_CONFIG`, run a match nothing has seen.

Judge by watching, not by the accuracy number. Do the boxes sit on robots? Do they disappear at
camera cuts instead of sliding across them? A high mAP with drifting boxes means leakage, not a
good model — and we already know both source datasets leaked before we rebuilt their splits.

**Done when:** boxes track robots through a real match and gap at broadcast cuts.

**1.6 passed on 2026-09-04**, on `2026tuis_qm29` with `robot-v2`: 423 frames analysed, boxes on
robots, 34 tracks, gaps at cuts. The C++ detector was checked against `detect_runner.py` on real
frames first -- 29 of 29 boxes matching at IoU 1.000 -- because two implementations of the same
arithmetic agreeing is the only cheap way to know the port is right.

Everything after this is improvement rather than construction.

---

## What the first end-to-end run turned up

Neither of these is a bug. They are things only a real match could have told us.

### The lower half of a frame can be a second view of the same field

`2026tuis_qm29` is shot from a balcony: above the handrail is the field, below it is another view
of the same match. The detector is right to fire in both -- they are robots -- but a robot seen
twice is counted twice, and **every one of the seven original tracks had boxes on both sides**,
so tracks were switching between two views of the same field. 323 of 1,423 boxes were in the
lower region.

The analyzer has no notion of a region of interest. It needs one, per video source, exactly as
2.1 already says the OCR crop must be. Until then, per-team counts from footage like this are
inflated and nothing in the output says so.

### The whole chain now runs, and it delivers nothing

A real match went the whole way on 2026-09-04: analyzer, database, API, the endpoints the web app
calls, a human re-attribution, and the spreadsheet. Every step returned 200. Forty-five tracks and
two events reached Postgres, eleven tracks carried a team, five of the six roster teams appeared,
a track was re-attributed and `?raw=true` still returned the original.

And the export wrote **zero rows**, because a spreadsheet row needs a team-attributed *event*, and
the analyzer emits exactly two events per match: `match_start` and `match_end`. Neither belongs to
a team. Per-team statistics come back all zeros for the same reason.

So the plumbing is finished and the content is missing. **Action extraction -- shots, cycles,
pickups -- is the gap between a working pipeline and scouting data**, and nothing downstream can
paper over it. Tracks alone say where robots were; they do not say what they did.

Two real bugs fell out of the run:

  * **The database was unreachable and had never been written to.** `DATABASE_URL` pointed at the
    Supabase pooler on port 5432, which is session mode and closed on these projects. It fails
    with "server closed the connection unexpectedly", which reads like an outage rather than a
    wrong port. 6543 works. Every table had zero rows, so nothing had ever run against it.
  * **The export reported success while writing nothing.** 200 with `rows_written: 0`. That is
    the same lie the endpoint already refuses when credentials are missing. It now returns 422
    and names the cause.

### Recall on 2026 footage is the real problem, and it is not one venue

Chasing the missed lower-right robots turned up something larger. Twelve venues sampled at random
returned **one to five detections per frame in a six-robot match**, on robots plainly visible to a
person. Every one of those twelve has the same stacked two-camera layout, so the second view is
not an Istanbul quirk -- it is the standard 2026 broadcast format, and the detector is weak on all
of it.

That reframes what v2's numbers meant. Its 0.864 recall was measured against 2023-24 images with
2023-24 framing, and it is honest for that. The 2026 season is shot differently, and no amount of
threshold or resolution work closes a gap of this kind.

It also raises the stakes for September. Tengen will be fed footage **we** record from the stands,
which is further again from 2023-24 broadcast framing than any of this. Labelled 2026 frames are
now the single highest-value thing anyone can contribute.

`ingest/collection/label_pack.py` builds a pack: frames drawn round-robin across every venue so a
budget is never spent on one arena, broadcast graphics filtered out, and the current detector's
boxes pre-filled so a labeller corrects rather than draws. The first pack is 400 frames from 59
venues with 1,402 proposals -- roughly 3.5 per frame, which is itself the measurement, since six
robots play every match.

### Some robots are invisible to the detector, and it is the viewpoint

In a frame where a human can plainly see them, robots in the lower-right region are not detected.
Three explanations were tested in order, and the first two were wrong:

1. **Not the threshold.** Lowering `score_threshold` from 0.35 to **0.05** returns exactly the
   same five boxes. They are not low-confidence detections being filtered out.
2. **Not the resolution.** Tiling at native resolution found nothing there either. Nor did feeding
   the model that region alone at 2x and 3x magnification -- **zero boxes at 0.05**, while a
   control crop of the main field returned three.
3. **The viewpoint.** That region is a second field shot from almost directly overhead: robots
   seen top-down, dark against a blue floor, among hundreds of yellow balls. Every image the
   detector has ever trained on is a broadcast three-quarter view. It is not failing to resolve
   these robots, it is failing to recognise them as robots.

So the fix is training data from that viewpoint, and it needs **human** labels -- the machine
labelling loop cannot propose what the detector cannot see. Logged as 2.9.

Tiling was built anyway, because measuring it turned up a different benefit: it does not touch
the overhead view but it finds robots the whole-frame pass misses elsewhere, and that is worth
4.8x the inference time. It also needed one non-obvious guard. Tile seams fell at x=960, and a
robot straddling one came back as a 0.26-confidence sliver beside the real box -- suppression
cannot remove that, because a sliver barely overlaps anything. Detections touching a tile edge
that is not also a frame edge are dropped; the overlapping neighbour tile sees the same robot
whole.

The lesson the project keeps relearning: measure which explanation is true before fixing any of
them. An afternoon of measurement killed two plausible fixes that would both have been built
first.

### Tracks are shorter now, and that is the honest number

Same detections, different partitioning: 7 tracks became 34, median duration 166s became 8s. The
7 were not better -- they were one long fabricated identity each. A fragmented track is something
a human fixes in the correction UI, which is what it is for. A wrong identity is silently wrong
forever, and nobody checks a number that looks reasonable.

---

## Phase 2 — Make the output trustworthy

1.6 has passed, so these are live.

| # | Task | Owner | Why it matters |
|---|---|---|---|
| 2.1 | **Scoreboard OCR** | Robert | **Half done.** `ingest/scoreboard.py` reads the *roster* — 6/6 teams and both alliances on 7/7 sampled frames, with no per-venue crop: the bar is found by looking for saturated red touching saturated blue, and team numbers are the digits that do not change while the score and timer do. The **match start time** from the timer is still open, and is the half this row was originally about. |
| 2.2 | ~~**Bumper OCR / team ID**~~ | Robert | **Done 2026-09-04.** `ingest/team_id.py`: roster narrows to six, bumper colour to three, and a hundred observations per track vote. 11 of 45 tracks on the real match, no red team ever attributed to a blue robot. Accuracy is 2.8. |
| 2.3 | **Accuracy check against TBA** | Justin | Blocked twice over: TBA has no data for the 2026 matches collected, and there are no scoring events to compare. |
| 2.12 | **Finish one camera's calibration** | Justin | **Blocked on a human, and now we know why.** Measured across all 60 sources: tags alone cannot calibrate any of them. `pose` is ill-conditioned — every 2026 tag sits within 0.57 m of one plane across a 16.5 m field, so the set is **3.8% out-of-plane** and focal length trades against distance; sweeping HFOV 10°–120° gives a monotonic slope, not a minimum, bottoming at 45 px on a camera steady to 0.7 px. `plane` needs 4 non-collinear tags at one height: 13 of the 14 sources that have 4+ steady at one height are at z=1.124 m, most measuring **0% off-line**. The one well-conditioned height (z=0.889 m, 93%) is never steadily visible. Recovering the camera centre from two plane homographies would dodge FOV entirely, but no source has 4 steady tags at two heights. Added **`--method carpet`** — fit hand-marked carpet points directly, no tags, no FOV. Needs a person who can identify field features to mark five on `fMSSL1Iy_ig` (drift 0.6 px, and the camera behind the job that already runs end to end). |
| 2.11 | ~~**Action extraction**~~ | Justin | **Done 2026-09-04.** `ingest/action_extraction.py`: phase boundaries anchored to the real match clock, immobility from track geometry. The chain now exports **5 rows with nobody logging anything**, where it exported 0. Shots stay unemitted until `point_values` in `contracts/seasons/2026.json` are real — they are all zero placeholders, and a made-up unit would poison every accuracy figure. Scoring *moments* are surfaced as a worklist meanwhile. | The pipeline runs end to end and exports zero rows, because the analyzer emits only `match_start` and `match_end`. Until a shot, a pickup or a cycle becomes an event, there is no scouting data at the end of any of this. |
| 2.4 | **More matches, `robot-v2`** | everyone | ~~Same loop as phase 1, into a **new** output folder.~~ **Done 2026-09-04:** 150 epochs on the AMD GPU via WSL2+ROCm, recall 0.819 to 0.864 on the same human-labelled split. |
| 2.5 | ~~**Region of interest per video source**~~ | Justin | **Done 2026-09-06.** `ingest/collection/view_region.py` and `calibrate_region.py`, honoured by the C++ analyzer via `regions_path`. **39 of 60 sources** composite a second view, seams from 0.42 to 0.78 — so no single crop line serves. The seam is not readable from one frame (row-discontinuity scores 4.55 on stacked frames against 4.49 on flat ones, i.e. chance), so it is calibrated per source from where robots are found. Filtering after inference beat cropping before it at 640 — 4.22 boxes per frame against 4.05 — and ties at 960; they cost the same either way. |
| 2.6 | ~~**Detector threshold on real footage**~~ | anyone | **Measured, and it is not the threshold.** Robots visible in the lower-right of a real frame are missed at 0.35 and still missed at **0.05** -- the model does not see them at all. See below. |
| 2.7 | ~~**Tiled inference**~~ | Justin | **Done 2026-09-04**, and it does *not* fix 2.6 — that was the wrong hypothesis, see below. It is still worth having: median robots on screen went 4 to 5 of 6, boxes +29%, longest track 150s to 170s. Costs 4.8x the inference (35s to 167s a match). `tile_size` in the detector config; 0 disables. |
| 2.9 | **Label 2026 footage** | everyone | **Pack is built and waiting**: `data/label-packs/v3-viewpoint`, 400 frames across 59 venues, 232 MB. Instructions ship inside it. This is the bottleneck and it parallelises across as many people as want to help. |
| 2.10 | ~~**Train robot-v3**~~ | Robert | **Done 2026-09-06.** 1,000 hand-reviewed images and a labelling tool of his own. On the held-out viewpoint pack, mean detections 3.47 -> **6.34** and blind frames 13 -> **0**. Exported at 960: the same weights at 640 score 5.61, so a third of the gain was input size. His own val split reports recall 0.954, but it is frames from the same few videos, so it measures near-duplicates; the honest numbers are the pack's. |
| 2.8 | **Bumper OCR accuracy** | Justin | 8 of 34 tracks attributed on the first real match. The scoreboard roster is solid (6/6 teams, 7/7 frames); the weak link is the digit read. Worth measuring against a hand-labelled match before tuning further. |

## Phase 3 — Operations

Independent of phases 1 and 2. Can happen in parallel any time.

| # | Task | Owner | Notes |
|---|---|---|---|
| 3.1 | **Google Sheets service account** | Robert | Create the Cloud service account, download the JSON key, share the sheet with its email as **Editor**, point `GOOGLE_APPLICATION_CREDENTIALS` at the file. Keep the JSON outside the repo. Everything else is already built — tabs are created automatically, sheet ID is already in `ingest/.env`. |
| 3.2 | **Ask about the classroom PCs** | Robert | Twenty machines with 5070 Tis. Cheap to ask; it's the difference between one training box and twenty labelling boxes. **Get explicit school permission, and do not bulk-download YouTube video on school hardware.** |
| 3.3 | **Decide where ingest runs** | Justin | Justin has said he'd rather not host it on his own PC. Options and trade-offs are in [docs/HOSTING.md](HOSTING.md). YouTube blocks datacentre IPs, which is the constraint that shapes the whole answer. |

## Phase 4 — Cheap experiments worth doing

Small, self-contained, and each one is genuinely uncertain — do them when you want a break from
the main line.

- **Is `gemma3:4b` earning its slot?** It takes image input but isn't trained for box grounding
  the way the two Qwen-VL models are — and those two are the same family, so gemma3 carries all
  the actual diversity in the ensemble. Per-model raw output is already saved, so this is ~20
  minutes on 50 frames using `compare-models`.
- **Try `iou_threshold: 0.40` instead of 0.50.** A robot in a wide field shot is often under 5% of
  frame width, where a few pixels of honest disagreement drops a real match below 0.5.
- **`jpeg_quality: 85` instead of 95** roughly halves frame storage with no meaningful detection
  loss.
- **SAM, if a machine can run it.** Note that this needs SAM **3**, not SAM 2: the integration
  works by text prompt ("FRC competition robot"), and text prompting is a SAM 3 feature. SAM 2
  takes only points/boxes/masks, so it cannot stand in.
- **SAM 2 video tracking — a different, probably better idea.** Draw boxes on one frame, let SAM 2
  propagate them through the match, sample at our frame timestamps. That converts per-frame
  labelling into per-match labelling. Nobody has built it; it has to run on the source MP4s rather
  than the 4-second-spaced frames, and camera cuts and robot-on-robot occlusion both break
  tracking. Worth doing for the real season dataset, not for a proof of concept that is already
  labelled. SAM 2 is also Apache 2.0 with no Hugging Face gating, unlike SAM 3.1.

---

## Nathaniel

Nathaniel is busy and his work moved to Robert. Nothing is assigned to him. If he frees up, the
best use of his time is **1.4 (reviewing boxes)** — it's the bottleneck, needs no setup, and
parallelises across as many people as want to help.

## Rules that hold across every phase

- Nothing in `/contracts/` changes without all three of us agreeing.
- Never commit `data/`, `ingest/.env`, ONNX weights, service-account JSON, or Hugging Face tokens.
- Never overwrite a model, a reviewed dataset, or raw model output. New version, new folder.
- Run `.\run.ps1 check` before you push. CI runs the same thing.
- When you change a model column, add the matching `ALTER TABLE` in `ingest/database.py` —
  `create_all()` only creates missing *tables*, never new columns on existing ones. The service
  now warns loudly at startup when the live schema has drifted.
