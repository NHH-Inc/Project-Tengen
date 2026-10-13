# Getting frames labelled

This is the bottleneck, and it is the one job in the project a machine genuinely cannot start.

On 2026 footage the detector returns **one to five robots per frame in a six-robot match**, and in
the lower of the two stacked camera views it returns nothing at all — not at a lower threshold,
not under tiled inference, not at 3x magnification. It is failing to *recognise* those robots, not
to resolve them. A machine labelling loop can only propose what the detector already sees, so this
one needs people.

It parallelises perfectly. Ten people doing forty images each is a couple of hours of total
effort, and it is worth more to the project right now than any model work.

---

## 1. Hand out the packs

Already built, one per person, none overlapping:

```
data\label-packs\v3-zips\tengen-labels-01.zip   ... through 10
```

**40 images each, ~23 MB each** — small enough for Discord, email, or a phone. Each covers 40
different venues, so every pack is a useful sample on its own even if the others never come back.

Send **one zip per person** and note who has which. Nothing else is needed; each zip contains its
own images, labels, instructions and manifest.

To make packs for a different number of people:

```powershell
python -m ingest.collection.split_pack --pack data\label-packs\v3-viewpoint --out data\label-packs\v3-zips --chunks 6 --no-compress
```

Or build a fresh pack of different size from the source video:

```powershell
python -m ingest.collection.label_pack --segments data\segments --out data\label-packs\v4 --frames 600 --model data\robot-v2.onnx
```

---

## 2. What labellers use

The packs are **YOLO format**, which every common tool reads. In rough order of friction:

| Tool | Install | Account | Notes |
|---|---|---|---|
| **makesense.ai** | none — runs in the browser | none | Drag the images in, drag `labels/` in as YOLO, export YOLO when done. Runs client-side, so nothing uploads. Best bet for school computers. |
| **LabelImg** | `pip install labelImg` | none | Desktop, offline, writes YOLO `.txt` straight back into the folder. Good if someone would rather not use a browser. |
| **Roboflow** | none | one project, yours | Best if you want everyone in one place and progress visible. You upload once and invite people, rather than sending zips. |

Whichever they use, **the class is `robot` and there is only one**. If a tool asks for a class
list, that is the whole list.

---

## 3. What to tell them

The full instructions are inside each zip as `README.md`. Say the important part out loud anyway,
because it is the one mistake that makes more data make things *worse*:

> **Box every robot in the image, or skip the image entirely.**
>
> Training treats anything you did not box as "definitely not a robot". So an image with four
> robots boxed and two missed does not merely fail to teach — it actively teaches the model that
> robots are background. Skipping an image costs us nothing. Guessing costs us the model.

Other things worth saying:

- **Most of the work is adding, not correcting.** Boxes are already drawn from our current
  detector, and it misses a lot. Expect to add more than you fix.
- **The small second camera view at the bottom matters most.** That is where the detector is
  completely blind, so those boxes are the most valuable in the pack.
- **Robots only.** Not goals, not the yellow balls, not carts, not people. Partly hidden robots
  count — box the part you can see. A robot shown on a screen *inside* the picture does not count;
  if you cannot tell, skip the image.
- **Close enough is fine.** A few pixels either way does not matter. A missed robot does.

---

## 4. Getting them back

Ask for the whole folder, or just the `labels/` folder inside it — that is all we read.
**Filenames must not change**; they are how each label finds its image.

Partial packs are welcome. Ask people to say how far they got, so an unfinished pack is not
mistaken for a finished one where the rest of the images genuinely had no robots.

---

## 5. Merging and retraining

Drop returned `labels/` folders back over the matching pack directory, then:

```powershell
python -m ingest.collection.dataset_merge --into data\datasets\frc-robots-v4 data\datasets\robot-yolo11n-reviewed-aug data\label-packs\<returned pack>
```

> **Never merge `v3-viewpoint` (or any held-out pack).** Since 2026-10-13 it is the yardstick
> every model is scored on ([MODELS.md](MODELS.md)). Training on it would inflate every later
> score. Labels returned for it are used as `--truth` for scoring only. The full loop is in
> [IMPROVING-THE-MODEL.md](IMPROVING-THE-MODEL.md).

Images whose labels are still empty are **dropped, not merged** — an unlabelled image is not a
negative example, and asserting "no robots here" about a frame nobody looked at is exactly the
poisoning described above.

Then train `robot-v3` into a **new** folder, the same way v2 was trained on the AMD GPU. Never
overwrite v2: it is the only model whose behaviour on real footage we have actually measured.

See [TRAINING.md](TRAINING.md) for the training run itself.
