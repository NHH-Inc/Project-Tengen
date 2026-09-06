"""Which part of the frame is the field, when a broadcast puts two views in one picture.

Many venues composite a second camera under the main one. Both show the same six robots, so the
detector finds each robot twice, the tracker makes two tracks of it, and -- because attribution is
per track -- a human is asked to type the same team number twice while every shot on that robot is
counted twice. A duplicate is not a harmless extra box; it is worse for the numbers than a miss.

There are two ways to act on that, and neither of the arguments for the appealing one survived
measurement. Cropping before inference should spend the model's whole input on the panel that
matters rather than two thirds of it, and should be cheaper for handing over a smaller image.
Across the stacked segments of the viewpoint pack:

    input   filter   crop     cropping ahead in
    640px    4.22    4.05     7 of 28 segments
    960px    4.50    4.52    15 of 38 segments

At 640 filtering wins outright, because the model was trained on whole frames and moving the
aspect ratio away from that costs more than the extra pixels return. At 960 they tie, which fits
that reading -- the resolution the crop was buying stops mattering once there is enough of it.

The speed argument is simply false: 68.7 ms per frame whole-frame, 70.0 filtering, 70.8 cropping.
Both letterbox into the same square, so the model does identical work either way.

So `FILTER` is the default, on the strength of the 640 result and a tie at 960. `CROP` is kept
because it is six lines and already tested, and because a model *trained* on cropped frames would
change the comparison -- not because it currently helps. Re-measure before preferring it.

**The seam cannot be read off a single frame.** The obvious signal -- the row where consecutive
rows stop resembling each other -- was measured across the 400-frame viewpoint pack and does not
separate the two cases at all: stacked frames score a median 4.55 and single-view frames 4.49, and
no threshold does better than chance. A scoreboard edge, a spectator rail and a carpet boundary
are all just as sharp as a panel join.

What does separate them is where robots are *found*. A single view puts them in one horizontal
band; a stacked one puts the same robots in two bands with a clear gap between. That needs the
detector, so this is a calibration pass over a handful of frames per source -- the same shape as
`calibrate.py` -- and not something decided per frame at inference time.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median

#: Fraction of frame height that must separate two clusters of robots before they are called two
#: views. Robots near and far in one view span a lot of height, so a small gap means nothing; the
#: pack's stacked frames sit far above this and its single-view frames far below.
MIN_GAP = 0.18
#: A seam outside this band is not a panel join. Above it the "gap" is the scoreboard, below it
#: the gap is a caption bar.
MIN_SEAM, MAX_SEAM = 0.30, 0.80
#: Fewer detections than this cannot show two clusters, so the frame abstains rather than voting.
MIN_BOXES = 4
#: Robots either side of the seam before the two groups are called two camera views. One stray box
#: is noise, and treating it as a panel crops most of the frame away.
MIN_CLUSTER = 2
#: Where a panel's robots may sit. A group hugging the very top of the frame is the scoreboard
#: misread -- team avatars in the alliance strip detect as robots on several real sources -- and
#: one pinned to the very bottom edge is a caption bar. Neither is a view of the field, and both
#: otherwise produce a seam whose midpoint lands innocently mid-frame.
#:
#: The lower bound is generous because composited views really do sit low: on the sources measured
#: here their robots run from 0.87 to 0.97, centroid about 0.93. An earlier 0.92 rejected them and
#: called genuinely stacked sources single-view. The majority vote in `calibrate` is the real
#: defence against a stray pair of boxes; this only rejects what cannot be a camera view at all.
MIN_PANEL_CENTRE, MAX_PANEL_CENTRE = 0.15, 0.97
#: How much of a source's sampled frames must read stacked before the source is called stacked.
#: Well below half: a frame with everyone in one panel is common and is evidence of nothing,
#: while a frame that genuinely shows two bands is hard to produce by accident.
MIN_STACKED_SHARE = 0.35
#: Keep a little of the lower panel's sliver rather than cutting exactly on the seam, since the
#: seam is a midpoint between two robots and a robot's box extends below its centre.
SEAM_PADDING = 0.02


#: How a region is applied. FILTER runs the model on the whole frame and drops boxes whose centre
#: falls outside; CROP runs it on the region alone. They agree on which robots count and disagree
#: on cost and accuracy -- see the module docstring for the measurement.
FILTER, CROP = "filter", "crop"


@dataclass(frozen=True)
class Region:
    """A normalized part of the source frame. The full frame is the identity."""

    x: float = 0.0
    y: float = 0.0
    w: float = 1.0
    h: float = 1.0
    mode: str = FILTER

    @property
    def is_full_frame(self) -> bool:
        return (self.x, self.y, self.w, self.h) == (0.0, 0.0, 1.0, 1.0)

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h, "mode": self.mode}

    @classmethod
    def from_dict(cls, data: dict | None) -> "Region":
        if not data:
            return cls()
        mode = str(data.get("mode", FILTER))
        if mode not in (FILTER, CROP):
            raise ValueError(f"region mode must be {FILTER!r} or {CROP!r}, not {mode!r}")
        region = cls(float(data.get("x", 0.0)), float(data.get("y", 0.0)),
                     float(data.get("w", 1.0)), float(data.get("h", 1.0)), mode)
        if region.w <= 0 or region.h <= 0:
            raise ValueError(f"region has no area: {data}")
        if region.x < 0 or region.y < 0 or region.x + region.w > 1.000001 \
                or region.y + region.h > 1.000001:
            raise ValueError(f"region falls outside the frame: {data}")
        return region

    def contains_centre(self, box: dict) -> bool:
        """Whether a box belongs to this region, judged by its centre.

        The centre, not the edges: a robot straddling the seam belongs to whichever view it is
        mostly in, and testing edges would either drop it from both or keep it in both.
        """
        cx, cy = box["x"] + box["w"] / 2, box["y"] + box["h"] / 2
        return self.x <= cx <= self.x + self.w and self.y <= cy <= self.y + self.h


def crop(image, region: Region):
    """The pixels of `region`. Returns the original array when the region is the whole frame."""
    if region.is_full_frame:
        return image
    h, w = image.shape[:2]
    x0, y0 = int(round(region.x * w)), int(round(region.y * h))
    # At least one pixel each way, so a degenerate region cannot hand the model an empty array.
    x1 = max(x0 + 1, min(w, int(round((region.x + region.w) * w))))
    y1 = max(y0 + 1, min(h, int(round((region.y + region.h) * h))))
    return image[y0:y1, x0:x1]


def map_from_region(box: dict, region: Region) -> dict:
    """Put a box detected inside a crop back into source-frame coordinates.

    Forgetting this is the quiet failure: every box still looks like a valid normalized box, and
    every robot is reported nearer the top of the frame than it is. Tracking still works, field
    positions do not, and nothing raises.
    """
    if region.is_full_frame:
        return box
    out = dict(box)
    out["x"] = region.x + box["x"] * region.w
    out["y"] = region.y + box["y"] * region.h
    out["w"] = box["w"] * region.w
    out["h"] = box["h"] * region.h
    return out


def seam_from_centres(centres: list[float]) -> float | None:
    """The gap between two vertical clusters of robots, or None if there is only one cluster."""
    if len(centres) < MIN_BOXES:
        return None
    ordered = sorted(centres)
    gap, seam = 0.0, None
    for above, below in zip(ordered, ordered[1:]):
        if below - above > gap:
            gap, seam = below - above, (above + below) / 2
    if gap < MIN_GAP or seam is None or not (MIN_SEAM <= seam <= MAX_SEAM):
        return None

    # A seam is only a seam if there is a camera view on each side of it. Two boxes stranded in
    # the scoreboard put their midpoint with the field innocently mid-frame, and cropping there
    # would throw away most of the picture without anything looking wrong.
    above = [c for c in ordered if c <= seam]
    below = [c for c in ordered if c > seam]
    if len(above) < MIN_CLUSTER or len(below) < MIN_CLUSTER:
        return None
    if not MIN_PANEL_CENTRE <= sum(above) / len(above):
        return None
    if not sum(below) / len(below) <= MAX_PANEL_CENTRE:
        return None
    return seam


def centres_of(boxes: list[dict]) -> list[float]:
    return [b["y"] + b["h"] / 2 for b in boxes]


def calibrate(frames_boxes: list[list[dict]]) -> dict:
    """Decide a source's region of interest from detections on several of its frames.

    Takes boxes rather than images or a detector so that the decision is testable without a model,
    and so a caller may sample frames however it likes.

    Returns the region alongside the evidence for it. A caller that cannot see why a crop was
    chosen cannot tell a real second view from a bad sample, and this crop silently discards a
    third of the picture if it is wrong.
    """
    seams = [s for s in (seam_from_centres(centres_of(b)) for b in frames_boxes) if s is not None]
    usable = sum(1 for b in frames_boxes if len(b) >= MIN_BOXES)
    share = len(seams) / usable if usable else 0.0

    if not seams or share < MIN_STACKED_SHARE:
        return {
            "region": Region(),
            "stacked": False,
            "frames": len(frames_boxes),
            "frames_usable": usable,
            "frames_stacked": len(seams),
            "reason": ("no second view found" if usable else
                       "too few detections to tell -- check the model before trusting this"),
        }

    seam = median(seams)
    return {
        "region": Region(0.0, 0.0, 1.0, min(1.0, seam + SEAM_PADDING)),
        "stacked": True,
        "frames": len(frames_boxes),
        "frames_usable": usable,
        "frames_stacked": len(seams),
        "seam": seam,
        "seam_spread": (min(seams), max(seams)),
        "reason": f"{len(seams)} of {usable} usable frames show two bands of robots",
    }
