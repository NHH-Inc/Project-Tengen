"""Turn "a robot" into "team 254": the roster from the scoreboard, the number off the bumper.

This is doc 0's 2.2, and it is what makes the output *scouting* data rather than object detection.

The hard part is not the OCR. A team number on a bumper in a wide broadcast shot is ten to
fifteen pixels tall, and no engine reads that reliably; per-frame accuracy here is poor and always
will be. What makes the problem tractable is that it is not an open-vocabulary question:

  * the scoreboard names the six teams in the match, so the answer is one of six, not any number;
  * the bumper colour says which alliance, narrowing six to three;
  * a track holds a hundred observations of the same robot, and they vote.

So a read that is wrong most of the time still produces a confident track. What must never happen
is a confident *wrong* answer, so a read only votes for a candidate it is strictly closer to than
to every other, a track's alliance is decided before its digits are scored, and a track with no
clear winner is left as null. Null is a fine answer -- the web app already has a first-class path
for a human to attribute an unidentified track, and doing that once moves every event on it.

Nothing here overwrites raw model output. `team_confidence` is how sure this was, which is the
number the UI needs to flag the tracks a human should look at first.
"""

from __future__ import annotations

import os
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass, field

# Tesseract is optional. Without it the analyzer still runs and every track stays unattributed,
# which is the honest degradation -- the same shape as a missing TBA key or a missing model.
try:  # pragma: no cover - exercised by whether the binary exists, not by the tests
    import pytesseract
except ImportError:  # pragma: no cover
    pytesseract = None

#: OpenCV hue is 0-179. FRC red bumpers photograph as magenta and land at 150-179, not at 0-10;
#: a band of "red means hue under 10" classified every red robot in a real match as blue.
RED_HUE = ((150, 179), (0, 12))
BLUE_HUE = ((95, 138),)

RING = 4                  #: px dilation used to sample what surrounds a pale blob
RING_COLOURED = 0.45      #: fraction of that ring that must be bumper colour for it to be a digit
MIN_BUMPER_SHARE = 0.02   #: of the crop; below this it is a distant robot's bumper, not this one's
MAX_BUMPER_SHARE = 0.60   #: above this the mask has run off into the background
TARGET_HEIGHT = 96        #: px the digit strip is scaled to before reading
BOX_MARGIN = 0.25         #: the detector box often clips the bumper, so widen before cropping

MIN_SIMILARITY = 0.60     #: below this a read is not evidence for any candidate
MIN_READ_MARGIN = 0.15    #: best roster candidate must clearly beat the runner-up
MIN_VOTES = 2.0           #: total vote weight before a track may be attributed at all
MIN_CONFIDENCE = 0.55     #: winning share below this is a coin toss, so the track stays null
MIN_VOTE_MARGIN = 0.15    #: winning share must also clear second place by this much


def tesseract_available() -> bool:
    """Whether digits can actually be read. Callers degrade rather than fail."""
    if pytesseract is None:
        return False
    command = os.environ.get("TESSERACT_CMD")
    if command:
        pytesseract.pytesseract.tesseract_cmd = command
    try:
        pytesseract.get_tesseract_version()
        return True
    except (EnvironmentError, subprocess.SubprocessError):
        return False


# --- pure scoring, testable without a video or an OCR engine ------------------------------------

def similarity(read: str, candidate: str) -> float:
    """How much of the candidate a read accounts for, 0 to 1.

    Length-normalised longest common subsequence. Tolerant of one dropped or hallucinated digit,
    which is the usual failure at this size -- "824" for 8242 -- while still separating 11244
    from 11281, which differ in their last two.
    """
    if not read or not candidate:
        return 0.0
    grid = [[0] * (len(candidate) + 1) for _ in range(len(read) + 1)]
    for i in range(1, len(read) + 1):
        for j in range(1, len(candidate) + 1):
            if read[i - 1] == candidate[j - 1]:
                grid[i][j] = grid[i - 1][j - 1] + 1
            else:
                grid[i][j] = max(grid[i - 1][j], grid[i][j - 1])
    return grid[len(read)][len(candidate)] / max(len(read), len(candidate))


def score_read(read: str, candidates: list[int]) -> tuple[int, float] | None:
    """The one candidate this read is closest to, or None if it does not choose.

    A read equidistant from two teams is evidence for neither. Returning it anyway is how an
    attribution system manufactures confidence it has not earned.
    """
    if not candidates:
        return None
    ranked = sorted(((similarity(read, str(c)), c) for c in candidates), reverse=True)
    best_score, best = ranked[0]
    runner = ranked[1][0] if len(ranked) > 1 else 0.0
    if best_score < MIN_SIMILARITY or best_score - runner < MIN_READ_MARGIN:
        return None
    return best, best_score


def decide_alliance(observations: list[str | None]) -> str | None:
    """A track's alliance, by majority of the frames that saw a colour at all.

    Decided for the whole track before any digit is scored. Choosing per frame lets a frame whose
    bumper was not found fall back to all six candidates, and a blue robot then collects votes for
    red teams -- which is exactly what happened on two tracks before this existed.
    """
    votes = Counter(o for o in observations if o)
    if not votes:
        return None
    (winner, count), = votes.most_common(1)
    return winner if count >= 0.6 * sum(votes.values()) else None


@dataclass
class TrackVote:
    """What the reads across one track add up to."""

    alliance: str | None = None
    tally: dict[int, float] = field(default_factory=dict)
    reads: int = 0

    @property
    def total(self) -> float:
        return sum(self.tally.values())

    def resolve(self) -> tuple[int | None, float]:
        """(team, confidence). Null when the evidence does not support a single answer."""
        if not self.tally or self.total < MIN_VOTES:
            return None, 0.0
        ranked = sorted(self.tally.items(), key=lambda kv: -kv[1])
        team, weight = ranked[0]
        confidence = weight / self.total
        runner_weight = ranked[1][1] if len(ranked) > 1 else 0.0
        vote_margin = (weight - runner_weight) / self.total
        if confidence < MIN_CONFIDENCE or vote_margin < MIN_VOTE_MARGIN:
            return None, round(confidence, 3)
        return team, round(confidence, 3)


def tally_track(reads: list[str], alliance: str | None, roster: dict[str, list[int]]) -> TrackVote:
    """Score every read for one track against that track's three candidates."""
    vote = TrackVote(alliance=alliance)
    candidates = roster.get(alliance) if alliance else None
    if not candidates:
        # No alliance means no elimination. Doc 0 anticipates this for a job with no TBA data;
        # the same fallback serves a track whose bumper colour was never legible.
        candidates = sorted({t for teams in roster.values() for t in teams})
    tally: dict[int, float] = defaultdict(float)
    for read in reads:
        vote.reads += 1
        chosen = score_read(read, candidates)
        if chosen is not None:
            tally[chosen[0]] += chosen[1]
    vote.tally = dict(tally)
    return vote


# --- image work ----------------------------------------------------------------------------------

def _hue_mask(hue, saturation, value, bands):
    import numpy as np

    strong = (saturation > 80) & (value > 45)
    out = np.zeros_like(strong)
    for low, high in bands:
        out |= (hue >= low) & (hue <= high)
    return (out & strong).astype("uint8") * 255


def alliance_masks(patch) -> dict:
    """Where the red bumper is, and where the blue one is, in one crop."""
    import cv2

    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return {"red": _hue_mask(h, s, v, RED_HUE), "blue": _hue_mask(h, s, v, BLUE_HUE)}


def find_bumper(patch):
    """(alliance, colour mask, bounding box) of the bumper in this crop, or None.

    The area ceiling is load-bearing. Some venues frame a second view of the field in the same
    shot, and there the floor itself is blue-grey; without a ceiling the "bumper" becomes the
    whole picture, which is what happened on a real crop.
    """
    import cv2
    import numpy as np

    total = patch.shape[0] * patch.shape[1]
    masks = alliance_masks(patch)
    best = None
    for alliance, mask in masks.items():
        # Close the gaps the digits punch in the slab, so the bumper is one component.
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        count, _, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
        for i in range(1, count):
            x, y, w, h, area = stats[i]
            if not (total * MIN_BUMPER_SHARE <= area <= total * MAX_BUMPER_SHARE):
                continue
            if w < h * 0.8:      # a bumper from any angle is wider than it is tall
                continue
            if area > (best[3] if best else 0):
                best = (alliance, mask, (x, y, w, h), area)
    return None if best is None else (best[0], best[1], best[2])


def digit_strip(patch, colour_mask, box):
    """A tight, upscaled grayscale image of just the number, or None.

    Locating the digits and reading them want different pictures. A pale blob is a digit only if
    what *surrounds* it is bumper -- a strut is ringed by floor and more metal -- and that ring
    test does what no brightness threshold could, because bumper white, aluminium and the field
    floor are all pale.

    But the reading is done on the original pixels. A binary mask of a ten-pixel glyph throws away
    the anti-aliasing that carries its shape, and every attempt to hand Tesseract a mask produced
    unreadable lumps.
    """
    import cv2
    import numpy as np

    x, y, w, h = box
    region = patch[y:y + h, x:x + w]
    colour = colour_mask[y:y + h, x:x + w]
    if region.size == 0:
        return None
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    pale = ((hsv[..., 2] > 110) & (hsv[..., 1] < 120)).astype(np.uint8) * 255

    count, labels, stats, _ = cv2.connectedComponentsWithStats(pale, connectivity=8)
    kernel = np.ones((RING * 2 + 1, RING * 2 + 1), np.uint8)
    candidates = []
    for i in range(1, count):
        cx, cy, cw, ch, area = stats[i]
        if ch < 4 or ch > h * 0.9 or area < 8 or cw > ch * 2.2:
            continue
        blob = (labels == i).astype(np.uint8) * 255
        ring = cv2.subtract(cv2.dilate(blob, kernel), blob)
        ring_px = int(ring.sum() / 255)
        if ring_px == 0:
            continue
        if int(cv2.bitwise_and(ring, colour).sum() / 255) / ring_px < RING_COLOURED:
            continue
        candidates.append((i, cx, cy, cw, ch))
    if len(candidates) < 2:
        return None

    # Digits of one number share a height; the bumper's own lit edge does not.
    median_h = float(np.median([c[4] for c in candidates]))
    digits = [c for c in candidates if abs(c[4] - median_h) <= max(2.0, median_h * 0.35)]
    if len(digits) < 2:
        return None

    x0 = max(0, min(c[1] for c in digits) - 2)
    y0 = max(0, min(c[2] for c in digits) - 2)
    x1 = min(w, max(c[1] + c[3] for c in digits) + 2)
    y1 = min(h, max(c[2] + c[4] for c in digits) + 2)
    strip = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)[y0:y1, x0:x1]
    if strip.size == 0 or strip.shape[0] < 4:
        return None

    scale = TARGET_HEIGHT / strip.shape[0]
    big = cv2.resize(strip, (max(8, int(strip.shape[1] * scale)), TARGET_HEIGHT),
                     interpolation=cv2.INTER_CUBIC)
    big = cv2.normalize(big, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.copyMakeBorder(big, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=int(np.median(big)))


def read_bumper(patch) -> tuple[str | None, list[str]]:
    """(alliance, digit strings) from one robot crop."""
    import cv2

    found = find_bumper(patch)
    if found is None:
        return None, []
    alliance, colour_mask, box = found
    if not tesseract_available():
        return alliance, []
    strip = digit_strip(patch, colour_mask, box)
    if strip is None:
        return alliance, []

    reads: list[str] = []
    for psm in ("13", "7", "8", "6"):
        text = pytesseract.image_to_string(
            strip, config=f"--psm {psm} -c tessedit_char_whitelist=0123456789").strip()
        for run in re.findall(r"\d+", text):
            if 3 <= len(run) <= 6:
                reads.append(run)
        if reads:
            break
    return alliance, reads


def crop_for(frame, box, margin: float = BOX_MARGIN):
    """The robot's box, widened. The detector often clips the bumper off the bottom."""
    h, w = frame.shape[:2]
    mx, my = box["w"] * margin, box["h"] * margin
    x0 = max(0, int((box["x"] - mx) * w))
    y0 = max(0, int((box["y"] - my) * h))
    x1 = min(w, int((box["x"] + box["w"] + mx) * w))
    y1 = min(h, int((box["y"] + box["h"] + my) * h))
    return frame[y0:y1, x0:x1]
