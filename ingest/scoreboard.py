"""The match roster, read off the broadcast scoreboard.

Doc 0 says `alliances` is what narrows bumper OCR to three candidates a side, and that a job whose
TBA lookup found nothing is still valid. That case is not rare -- the first real match analysed
had `alliances: null`, because TBA has no data for it -- and without a roster bumper OCR is an
open-vocabulary problem it cannot win. The graphic on screen already names all six teams, so it
is the fallback, and doc 0 already reserves `scoreboard_ocr` as a source for exactly this.

Two ideas do the work, and neither needs a per-venue crop region:

  * The alliance bar is the one place on screen where a wide band of saturated red sits directly
    beside a wide band of saturated blue. Finding that pair locates both cells, and which cell a
    number came from IS its alliance -- no guessing from left and right.

  * Inside a cell, the team numbers are the digits that do not change. The score changes, the
    timer changes, the ranking changes. Sampling frames across the match separates them with no
    knowledge of the layout at all.

Handing a whole cell to Tesseract does not work: the team logos beside each number are shapes it
tries to read, and the score is three times the height of the team numbers, so there is no
consistent line. Glyphs are isolated instead, grouped by the height they share, and each number is
read on its own -- a much easier question than reading a layout.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict

from .team_id import tesseract_available

try:  # pragma: no cover
    import pytesseract
except ImportError:  # pragma: no cover
    pytesseract = None

STRIP_SHARE = 0.20    #: top fraction of the frame the graphic can occupy
MIN_RUN = 120         #: px; a narrower band of colour is an icon or a side panel, not a cell
UPSCALE = 4
TEAM_DIGITS = (3, 5)  #: an FRC team number, so not a score and not a fragment
ALLIANCE_SIZE = 3


def _runs(present, threshold, min_run=MIN_RUN):
    """Contiguous column ranges where a colour covers most of the bar's height."""
    on = present > threshold
    out, start = [], None
    for x, flag in enumerate(on):
        if flag and start is None:
            start = x
        elif not flag and start is not None:
            if x - start >= min_run:
                out.append((start, x - 1))
            start = None
    if start is not None and len(on) - start >= min_run:
        out.append((start, len(on) - 1))
    return out


def find_cells(frame):
    """{"red": (x0, x1, y0, y1), "blue": ...} for the alliance bar, or None."""
    import cv2
    import numpy as np

    h, w = frame.shape[:2]
    strip = frame[0:int(h * STRIP_SHARE)]
    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
    hh, ss, vv = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    strong = (ss > 120) & (vv > 70)
    red = strong & ((hh < 10) | (hh > 170))
    blue = strong & (hh > 100) & (hh < 130)

    bar_rows = [y for y in range(strip.shape[0])
                if red[y].sum() > w * 0.10 and blue[y].sum() > w * 0.10]
    if not bar_rows:
        return None
    y0, y1 = min(bar_rows), max(bar_rows)
    height = y1 - y0 + 1
    red_runs = _runs(red[y0:y1 + 1].sum(axis=0), height * 0.5)
    blue_runs = _runs(blue[y0:y1 + 1].sum(axis=0), height * 0.5)
    if not red_runs or not blue_runs:
        return None
    # The bar is the red run and the blue run that touch. Side panels sit far from both.
    r, b = min(((r, b) for r in red_runs for b in blue_runs),
               key=lambda rb: min(abs(rb[0][1] - rb[1][0]), abs(rb[1][1] - rb[0][0])))
    return {"red": (r[0], r[1], y0, y1), "blue": (b[0], b[1], y0, y1)}


def read_cell(frame, cell) -> list[int]:
    """Every team-number-shaped group of white glyphs in one alliance cell."""
    import cv2
    import numpy as np

    if not tesseract_available():
        return []
    x0, x1, y0, y1 = cell
    patch = frame[y0:y1 + 1, x0:x1 + 1]
    if patch.size == 0:
        return []
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    # White text on a saturated ground. A team logo is coloured, so it drops out here.
    white = ((hsv[..., 2] > 150) & (hsv[..., 1] < 90)).astype(np.uint8) * 255

    count, _, stats, _ = cv2.connectedComponentsWithStats(white, connectivity=8)
    glyphs = [(x, y, w, h) for i in range(1, count)
              for x, y, w, h, area in [stats[i]]
              if h >= 6 and area >= 12 and w <= h * 2.5]
    if not glyphs:
        return []

    # Team-number digits share a height; the score is far taller. The commonest height is theirs.
    mode = Counter(g[3] for g in glyphs).most_common(1)[0][0]
    same = [g for g in glyphs if abs(g[3] - mode) <= max(2, mode * 0.25)]
    if not same:
        return []

    same.sort(key=lambda g: g[0])
    gap_limit = mode * 0.9
    groups, current = [], [same[0]]
    for g in same[1:]:
        prev = current[-1]
        if g[0] - (prev[0] + prev[2]) > gap_limit or abs(g[1] - prev[1]) > mode * 0.5:
            groups.append(current)
            current = [g]
        else:
            current.append(g)
    groups.append(current)

    out = []
    for group in groups:
        gx0 = min(g[0] for g in group)
        gy0 = min(g[1] for g in group)
        gx1 = max(g[0] + g[2] for g in group)
        gy1 = max(g[1] + g[3] for g in group)
        crop = white[max(0, gy0 - 3):gy1 + 3, max(0, gx0 - 3):gx1 + 3]
        if crop.size == 0:
            continue
        big = cv2.resize(crop, (crop.shape[1] * UPSCALE, crop.shape[0] * UPSCALE),
                         interpolation=cv2.INTER_CUBIC)
        big = cv2.copyMakeBorder(big, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=0)
        text = pytesseract.image_to_string(
            cv2.bitwise_not(big),
            config="--psm 7 -c tessedit_char_whitelist=0123456789").strip()
        digits = re.sub(r"\D", "", text)
        if TEAM_DIGITS[0] <= len(digits) <= TEAM_DIGITS[1]:
            out.append(int(digits))
    return out


def roster_from_counts(counts: dict[str, dict[int, int]], frames: int) -> dict[str, list[int]]:
    """The three most persistent numbers per alliance, and only if they really persisted.

    A real majority is required so a missing team stays missing rather than being filled in with
    the best available piece of noise. Two teams and a null is a better answer than three teams
    one of which is invented.
    """
    roster = {}
    for alliance in ("red", "blue"):
        ranked = sorted(counts.get(alliance, {}).items(), key=lambda kv: (-kv[1], kv[0]))
        roster[alliance] = sorted(n for n, c in ranked[:ALLIANCE_SIZE] if c > frames / 2)
    return roster


def read_roster(video_path, sample_times, fps: float) -> dict[str, list[int]]:
    """Sample the scoreboard across the match and keep what stayed constant."""
    import cv2

    wanted = {int(round(t * fps)) for t in sample_times}
    counts: dict[str, dict[int, int]] = {"red": defaultdict(int), "blue": defaultdict(int)}
    frames = 0
    capture = cv2.VideoCapture(str(video_path))
    index, last = 0, (max(wanted) if wanted else -1)
    while index <= last:
        ok, frame = capture.read()
        if not ok:
            break
        if index in wanted:
            cells = find_cells(frame)
            if cells is not None:
                frames += 1
                for alliance, cell in cells.items():
                    for number in read_cell(frame, cell):
                        counts[alliance][number] += 1
        index += 1
    capture.release()
    if frames == 0:
        return {"red": [], "blue": []}
    return roster_from_counts(counts, frames)


#: Dilation used to sample what surrounds a glyph, in pixels.
SCORE_RING = 6


def _surrounding_colour(hsv, component):
    """red, blue or white, from the hue immediately around a glyph."""
    import cv2
    import numpy as np

    ring = cv2.subtract(
        cv2.dilate(component, np.ones((SCORE_RING * 2 + 1,) * 2, np.uint8)), component)
    selected = ring > 0
    if selected.sum() < 20:
        return None
    hue, sat = hsv[..., 0][selected], hsv[..., 1][selected]
    strong = sat > 90
    if strong.sum() < 0.25 * selected.sum():
        return "white"
    coloured = hue[strong]
    red = int(((coloured < 10) | (coloured > 170)).sum())
    blue = int(((coloured > 100) & (coloured < 135)).sum())
    return "red" if red > blue else "blue"


def _bar_columns(cells) -> slice:
    """Both alliance cells and everything between them, whichever side red is on.

    Venues disagree: Istanbul put red on the left, the 2026 Championship put blue there. Taking
    red's left edge and blue's right edge assumed the former, and on a blue-left broadcast it
    produced an empty strip, so the timer and both scores read as nothing for the whole match.
    """
    left = min(cells["red"][0], cells["blue"][0])
    right = max(cells["red"][1], cells["blue"][1])
    return slice(left, right + 1)


def _bar_rows(cells) -> slice:
    top = min(cells["red"][2], cells["blue"][2])
    bottom = max(cells["red"][3], cells["blue"][3])
    return slice(top, bottom + 1)


def read_alliance_scores(frame) -> tuple[int | None, int | None]:
    """(red, blue) as shown on the scoreboard, or None for either that could not be read.

    Read across the whole bar rather than inside the alliance cells. The score's white digits are
    tall enough to break the run of alliance colour that defines a cell, so the cell boundary
    stops exactly where the score starts -- searching inside one finds the 30x30 team logos and
    calls them the score.

    The three big numbers on the bar are the two scores and the timer between them. They are
    separated by the colour immediately around each, which is the same reasoning that finds bumper
    digits: the glyph is not distinctive, its background is.

    A score is not a shot count. This reports what the broadcast says; turning it into scoring
    events is `action_extraction`'s job, and needs the season's point values to go further.
    """
    import re

    import cv2
    import numpy as np

    if not tesseract_available():
        return None, None
    cells = find_cells(frame)
    if not cells:
        return None, None

    bar = frame[_bar_rows(cells), _bar_columns(cells)]
    if bar.size == 0:
        return None, None

    hsv = cv2.cvtColor(bar, cv2.COLOR_BGR2HSV)
    white = ((hsv[..., 2] > 150) & (hsv[..., 1] < 90)).astype(np.uint8) * 255
    count, labels, stats, _ = cv2.connectedComponentsWithStats(white, connectivity=8)

    bar_height = bar.shape[0]
    big = []
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        # A score digit is tall and taller than it is wide. A team logo is square; a team number
        # is under half this height.
        if h >= bar_height * 0.28 and w < h * 0.85 and area >= 60:
            big.append((i, x, y, w, h))

    grouped: dict[str, list] = {"red": [], "blue": []}
    for i, x, y, w, h in big:
        where = _surrounding_colour(hsv, (labels == i).astype(np.uint8) * 255)
        if where in grouped:
            grouped[where].append((i, x, y, w, h))

    out: dict[str, int | None] = {}
    for side in ("red", "blue"):
        items = sorted(grouped[side], key=lambda g: g[1])
        if not items:
            out[side] = None
            continue
        keep = np.zeros_like(white)
        for i, *_ in items:
            keep[labels == i] = 255
        x0 = max(0, min(g[1] for g in items) - 3)
        y0 = max(0, min(g[2] for g in items) - 3)
        x1 = max(g[1] + g[3] for g in items) + 3
        y1 = max(g[2] + g[4] for g in items) + 3
        crop = keep[y0:y1, x0:x1]
        if crop.size == 0:
            out[side] = None
            continue
        big_crop = cv2.resize(crop, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC)
        big_crop = cv2.copyMakeBorder(big_crop, 25, 25, 25, 25, cv2.BORDER_CONSTANT, value=0)
        text = pytesseract.image_to_string(
            cv2.bitwise_not(big_crop),
            config="--psm 7 -c tessedit_char_whitelist=0123456789").strip()
        digits = re.sub(r"\D", "", text)
        out[side] = int(digits) if digits and len(digits) <= 3 else None
    return out.get("red"), out.get("blue")


def read_match_timer(frame) -> int | None:
    """Seconds remaining on the match clock, from the timer between the two scores.

    This is doc 0's 2.1, and it is what makes every phase boundary real rather than assumed. A
    clip is not a match: this one runs 215 seconds around 150 seconds of play, so timing phases
    from the start of the file puts auto, teleop and endgame in the wrong places and attributes
    events to the wrong period.

    The timer is already isolated by `read_alliance_scores` -- it is the big number whose
    surroundings are white rather than alliance-coloured -- so this only has to read it. Digits
    only: the colon is dropped and the halves are recovered by length, because a colon at this
    size is two specks that Tesseract reads as a 1, an 8, or nothing at all.
    """
    import re

    import cv2
    import numpy as np

    if not tesseract_available():
        return None
    cells = find_cells(frame)
    if not cells:
        return None

    bar = frame[_bar_rows(cells), _bar_columns(cells)]
    if bar.size == 0:
        return None

    hsv = cv2.cvtColor(bar, cv2.COLOR_BGR2HSV)
    # The timer is the one number on the bar that is DARK on light: it sits in a white box
    # between the two scores, which are light on alliance colour. Masking for white finds the
    # box, not the digits -- the first attempt read nothing on all 22 samples for exactly that.
    white_box = ((hsv[..., 2] > 150) & (hsv[..., 1] < 90)).astype(np.uint8) * 255
    dark = ((hsv[..., 2] < 110) & (hsv[..., 1] < 120)).astype(np.uint8) * 255
    # Only dark pixels sitting inside the white box are timer digits; everything else dark on the
    # bar is a team logo or the gap between cells.
    inside = cv2.erode(white_box, np.ones((9, 9), np.uint8))
    inside = cv2.dilate(inside, np.ones((25, 25), np.uint8))
    dark = cv2.bitwise_and(dark, inside)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)

    bar_height = bar.shape[0]
    timer = []
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if h >= bar_height * 0.28 and w < h * 1.1 and area >= 40:
            timer.append((i, x, y, w, h))
    if not timer:
        return None

    timer.sort(key=lambda g: g[1])
    keep = np.zeros_like(dark)
    for i, *_ in timer:
        keep[labels == i] = 255
    x0 = max(0, min(g[1] for g in timer) - 3)
    y0 = max(0, min(g[2] for g in timer) - 3)
    x1 = max(g[1] + g[3] for g in timer) + 3
    y1 = max(g[2] + g[4] for g in timer) + 3
    crop = keep[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    big = cv2.resize(crop, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC)
    big = cv2.copyMakeBorder(big, 25, 25, 25, 25, cv2.BORDER_CONSTANT, value=0)
    text = pytesseract.image_to_string(
        cv2.bitwise_not(big), config="--psm 7 -c tessedit_char_whitelist=0123456789").strip()
    return timer_seconds(re.sub(r"\D", "", text))


def timer_seconds(digits: str) -> int | None:
    """Turn the digits of a match clock into seconds remaining.

    The colon never survives OCR at this size, so the split is recovered by length: the last two
    digits are always seconds, and whatever precedes them is minutes. "220" is 2:20, not 220
    seconds, and reading it as the latter would put the match start over three minutes wrong.
    """
    if not digits or len(digits) > 4:
        return None
    if len(digits) <= 2:
        seconds, minutes = int(digits), 0
    else:
        seconds, minutes = int(digits[-2:]), int(digits[:-2])
    if seconds > 59 or minutes > 15:
        return None
    return minutes * 60 + seconds
