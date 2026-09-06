"""Draw the Tengen emoji set and server icon as PNG, straight from geometry.

Discord will not take SVG, and converting one needs a rendering library that is awkward to install
on Windows. Everything here is geometric anyway -- hexagons, stars, boxes -- so it is drawn
directly with Pillow instead, which removes the conversion step entirely.

Pillow does not antialias shape edges, so every image is drawn at 4x and downsampled with LANCZOS.
Without that, a 128px hexagon has visibly stepped diagonals, which is exactly the sort of thing
that looks fine in isolation and cheap next to real emoji in a picker.

The set is chosen for what this project actually talks about: builds passing and failing, a
bounding box, a track, a detection. Generic reaction emoji are already covered by Discord.
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent / "emoji"
SS = 4  # supersample factor

NAVY = (22, 35, 58, 255)
STEEL = (195, 206, 217, 255)
STEEL_DIM = (143, 160, 179, 255)
WHITE = (242, 246, 250, 255)
GREEN = (46, 204, 113, 255)
RED = (231, 76, 60, 255)
AMBER = (230, 126, 34, 255)
PURPLE = (155, 89, 182, 255)
BLUE = (52, 152, 219, 255)


def canvas(size: int, bg=(0, 0, 0, 0)):
    img = Image.new("RGBA", (size * SS, size * SS), bg)
    return img, ImageDraw.Draw(img)


def finish(img: Image.Image, size: int, name: str) -> Path:
    out = img.resize((size, size), Image.LANCZOS)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.png"
    out.save(path, "PNG", optimize=True)
    return path


def hexagon(cx: float, cy: float, r: float) -> list[tuple[float, float]]:
    """Flat-top hexagon, same orientation as the logo."""
    return [(cx + r * math.cos(math.radians(a)), cy + r * math.sin(math.radians(a)))
            for a in (-60, -120, 180, 120, 60, 0)]


def star8(cx: float, cy: float, long_r: float, short_r: float, waist: float):
    """The logo's eight-point star, as a list of kite polygons."""
    kites = []
    for i in range(8):
        angle = math.radians(i * 45)
        reach = long_r if i % 2 == 0 else short_r
        tip = (cx + reach * math.cos(angle), cy + reach * math.sin(angle))
        left = (cx + waist * math.cos(angle + math.pi / 4),
                cy + waist * math.sin(angle + math.pi / 4))
        right = (cx + waist * math.cos(angle - math.pi / 4),
                 cy + waist * math.sin(angle - math.pi / 4))
        kites.append(([tip, left, (cx, cy), right], i % 2 == 0))
    return kites


def draw_mark(d: ImageDraw.ImageDraw, cx, cy, r, ring=True, width_scale=1.0, fill=None):
    """The Tengen mark: hexagon, star, centre node.

    `fill` matters more than it looks. Steel on transparent disappears against a light theme --
    fine on Discord's dark default, invisible for anyone on light. Filling the hexagon navy gives
    the mark its own background so it reads the same either way, which is the whole job of an
    emoji.
    """
    lw = max(1, int(r * 0.085 * width_scale))
    d.polygon(hexagon(cx, cy, r), fill=fill, outline=STEEL, width=lw)
    if ring:
        rr = r * 0.62
        d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], outline=STEEL_DIM,
                  width=max(1, int(r * 0.018)))
    for kite, is_cardinal in star8(cx, cy, r * 0.92, r * 0.34, r * 0.10):
        d.polygon(kite, fill=STEEL if is_cardinal else STEEL_DIM)
    nr = r * 0.10
    d.ellipse([cx - nr, cy - nr, cx + nr, cy + nr], fill=NAVY, outline=STEEL,
              width=max(1, int(r * 0.045)))


# --- the set ---------------------------------------------------------------------------------

def emoji_tengen(size=128):
    img, d = canvas(size)
    s = size * SS
    draw_mark(d, s / 2, s / 2, s * 0.44, fill=NAVY)
    return finish(img, size, "tengen")


def _badge(name, colour, glyph, size=128):
    """A hexagon in an accent colour with a simple glyph inside."""
    img, d = canvas(size)
    s = size * SS
    cx = cy = s / 2
    r = s * 0.44
    d.polygon(hexagon(cx, cy, r), fill=NAVY, outline=colour, width=int(s * 0.055))
    glyph(d, cx, cy, s, colour)
    return finish(img, size, name)


def _tick(d, cx, cy, s, colour):
    w = int(s * 0.075)
    d.line([(cx - s * 0.17, cy + s * 0.02), (cx - s * 0.04, cy + s * 0.15),
            (cx + s * 0.19, cy - s * 0.16)], fill=colour, width=w, joint="curve")


def _cross(d, cx, cy, s, colour):
    w = int(s * 0.075)
    k = s * 0.15
    d.line([(cx - k, cy - k), (cx + k, cy + k)], fill=colour, width=w)
    d.line([(cx - k, cy + k), (cx + k, cy - k)], fill=colour, width=w)


def _dots(d, cx, cy, s, colour):
    r = s * 0.045
    for dx in (-s * 0.15, 0, s * 0.15):
        d.ellipse([cx + dx - r, cy - r, cx + dx + r, cy + r], fill=colour)


def _merge(d, cx, cy, s, colour):
    """A branch rejoining the trunk.

    The first attempt drew the arc sweeping outward, so the branch curved away and dead-ended in
    empty space -- a fork, not a merge, and visibly wrong once blown up. The arc has to be centred
    ON the trunk at the branch's height, so sweeping from 0 to 90 degrees lands it back on the
    trunk line rather than beside it.
    """
    w = int(s * 0.055)
    r_dot = s * 0.055
    trunk_x = cx - s * 0.11
    top_y, bottom_y = cy - s * 0.19, cy + s * 0.19
    branch_x, branch_y = cx + s * 0.15, cy - s * 0.13
    radius = branch_x - trunk_x

    d.line([(trunk_x, top_y), (trunk_x, bottom_y)], fill=colour, width=w)
    # Centre on the trunk, at the branch's height: 0deg is the branch, 90deg is the trunk below.
    d.arc([trunk_x - radius, branch_y - radius, trunk_x + radius, branch_y + radius],
          0, 90, fill=colour, width=w)
    for px, py in ((trunk_x, top_y), (trunk_x, bottom_y), (branch_x, branch_y)):
        d.ellipse([px - r_dot, py - r_dot, px + r_dot, py + r_dot], fill=colour)


def emoji_bbox(size=128):
    """A detection box on a robot silhouette. The thing this project exists to draw."""
    img, d = canvas(size)
    s = size * SS
    body = [s * 0.34, s * 0.42, s * 0.66, s * 0.68]
    d.rounded_rectangle(body, radius=s * 0.03, fill=STEEL_DIM)
    wheel = s * 0.055
    for wx in (s * 0.40, s * 0.60):
        d.ellipse([wx - wheel, s * 0.66 - wheel, wx + wheel, s * 0.66 + wheel], fill=NAVY)
    box = [s * 0.22, s * 0.28, s * 0.78, s * 0.78]
    d.rectangle(box, outline=GREEN, width=int(s * 0.045))
    corner = s * 0.10
    for (x, y, dx, dy) in ((box[0], box[1], 1, 1), (box[2], box[1], -1, 1),
                           (box[0], box[3], 1, -1), (box[2], box[3], -1, -1)):
        d.line([(x, y), (x + corner * dx, y)], fill=WHITE, width=int(s * 0.05))
        d.line([(x, y), (x, y + corner * dy)], fill=WHITE, width=int(s * 0.05))
    return finish(img, size, "bbox")


def emoji_track(size=128):
    """A trajectory: what tracking produces when it works."""
    img, d = canvas(size)
    s = size * SS
    pts = [(s * 0.16, s * 0.70), (s * 0.34, s * 0.44), (s * 0.52, s * 0.58), (s * 0.84, s * 0.26)]
    d.line(pts, fill=BLUE, width=int(s * 0.055), joint="curve")
    for i, (px, py) in enumerate(pts):
        r = s * 0.055 if i == len(pts) - 1 else s * 0.038
        d.ellipse([px - r, py - r, px + r, py + r],
                  fill=WHITE if i == len(pts) - 1 else BLUE)
    return finish(img, size, "track")


def emoji_stats(size=128):
    img, d = canvas(size)
    s = size * SS
    for i, (h, c) in enumerate(((0.30, STEEL_DIM), (0.48, STEEL), (0.66, GREEN))):
        x = s * (0.24 + i * 0.20)
        d.rounded_rectangle([x, s * (0.80 - h), x + s * 0.13, s * 0.80],
                            radius=s * 0.02, fill=c)
    return finish(img, size, "stats")


def server_icon(size=512):
    img, d = canvas(size, bg=NAVY)
    s = size * SS
    # Rounded square, since Discord masks to a circle and a square bleeds at the corners anyway.
    mask = Image.new("L", (s, s), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, s, s], radius=int(s * 0.18), fill=255)
    img.putalpha(mask)
    draw_mark(d, s / 2, s / 2, s * 0.36, width_scale=1.15)
    return finish(img, size, "server-icon")


def bot_avatar(size=1024):
    """The mark on a full-bleed navy square, sized for Discord's circular crop.

    Two differences from `server_icon`, both because of that crop. The square is filled edge to
    edge rather than rounded, since the corners are discarded and a rounded rect only risks a
    sliver of transparency at the circle's tangent points. And the mark is drawn larger, because
    the crop removes about a fifth of the visible area and a mark sized for the square reads
    undersized once it lands in the circle.

    1024 rather than 512: Discord serves avatars at up to 512 on retina, and downscaling from
    double that keeps the star's thin kites clean.
    """
    img, d = canvas(size, bg=NAVY)
    s = size * SS
    draw_mark(d, s / 2, s / 2, s * 0.40, width_scale=1.15)
    return finish(img, size, "bot-avatar")


if __name__ == "__main__":
    made = [
        emoji_tengen(),
        _badge("tengen_pass", GREEN, _tick),
        _badge("tengen_fail", RED, _cross),
        _badge("tengen_building", AMBER, _dots),
        _badge("tengen_merged", PURPLE, _merge),
        emoji_bbox(),
        emoji_track(),
        emoji_stats(),
        server_icon(),
        bot_avatar(),
    ]
    for path in made:
        kb = path.stat().st_size / 1024
        limit = 8192 if path.stem in ("server-icon", "bot-avatar") else 256
        flag = "ok" if kb < limit else "TOO BIG"
        print(f"  {path.name:<22} {kb:6.1f} KB  {flag}")


# --- utility set ------------------------------------------------------------------------------
# The genuinely useful emoji on sites like emoji.gg are coloured bullets, ticks and status dots.
# They are also trivial geometry, so generating them here gives a set that matches the brand
# exactly and carries no licensing question -- rather than a grab-bag of other people's art in
# eight different styles, uploaded without attribution.

ACCENTS = {
    "green": GREEN, "red": RED, "amber": AMBER, "blue": BLUE,
    "purple": PURPLE, "steel": STEEL,
}


def emoji_dot(name: str, colour, size=128):
    img, d = canvas(size)
    s = size * SS
    r = s * 0.30
    d.ellipse([s/2 - r, s/2 - r, s/2 + r, s/2 + r], fill=colour)
    return finish(img, size, name)


def emoji_arrow(name: str, colour, rotate=0, size=128):
    img, d = canvas(size)
    s = size * SS
    d.polygon([(s*0.30, s*0.22), (s*0.72, s*0.50), (s*0.30, s*0.78),
               (s*0.30, s*0.62), (s*0.50, s*0.50), (s*0.30, s*0.38)], fill=colour)
    if rotate:
        img = img.rotate(rotate, resample=Image.BICUBIC)
    return finish(img, size, name)


def emoji_priority(name: str, colour, bars: int, size=128):
    """One, two or three rising bars. Reads as severity without needing words."""
    img, d = canvas(size)
    s = size * SS
    for i in range(3):
        h = 0.22 + i * 0.18
        c = colour if i < bars else (255, 255, 255, 40)
        x = s * (0.24 + i * 0.20)
        d.rounded_rectangle([x, s*(0.80 - h), x + s*0.13, s*0.80], radius=s*0.02, fill=c)
    return finish(img, size, name)
