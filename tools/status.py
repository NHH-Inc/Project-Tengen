"""Status graphics for the README and Discord, generated from the registry and the catalogue.

    python -m tools.status

Writes docs/status/:
  status.svg / .png   two treemaps and a history chart, in the style of AnyPS5's status:
                        - held-out frames, one cell per frame, one block per 2026 event, green where
                          the current model finds all six robots;
                        - strategy features, one cell per feature of the plan's catalogue, one
                          block per category, green where built;
                        - daily history of both percentages.
  badge-*.svg         small shields for the README

The picture is built once as a list of shapes and rendered twice: SVG for GitHub (hover titles
work there) and PNG for Discord, which does not show SVG. One scene means the two cannot drift.
Every number comes from training/registry and ingest/strategy/catalogue.py.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from html import escape
from math import ceil, sqrt

from ingest.strategy.catalogue import FEATURES, built
from tools.models import ROOT, models_with_scores

OUT = ROOT / "docs" / "status"
BACKGROUND = "#0d1117"
TEXT = "#ffffff"
MUTED = "#8b949e"
GRID = "#30363d"
DONE = "#2ea043"
TODO = "#6e7681"
MODEL_COLOUR = "#388bfd"
FEATURE_COLOUR = "#2ea043"
WIDTH, HEIGHT = 1100, 800


# ----------------------------------------------------------------------------- scene

@dataclass
class Rect:
    x: float
    y: float
    w: float
    h: float
    fill: str
    stroke: str = BACKGROUND
    stroke_width: float = 0.5
    title: str | None = None
    opacity: float = 1.0


@dataclass
class Text:
    x: float
    y: float
    text: str
    size: int = 13
    fill: str = TEXT
    anchor: str = "start"      # start | middle | end
    rotate: float = 0.0        # degrees, about (x, y)
    bold: bool = False
    outline: bool = False      # dark halo, for labels drawn over cells


@dataclass
class Line:
    points: list[tuple[float, float]]
    stroke: str
    width: float = 1.0


@dataclass
class Area:
    points: list[tuple[float, float]]
    fill: str
    opacity: float = 0.15


@dataclass
class Dot:
    x: float
    y: float
    r: float
    fill: str


@dataclass
class Scene:
    width: int
    height: int
    shapes: list = field(default_factory=list)

    def add(self, *shapes):
        self.shapes.extend(shapes)


# ----------------------------------------------------------------------------- treemap

def squarify(values: list[float], x: float, y: float, w: float, h: float) -> list[tuple]:
    """Bruls' squarified treemap: rectangles proportional to values, as square as possible."""
    total = sum(values)
    if not values or total <= 0:
        return []
    scaled = [v * w * h / total for v in values]
    rects: list[tuple] = []

    def worst(row, side):
        s = sum(row)
        return max(max(side * side * r / (s * s), (s * s) / (side * side * r)) for r in row)

    index = 0
    while index < len(scaled):
        side = min(w, h)
        row = [scaled[index]]
        index += 1
        while index < len(scaled) and worst(row + [scaled[index]], side) <= worst(row, side):
            row.append(scaled[index])
            index += 1
        s = sum(row)
        if w >= h:  # lay the row down the left edge
            column = s / h
            offset = y
            for r in row:
                rects.append((x, offset, column, r / column))
                offset += r / column
            x, w = x + column, w - column
        else:       # lay the row along the top edge
            band = s / w
            offset = x
            for r in row:
                rects.append((offset, y, r / band, band))
                offset += r / band
            y, h = y + band, h - band
    return rects


def cell_grid(n: int, w: float, h: float) -> tuple[int, int]:
    """Columns and rows that fit n roughly square cells into w x h."""
    columns = max(1, round(sqrt(n * w / h))) if h > 0 else n
    rows = ceil(n / columns)
    while rows * columns < n:
        columns += 1
    return columns, rows


def treemap(scene: Scene, x: float, y: float, w: float, h: float, title: str,
            groups: list[tuple[str, list[tuple[bool, str, str | None]]]]) -> None:
    """groups: (label, [(done, hover title, cell label or None)]). Done cells first, like AnyPS5."""
    scene.add(Text(x, y - 10, title, size=16))
    sizes = [len(cells) for _, cells in groups]
    for (label, cells), (gx, gy, gw, gh) in zip(groups, squarify(sizes, x, y, w, h)):
        cells = sorted(cells, key=lambda c: not c[0])
        columns, rows = cell_grid(len(cells), gw, gh)
        ch = gh / rows
        for i, (done, hover, cell_label) in enumerate(cells):
            row, col = divmod(i, columns)
            # The last row stretches across the block, so every block is filled edge to edge.
            in_row = columns if row < rows - 1 else len(cells) - columns * (rows - 1)
            cw = gw / in_row
            cx, cy = gx + col * cw, gy + row * ch
            scene.add(Rect(cx, cy, cw, ch, DONE if done else TODO, title=hover))
            if cell_label and cw > 30 and ch > 30:
                # Bottom-left, so the block's own label at the top-left never covers it.
                scene.add(Text(cx + 4, cy + ch - 6, cell_label, size=11, outline=True))
        scene.add(Rect(gx, gy, gw, gh, "none", stroke=BACKGROUND, stroke_width=2,
                       title=f"{label}: {sum(c[0] for c in cells)}/{len(cells)}"))
        if gw > 34 and gh > 16:
            scene.add(Text(gx + 4, gy + 14, label, size=12, outline=True))


def event_groups(per_frame: dict[str, int]) -> list:
    path = ROOT / "training" / "registry" / "heldout_events.json"
    events = json.loads(path.read_text(encoding="utf-8"))["events"] if path.exists() else {}
    groups: dict[str, list] = {}
    for stem, count in sorted(per_frame.items()):
        event = events.get(stem[:11], stem[:11])
        label = event[4:].upper() if event[:4].isdigit() else event
        groups.setdefault(label, []).append((count >= 6, f"{stem}: {count} robots", None))
    # Largest blocks first: squarify lays them out best that way, as AnyPS5's libc/libkernel.
    return sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))


def feature_groups() -> list:
    groups: dict[str, list] = {}
    for category, label, description, added in FEATURES:
        hover = f"{description}: " + (f"built {added}" if added else "not built yet")
        groups.setdefault(category, []).append((added is not None, hover, label))
    return sorted(groups.items(), key=lambda kv: -len(kv[1]))


# ----------------------------------------------------------------------------- history

def history_series(models: list[dict], start: date, end: date) -> tuple[list, list, list]:
    """Daily values: the then-current model's held-out score, and the share of features built."""
    trained = sorted((m for m in models if m["heldout"] and not m.get("export_of")),
                     key=lambda m: m["trained_on"])
    days, model_values, feature_values = [], [], []
    day = start
    while day <= end:
        stamp = day.isoformat()
        current = [m for m in trained if m["trained_on"] <= stamp]
        days.append(day)
        model_values.append(current[-1]["heldout"]["full_share"] if current else None)
        feature_values.append(built(stamp) / len(FEATURES))
        day += timedelta(days=1)
    return days, model_values, feature_values


def chart(scene: Scene, x: float, y: float, w: float, h: float, days: list,
          series: list[tuple[str, str, list]]) -> None:
    for pct in range(0, 101, 10):
        gy = y + h * (1 - pct / 100)
        scene.add(Line([(x, gy), (x + w, gy)], GRID), Text(x - 8, gy + 4, f"{pct}%", size=12, fill=TEXT, anchor="end"))
    step = w / max(1, len(days) - 1)
    for i, day in enumerate(days):
        gx = x + i * step
        scene.add(Line([(gx, y), (gx, y + h)], GRID, 0.5))
        if True:
            scene.add(Text(gx - 2, y + h + 12, day.strftime("%m-%d"), size=11, fill=MUTED, anchor="end", rotate=-60))
    for name, colour_hex, values in series:
        points = [(x + i * step, y + h * (1 - v)) for i, v in enumerate(values) if v is not None]
        if not points:
            continue
        scene.add(Area(points + [(points[-1][0], y + h), (points[0][0], y + h)], colour_hex, 0.12))
        scene.add(Line(points, colour_hex, 2))
        for px, py in points:
            scene.add(Dot(px, py, 3, colour_hex))
        scene.add(Text(points[-1][0] + 10, points[-1][1] + 4, f"{name} {values[-1]:.1%}", size=13,
                       fill=colour_hex, bold=True))


def _titles(models: list[dict]) -> tuple[str, str]:
    current = next(m for m in models if m["status"] == "current")
    h = current["heldout"]
    n = built()
    return (f"Held-out frames: {current['id']} finds all six in {h['full_share']:.1%} ({h['full_frames']}/{h['frames']})",
            f"Strategy features: {n / len(FEATURES):.1%} ({n}/{len(FEATURES)})")


def _history(models: list[dict], today: date | None):
    first = min(date.fromisoformat(m["trained_on"]) for m in models)
    return history_series(models, first, today or date.today())


def _treemaps(scene: Scene, models: list[dict], top: float = 34) -> None:
    current = next(m for m in models if m["status"] == "current")
    held, features = _titles(models)
    treemap(scene, 6, top, 528, 300, held, event_groups(current["heldout"]["per_frame"]))
    treemap(scene, 564, top, 528, 300, features, feature_groups())


def _chart(scene: Scene, models: list[dict], top: float, today: date | None) -> None:
    days, model_values, feature_values = _history(models, today)
    chart(scene, 60, top, 900, 360, days,
          [("Model", MODEL_COLOUR, model_values), ("Features", FEATURE_COLOUR, feature_values)])


def build_scene(models: list[dict], today: date | None = None) -> Scene:
    """Treemaps above the history chart: the README and Discord picture."""
    scene = Scene(WIDTH, HEIGHT)
    scene.add(Rect(0, 0, WIDTH, HEIGHT, BACKGROUND, stroke=BACKGROUND, stroke_width=0))
    _treemaps(scene, models)
    _chart(scene, models, 370, today)
    return scene


def treemap_scene(models: list[dict]) -> Scene:
    """The two treemaps alone: the site's progress page, like AnyPS5's progress.svg."""
    scene = Scene(WIDTH, 344)
    scene.add(Rect(0, 0, WIDTH, 344, BACKGROUND, stroke=BACKGROUND, stroke_width=0))
    _treemaps(scene, models)
    return scene


def history_scene(models: list[dict], today: date | None = None) -> Scene:
    """The history chart alone: the site's history page."""
    scene = Scene(WIDTH, 440)
    scene.add(Rect(0, 0, WIDTH, 440, BACKGROUND, stroke=BACKGROUND, stroke_width=0))
    _chart(scene, models, 20, today)
    return scene


# ----------------------------------------------------------------------------- renderers

def to_svg(scene: Scene) -> str:
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {scene.width} {scene.height}" '
           f'width="{scene.width}" height="{scene.height}" font-family="Segoe UI,Helvetica,Arial,sans-serif">']
    for s in scene.shapes:
        if isinstance(s, Rect):
            tip = f"<title>{escape(s.title)}</title>" if s.title else ""
            out.append(f'<rect x="{s.x:.2f}" y="{s.y:.2f}" width="{s.w:.2f}" height="{s.h:.2f}" fill="{s.fill}" '
                       f'stroke="{s.stroke}" stroke-width="{s.stroke_width}">{tip}</rect>')
        elif isinstance(s, Text):
            anchor = {"start": "start", "middle": "middle", "end": "end"}[s.anchor]
            extra = ' paint-order="stroke" stroke="#0d1117" stroke-width="3"' if s.outline else ""
            rotate = f' transform="rotate({s.rotate} {s.x:.1f} {s.y:.1f})"' if s.rotate else ""
            weight = ' font-weight="600"' if s.bold else ""
            out.append(f'<text x="{s.x:.1f}" y="{s.y:.1f}" font-size="{s.size}" fill="{s.fill}" '
                       f'text-anchor="{anchor}"{weight}{extra}{rotate}>{escape(s.text)}</text>')
        elif isinstance(s, Line):
            pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in s.points)
            out.append(f'<polyline points="{pts}" fill="none" stroke="{s.stroke}" stroke-width="{s.width}"/>')
        elif isinstance(s, Area):
            pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in s.points)
            out.append(f'<polygon points="{pts}" fill="{s.fill}" fill-opacity="{s.opacity}"/>')
        elif isinstance(s, Dot):
            out.append(f'<circle cx="{s.x:.1f}" cy="{s.y:.1f}" r="{s.r}" fill="{s.fill}"/>')
    out.append("</svg>")
    return "\n".join(out)


def _font(size: int, bold: bool = False):
    from PIL import ImageFont
    names = ("seguisb.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf") if bold else ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf")
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def to_png(scene: Scene, path, scale: int = 2) -> None:
    """Drawn at twice the size and downsampled, so edges and text are smooth."""
    from PIL import Image, ImageColor, ImageDraw

    S = scale
    image = Image.new("RGBA", (scene.width * S, scene.height * S), BACKGROUND)
    draw = ImageDraw.Draw(image)
    for s in scene.shapes:
        if isinstance(s, Rect):
            box = [s.x * S, s.y * S, (s.x + s.w) * S - 1, (s.y + s.h) * S - 1]
            draw.rectangle(box, fill=None if s.fill == "none" else s.fill,
                           outline=s.stroke if s.stroke_width else None, width=max(1, round(s.stroke_width * S)))
        elif isinstance(s, Area):
            overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
            r, g, b = ImageColor.getrgb(s.fill)
            ImageDraw.Draw(overlay).polygon([(x * S, y * S) for x, y in s.points], fill=(r, g, b, int(255 * s.opacity)))
            image = Image.alpha_composite(image, overlay)
            draw = ImageDraw.Draw(image)
        elif isinstance(s, Line):
            draw.line([(x * S, y * S) for x, y in s.points], fill=s.stroke, width=max(1, round(s.width * S)))
        elif isinstance(s, Dot):
            draw.ellipse([(s.x - s.r) * S, (s.y - s.r) * S, (s.x + s.r) * S, (s.y + s.r) * S], fill=s.fill)
        elif isinstance(s, Text):
            font = _font(s.size * S, s.bold)
            anchor = {"start": "ls", "middle": "ms", "end": "rs"}[s.anchor]
            stroke = {"stroke_width": 3, "stroke_fill": BACKGROUND} if s.outline else {}
            if s.rotate:
                import math
                width = int(draw.textlength(s.text, font=font)) + 8
                tall = s.size * S * 2
                anchor_point = (width - 4, int(tall * 0.7))   # where the text's end sits in the tile
                tile = Image.new("RGBA", (width, tall), (0, 0, 0, 0))
                ImageDraw.Draw(tile).text(anchor_point, s.text, fill=s.fill, font=font, anchor="rs")
                theta = math.radians(-s.rotate)                # PIL turns counter-clockwise
                rotated = tile.rotate(-s.rotate, expand=True, resample=Image.BICUBIC)
                # Where the anchor point went: rotate it about the centre, as PIL did the pixels.
                dx, dy = anchor_point[0] - width / 2, anchor_point[1] - tall / 2
                ax = rotated.width / 2 + dx * math.cos(theta) + dy * math.sin(theta)
                ay = rotated.height / 2 - dx * math.sin(theta) + dy * math.cos(theta)
                image.alpha_composite(rotated, (int(round(s.x * S - ax)), int(round(s.y * S - ay))))
                draw = ImageDraw.Draw(image)
            else:
                draw.text((s.x * S, s.y * S), s.text, fill=s.fill, font=font, anchor=anchor, **stroke)
    image = image.resize((scene.width, scene.height), Image.LANCZOS).convert("RGB")
    image.save(path)


# ----------------------------------------------------------------------------- badges

def badge(label: str, value: str, colour_hex: str) -> str:
    """A shields.io-style flat badge, so the README needs no external badge service for these."""
    left, right = 6 * len(label) + 12, 6 * len(value) + 14
    width = left + right
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="20" role="img" '
            f'aria-label="{escape(label)}: {escape(value)}"><title>{escape(label)}: {escape(value)}</title>'
            f'<linearGradient id="s" x2="0" y2="100%"><stop offset="0" stop-color="#bbb" stop-opacity=".1"/>'
            f'<stop offset="1" stop-opacity=".1"/></linearGradient><clipPath id="r"><rect width="{width}" '
            f'height="20" rx="3" fill="#fff"/></clipPath><g clip-path="url(#r)"><rect width="{left}" '
            f'height="20" fill="#555"/><rect x="{left}" width="{right}" height="20" fill="{colour_hex}"/>'
            f'<rect width="{width}" height="20" fill="url(#s)"/></g><g fill="#fff" text-anchor="middle" '
            f'font-family="Verdana,Geneva,DejaVu Sans,sans-serif" font-size="11">'
            f'<text x="{left / 2}" y="14">{escape(label)}</text>'
            f'<text x="{left + right / 2}" y="14">{escape(value)}</text></g></svg>')


def score_colour(share: float) -> str:
    return "#4c1" if share >= 0.9 else "#97ca00" if share >= 0.6 else "#dfb317" if share >= 0.4 else "#e05d44"


# ----------------------------------------------------------------------------- main

def build() -> list:
    OUT.mkdir(parents=True, exist_ok=True)
    models = models_with_scores()
    current = next(m for m in models if m["status"] == "current")
    if not current["heldout"]:
        raise SystemExit(f"{current['id']} has no held-out score yet: run python -m tools.models evaluate")
    scene = build_scene(models)
    written = []

    def write(name: str, text: str) -> None:
        (OUT / name).write_text(text + "\n", encoding="utf-8")
        written.append(OUT / name)

    write("status.svg", to_svg(scene))
    try:
        to_png(scene, OUT / "status.png")
        written.append(OUT / "status.png")
    except ImportError:
        # Pillow is only needed for Discord's PNG; the site and README use the SVG.
        print("Pillow not installed: skipped status.png")
    share = current["heldout"]["full_share"]
    features = built() / len(FEATURES)
    write("badge-heldout.svg", badge("held-out*", f"{share:.1%}", score_colour(share)))
    write("badge-features.svg", badge("features", f"{features:.1%}", score_colour(features)))
    write("badge-model.svg", badge("model", f"{current['id']} · {current['architecture']}", "#2f81f7"))
    pipeline = (current.get("pipeline") or {}).get("all_six_tracked_share")
    if pipeline is not None:
        write("badge-tracking.svg", badge("tracked", f"{pipeline:.0%}", score_colour(pipeline)))
    # Superseded by status.svg/png; removed so nothing links to a stale picture.
    for old in ("model-map.svg", "model-map.png", "history.svg", "history.png", "badge-models.svg"):
        (OUT / old).unlink(missing_ok=True)
    return written


def main() -> int:
    for path in build():
        print(path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
