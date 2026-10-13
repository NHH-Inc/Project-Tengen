"""Status graphics for the README and Discord, generated from the model registry.

    python -m tools.status

Writes docs/status/:
  model-map.svg / .png   one square per held-out frame, grouped by venue, coloured by how many
                         robots the current model finds there (hover a square for its frame)
  history.svg / .png     every model's held-out score, oldest to newest
  badge-*.svg            small shields for the README

SVG for GitHub, where hover titles work; PNG for Discord, which does not render SVG. Everything
is derived from training/registry, so the pictures cannot drift from the numbers.
"""

from __future__ import annotations

from collections import OrderedDict
from html import escape
from pathlib import Path

from tools.models import ROOT, load, models_with_scores

OUT = ROOT / "docs" / "status"
BACKGROUND = "#0d1117"
TEXT = "#e6edf3"
MUTED = "#8b949e"
#: Robots found in a frame -> colour. Six play; a stacked broadcast can show twelve.
BANDS = [(6, "#2ea043", "all six or more"), (3, "#d29922", "three to five"),
         (1, "#db6d28", "one or two"), (0, "#da3633", "none")]


def colour(count: int) -> str:
    return next(c for floor, c, _ in BANDS if count >= floor)


def segment_of(stem: str) -> str:
    return "_".join(stem.split("_")[:-1])


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
    return "#2ea043" if share >= 0.8 else "#97ca00" if share >= 0.6 else "#d29922" if share >= 0.4 else "#da3633"


# ----------------------------------------------------------------------------- the map

def map_layout(per_frame: dict[str, int]):
    """Venues in order of how well the model does on them, frames within each in time order."""
    groups: dict[str, list[tuple[str, int]]] = {}
    for stem in sorted(per_frame):
        groups.setdefault(segment_of(stem), []).append((stem, per_frame[stem]))
    order = sorted(groups.items(), key=lambda kv: -sum(min(n, 6) for _, n in kv[1]) / len(kv[1]))
    return OrderedDict(order)


CELL, GAP, VENUE_GAP, TOP, MAP_WIDTH = 14, 2, 8, 36, 1000


def map_cells(per_frame: dict[str, int], width: int = MAP_WIDTH):
    """(x, y, count, stem, venue) for every frame: venues flow left to right as small blocks,
    wrapping like words, best venues first, so the map reads from green to red."""
    cells, x, y = [], 4, TOP
    for venue, frames in map_layout(per_frame).items():
        block = len(frames) * (CELL + GAP)
        if x + block > width - 4 and x > 4:
            x, y = 4, y + CELL + GAP + 6
        for index, (stem, count) in enumerate(frames):
            cells.append((x + index * (CELL + GAP), y, count, stem, venue))
        x += block + VENUE_GAP
    return cells, y + CELL + 40


def _legend(per_frame):
    return [(hex_colour, f"{label}: {sum(1 for c in per_frame.values() if colour(c) == hex_colour)}")
            for _, hex_colour, label in BANDS]


def model_map_svg(model: dict, width: int = MAP_WIDTH) -> str:
    h = model["heldout"]
    cells, height = map_cells(h["per_frame"], width)
    venues = map_layout(h["per_frame"])
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" '
             f'height="{height}" font-family="Segoe UI,Helvetica,Arial,sans-serif">',
             f'<rect width="{width}" height="{height}" fill="{BACKGROUND}"/>',
             f'<text x="4" y="22" font-size="16" fill="{TEXT}">{escape(model["name"])}: all six robots found in '
             f'{h["full_share"]:.1%} of {h["frames"]} held-out frames, across {len(venues)} broadcasts</text>']
    current_venue = None
    for x, y, count, stem, venue in cells:
        if venue != current_venue:
            if current_venue is not None:
                parts.append("</g>")
            frames = venues[venue]
            parts.append(f'<g><title>{escape(venue)}: mean {sum(n for _, n in frames) / len(frames):.1f} '
                         f'robots over {len(frames)} frames</title>')
            current_venue = venue
        parts.append(f'<rect x="{x}" y="{y}" width="{CELL}" height="{CELL}" rx="2" fill="{colour(count)}">'
                     f'<title>{escape(stem)}: {count} robots</title></rect>')
    parts.append("</g>")
    x = 4
    for hex_colour, text in _legend(h["per_frame"]):
        parts.append(f'<rect x="{x}" y="{height - 22}" width="11" height="11" rx="2" fill="{hex_colour}"/>')
        parts.append(f'<text x="{x + 16}" y="{height - 12}" font-size="12" fill="{MUTED}">{text}</text>')
        x += 30 + 7 * len(text)
    parts.append(f'<text x="{width - 4}" y="{height - 12}" font-size="12" fill="{MUTED}" text-anchor="end">'
                 f'one square per frame, one block per broadcast, best first</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def model_map_png(model: dict, path: Path, width: int = MAP_WIDTH) -> None:
    from PIL import Image, ImageDraw

    h = model["heldout"]
    cells, height = map_cells(h["per_frame"], width)
    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    font, small = _font(16), _font(12)
    draw.text((4, 8), f"{model['name']}: all six robots found in {h['full_share']:.1%} of {h['frames']} "
                      f"held-out frames", fill=TEXT, font=font)
    for x, y, count, _, _ in cells:
        draw.rounded_rectangle([x, y, x + CELL - 1, y + CELL - 1], radius=2, fill=colour(count))
    x = 4
    for hex_colour, text in _legend(h["per_frame"]):
        draw.rounded_rectangle([x, height - 22, x + 11, height - 11], radius=2, fill=hex_colour)
        draw.text((x + 16, height - 24), text, fill=MUTED, font=small)
        x += 30 + int(draw.textlength(text, font=small))
    image.save(path)


def _font(size: int):
    from PIL import ImageFont
    for name in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


# ----------------------------------------------------------------------------- history

def history_svg(models: list[dict], width: int = 1000, height: int = 300) -> str:
    scored = [m for m in models if m["heldout"]]
    left, right, top, bottom = 60, 20, 40, 60
    plot_w, plot_h = width - left - right, height - top - bottom
    slot = plot_w / max(1, len(scored))
    bar = min(90, slot * 0.55)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" '
             f'height="{height}" font-family="Segoe UI,Helvetica,Arial,sans-serif">',
             f'<rect width="{width}" height="{height}" fill="{BACKGROUND}"/>',
             f'<text x="4" y="22" font-size="16" fill="{TEXT}">Held-out frames where the model finds all six '
             f'robots, by version</text>']
    for pct in (0, 25, 50, 75, 100):
        y = top + plot_h * (1 - pct / 100)
        parts.append(f'<line x1="{left}" x2="{width - right}" y1="{y}" y2="{y}" stroke="#30363d"/>')
        parts.append(f'<text x="{left - 8}" y="{y + 4}" font-size="12" fill="{MUTED}" text-anchor="end">{pct}%</text>')
    for index, m in enumerate(scored):
        share = m["heldout"]["full_share"]
        x = left + slot * index + (slot - bar) / 2
        y = top + plot_h * (1 - share)
        fill = "#2f81f7" if m["status"] == "current" else "#6e7681"
        parts.append(f'<g><title>{escape(m["name"])}: {share:.1%} (mean {m["heldout"]["mean_robots"]:.2f} robots, '
                     f'{m["heldout"]["blind_frames"]} blind frames)</title>'
                     f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar:.1f}" height="{top + plot_h - y:.1f}" rx="3" fill="{fill}"/></g>')
        parts.append(f'<text x="{x + bar / 2:.1f}" y="{y - 6:.1f}" font-size="13" fill="{TEXT}" '
                     f'text-anchor="middle">{share:.1%}</text>')
        parts.append(f'<text x="{x + bar / 2:.1f}" y="{top + plot_h + 18}" font-size="13" fill="{TEXT}" '
                     f'text-anchor="middle">{escape(m["id"])}</text>')
        parts.append(f'<text x="{x + bar / 2:.1f}" y="{top + plot_h + 34}" font-size="11" fill="{MUTED}" '
                     f'text-anchor="middle">{escape(m["trained_on"])}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def history_png(models: list[dict], path: Path, width: int = 1000, height: int = 300) -> None:
    from PIL import Image, ImageDraw

    scored = [m for m in models if m["heldout"]]
    left, right, top, bottom = 60, 20, 40, 60
    plot_w, plot_h = width - left - right, height - top - bottom
    slot = plot_w / max(1, len(scored))
    bar = min(90, slot * 0.55)
    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    font, small = _font(15), _font(12)
    draw.text((4, 10), "Held-out frames where the model finds all six robots, by version", fill=TEXT, font=font)
    for pct in (0, 25, 50, 75, 100):
        y = top + plot_h * (1 - pct / 100)
        draw.line([left, y, width - right, y], fill="#30363d")
        draw.text((left - 8, y - 7), f"{pct}%", fill=MUTED, font=small, anchor="ra")
    for index, m in enumerate(scored):
        share = m["heldout"]["full_share"]
        x = left + slot * index + (slot - bar) / 2
        y = top + plot_h * (1 - share)
        draw.rounded_rectangle([x, y, x + bar, top + plot_h], radius=3,
                               fill="#2f81f7" if m["status"] == "current" else "#6e7681")
        draw.text((x + bar / 2, y - 6), f"{share:.1%}", fill=TEXT, font=small, anchor="ms")
        draw.text((x + bar / 2, top + plot_h + 18), m["id"], fill=TEXT, font=small, anchor="ms")
        draw.text((x + bar / 2, top + plot_h + 34), m["trained_on"], fill=MUTED, font=small, anchor="ms")
    image.save(path)


# ----------------------------------------------------------------------------- main

def current(models: list[dict]) -> dict:
    return next(m for m in models if m["status"] == "current")


def build() -> list[Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    models = models_with_scores()
    model = current(models)
    if not model["heldout"]:
        raise SystemExit(f"{model['id']} has no held-out score yet: run python -m tools.models evaluate")
    written = []

    def write(name: str, text: str) -> None:
        path = OUT / name
        path.write_text(text + "\n", encoding="utf-8")
        written.append(path)

    write("model-map.svg", model_map_svg(model))
    write("history.svg", history_svg(models))
    model_map_png(model, OUT / "model-map.png")
    history_png(models, OUT / "history.png")
    written += [OUT / "model-map.png", OUT / "history.png"]

    share = model["heldout"]["full_share"]
    write("badge-model.svg", badge("model", f"{model['id']} · {model['architecture']}", "#2f81f7"))
    write("badge-heldout.svg", badge("held-out: all 6 found", f"{share:.1%}", score_colour(share)))
    pipeline = (model.get("pipeline") or {}).get("all_six_tracked_share")
    if pipeline is not None:
        write("badge-tracking.svg", badge("match: all 6 tracked", f"{pipeline:.0%}", score_colour(pipeline)))
    write("badge-models.svg", badge("models trained", str(len({m['weights'] for m in models})), "#8957e5"))
    return written


def main() -> int:
    for path in build():
        print(path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
