"""Interactive FRC robot box reviewer.

The tool is intentionally standalone: it uses Tkinter for the UI and Pillow for
loading and scaling common image formats. Each image must have annotation data
from either a matching JSON sidecar (normally ``image.json``) or a folder-level
``annotations.json`` file whose ``images`` list names that image.

Boxes are stored using normalized top-left coordinates (``x``, ``y``, ``w``,
``h``). New or edited boxes also carry a ``color`` field so red/blue labels
survive a restart.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import tkinter as tk
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Iterable

try:
    from PIL import Image, ImageTk
except ImportError as exc:  # pragma: no cover - exercised when launching without Pillow
    raise SystemExit(
        "Pillow is required. Install it with: python -m pip install -r requirements-labeler.txt"
    ) from exc


IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SKIP_DIRECTORY_NAMES = {".git", ".codex-doc-review", "previews", "__pycache__"}
RED = "red"
BLUE = "blue"
NEUTRAL = "neutral"
VALID_COLORS = {RED, BLUE, NEUTRAL}
HANDLE_HALF_SIZE = 6


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def natural_key(path: Path) -> tuple[tuple[tuple[int, Any], ...], str]:
    # Tokenize only the stem so frame.jpg sorts before frame2.jpg, while frame2
    # still sorts before frame10. Type tags prevent int/string comparisons.
    tokens = tuple(
        (1, int(part)) if part.isdigit() else (0, part.lower())
        for part in re.split(r"(\d+)", path.stem)
        if part
    )
    return tokens, path.suffix.lower()


def atomic_write_text(path: Path, text: str) -> None:
    """Write a file through a sibling temporary file so autosave is recoverable."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def numeric(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def clamp_box(x: float, y: float, w: float, h: float) -> tuple[float, float, float, float] | None:
    """Clamp a normalized box to the image and reject empty/invalid boxes."""

    x = max(0.0, min(1.0, x))
    y = max(0.0, min(1.0, y))
    w = max(0.0, min(1.0 - x, w))
    h = max(0.0, min(1.0 - y, h))
    if w <= 0.001 or h <= 0.001:
        return None
    return x, y, w, h


def color_from_mapping(mapping: dict[str, Any]) -> str:
    values = [
        mapping.get("color"),
        mapping.get("alliance"),
        mapping.get("side"),
        mapping.get("team_color"),
        mapping.get("teamColor"),
    ]
    if isinstance(mapping.get("team"), str):
        values.append(mapping["team"])
    text = " ".join(str(value).lower() for value in values if value is not None)
    if "red" in text:
        return RED
    if "blue" in text:
        return BLUE
    return NEUTRAL


@dataclass
class Box:
    x: float
    y: float
    w: float
    h: float
    color: str = NEUTRAL
    confidence: float | None = None
    original: dict[str, Any] = field(default_factory=dict)
    image_width: int | None = field(default=None, repr=False)
    image_height: int | None = field(default=None, repr=False)

    @classmethod
    def from_mapping(cls, value: Any, image_width: int, image_height: int) -> Box | None:
        if not isinstance(value, dict) or image_width <= 0 or image_height <= 0:
            return None

        coordinates: tuple[float, float, float, float] | None = None
        normalized_bbox = value.get("bbox_normalized")
        if isinstance(normalized_bbox, dict):
            raw = [
                numeric(normalized_bbox.get("x")),
                numeric(normalized_bbox.get("y")),
                numeric(normalized_bbox.get("width", normalized_bbox.get("w"))),
                numeric(normalized_bbox.get("height", normalized_bbox.get("h"))),
            ]
            if all(item is not None for item in raw):
                coordinates = tuple(float(item) for item in raw)  # type: ignore[arg-type]
        if coordinates is None and isinstance(value.get("bbox"), (list, tuple)) and len(value["bbox"]) >= 4:
            raw = [numeric(item) for item in value["bbox"][:4]]
            if all(item is not None for item in raw):
                bx, by, bw, bh = (float(item) for item in raw)  # type: ignore[arg-type]
                if max(abs(bx), abs(by), abs(bw), abs(bh)) > 1.0:
                    coordinates = (bx / image_width, by / image_height, bw / image_width, bh / image_height)
                else:
                    coordinates = (bx, by, bw, bh)

        if coordinates is None and all(key in value for key in ("x", "y")):
            width_value = value.get("w", value.get("width"))
            height_value = value.get("h", value.get("height"))
            raw = [numeric(value.get("x")), numeric(value.get("y")), numeric(width_value), numeric(height_value)]
            if all(item is not None for item in raw):
                bx, by, bw, bh = (float(item) for item in raw)  # type: ignore[arg-type]
                if max(abs(bx), abs(by), abs(bw), abs(bh)) > 1.0:
                    coordinates = (bx / image_width, by / image_height, bw / image_width, bh / image_height)
                else:
                    coordinates = (bx, by, bw, bh)

        center_keys = (("center_x", "center_y"), ("x_center", "y_center"), ("cx", "cy"))
        if coordinates is None:
            for x_key, y_key in center_keys:
                if x_key in value and y_key in value:
                    width_value = value.get("w", value.get("width"))
                    height_value = value.get("h", value.get("height"))
                    raw = [numeric(value.get(x_key)), numeric(value.get(y_key)), numeric(width_value), numeric(height_value)]
                    if all(item is not None for item in raw):
                        cx, cy, bw, bh = (float(item) for item in raw)  # type: ignore[arg-type]
                        if max(abs(cx), abs(cy), abs(bw), abs(bh)) > 1.0:
                            cx, cy, bw, bh = cx / image_width, cy / image_height, bw / image_width, bh / image_height
                        coordinates = (cx - bw / 2.0, cy - bh / 2.0, bw, bh)
                        break

        if coordinates is None and all(key in value for key in ("x1", "y1", "x2", "y2")):
            raw = [numeric(value.get(key)) for key in ("x1", "y1", "x2", "y2")]
            if all(item is not None for item in raw):
                x1, y1, x2, y2 = (float(item) for item in raw)  # type: ignore[arg-type]
                if max(abs(x1), abs(y1), abs(x2), abs(y2)) > 1.0:
                    x1, x2 = x1 / image_width, x2 / image_width
                    y1, y2 = y1 / image_height, y2 / image_height
                coordinates = (x1, y1, x2 - x1, y2 - y1)

        if coordinates is None and all(key in value for key in ("left", "top", "right", "bottom")):
            raw = [numeric(value.get(key)) for key in ("left", "top", "right", "bottom")]
            if all(item is not None for item in raw):
                left, top, right, bottom = (float(item) for item in raw)  # type: ignore[arg-type]
                if max(abs(left), abs(top), abs(right), abs(bottom)) > 1.0:
                    left, right = left / image_width, right / image_width
                    top, bottom = top / image_height, bottom / image_height
                coordinates = (left, top, right - left, bottom - top)

        if coordinates is None:
            return None
        normalized = clamp_box(*coordinates)
        if normalized is None:
            return None
        confidence = numeric(value.get("confidence", value.get("score")))
        return cls(
            *normalized,
            color=color_from_mapping(value),
            confidence=confidence,
            original=dict(value),
            image_width=image_width,
            image_height=image_height,
        )

    def as_mapping(self) -> dict[str, Any]:
        """Return a normalized project-style box while retaining input metadata."""

        result = dict(self.original)
        has_normalized_bbox = isinstance(result.get("bbox_normalized"), dict)
        if not has_normalized_bbox:
            result.update({
                "x": round(self.x, 6),
                "y": round(self.y, 6),
                "w": round(self.w, 6),
                "h": round(self.h, 6),
            })
        else:
            normalized_bbox = dict(result["bbox_normalized"])
            normalized_bbox.update({
                "x": round(self.x, 6),
                "y": round(self.y, 6),
                "width": round(self.w, 6),
                "height": round(self.h, 6),
            })
            result["bbox_normalized"] = normalized_bbox
        if isinstance(result.get("bbox"), (list, tuple)) and len(result["bbox"]) >= 4:
            values = [numeric(item) for item in result["bbox"][:4]]
            was_pixel_format = any(item is not None and abs(item) > 1.0 for item in values)
            if was_pixel_format and self.image_width and self.image_height:
                result["bbox"] = [
                    round(self.x * self.image_width, 6),
                    round(self.y * self.image_height, 6),
                    round(self.w * self.image_width, 6),
                    round(self.h * self.image_height, 6),
                ]
            else:
                result["bbox"] = [round(self.x, 6), round(self.y, 6), round(self.w, 6), round(self.h, 6)]
        if isinstance(result.get("bbox_xyxy"), (list, tuple)) and len(result["bbox_xyxy"]) >= 4:
            if self.image_width and self.image_height:
                result["bbox_xyxy"] = [
                    round(self.x * self.image_width),
                    round(self.y * self.image_height),
                    round((self.x + self.w) * self.image_width),
                    round((self.y + self.h) * self.image_height),
                ]
        if "alliance" in result:
            result["alliance"] = self.color if self.color in (RED, BLUE) else None
        else:
            result["color"] = self.color
        if "label" not in result:
            result.setdefault("class_name", "robot")
        if self.confidence is not None:
            result["confidence"] = round(max(0.0, min(1.0, self.confidence)), 6)
        return result


class AnnotationRepository:
    """Find image/JSON pairs and persist all edits back to that JSON file."""

    def __init__(self, root: Path):
        self.root = root.expanduser().resolve()
        if not self.root.is_dir():
            raise NotADirectoryError(f"Image folder does not exist: {self.root}")
        self.sidecar_state: dict[Path, tuple[Path, Any]] = {}
        self.annotation_manifest_path: Path | None = None
        self.annotation_manifest: dict[str, Any] | None = None
        self.annotation_rows_by_image: dict[Path, dict[str, Any]] = {}
        self.records: list[Path] = []
        self._load_annotation_manifest()
        self._discover_images()

    def _load_annotation_manifest(self) -> None:
        """Load the export format used by ``robot_image_exports`` folders."""

        candidate = self.root / "annotations.json"
        if not candidate.is_file():
            return
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict) or not isinstance(payload.get("images"), list):
            return
        rows_by_image: dict[Path, dict[str, Any]] = {}
        for row in payload["images"]:
            if not isinstance(row, dict):
                continue
            image_name = row.get("image", row.get("image_path", row.get("file_name")))
            if not isinstance(image_name, str):
                continue
            image_path = (self.root / image_name).resolve()
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                rows_by_image[image_path] = row
        if rows_by_image:
            self.annotation_manifest_path = candidate
            self.annotation_manifest = payload
            self.annotation_rows_by_image = rows_by_image

    def _discover_images(self) -> None:
        if self.annotation_rows_by_image:
            self.records = sorted(self.annotation_rows_by_image, key=natural_key)
            return
        paths: list[Path] = []
        for path in self.root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            try:
                relative_parts = path.relative_to(self.root).parts[:-1]
            except ValueError:
                relative_parts = ()
            if any(part.lower() in SKIP_DIRECTORY_NAMES for part in relative_parts):
                continue
            image_path = path.resolve()
            state = self._read_sidecar(image_path)
            if state is None:
                continue
            self.sidecar_state[image_path] = state
            paths.append(image_path)
        self.records = sorted(set(paths), key=natural_key)

    def _sidecar_candidates(self, image_path: Path) -> Iterable[Path]:
        yield image_path.with_suffix(".json")
        # Kept for users who edited a folder with the first version of the app.
        yield image_path.with_suffix(".boxes.json")

    @staticmethod
    def _extract_sidecar_boxes(payload: Any, image_path: Path, root: Path) -> list[Any] | None:
        if isinstance(payload, list):
            return payload
        if not isinstance(payload, dict):
            return None
        for key in ("boxes", "objects", "detections"):
            if isinstance(payload.get(key), list):
                return payload[key]
        for key in (str(image_path), image_path.name, image_path.stem):
            value = payload.get(key)
            if isinstance(value, list):
                return value
            if isinstance(value, dict):
                for nested_key in ("boxes", "objects", "detections"):
                    if isinstance(value.get(nested_key), list):
                        return value[nested_key]
        try:
            relative = str(image_path.relative_to(root)).replace("\\", "/")
        except ValueError:
            relative = image_path.name
        value = payload.get(relative)
        if isinstance(value, list):
            return value
        if isinstance(value, dict) and isinstance(value.get("boxes"), list):
            return value["boxes"]
        return None

    @staticmethod
    def _parse_box_list(values: Iterable[Any], image_width: int, image_height: int) -> list[Box]:
        boxes: list[Box] = []
        for value in values:
            box = Box.from_mapping(value, image_width, image_height)
            if box is not None:
                boxes.append(box)
        return boxes

    def _read_sidecar(self, image_path: Path) -> tuple[Path, Any] | None:
        for sidecar in self._sidecar_candidates(image_path):
            if not sidecar.is_file():
                continue
            try:
                payload = json.loads(sidecar.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            if self._extract_sidecar_boxes(payload, image_path, self.root) is not None:
                return sidecar, payload
        return None

    def load_boxes(self, image_path: Path, image_width: int, image_height: int) -> list[Box]:
        image_path = image_path.resolve()
        manifest_row = self.annotation_rows_by_image.get(image_path)
        if manifest_row is not None:
            values = manifest_row.get("boxes")
            if not isinstance(values, list):
                raise ValueError(f"{image_path.name}'s annotations entry must contain a boxes list")
            return self._parse_box_list(values, image_width, image_height)
        state = self.sidecar_state.get(image_path) or self._read_sidecar(image_path)
        if state is None:
            raise FileNotFoundError(f"A matching JSON annotation file is required for {image_path.name}")
        self.sidecar_state[image_path] = state
        _sidecar, payload = state
        values = self._extract_sidecar_boxes(payload, image_path, self.root)
        assert values is not None
        return self._parse_box_list(values, image_width, image_height)

    @staticmethod
    def _replace_sidecar_boxes(payload: Any, image_path: Path, root: Path, boxes: list[dict[str, Any]]) -> Any:
        if isinstance(payload, list):
            return boxes
        if not isinstance(payload, dict):
            raise ValueError(f"{image_path.name}'s JSON must contain a box list")
        for key in ("boxes", "objects", "detections"):
            if isinstance(payload.get(key), list):
                payload[key] = boxes
                payload["updated_at"] = utc_now()
                return payload
        try:
            relative = str(image_path.relative_to(root)).replace("\\", "/")
        except ValueError:
            relative = image_path.name
        for key in (str(image_path), image_path.name, image_path.stem, relative):
            value = payload.get(key)
            if isinstance(value, list):
                payload[key] = boxes
                payload["updated_at"] = utc_now()
                return payload
            if isinstance(value, dict):
                for nested_key in ("boxes", "objects", "detections"):
                    if isinstance(value.get(nested_key), list):
                        value[nested_key] = boxes
                        payload["updated_at"] = utc_now()
                        return payload
        raise ValueError(f"{image_path.name}'s JSON does not contain an editable box list")

    def save_boxes(self, image_path: Path, boxes: list[Box]) -> Path:
        image_path = image_path.resolve()
        serialized = [box.as_mapping() for box in boxes]
        manifest_row = self.annotation_rows_by_image.get(image_path)
        if manifest_row is not None:
            manifest_row["boxes"] = serialized
            assert self.annotation_manifest_path is not None and self.annotation_manifest is not None
            atomic_write_text(
                self.annotation_manifest_path,
                json.dumps(self.annotation_manifest, indent=2, sort_keys=False) + "\n",
            )
            return self.annotation_manifest_path
        state = self.sidecar_state.get(image_path)
        if state is None:
            raise FileNotFoundError(f"A matching JSON annotation file is required for {image_path.name}")
        sidecar, payload = state
        output = self._replace_sidecar_boxes(payload, image_path, self.root, serialized)
        self.sidecar_state[image_path] = (sidecar, output)
        atomic_write_text(sidecar, json.dumps(output, indent=2, sort_keys=True) + "\n")
        return sidecar

    def new_box(self, image_path: Path, x: float, y: float, w: float, h: float, color: str, image_width: int, image_height: int) -> Box:
        """Create a box matching the annotation format used by its image."""

        if image_path.resolve() in self.annotation_rows_by_image:
            return Box(
                x,
                y,
                w,
                h,
                color=color,
                original={
                    "label": "robot",
                    "alliance": color,
                    "bbox_xyxy": [0, 0, 0, 0],
                    "bbox_normalized": {"x": x, "y": y, "width": w, "height": h},
                },
                image_width=image_width,
                image_height=image_height,
            )
        return Box(x, y, w, h, color=color, image_width=image_width, image_height=image_height)


class RobotBoxLabeler(tk.Tk):
    def __init__(self, repository: AnnotationRepository, click_width: float = 0.12, click_height: float = 0.18, midpoint: float = 0.5):
        super().__init__()
        self.repository = repository
        self.records = repository.records
        self.click_width = max(0.01, min(1.0, click_width))
        self.click_height = max(0.01, min(1.0, click_height))
        self.midpoint = max(0.0, min(1.0, midpoint))
        self.index = 0
        self.image: Image.Image | None = None
        self.photo: ImageTk.PhotoImage | None = None
        self.boxes: list[Box] = []
        self.selected_index: int | None = None
        self.drag_start: tuple[float, float] | None = None
        self.mouse_operation: str | None = None
        self.mouse_down_hit: int | None = None
        self.resize_box_index: int | None = None
        self.resize_handle: str | None = None
        self.preview_rect: int | None = None
        self.image_offset = (0.0, 0.0)
        self.display_size = (0, 0)
        self.last_saved_path: Path | None = None

        self.title("FRC Robot Box Labeler")
        self.geometry("1280x820")
        self.minsize(800, 560)
        self.configure(bg="#0f172a")
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.bind("<Left>", lambda _event: self.previous_image())
        self.bind("<Right>", lambda _event: self.next_image())
        self.bind("<Delete>", lambda _event: self.delete_selected())
        self.bind("<BackSpace>", lambda _event: self.delete_selected())
        self.bind("<KeyPress-r>", lambda _event: self.choose_color(RED))
        self.bind("<KeyPress-b>", lambda _event: self.choose_color(BLUE))
        self.bind("<KeyPress-a>", lambda _event: self.choose_color("auto"))
        self.after(50, self._load_current)

    def _build_ui(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Toolbar.TFrame", background="#111827")
        style.configure("Toolbar.TLabel", background="#111827", foreground="#cbd5e1")
        style.configure("Hint.TLabel", background="#0f172a", foreground="#94a3b8")
        style.configure("Title.TLabel", background="#111827", foreground="#f8fafc", font=("Segoe UI", 12, "bold"))

        toolbar = ttk.Frame(self, style="Toolbar.TFrame", padding=(14, 12))
        toolbar.pack(fill="x")
        ttk.Label(toolbar, text="FRC ROBOT BOX LABELER", style="Title.TLabel").pack(side="left", padx=(0, 18))
        ttk.Button(toolbar, text="Open folder…", command=self.open_folder).pack(side="left")
        self.file_var = tk.StringVar(value="No image loaded")
        ttk.Label(toolbar, textvariable=self.file_var, style="Toolbar.TLabel").pack(side="left", padx=14)
        self.save_var = tk.StringVar(value="Ready")
        ttk.Label(toolbar, textvariable=self.save_var, style="Toolbar.TLabel").pack(side="right")

        color_frame = ttk.Frame(self, style="Toolbar.TFrame", padding=(14, 0, 14, 10))
        color_frame.pack(fill="x")
        ttk.Label(color_frame, text="New box color:", style="Toolbar.TLabel").pack(side="left", padx=(0, 8))
        self.color_mode = tk.StringVar(value="auto")
        ttk.Radiobutton(color_frame, text="Auto  (left = blue, right = red)", variable=self.color_mode, value="auto").pack(side="left", padx=4)
        ttk.Radiobutton(color_frame, text="Red", variable=self.color_mode, value=RED).pack(side="left", padx=4)
        ttk.Radiobutton(color_frame, text="Blue", variable=self.color_mode, value=BLUE).pack(side="left", padx=4)
        ttk.Button(color_frame, text="Recolor selected", command=self.recolor_selected).pack(side="right")

        canvas_frame = tk.Frame(self, bg="#0f172a")
        canvas_frame.pack(fill="both", expand=True, padx=12, pady=(2, 0))
        self.canvas = tk.Canvas(canvas_frame, bg="#020617", highlightthickness=1, highlightbackground="#334155", cursor="crosshair")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _event: self._redraw())
        self.canvas.bind("<ButtonPress-1>", self._mouse_down)
        self.canvas.bind("<B1-Motion>", self._mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self._mouse_up)
        self.canvas.bind("<Button-3>", self._right_click)

        footer = ttk.Frame(self, style="Toolbar.TFrame", padding=(14, 10))
        footer.pack(fill="x")
        ttk.Button(footer, text="← Previous", command=self.previous_image).pack(side="left")
        ttk.Button(footer, text="Next →", command=self.next_image).pack(side="left", padx=(8, 18))
        self.box_var = tk.StringVar(value="0 boxes")
        ttk.Label(footer, textvariable=self.box_var, style="Toolbar.TLabel").pack(side="left")
        ttk.Label(
            footer,
            text="Click a box to select it • drag a white corner handle to resize • drag anywhere else to draw • right-click/Delete removes • arrows navigate",
            style="Hint.TLabel",
        ).pack(side="right")

    def _load_current(self) -> None:
        if not self.records:
            self.file_var.set("No image/JSON pairs found")
            self.save_var.set("Each image must have a matching .json file containing a box list")
            return
        image_path = self.records[self.index]
        try:
            with Image.open(image_path) as opened:
                self.image = opened.convert("RGB")
        except (OSError, ValueError) as exc:
            self.image = None
            self.boxes = []
            self.file_var.set(image_path.name)
            self.save_var.set(f"Could not open image: {exc}")
            self._redraw()
            return
        self.boxes = self.repository.load_boxes(image_path, self.image.width, self.image.height)
        self.selected_index = None
        self.drag_start = None
        self.mouse_operation = None
        self.mouse_down_hit = None
        self.resize_box_index = None
        self.resize_handle = None
        self.last_saved_path = None
        self.file_var.set(f"{image_path.name}   ({self.index + 1} / {len(self.records)})")
        self.save_var.set("Loaded • edits save automatically")
        self._redraw()

    def _redraw(self) -> None:
        self.canvas.delete("all")
        if self.image is None:
            return
        available_width = max(40, self.canvas.winfo_width() - 22)
        available_height = max(40, self.canvas.winfo_height() - 22)
        scale = min(available_width / self.image.width, available_height / self.image.height)
        display_width = max(1, int(self.image.width * scale))
        display_height = max(1, int(self.image.height * scale))
        self.display_size = (display_width, display_height)
        offset_x = (self.canvas.winfo_width() - display_width) / 2
        offset_y = (self.canvas.winfo_height() - display_height) / 2
        self.image_offset = (offset_x, offset_y)
        resized = self.image.resize((display_width, display_height), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(resized)
        self.canvas.create_image(offset_x, offset_y, image=self.photo, anchor="nw", tags=("image",))
        for box_index, box in enumerate(self.boxes):
            self._draw_box(box_index, box)
        self.box_var.set(f"{len(self.boxes)} box" + ("es" if len(self.boxes) != 1 else ""))

    def _draw_box(self, index: int, box: Box) -> None:
        x1, y1 = self._normalized_to_canvas(box.x, box.y)
        x2, y2 = self._normalized_to_canvas(box.x + box.w, box.y + box.h)
        color = {RED: "#ef4444", BLUE: "#3b82f6", NEUTRAL: "#e2e8f0"}.get(box.color, "#e2e8f0")
        selected = index == self.selected_index
        self.canvas.create_rectangle(
            x1, y1, x2, y2,
            outline="#facc15" if selected else color,
            width=4 if selected else 3,
            dash=(7, 4) if box.color == NEUTRAL and not selected else (),
            tags=(f"box-{index}", "box"),
        )
        label = box.color.upper()
        if box.color == NEUTRAL:
            label = "ROBOT"
        self.canvas.create_text(
            x1 + 5, y1 + 4, text=label, anchor="nw",
            fill="#f8fafc" if box.color != NEUTRAL else "#0f172a",
            font=("Segoe UI", 9, "bold"), tags=(f"box-label-{index}", "box-label"),
        )
        if selected:
            for handle_x, handle_y in self._resize_handles(box).values():
                self.canvas.create_rectangle(
                    handle_x - HANDLE_HALF_SIZE,
                    handle_y - HANDLE_HALF_SIZE,
                    handle_x + HANDLE_HALF_SIZE,
                    handle_y + HANDLE_HALF_SIZE,
                    fill="#f8fafc",
                    outline="#0f172a",
                    width=2,
                    tags=("resize-handle",),
                )

    def _canvas_to_normalized(self, canvas_x: float, canvas_y: float) -> tuple[float, float] | None:
        if self.image is None:
            return None
        offset_x, offset_y = self.image_offset
        display_width, display_height = self.display_size
        if display_width <= 0 or display_height <= 0:
            return None
        x = (canvas_x - offset_x) / display_width
        y = (canvas_y - offset_y) / display_height
        if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
            return None
        return x, y

    def _normalized_to_canvas(self, x: float, y: float) -> tuple[float, float]:
        offset_x, offset_y = self.image_offset
        display_width, display_height = self.display_size
        return offset_x + x * display_width, offset_y + y * display_height

    def _box_at(self, normalized: tuple[float, float]) -> int | None:
        x, y = normalized
        for index in range(len(self.boxes) - 1, -1, -1):
            box = self.boxes[index]
            if box.x <= x <= box.x + box.w and box.y <= y <= box.y + box.h:
                return index
        return None

    def _resize_handles(self, box: Box) -> dict[str, tuple[float, float]]:
        left, top = self._normalized_to_canvas(box.x, box.y)
        right, bottom = self._normalized_to_canvas(box.x + box.w, box.y + box.h)
        return {
            "nw": (left, top),
            "ne": (right, top),
            "se": (right, bottom),
            "sw": (left, bottom),
        }

    def _resize_handle_at(self, canvas_x: float, canvas_y: float) -> str | None:
        if self.selected_index is None or not (0 <= self.selected_index < len(self.boxes)):
            return None
        for handle, (handle_x, handle_y) in self._resize_handles(self.boxes[self.selected_index]).items():
            if abs(canvas_x - handle_x) <= HANDLE_HALF_SIZE + 3 and abs(canvas_y - handle_y) <= HANDLE_HALF_SIZE + 3:
                return handle
        return None

    def _mouse_down(self, event: tk.Event) -> None:
        normalized = self._canvas_to_normalized(event.x, event.y)
        if normalized is None:
            return
        resize_handle = self._resize_handle_at(event.x, event.y)
        self.drag_start = normalized
        self.preview_rect = None
        self.resize_box_index = self.selected_index if resize_handle is not None else None
        self.resize_handle = resize_handle
        self.mouse_down_hit = None
        if resize_handle is not None:
            self.mouse_operation = "resize"
            return
        hit = self._box_at(normalized)
        self.mouse_down_hit = hit
        self.mouse_operation = "potential_draw" if hit is not None else "draw"
        if hit is not None:
            self.selected_index = hit
        else:
            self.selected_index = None
        self._redraw()

    def _mouse_move(self, event: tk.Event) -> None:
        if self.drag_start is None or self.mouse_operation is None:
            return
        normalized = self._canvas_to_normalized(event.x, event.y)
        if normalized is None:
            return
        if self.mouse_operation == "resize":
            self._resize_selected(normalized)
            self._redraw()
            return
        start_x, start_y = self.drag_start
        x = min(start_x, normalized[0])
        y = min(start_y, normalized[1])
        width = abs(normalized[0] - start_x)
        height = abs(normalized[1] - start_y)
        self._redraw()
        if width > 0.005 and height > 0.005:
            x1, y1 = self._normalized_to_canvas(x, y)
            x2, y2 = self._normalized_to_canvas(x + width, y + height)
            self.preview_rect = self.canvas.create_rectangle(x1, y1, x2, y2, outline="#facc15", width=3, dash=(7, 4))

    def _resize_selected(self, normalized: tuple[float, float]) -> None:
        if self.resize_box_index is None or self.resize_handle is None:
            return
        if not (0 <= self.resize_box_index < len(self.boxes)):
            return
        box = self.boxes[self.resize_box_index]
        left, top = box.x, box.y
        right, bottom = box.x + box.w, box.y + box.h
        pointer_x = max(0.0, min(1.0, normalized[0]))
        pointer_y = max(0.0, min(1.0, normalized[1]))
        minimum_size = 0.005
        if "w" in self.resize_handle:
            left = max(0.0, min(pointer_x, right - minimum_size))
        else:
            right = min(1.0, max(pointer_x, left + minimum_size))
        if "n" in self.resize_handle:
            top = max(0.0, min(pointer_y, bottom - minimum_size))
        else:
            bottom = min(1.0, max(pointer_y, top + minimum_size))
        box.x, box.y, box.w, box.h = left, top, right - left, bottom - top

    def _mouse_up(self, event: tk.Event) -> None:
        if self.drag_start is None:
            return
        normalized = self._canvas_to_normalized(event.x, event.y)
        start = self.drag_start
        operation = self.mouse_operation
        hit = self.mouse_down_hit
        self.drag_start = None
        self.mouse_operation = None
        self.mouse_down_hit = None
        self.resize_box_index = None
        self.resize_handle = None
        if self.preview_rect is not None:
            self.preview_rect = None
        if normalized is None:
            self._redraw()
            return
        if operation == "resize":
            self._save_current("Box resized")
            self._redraw()
            return
        x = min(start[0], normalized[0])
        y = min(start[1], normalized[1])
        width = abs(normalized[0] - start[0])
        height = abs(normalized[1] - start[1])
        if width < 0.01 or height < 0.01:
            if hit is not None:
                self._redraw()
                return
            width = self.click_width
            height = self.click_height
            x = start[0] - width / 2.0
            y = start[1] - height / 2.0
        clamped = clamp_box(x, y, width, height)
        if clamped is None:
            self._redraw()
            return
        color = self._color_for_click((start[0] + normalized[0]) / 2.0)
        assert self.image is not None
        self.boxes.append(
            self.repository.new_box(
                self.records[self.index],
                *clamped,
                color,
                self.image.width,
                self.image.height,
            )
        )
        self.selected_index = len(self.boxes) - 1
        self._save_current("Box added")
        self._redraw()

    def _right_click(self, event: tk.Event) -> None:
        normalized = self._canvas_to_normalized(event.x, event.y)
        if normalized is None:
            return
        hit = self._box_at(normalized)
        if hit is None:
            return
        self.selected_index = hit
        self.delete_selected()

    def _color_for_click(self, normalized_x: float) -> str:
        mode = self.color_mode.get()
        if mode in (RED, BLUE):
            return mode
        return BLUE if normalized_x < self.midpoint else RED

    def choose_color(self, color: str) -> None:
        self.color_mode.set(color)
        if color in (RED, BLUE):
            self.recolor_selected()

    def recolor_selected(self) -> None:
        if self.selected_index is None or not (0 <= self.selected_index < len(self.boxes)):
            self.save_var.set("Select a box first, then choose its color")
            return
        box = self.boxes[self.selected_index]
        # Auto mode was previously a no-op here, which made the main button
        # appear broken unless Red or Blue was selected first.
        color = self._color_for_click(box.x + box.w / 2.0)
        self.boxes[self.selected_index].color = color
        self._save_current("Box recolored")
        self._redraw()

    def delete_selected(self) -> None:
        if self.selected_index is None or not (0 <= self.selected_index < len(self.boxes)):
            return
        del self.boxes[self.selected_index]
        self.selected_index = None
        self._save_current("Box deleted")
        self._redraw()

    def _save_current(self, message: str = "Saved") -> None:
        if not self.records or self.image is None:
            return
        try:
            path = self.repository.save_boxes(self.records[self.index], self.boxes)
        except (OSError, ValueError) as exc:
            self.save_var.set(f"Save failed: {exc}")
            messagebox.showerror("Autosave failed", str(exc), parent=self)
            return
        self.last_saved_path = path
        self.save_var.set(f"{message} • autosaved to {path.name}")

    def previous_image(self) -> None:
        if not self.records or self.index <= 0:
            return
        self._save_current()
        self.index -= 1
        self._load_current()

    def next_image(self) -> None:
        if not self.records or self.index >= len(self.records) - 1:
            return
        self._save_current()
        self.index += 1
        self._load_current()

    def open_folder(self) -> None:
        selected = filedialog.askdirectory(parent=self, title="Choose the competition image folder")
        if not selected:
            return
        try:
            repository = AnnotationRepository(Path(selected))
        except (OSError, ValueError) as exc:
            messagebox.showerror("Could not open folder", str(exc), parent=self)
            return
        self.repository = repository
        self.records = repository.records
        self.index = 0
        self.image = None
        self.boxes = []
        self._load_current()

    def _close(self) -> None:
        self._save_current()
        self.destroy()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review and label FRC robot boxes on competition images.")
    parser.add_argument("folder", nargs="?", help="folder containing competition images (a chooser opens if omitted)")
    parser.add_argument("--click-width", type=float, default=0.12, help="default box width for a single click, as a fraction of image width")
    parser.add_argument("--click-height", type=float, default=0.18, help="default box height for a single click, as a fraction of image height")
    parser.add_argument("--midpoint", type=float, default=0.5, help="normalized left/right split used by Auto color mode")
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    if args.folder:
        folder = Path(args.folder)
    else:
        chooser = tk.Tk()
        chooser.withdraw()
        selected = filedialog.askdirectory(title="Choose the competition image folder")
        chooser.destroy()
        if not selected:
            return 0
        folder = Path(selected)
    try:
        repository = AnnotationRepository(folder)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1
    if not repository.records:
        print(f"No image/JSON pairs found in {repository.root}")
        return 1
    app = RobotBoxLabeler(repository, args.click_width, args.click_height, args.midpoint)
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
