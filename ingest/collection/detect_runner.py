"""Run one or more ONNX detectors over a collection's frames and fuse the results.

This is the step that turns a trained model into labels for our own footage. It reads the ONNX
export rather than a framework checkpoint on purpose:

  * onnxruntime is MIT, so nothing AGPL has to be installed to label data;
  * it is a fraction of the install size of a training framework; and
  * it is the SAME file the C++ analyzer loads. A model that produces sensible boxes here is a
    model that will load there, which removes a whole class of "works in Python, fails in C++"
    surprise from the handoff.

Frames the quality filter rejected are skipped. Running a detector over a FIRST logo wastes time
and, worse, produces boxes on it -- which is exactly how the previous labelling round poisoned
itself.

Detectors are pluggable so the pipeline can be exercised without a model present: anything with
`.name` and `.detect(image_bgr) -> list[box]` will do.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .box_fusion import estimate_source_weights, fuse_frame
from .view_region import CROP, Region, crop, map_from_region


class Detector(Protocol):
    name: str

    def detect(self, image_bgr) -> list[dict]:
        """Return boxes as normalised {x, y, w, h, confidence} with x/y at the top-left."""
        ...


class RegionCropped:
    """Runs the wrapped decode on a crop of the frame and puts the boxes back where they belong.

    Both detector families inherit this rather than each cropping for itself. They already differ
    on normalisation, letterboxing and output layout; a third thing for them to disagree about --
    silently, since a mis-mapped box is still a valid box -- is not worth the duplication.
    """

    def detect(self, image_bgr) -> list[dict]:
        region = getattr(self, "region", None) or Region()
        if region.is_full_frame:
            return self._detect_cropped(image_bgr)
        if region.mode == CROP:
            boxes = self._detect_cropped(crop(image_bgr, region))
            return [map_from_region(b, region) for b in boxes]
        return [b for b in self._detect_cropped(image_bgr) if region.contains_centre(b)]


@dataclass
class OnnxDetector(RegionCropped):
    """A YOLO-family ONNX model.

    Output layout is not guessed. YOLOv8/11 export a single tensor shaped (1, 4+nc, anchors);
    older exports use (1, anchors, 4+nc). Both are handled by checking which axis matches the
    expected 4+nc width, because silently transposing the wrong one produces boxes that look
    plausible and are meaningless.
    """

    model_path: str
    name: str = "onnx"
    confidence_threshold: float = 0.25
    #: Read from the model itself when left as None. Getting this wrong silently mis-scales every
    #: box: an export trained at 960 fed 640-sized input produces plausible, wrong coordinates.
    input_size: int | None = None
    #: IoU above which two candidates are treated as the same object. Raw YOLO output contains
    #: thousands of overlapping candidates per object; without suppression one robot arrives as
    #: seven boxes and every count downstream is meaningless.
    nms_iou: float = 0.50
    #: The part of the frame that holds the field, for sources that composite a second camera view
    #: into one picture and would otherwise have every robot counted twice. See view_region for
    #: which of its two modes is the default and the measurement behind that.
    region: "Region | None" = None
    _session: object = None

    def __post_init__(self):
        import onnxruntime as ort
        self._session = ort.InferenceSession(
            self.model_path, providers=["CPUExecutionProvider"]
        )
        if self.input_size is None:
            # The export records its own training resolution. Trust that over a default.
            shape = self._session.get_inputs()[0].shape
            self.input_size = int(shape[2]) if isinstance(shape[2], int) else 640

    def _detect_cropped(self, image_bgr) -> list[dict]:
        import numpy as np
        import cv2

        h0, w0 = image_bgr.shape[:2]
        # Letterbox to a square so aspect ratio is preserved; stretching moves every box.
        scale = self.input_size / max(h0, w0)
        nh, nw = int(round(h0 * scale)), int(round(w0 * scale))
        resized = cv2.resize(image_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        top, left = (self.input_size - nh) // 2, (self.input_size - nw) // 2
        canvas[top:top + nh, left:left + nw] = resized

        blob = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        name = self._session.get_inputs()[0].name
        raw = self._session.run(None, {name: blob})[0]

        preds = np.squeeze(raw)
        if preds.ndim != 2:
            return []
        # (4+nc, anchors) -> (anchors, 4+nc). One class means width 5.
        if preds.shape[0] < preds.shape[1]:
            preds = preds.T

        boxes = []
        for row in preds:
            score = float(row[4:].max()) if row.shape[0] > 4 else 0.0
            if score < self.confidence_threshold:
                continue
            cx, cy, bw, bh = (float(v) for v in row[:4])
            # Undo letterbox, then normalise against the original frame.
            x = (cx - bw / 2 - left) / scale
            y = (cy - bh / 2 - top) / scale
            boxes.append({
                "x": max(0.0, min(1.0, x / w0)),
                "y": max(0.0, min(1.0, y / h0)),
                "w": max(0.0, min(1.0, bw / scale / w0)),
                "h": max(0.0, min(1.0, bh / scale / h0)),
                "confidence": score,
            })
        return non_max_suppression(boxes, self.nms_iou)


@dataclass
class RfDetrDetector(RegionCropped):
    """An RF-DETR ONNX export.

    The output format is DETR's, not YOLO's, and the decode matches analysis/src/RFDetrDetector.cpp
    exactly -- that C++ file is the authority for how this project reads an RF-DETR export, and two
    decoders that disagree would produce different boxes from one model. Specifically:

      * two outputs, `dets` (1, queries, 4) in normalised cxcywh and `labels` (1, queries, classes);
      * a fixed query set, so there is no confidence-free anchor grid to threshold -- every query
        is scored and the low ones dropped;
      * ImageNet mean/std normalisation after /255, where YOLO uses a bare /255; and
      * a plain resize to square, where our YOLO path letterboxes. RF-DETR trains on the stretched
        square, so matching that at inference is correct rather than a shortcut.
    """

    model_path: str
    name: str = "rfdetr"
    confidence_threshold: float = 0.25
    input_size: int = 640
    robot_class_id: int = 0        # this export puts robot at class 0; class 1 is inert
    nms_iou: float = 0.50
    region: "Region | None" = None
    _session: object = None

    def __post_init__(self):
        import onnxruntime as ort
        self._session = ort.InferenceSession(
            self.model_path, providers=["CPUExecutionProvider"]
        )

    def _detect_cropped(self, image_bgr) -> list[dict]:
        import numpy as np
        import cv2

        mean = np.array([0.485, 0.456, 0.406], np.float32)
        std = np.array([0.229, 0.224, 0.225], np.float32)
        resized = cv2.resize(image_bgr[:, :, ::-1], (self.input_size, self.input_size),
                             interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        blob = ((resized - mean) / std).transpose(2, 0, 1)[None]

        name = self._session.get_inputs()[0].name
        dets, labels = self._session.run(None, {name: blob})
        dets, labels = dets[0], labels[0]

        scores = 1.0 / (1.0 + np.exp(-labels[:, self.robot_class_id]))
        boxes = []
        for i in range(dets.shape[0]):
            score = float(scores[i])
            if score < self.confidence_threshold:
                continue
            cx, cy, w, h = (float(v) for v in dets[i])
            x = cx - w / 2.0
            y = cy - h / 2.0
            boxes.append({
                "x": max(0.0, min(1.0, x)),
                "y": max(0.0, min(1.0, y)),
                "w": max(0.0, min(1.0, w)),
                "h": max(0.0, min(1.0, h)),
                "confidence": score,
            })
        return non_max_suppression(boxes, self.nms_iou)


def non_max_suppression(boxes: list[dict], iou_threshold: float = 0.50) -> list[dict]:
    """Collapse overlapping candidates to one box per object.

    A raw YOLO tensor holds a prediction for every anchor, so a single robot produces a cluster of
    near-identical boxes. Without this, counts are inflated several-fold and box fusion sees one
    detector's repeated opinion as if it were corroboration.

    Greedy and standard: take the most confident box, drop everything overlapping it beyond the
    threshold, repeat.
    """
    from .box_fusion import iou as _iou

    remaining = sorted(boxes, key=lambda b: -b["confidence"])
    kept: list[dict] = []
    while remaining:
        best = remaining.pop(0)
        kept.append(best)
        remaining = [b for b in remaining if _iou(best, b) < iou_threshold]
    return kept


def usable_frames(collection: Path) -> list[dict]:
    """Frames the quality filter kept. Older collections predate the flag and are all kept."""
    rows = []
    for line in (collection / "frames.jsonl").read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("quality_ok", True):
            rows.append(row)
    return rows


def run_detectors(
    collection: Path,
    detectors: list[Detector],
    limit: int | None = None,
) -> dict[str, list[dict]]:
    """Detect over one collection. Returns frame_id -> {detector name: boxes}."""
    import cv2

    frames = usable_frames(collection)
    if limit is not None:
        frames = frames[:limit]

    per_frame: dict[str, dict[str, list[dict]]] = {}
    for frame in frames:
        image = cv2.imread(str(collection / frame["image_path"]))
        if image is None:
            continue
        per_frame[frame["frame_id"]] = {
            d.name: d.detect(image) for d in detectors
        }
    return per_frame


def fuse_and_write(
    collection: Path,
    per_frame: dict[str, dict[str, list[dict]]],
    output_name: str = "detector-consensus.jsonl",
) -> Path:
    """Fuse each frame's detectors and write one row per frame.

    Source weights are estimated across the whole collection first, so a detector that habitually
    invents boxes is already demoted by the time any individual frame is scored. Estimating
    per-frame would leave nothing to learn from.
    """
    weights = estimate_source_weights(per_frame.values())

    rows = []
    for frame_id, proposals in sorted(per_frame.items()):
        fused = fuse_frame(proposals, source_weights=weights)
        rows.append({
            "frame_id": frame_id,
            "detectors": sorted(proposals),
            "source_weights": {k: round(v, 4) for k, v in weights.items()},
            "status": "proposed",
            "human_review_required": True,
            "boxes": [b.as_dict() for b in fused],
        })

    out = collection / output_name
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    tmp.replace(out)
    return out
