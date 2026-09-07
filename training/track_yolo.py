"""Run a trained YOLO detector with ByteTrack or BoT-SORT and emit persistent robot tracks."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path


# Normalized coordinates of the yellow-marked upper broadcast panel in the supplied reference.
# The small margins keep the yellow markup itself out of the model input.
DEFAULT_CROP = (0.02, 0.035, 0.98, 0.66)  # left, top, right, bottom
# A sudden identity swap or bad calibration can manufacture an impossible velocity. Keep that
# out of scouting metrics instead of presenting it as a very fast robot.
MAX_PLAUSIBLE_FTPS = 20.0
# A detection has to move farther than detector jitter before it is allowed into the public
# track stream. The separate five-second timer removes detections that later become stationary.
STATIONARY_CENTER_THRESHOLD = 0.018
# A stationary false positive is allowed a short grace period because detector startup can
# produce a few duplicate boxes.  Once it has been still this long, it is suppressed for the
# rest of the match instead of being rediscovered every frame.
STATIONARY_SUPPRESSION_SECONDS = 5.0
# ByteTrack is deliberately motion-first.  This lightweight appearance memory sits after it and
# stitches a newly-issued raw id back onto a recently-lost robot when the crop still looks alike.
# Keep this longer than the requested one-second dropout: the detector can miss several frames at
# an edge and the new raw id may not arrive until the object is fully back in view.
DEFAULT_REID_MEMORY_SECONDS = 5.0
DEFAULT_REID_APPEARANCE_THRESHOLD = 0.60
DEFAULT_REID_MAX_CENTER_DISTANCE = 0.60
DEFAULT_REID_SCORE_MARGIN = 0.08
DEFAULT_REID_MAX_NORMALIZED_SPEED = 0.75
DEFAULT_REID_MOTION_SLACK = 0.045
DEFAULT_REID_EDGE_THRESHOLD = 0.10
DEFAULT_REID_TEMPLATE_GALLERY_SIZE = 5
DEFAULT_REID_TEMPLATE_CONFIRMATION_FRAMES = 3
DEFAULT_REID_TEMPLATE_MIN_CONFIDENCE = 0.50
DEFAULT_REID_ALLIANCE_LOCK_SECONDS = 5.0
DEFAULT_REID_ALLIANCE_LOCK_MARGIN_SECONDS = 2.0
DEFAULT_REID_ALLIANCE_EVIDENCE_MAX_GAP_SECONDS = 0.25
DEFAULT_STARTUP_POSITION_SECONDS = 2.0
DEFAULT_STARTUP_SPLIT_X = 0.50
MAX_ROBOTS = 6


def track_has_motion(
    boxes: list[dict[str, object]],
    minimum_displacement: float = STATIONARY_CENTER_THRESHOLD,
) -> bool:
    """Return whether a track moved farther than normal detector-box jitter.

    The fallback is image space because homography is optional.  Scale the threshold by the
    object's own box size so tiny confidence jitter on a large static field prop is not mistaken
    for robot travel.
    """

    if any(
        isinstance(box.get("speed_ftps"), (int, float))
        and float(box["speed_ftps"]) >= 0.5
        for box in boxes
    ):
        return True
    if len(boxes) < 2:
        return False
    centres = [
        (float(box["x"]) + float(box["w"]) / 2.0,
         float(box["y"]) + float(box["h"]) / 2.0)
        for box in boxes
    ]
    widths = sorted(float(box["w"]) for box in boxes)
    heights = sorted(float(box["h"]) for box in boxes)
    middle = len(boxes) // 2
    box_diagonal = math.hypot(widths[middle], heights[middle])
    required = max(minimum_displacement, box_diagonal * 0.18)
    span = math.hypot(
        max(point[0] for point in centres) - min(point[0] for point in centres),
        max(point[1] for point in centres) - min(point[1] for point in centres),
    )
    return span >= required


def startup_alliance(
    detected_alliance: str | None,
    center_x: float,
    timestamp: float,
    startup_seconds: float = DEFAULT_STARTUP_POSITION_SECONDS,
    split_x: float = DEFAULT_STARTUP_SPLIT_X,
) -> str | None:
    """Apply the known broadcast orientation only during the opening position window.

    The raw bumper-colour observation is preserved separately. Once the opening window ends,
    alliance comes from visual evidence again because robots can cross the field midpoint.
    """

    if startup_seconds > 0.0 and 0.0 <= timestamp < startup_seconds:
        return "blue" if center_x < split_x else "red"
    return detected_alliance


def image_window_has_motion(
    boxes: list[dict[str, object]],
    minimum_displacement: float = 0.012,
) -> bool:
    """Detect real image-space travel while ignoring one-frame box jitter.

    A max/min span is useful for deciding whether a complete track ever moved, but it is too
    sensitive for a five-second stationary timer: one bad detector box would reset that timer.
    Compare the median centre of the first and second half of a short window instead.
    """

    if len(boxes) < 4:
        return False

    def median(values: list[float]) -> float:
        ordered = sorted(values)
        middle = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[middle]
        return (ordered[middle - 1] + ordered[middle]) / 2.0

    midpoint = len(boxes) // 2
    first = boxes[:midpoint]
    second = boxes[midpoint:]
    first_center = (
        median([float(box["x"]) + float(box["w"]) / 2.0 for box in first]),
        median([float(box["y"]) + float(box["h"]) / 2.0 for box in first]),
    )
    second_center = (
        median([float(box["x"]) + float(box["w"]) / 2.0 for box in second]),
        median([float(box["y"]) + float(box["h"]) / 2.0 for box in second]),
    )
    widths = sorted(float(box["w"]) for box in boxes)
    heights = sorted(float(box["h"]) for box in boxes)
    middle = len(boxes) // 2
    box_diagonal = math.hypot(widths[middle], heights[middle])
    required = max(minimum_displacement, box_diagonal * 0.10)
    return math.hypot(
        second_center[0] - first_center[0],
        second_center[1] - first_center[1],
    ) >= required


def eligible_track_ids(
    tracks: dict[int, list[dict[str, object]]],
    alliances: dict[int, str | None] | None,
    suppressed_track_ids: set[int] | None = None,
) -> set[int]:
    """Tracks allowed into video overlays and Contract C output.

    A stationary detection is usually field hardware, signage, or another persistent false
    positive. The runner suppresses it after five seconds, and the explicit set keeps it hidden
    even if the detector later rediscovers the same object. A genuinely moving robot remains
    visible even when its bumper colour is temporarily unreadable.
    """

    colours = alliances or {}
    suppressed = suppressed_track_ids or set()
    return {
        track_id
        for track_id, boxes in tracks.items()
        # A confirmed alliance colour is the signal that this is a real robot.  Do not remove
        # a robot merely because it waits in place; only stationary, colourless false positives
        # (the laptop/field-hardware case) are suppressed.
        if colours.get(track_id) in {"red", "blue"}
        or (track_id not in suppressed and track_has_motion(boxes))
    }


def appearance_similarity(left: list[float], right: list[float]) -> float:
    """Cosine similarity for two already-normalised appearance descriptors."""

    if not left or len(left) != len(right):
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return 0.0
    return numerator / (left_norm * right_norm)


def appearance_descriptor(image, raw_box: list[float]) -> list[float] | None:
    """Build a spatial colour-and-shape signature from one robot crop.

    The old descriptor was dominated by the bumper: one HSV histogram and four mean-colour cells
    made all three robots on an alliance look alike.  This remains CPU-cheap, but uses the upper
    body for colour histograms, a 4x4 Lab layout, and cell-wise gradient orientations.  Bumper
    colour is handled separately as a compatibility gate and therefore need not dominate identity.
    """

    import cv2
    import numpy as np

    height, width = image.shape[:2]
    left, top, right, bottom = (int(round(value)) for value in raw_box)
    left = max(0, min(width - 1, left))
    right = max(left + 1, min(width, right))
    top = max(0, min(height - 1, top))
    bottom = max(top + 1, min(height, bottom))
    crop = image[top:bottom, left:right]
    if crop.size == 0:
        return None
    resized_bgr = cv2.resize(crop, (32, 32), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # The lower fifth normally contains the alliance bumper. Excluding it keeps the histogram
    # focused on mechanisms, sponsor panels and robot structure that distinguish same-colour bots.
    body_hsv = hsv[:26]
    hsv_histogram = cv2.calcHist(
        [body_hsv], [0, 1], None, [16, 4], [0, 180, 0, 256]
    ).reshape(-1).astype(np.float32)
    hsv_histogram /= max(float(np.linalg.norm(hsv_histogram)), 1e-12)
    lab_histograms = [
        cv2.calcHist([lab[:26]], [channel], None, [8], [0, 256]).reshape(-1)
        for channel in range(3)
    ]
    for histogram in lab_histograms:
        histogram /= max(float(np.linalg.norm(histogram)), 1e-12)

    layout: list[float] = []
    for y0 in range(0, 32, 8):
        for x0 in range(0, 32, 8):
            cell = lab[y0:y0 + 8, x0:x0 + 8].astype(np.float32) / 255.0
            layout.extend(cell.mean(axis=(0, 1)).tolist())
            layout.extend(cell.std(axis=(0, 1)).tolist())

    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude, angle = cv2.cartToPolar(gradient_x, gradient_y, angleInDegrees=True)
    gradient_layout: list[float] = []
    for y0 in range(0, 32, 8):
        for x0 in range(0, 32, 8):
            cell_angles = angle[y0:y0 + 8, x0:x0 + 8]
            cell_magnitude = magnitude[y0:y0 + 8, x0:x0 + 8]
            histogram, _ = np.histogram(
                cell_angles, bins=8, range=(0.0, 360.0), weights=cell_magnitude
            )
            histogram = histogram.astype(np.float32)
            histogram /= max(float(np.linalg.norm(histogram)), 1e-12)
            gradient_layout.extend(histogram.tolist())

    descriptor = np.concatenate((
        hsv_histogram,
        *(histogram.astype(np.float32) for histogram in lab_histograms),
        np.asarray(layout, dtype=np.float32) * 0.45,
        np.asarray(gradient_layout, dtype=np.float32) * 0.35,
    ))
    if float(np.linalg.norm(descriptor)) <= 1e-12:
        return None
    descriptor /= float(np.linalg.norm(descriptor))
    return descriptor.astype(float).tolist()


@dataclass(frozen=True)
class ReIDConfig:
    """All custom identity thresholds in one injectable, CLI-configurable object."""

    memory_seconds: float = DEFAULT_REID_MEMORY_SECONDS
    appearance_threshold: float = DEFAULT_REID_APPEARANCE_THRESHOLD
    max_center_distance: float = DEFAULT_REID_MAX_CENTER_DISTANCE
    score_margin: float = DEFAULT_REID_SCORE_MARGIN
    max_normalized_speed: float = DEFAULT_REID_MAX_NORMALIZED_SPEED
    motion_slack: float = DEFAULT_REID_MOTION_SLACK
    edge_threshold: float = DEFAULT_REID_EDGE_THRESHOLD
    template_gallery_size: int = DEFAULT_REID_TEMPLATE_GALLERY_SIZE
    template_confirmation_frames: int = DEFAULT_REID_TEMPLATE_CONFIRMATION_FRAMES
    template_min_detection_confidence: float = DEFAULT_REID_TEMPLATE_MIN_CONFIDENCE
    max_robots: int = MAX_ROBOTS
    minimum_score: float = 0.54
    continuity_similarity: float = 0.32
    strong_motion_score: float = 0.82
    template_update_similarity: float = 0.72
    template_update_score: float = 0.72
    maximum_field_speed_ftps: float = MAX_PLAUSIBLE_FTPS
    field_motion_slack_feet: float = 2.0
    appearance_weight: float = 0.58
    motion_weight: float = 0.34
    edge_weight: float = 0.08
    missing_appearance_motion_weight: float = 0.82
    edge_neutral_score: float = 0.50
    raw_continuity_bonus: float = 0.06
    velocity_update_weight: float = 0.45
    template_distinct_similarity: float = 0.995
    alliance_lock_seconds: float = DEFAULT_REID_ALLIANCE_LOCK_SECONDS
    alliance_lock_margin_seconds: float = DEFAULT_REID_ALLIANCE_LOCK_MARGIN_SECONDS
    alliance_evidence_max_gap_seconds: float = DEFAULT_REID_ALLIANCE_EVIDENCE_MAX_GAP_SECONDS
    alliance_evidence_min_detection_confidence: float = 0.50
    alliance_evidence_min_assignment_score: float = 0.70
    startup_position_seconds: float = DEFAULT_STARTUP_POSITION_SECONDS
    startup_split_x: float = DEFAULT_STARTUP_SPLIT_X


@dataclass(frozen=True)
class ReIDDetection:
    raw_track_id: int
    descriptor: list[float] | None
    center: tuple[float, float]
    alliance: str | None
    edge: str | None = None
    field_center: tuple[float, float] | None = None
    confidence: float = 1.0
    alliance_is_authoritative: bool = False


@dataclass
class _IdentityState:
    last_seen: float
    center: tuple[float, float]
    velocity: tuple[float, float] = (0.0, 0.0)
    alliance: str | None = None
    templates: list[list[float]] = field(default_factory=list)
    last_raw_id: int | None = None
    last_edge: str | None = None
    field_center: tuple[float, float] | None = None
    field_velocity: tuple[float, float] = (0.0, 0.0)
    camera_offset: tuple[float, float] = (0.0, 0.0)
    pending_raw_id: int | None = None
    pending_template: list[float] | None = None
    pending_count: int = 0
    alliance_evidence_seconds: dict[str, float] = field(
        default_factory=lambda: {"red": 0.0, "blue": 0.0}
    )


class AppearanceTrackMemory:
    """Globally map tracker tracklets onto at most six stable robot identities.

    The public name is retained for compatibility, but assignment is now frame-level.  Invalid or
    ambiguous observations return ``None`` and remain available in the raw-tracklet sidecar rather
    than being forced into (and corrupting) an existing robot track.
    """

    def __init__(
        self,
        memory_seconds: float = DEFAULT_REID_MEMORY_SECONDS,
        appearance_threshold: float = DEFAULT_REID_APPEARANCE_THRESHOLD,
        max_center_distance: float = DEFAULT_REID_MAX_CENTER_DISTANCE,
        *,
        config: ReIDConfig | None = None,
    ):
        self.config = config or ReIDConfig(
            memory_seconds=memory_seconds,
            appearance_threshold=appearance_threshold,
            max_center_distance=max_center_distance,
        )
        self.memory_seconds = self.config.memory_seconds
        self.appearance_threshold = self.config.appearance_threshold
        self.max_center_distance = self.config.max_center_distance
        self.raw_to_stable: dict[int, int] = {}
        self.states: dict[int, _IdentityState] = {}
        self.camera_offset = (0.0, 0.0)

    def _available_identity(self) -> int | None:
        for stable_id in range(1, min(self.config.max_robots, MAX_ROBOTS) + 1):
            if stable_id not in self.states:
                return stable_id
        return None

    def _edge(self, center: tuple[float, float]) -> str | None:
        x, y = center
        distances = {"left": x, "right": 1.0 - x, "top": y, "bottom": 1.0 - y}
        edge, distance = min(distances.items(), key=lambda item: item[1])
        return edge if distance <= self.config.edge_threshold else None

    def _appearance(self, detection: ReIDDetection, state: _IdentityState) -> float | None:
        if detection.descriptor is None or not state.templates:
            return None
        return max(appearance_similarity(detection.descriptor, template) for template in state.templates)

    def _record_alliance_evidence(
        self,
        state: _IdentityState,
        timestamp: float,
        detection: ReIDDetection,
        assignment_score: float,
    ) -> None:
        """Accumulate trusted observation time, then permanently lock a clear alliance."""

        if state.alliance is not None or detection.alliance not in {"red", "blue"}:
            return
        if detection.alliance_is_authoritative:
            state.alliance = detection.alliance
            return
        if detection.confidence < self.config.alliance_evidence_min_detection_confidence:
            return
        if assignment_score < self.config.alliance_evidence_min_assignment_score:
            return
        observed_seconds = min(
            max(0.0, timestamp - state.last_seen),
            self.config.alliance_evidence_max_gap_seconds,
        )
        state.alliance_evidence_seconds[detection.alliance] += observed_seconds
        ordered = sorted(
            state.alliance_evidence_seconds.items(), key=lambda item: item[1], reverse=True
        )
        (best_alliance, best_seconds), (_, second_seconds) = ordered
        if (
            best_seconds >= self.config.alliance_lock_seconds
            and best_seconds - second_seconds >= self.config.alliance_lock_margin_seconds
        ):
            state.alliance = best_alliance

    def _pair_score(
        self,
        timestamp: float,
        detection: ReIDDetection,
        stable_id: int,
        state: _IdentityState,
    ) -> float | None:
        gap = timestamp - state.last_seen
        if gap <= 0.0 or gap > self.config.memory_seconds:
            return None
        if detection.alliance and state.alliance and detection.alliance != state.alliance:
            return None

        # Prefer field coordinates when calibration is available. Image-space speed is otherwise
        # evaluated after subtracting any supplied global camera motion.
        if detection.field_center is not None and state.field_center is not None:
            dx = detection.field_center[0] - state.field_center[0]
            dy = detection.field_center[1] - state.field_center[1]
            allowed = self.config.field_motion_slack_feet + self.config.maximum_field_speed_ftps * gap
            if math.hypot(dx, dy) > allowed:
                return None
            predicted = (
                state.field_center[0] + state.field_velocity[0] * gap,
                state.field_center[1] + state.field_velocity[1] * gap,
            )
            residual = math.hypot(
                detection.field_center[0] - predicted[0], detection.field_center[1] - predicted[1]
            )
        else:
            camera_dx = self.camera_offset[0] - state.camera_offset[0]
            camera_dy = self.camera_offset[1] - state.camera_offset[1]
            dx = detection.center[0] - state.center[0] - camera_dx
            dy = detection.center[1] - state.center[1] - camera_dy
            allowed = min(
                self.config.max_center_distance,
                self.config.motion_slack + self.config.max_normalized_speed * gap,
            )
            if math.hypot(dx, dy) > allowed:
                return None
            predicted = (
                state.center[0] + state.velocity[0] * gap + camera_dx,
                state.center[1] + state.velocity[1] * gap + camera_dy,
            )
            residual = math.hypot(
                detection.center[0] - predicted[0], detection.center[1] - predicted[1]
            )
        motion_score = max(0.0, 1.0 - residual / max(allowed, 1e-9))

        current_edge = detection.edge or self._edge(detection.center)
        raw_continuity = self.raw_to_stable.get(detection.raw_track_id) == stable_id
        if not raw_continuity and state.last_edge and current_edge and state.last_edge != current_edge:
            # A robot that vanished through one boundary cannot make its first observation back at
            # the opposite boundary without also violating the physical model in many wide shots.
            return None
        edge_score = (
            1.0
            if state.last_edge and current_edge == state.last_edge
            else self.config.edge_neutral_score
        )

        appearance = self._appearance(detection, state)
        if raw_continuity and appearance is not None and appearance < self.config.continuity_similarity:
            return None  # the tracker has recycled this raw id for a visibly different object
        if not raw_continuity:
            if appearance is not None and appearance < self.config.appearance_threshold:
                return None
            if appearance is None and motion_score < self.config.strong_motion_score:
                return None

        if appearance is None:
            score = (
                self.config.missing_appearance_motion_weight * motion_score
                + (1.0 - self.config.missing_appearance_motion_weight) * edge_score
            )
        else:
            score = (
                self.config.appearance_weight * appearance
                + self.config.motion_weight * motion_score
                + self.config.edge_weight * edge_score
            )
        if raw_continuity:
            score = min(1.0, score + self.config.raw_continuity_bonus)
        return score if score >= self.config.minimum_score else None

    @staticmethod
    def _best_assignment(
        scores: list[list[float | None]],
        forbidden: tuple[int, int] | None = None,
    ) -> tuple[float, dict[int, int]]:
        """Exact maximum-weight one-to-one assignment; the problem is capped at six identities."""

        @lru_cache(maxsize=None)
        def solve(row: int, used_mask: int) -> tuple[float, tuple[tuple[int, int], ...]]:
            if row == len(scores):
                return 0.0, ()
            best_score, best_pairs = solve(row + 1, used_mask)  # abstention is always allowed
            for column, score in enumerate(scores[row]):
                bit = 1 << column
                if score is None or used_mask & bit or forbidden == (row, column):
                    continue
                remaining_score, remaining_pairs = solve(row + 1, used_mask | bit)
                candidate_score = score + remaining_score
                if candidate_score > best_score + 1e-12:
                    best_score = candidate_score
                    best_pairs = ((row, column), *remaining_pairs)
            return best_score, best_pairs

        total, pairs = solve(0, 0)
        return total, dict(pairs)

    def _update_state(
        self,
        stable_id: int,
        timestamp: float,
        detection: ReIDDetection,
        assignment_score: float,
    ) -> None:
        state = self.states.get(stable_id)
        if state is None:
            self.states[stable_id] = _IdentityState(
                last_seen=timestamp,
                center=detection.center,
                # A known starting side is authoritative. Ordinary bumper-colour reads remain
                # provisional until sustained evidence performs the irreversible lock.
                alliance=(detection.alliance if detection.alliance_is_authoritative else None),
                templates=[detection.descriptor] if detection.descriptor is not None else [],
                last_raw_id=detection.raw_track_id,
                last_edge=detection.edge or self._edge(detection.center),
                field_center=detection.field_center,
                camera_offset=self.camera_offset,
            )
            return

        gap = timestamp - state.last_seen
        if gap > 0.0:
            new_weight = self.config.velocity_update_weight
            old_weight = 1.0 - new_weight
            measured = (
                (detection.center[0] - state.center[0]) / gap,
                (detection.center[1] - state.center[1]) / gap,
            )
            state.velocity = (
                old_weight * state.velocity[0] + new_weight * measured[0],
                old_weight * state.velocity[1] + new_weight * measured[1],
            )
            if detection.field_center is not None and state.field_center is not None:
                measured_field = (
                    (detection.field_center[0] - state.field_center[0]) / gap,
                    (detection.field_center[1] - state.field_center[1]) / gap,
                )
                state.field_velocity = (
                    old_weight * state.field_velocity[0] + new_weight * measured_field[0],
                    old_weight * state.field_velocity[1] + new_weight * measured_field[1],
                )

        descriptor_similarity = self._appearance(detection, state)
        eligible_template = (
            detection.descriptor is not None
            and descriptor_similarity is not None
            and descriptor_similarity >= self.config.template_update_similarity
            and assignment_score >= self.config.template_update_score
            and detection.confidence >= self.config.template_min_detection_confidence
        )
        if eligible_template:
            if state.pending_raw_id == detection.raw_track_id:
                state.pending_count += 1
            else:
                state.pending_raw_id = detection.raw_track_id
                state.pending_count = 1
            state.pending_template = detection.descriptor
            if state.pending_count >= self.config.template_confirmation_frames:
                # Store diverse, confirmed exemplars. Never blend an uncertain crop into every
                # future comparison; a single wrong association therefore cannot poison memory.
                if all(
                    appearance_similarity(detection.descriptor, template)
                    < self.config.template_distinct_similarity
                    for template in state.templates
                ):
                    state.templates.append(detection.descriptor)
                    del state.templates[:-self.config.template_gallery_size]
                state.pending_count = 0
                state.pending_template = None
        else:
            state.pending_raw_id = None
            state.pending_template = None
            state.pending_count = 0

        self._record_alliance_evidence(state, timestamp, detection, assignment_score)
        state.last_seen = timestamp
        state.center = detection.center
        state.field_center = detection.field_center
        state.camera_offset = self.camera_offset
        state.last_edge = detection.edge or self._edge(detection.center)
        state.last_raw_id = detection.raw_track_id

    def resolve_frame(
        self,
        timestamp: float,
        detections: list[ReIDDetection],
        camera_motion: tuple[float, float] | None = None,
    ) -> list[int | None]:
        """Resolve one complete frame with global assignment and explicit abstention."""

        if camera_motion is not None:
            self.camera_offset = (
                self.camera_offset[0] + camera_motion[0], self.camera_offset[1] + camera_motion[1]
            )
        # Expire only the raw-id shortcut. Long-lived state remains as immutable history and is
        # never silently reassigned, while re-identification candidates obey the memory window.
        for raw_id, stable_id in list(self.raw_to_stable.items()):
            state = self.states.get(stable_id)
            if state is None or timestamp - state.last_seen > self.config.memory_seconds:
                self.raw_to_stable.pop(raw_id, None)

        candidate_ids = [
            stable_id for stable_id, state in sorted(self.states.items())
            if 0.0 < timestamp - state.last_seen <= self.config.memory_seconds
        ]
        scores = [
            [
                self._pair_score(timestamp, detection, stable_id, self.states[stable_id])
                for stable_id in candidate_ids
            ]
            for detection in detections
        ]
        best_total, proposed = self._best_assignment(scores)
        accepted: dict[int, int] = {}
        for row, column in proposed.items():
            alternative_total, _ = self._best_assignment(scores, forbidden=(row, column))
            if best_total - alternative_total + 1e-12 >= self.config.score_margin:
                accepted[row] = column

        resolved: list[int | None] = [None] * len(detections)
        for row, column in accepted.items():
            stable_id = candidate_ids[column]
            resolved[row] = stable_id
            self.raw_to_stable[detections[row].raw_track_id] = stable_id
            self._update_state(stable_id, timestamp, detections[row], float(scores[row][column]))

        for row, detection in enumerate(detections):
            if resolved[row] is not None:
                continue
            # If any identity produced a viable score, the evidence was globally ambiguous. Keep
            # this raw tracklet tentative instead of consuming a robot slot or corrupting a track.
            if any(score is not None for score in scores[row]):
                self.raw_to_stable.pop(detection.raw_track_id, None)
                continue
            stable_id = self._available_identity()
            if stable_id is None:
                self.raw_to_stable.pop(detection.raw_track_id, None)
                continue
            resolved[row] = stable_id
            self.raw_to_stable[detection.raw_track_id] = stable_id
            self._update_state(stable_id, timestamp, detection, 1.0)
        return resolved

    def resolve(
        self,
        raw_track_id: int,
        timestamp: float,
        descriptor: list[float] | None,
        center: tuple[float, float],
        alliance: str | None,
        used_stable_ids: set[int] | None = None,
        present_raw_ids: set[int] | None = None,
    ) -> int | None:
        """Compatibility wrapper for callers that truly have only one detection."""

        del used_stable_ids, present_raw_ids
        return self.resolve_frame(
            timestamp,
            [ReIDDetection(raw_track_id, descriptor, center, alliance)],
        )[0]


def contract_track_records(
    tracks: dict[int, list[dict[str, object]]],
    fps: float,
    alliances: dict[int, str | None] | None = None,
    position_source: str | None = None,
    visible_track_ids: set[int] | None = None,
    raw_track_ids: dict[int, set[int]] | None = None,
) -> list[dict[str, object]]:
    if fps <= 0:
        raise ValueError("fps must be positive")
    frame_interval = 1.0 / fps
    records = []
    visible = visible_track_ids if visible_track_ids is not None else eligible_track_ids(tracks, alliances)
    for track_id, boxes in sorted(tracks.items()):
        if track_id not in visible or not 1 <= track_id <= MAX_ROBOTS:
            continue
        gaps = []
        for previous, current in zip(boxes, boxes[1:]):
            if current["t"] - previous["t"] > frame_interval * 1.5:
                near_edge = (
                    float(previous["x"]) <= 0.03
                    or float(previous["x"]) + float(previous["w"]) >= 0.97
                    or float(previous["y"]) <= 0.03
                    or float(previous["y"]) + float(previous["h"]) >= 0.97
                    or float(current["x"]) <= 0.03
                    or float(current["x"]) + float(current["w"]) >= 0.97
                    or float(current["y"]) <= 0.03
                    or float(current["y"]) + float(current["h"]) >= 0.97
                )
                gaps.append({
                    "start": round(previous["t"] + frame_interval, 6),
                    "end": round(current["t"] - frame_interval, 6),
                    "reason": "out_of_frame" if near_edge else "occlusion",
                })
        record: dict[str, object] = {
            "schema_version": 3,
            "track_id": track_id,
            "robot_name": f"robot{track_id}",
            "team": None,
            "alliance": (alliances or {}).get(track_id),
            "team_confidence": None,
            "boxes": boxes,
            "gaps": gaps,
        }
        if position_source:
            record["position_source"] = position_source
        if raw_track_ids is not None:
            record["raw_track_ids"] = sorted(raw_track_ids.get(track_id, set()))
        records.append(record)
    return records


def write_track_records(
    path: Path,
    tracks: dict[int, list[dict[str, object]]],
    fps: float,
    alliances: dict[int, str | None] | None = None,
    position_source: str | None = None,
    visible_track_ids: set[int] | None = None,
    raw_track_ids: dict[int, set[int]] | None = None,
) -> None:
    """Atomically publish the tracks accumulated so far for the live UI overlay."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in contract_track_records(
            tracks, fps, alliances, position_source, visible_track_ids, raw_track_ids
        ):
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    # The browser polls the partial file while the worker is writing it. Windows can briefly
    # keep that read handle open, so retry the atomic swap instead of aborting the whole run.
    for attempt in range(30):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 29:
                raise
            time.sleep(0.1)


def write_raw_tracklets(path: Path, raw_tracklets: dict[int, dict[str, object]]) -> None:
    """Atomically write the immutable tracker observations used by the identity layer."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for fragment_id in sorted(raw_tracklets):
            handle.write(json.dumps(raw_tracklets[fragment_id], sort_keys=True) + "\n")
    temporary.replace(path)


def bumper_color(image, raw_box: list[float]) -> str | None:
    """Infer red/blue from the lower part of a robot box; return None when ambiguous."""
    import cv2

    height, width = image.shape[:2]
    left, top, right, bottom = (int(round(value)) for value in raw_box)
    left = max(0, min(width - 1, left))
    right = max(left + 1, min(width, right))
    # Use the central lower bumper, not the full width of the box.  At the edge of a broadcast
    # crop a laptop, scoreboard, or field hardware can otherwise borrow red/blue pixels from the
    # adjacent overlay and be mistaken for a coloured robot.
    horizontal_margin = int((right - left) * 0.15)
    left = min(right - 1, left + horizontal_margin)
    right = max(left + 1, right - horizontal_margin)
    top = max(0, min(height - 1, top + int((bottom - top) * 0.60)))
    bottom = max(top + 1, min(height, bottom))
    region = image[top:bottom, left:right]
    if region.size == 0:
        return None

    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    vivid = (saturation > 75) & (value > 55)
    hue = hsv[:, :, 0]
    red = vivid & ((hue <= 10) | (hue >= 170))
    blue = vivid & (hue >= 95) & (hue <= 135)
    red_count = int(red.sum())
    blue_count = int(blue.sum())
    minimum = max(8, int(region.shape[0] * region.shape[1] * 0.015))
    if red_count >= minimum and red_count >= blue_count * 1.25:
        return "red"
    if blue_count >= minimum and blue_count >= red_count * 1.25:
        return "blue"
    return None


def resolved_alliances(votes: dict[int, dict[str, int]]) -> dict[int, str | None]:
    """Keep a bumper color when repeated evidence gives it a strict majority.

    A 60% cutoff was too conservative for broadcast compression and occlusion: robots that were
    visibly coloured in individual frames could end up colourless after a whole-match vote and
    then be removed by the stationary false-positive filter. Ties remain unknown.
    """
    resolved: dict[int, str | None] = {}
    for track_id, counts in votes.items():
        total = counts.get("red", 0) + counts.get("blue", 0)
        if total < 3:
            resolved[track_id] = None
            continue
        colour, count = max(counts.items(), key=lambda item: item[1])
        other = total - count
        resolved[track_id] = colour if count > other else None
    return resolved


def crop_bounds(width: int, height: int, crop: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    left, top, right, bottom = crop
    if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
        raise ValueError("crop must be normalized as left top right bottom within 0..1")
    x0 = max(0, min(width - 1, int(round(width * left))))
    y0 = max(0, min(height - 1, int(round(height * top))))
    x1 = max(x0 + 1, min(width, int(round(width * right))))
    y1 = max(y0 + 1, min(height, int(round(height * bottom))))
    # Keep the encoded model input compatible with common YUV 4:2:0 MP4 codecs.
    if (x1 - x0) % 2:
        x1 = x1 + 1 if x1 < width else x1 - 1
    if (y1 - y0) % 2:
        y1 = y1 + 1 if y1 < height else y1 - 1
    return x0, y0, x1, y1


def crop_box_to_source(
    box: tuple[float, float, float, float],
    crop: tuple[float, float, float, float],
    cropped_width: int,
    cropped_height: int,
) -> tuple[float, float, float, float]:
    """Convert a normalized box in the model crop back to full-source normalization."""

    left, top, right, bottom = crop
    crop_width = right - left
    crop_height = bottom - top
    x, y, width, height = box
    return (
        left + x * crop_width,
        top + y * crop_height,
        width * crop_width,
        height * crop_height,
    )


def edge_for_box(
    box: tuple[float, float, float, float], threshold: float = DEFAULT_REID_EDGE_THRESHOLD
) -> str | None:
    """Nearest frame edge touched by a normalized x/y/width/height box."""

    x, y, width, height = box
    distances = {
        "left": x,
        "right": 1.0 - (x + width),
        "top": y,
        "bottom": 1.0 - (y + height),
    }
    edge, distance = min(distances.items(), key=lambda item: item[1])
    return edge if distance <= threshold else None


def field_position_for_box(
    box: tuple[float, float, float, float],
    mapper,
    crop: tuple[float, float, float, float],
    cropped_width: int,
    cropped_height: int,
) -> tuple[float, float] | None:
    """Map a normalized model-crop box to its carpet footprint, when calibration is valid."""

    if mapper is None:
        return None
    try:
        source_box = crop_box_to_source(box, crop, cropped_width, cropped_height)
        image_width = cropped_width / (crop[2] - crop[0])
        image_height = cropped_height / (crop[3] - crop[1])
        field_x, field_y = mapper.box_to_field(*source_box, image_width, image_height)
        pixel_x = (source_box[0] + source_box[2] / 2.0) * image_width
        pixel_y = (source_box[1] + source_box[3]) * image_height
    except (KeyError, TypeError, ValueError):
        return None
    if not (math.isfinite(field_x) and math.isfinite(field_y)) or not mapper.on_field(pixel_x, pixel_y):
        return None
    return field_x, field_y


def add_field_motion(
    sample: dict[str, object],
    mapper,
    crop: tuple[float, float, float, float],
    cropped_width: int,
    cropped_height: int,
    history: dict[int, list[tuple[float, float, float]]],
    max_gap_seconds: float,
) -> None:
    """Attach carpet position and finite-difference motion to one tracked sample."""

    track_id = sample.get("track_id")
    if mapper is None or not isinstance(track_id, int):
        return
    bbox = sample.get("bbox_normalized")
    if not isinstance(bbox, dict):
        return
    position = field_position_for_box(
        (float(bbox["x"]), float(bbox["y"]), float(bbox["width"]), float(bbox["height"])),
        mapper,
        crop,
        cropped_width,
        cropped_height,
    )
    if position is None:
        return
    field_x, field_y = position

    timestamp = float(sample["t"])
    sample["field_x"] = round(field_x, 4)
    sample["field_y"] = round(field_y, 4)
    observations = history.setdefault(track_id, [])
    previous = observations[-1] if observations else None
    if previous is not None:
        previous_time, previous_x, previous_y = previous
        dt = timestamp - previous_time
        if 0.0 < dt <= max_gap_seconds:
            velocity_x = (field_x - previous_x) / dt
            velocity_y = (field_y - previous_y) / dt
            speed = math.hypot(velocity_x, velocity_y)
            if speed <= MAX_PLAUSIBLE_FTPS:
                sample["velocity_x_ftps"] = round(velocity_x, 4)
                sample["velocity_y_ftps"] = round(velocity_y, 4)
                sample["speed_ftps"] = round(speed, 4)
                sample["motion_heading_rad"] = round(math.atan2(velocity_y, velocity_x), 6)
            else:
                # Preserve the valid current position, but do not let a bad jump contaminate the
                # next finite difference either.
                observations.clear()
    observations.append((timestamp, field_x, field_y))
    if len(observations) > 4:
        del observations[:-4]


def build_cropped_video(source: Path, destination: Path, crop: tuple[float, float, float, float]) -> None:
    """Materialize the marked panel so Ultralytics never receives the lower broadcast panel."""
    import cv2

    capture = cv2.VideoCapture(str(source))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(f"Could not read video dimensions: {source}")
    x0, y0, x1, y1 = crop_bounds(width, height, crop)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.unlink(missing_ok=True)
    writer = cv2.VideoWriter(
        str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (x1 - x0, y1 - y0)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create cropped model input: {destination}")
    frames = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(frame[y0:y1, x0:x1])
            frames += 1
    finally:
        capture.release()
        writer.release()
    if frames == 0:
        raise RuntimeError(f"Cropped model input contains no frames: {destination}")


STREAM_SELECTOR = "bv[height<=1080][ext=mp4][vcodec^=avc1]/bv[height<=1080][ext=mp4]/bv[height<=1080]"


def stream_source_fps(url: str) -> float:
    """Read stream metadata without fetching media."""
    import yt_dlp

    with yt_dlp.YoutubeDL({
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "skip_download": True,
        "format": STREAM_SELECTOR,
    }) as ydl:
        info = ydl.extract_info(url, download=False)
    return float(info.get("fps") or 30.0)


def stream_source_frames(url: str):
    """Yield full BGR frames from a yt-dlp/FFmpeg pipe without saving the source video."""
    try:
        import av
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError("The stream mode requires PyAV and yt-dlp in the vision environment") from exc

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg must be on PATH for YouTube stream mode")

    # A metadata-only yt-dlp call gives the source frame rate for snapshot timestamps. The
    # media itself is fetched by the separate stdout pipe below.
    # DASH MP4 is not seekable when sent to stdout. FFmpeg reads that pipe and emits a
    # streamable Matroska pipe; MJPEG keeps the decoder path broadly compatible with the
    # installed OpenCV/Ultralytics stack while still never touching disk.
    yt_process = subprocess.Popen(
        [
            sys.executable, "-m", "yt_dlp", "--quiet", "--no-warnings", "--no-playlist",
            "--format", STREAM_SELECTOR, "--output", "-", url,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert yt_process.stdout is not None
    ffmpeg_process = subprocess.Popen(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
            "-map", "0:v:0", "-an", "-c:v", "mjpeg", "-f", "matroska", "pipe:1",
        ],
        stdin=yt_process.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    yt_process.stdout.close()
    assert ffmpeg_process.stdout is not None
    try:
        with av.open(ffmpeg_process.stdout, mode="r", format="matroska") as container:
            for frame in container.decode(video=0):
                yield frame.to_ndarray(format="bgr24")
    finally:
        if ffmpeg_process.poll() is None:
            ffmpeg_process.terminate()
        if yt_process.poll() is None:
            yt_process.terminate()
        ffmpeg_stderr = ffmpeg_process.stderr.read().decode("utf-8", errors="replace") if ffmpeg_process.stderr else ""
        yt_stderr = yt_process.stderr.read().decode("utf-8", errors="replace") if yt_process.stderr else ""
        ffmpeg_process.wait(timeout=30)
        yt_process.wait(timeout=30)
        if ffmpeg_process.returncode not in (0, -15, 15) and ffmpeg_stderr.strip():
            raise RuntimeError(f"ffmpeg stream failed: {ffmpeg_stderr.strip()[-1000:]}")
        if yt_process.returncode not in (0, -15, 15) and yt_stderr.strip():
            raise RuntimeError(f"yt-dlp stream failed: {yt_stderr.strip()[-1000:]}")


def stream_cropped_frames(url: str, crop: tuple[float, float, float, float]):
    """Yield cropped BGR frames from yt-dlp/FFmpeg pipes without saving the source video."""

    for image in stream_source_frames(url):
        height, width = image.shape[:2]
        x0, y0, x1, y1 = crop_bounds(width, height, crop)
        yield image[y0:y1, x0:x1]


def stream_homography(
    url: str,
    layout_path: Path,
    *,
    hfov_deg: float = 70.0,
    samples: int = 12,
    max_frames: int = 900,
    region: tuple[float, float] = (0.0, 0.68),
) -> dict[str, object] | None:
    """Fit a carpet mapping from AprilTags while keeping the source stream in memory only."""

    from ingest.collection.apriltag_layout import load_layout
    from ingest.collection.calibrate import UPSCALE, TagSighting, build_detector, steady_tags
    from ingest.collection import homography as homography_module
    import cv2

    layout = load_layout(layout_path)
    target_indices = {
        int(round(max_frames * (0.08 + 0.84 * i / max(1, samples - 1))))
        for i in range(samples)
    }
    detector = build_detector()
    sightings: dict[int, TagSighting] = {}
    frames_read = 0
    width = height = 0
    for index, frame in enumerate(stream_source_frames(url)):
        if index > max(target_indices):
            break
        if index not in target_indices:
            continue
        frames_read += 1
        height, width = frame.shape[:2]
        region_top = max(0, min(height - 1, int(round(region[0] * height))))
        region_bottom = max(region_top + 1, min(height, int(round(region[1] * height))))
        grey = cv2.cvtColor(frame[region_top:region_bottom], cv2.COLOR_BGR2GRAY)
        image = cv2.resize(grey, None, fx=UPSCALE, fy=UPSCALE,
                           interpolation=cv2.INTER_CUBIC)
        corners, ids, _ = detector.detectMarkers(image)
        for corner, tag_id in zip(corners, (ids.flatten() if ids is not None else [])):
            centre = corner.reshape(4, 2).mean(axis=0) / UPSCALE
            centre[1] += region_top
            seen = sightings.setdefault(int(tag_id), TagSighting(int(tag_id)))
            seen.xs.append(float(centre[0]))
            seen.ys.append(float(centre[1]))

    observed, _notes = steady_tags(sightings)
    used_ids = sorted(tag_id for tag_id in observed if tag_id in layout.tags)
    image_points = [observed[tag_id] for tag_id in used_ids]
    field_points_3d = [
        (layout.tags[tag_id].x_ft, layout.tags[tag_id].y_ft, layout.tags[tag_id].z_ft)
        for tag_id in used_ids
    ]
    if width <= 0 or height <= 0 or len(image_points) < 4:
        return None

    low = max(25.0, float(hfov_deg) - 30.0)
    high = min(120.0, float(hfov_deg) + 30.0)
    candidates = []
    for candidate in [float(hfov_deg), *(low + 2.0 * i for i in range(int((high - low) / 2.0) + 1))]:
        attempt = homography_module.solve_camera_pose(
            image_points, field_points_3d, width, height, candidate,
            layout.length_ft, layout.width_ft,
        )
        if attempt is not None:
            candidates.append(attempt)
    if not candidates:
        return None
    mapper, pose = min(candidates, key=lambda item: item[1]["reprojection_px"])
    best_hfov = float(pose["hfov_deg"])
    refinements = []
    for offset in range(-8, 9):
        attempt = homography_module.solve_camera_pose(
            image_points, field_points_3d, width, height, best_hfov + offset * 0.25,
            layout.length_ft, layout.width_ft,
        )
        if attempt is not None:
            refinements.append(attempt)
    if refinements:
        mapper, pose = min(refinements, key=lambda item: item[1]["reprojection_px"])
    return {
        "mapping_source": "carpet_pose",
        "field_length_ft": layout.length_ft,
        "field_width_ft": layout.width_ft,
        "plane_height_ft": 0.0,
        "point_count": len(image_points),
        "has_redundancy": len(image_points) >= 7,
        "trustworthy": bool(mapper.trustworthy),
        "tags_used": used_ids,
        "frames_sampled": frames_read,
        "matrix": mapper.matrix,
        "pose": pose,
    }


def colored_detections(result, stable_ids=None, visible_track_ids: set[int] | None = None):
    """Draw only public detections, using stable ids and inferred bumper colours."""
    import cv2

    source_image = result.orig_img
    image = source_image.copy()
    if result.boxes is None:
        return image
    boxes = result.boxes.xyxy.detach().cpu().tolist()
    confidences = result.boxes.conf.detach().cpu().tolist()
    identifiers = (
        result.boxes.id.detach().cpu().tolist()
        if result.boxes.id is not None else [None] * len(boxes)
    )
    if stable_ids is not None:
        identifiers = stable_ids
    palette = {
        "red": (55, 75, 225),
        "blue": (225, 120, 55),
        None: (155, 155, 155),
    }
    for raw_box, confidence, identifier in zip(boxes, confidences, identifiers):
        if identifier is None:
            continue
        stable_id = int(identifier)
        if visible_track_ids is not None and stable_id not in visible_track_ids:
            continue
        left, top, right, bottom = (int(round(value)) for value in raw_box)
        alliance = bumper_color(source_image, raw_box)
        colour = palette[alliance]
        cv2.rectangle(image, (left, top), (right, bottom), colour, 3)
        label = "robot"
        label += f" id:{stable_id}"
        label += f" {float(confidence):.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
        )
        label_top = max(0, top - text_height - baseline - 5)
        cv2.rectangle(
            image,
            (left, label_top),
            (min(image.shape[1], left + text_width + 10), top),
            colour,
            -1,
        )
        cv2.putText(
            image,
            label,
            (left + 5, max(text_height, top - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
    return image


def detection_records(
    result,
    stable_ids=None,
    visible_track_ids: set[int] | None = None,
) -> list[dict[str, object]]:
    """Return metadata for public detections on one clean source frame."""
    if result.boxes is None:
        return []
    image = result.orig_img
    height, width = image.shape[:2]
    boxes = result.boxes.xyxy.detach().cpu().tolist()
    confidences = result.boxes.conf.detach().cpu().tolist()
    identifiers = (
        result.boxes.id.detach().cpu().tolist()
        if result.boxes.id is not None else [None] * len(boxes)
    )
    if stable_ids is not None:
        identifiers = stable_ids
    records: list[dict[str, object]] = []
    for raw_box, confidence, identifier in zip(boxes, confidences, identifiers):
        if identifier is None:
            continue
        stable_id = int(identifier)
        if visible_track_ids is not None and stable_id not in visible_track_ids:
            continue
        left, top, right, bottom = (float(value) for value in raw_box)
        alliance = bumper_color(image, raw_box)
        records.append({
            "label": "robot",
            "track_id": stable_id,
            "confidence": round(float(confidence), 6),
            "alliance": alliance,
            "bbox_xyxy": [
                int(round(left)), int(round(top)), int(round(right)), int(round(bottom)),
            ],
            "bbox_normalized": {
                "x": round(left / width, 6),
                "y": round(top / height, 6),
                "width": round((right - left) / width, 6),
                "height": round((bottom - top) / height, 6),
            },
        })
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--homography", help="calibration JSON for carpet positions and speed")
    parser.add_argument(
        "--auto-homography",
        action="store_true",
        help="fit a carpet mapping from AprilTags in the stream without writing media to disk",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", help="local video file")
    source.add_argument("--stream-url", help="YouTube URL; media is consumed as a pipe")
    parser.add_argument("--output", required=True, help="JSONL track samples")
    parser.add_argument("--tracker", choices=("bytetrack", "botsort"), default="bytetrack")
    parser.add_argument("--confidence", type=float, default=0.20)
    parser.add_argument("--image-size", type=int, default=960)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--reid-memory-seconds",
        type=float,
        default=DEFAULT_REID_MEMORY_SECONDS,
        help="seconds a lost robot remains eligible for appearance re-identification",
    )
    parser.add_argument(
        "--reid-appearance-threshold",
        type=float,
        default=DEFAULT_REID_APPEARANCE_THRESHOLD,
        help="minimum cosine appearance similarity for stitching a new raw id",
    )
    parser.add_argument(
        "--reid-max-distance",
        type=float,
        default=DEFAULT_REID_MAX_CENTER_DISTANCE,
        help="maximum normalized image-centre travel during a re-identification gap",
    )
    parser.add_argument(
        "--reid-score-margin", type=float, default=DEFAULT_REID_SCORE_MARGIN,
        help="minimum global-score loss required before a proposed merge is accepted",
    )
    parser.add_argument(
        "--reid-max-speed", type=float, default=DEFAULT_REID_MAX_NORMALIZED_SPEED,
        help="maximum normalized image travel per second when field calibration is unavailable",
    )
    parser.add_argument(
        "--reid-edge-threshold", type=float, default=DEFAULT_REID_EDGE_THRESHOLD,
        help="normalized boundary band used for edge-consistent exits and re-entry",
    )
    parser.add_argument(
        "--reid-template-gallery-size", type=int, default=DEFAULT_REID_TEMPLATE_GALLERY_SIZE,
        help="maximum confirmed appearance exemplars retained per robot",
    )
    parser.add_argument(
        "--reid-template-confirmation-frames",
        type=int,
        default=DEFAULT_REID_TEMPLATE_CONFIRMATION_FRAMES,
        help="consecutive accepted frames before a new appearance exemplar is stored",
    )
    parser.add_argument(
        "--reid-template-min-confidence",
        type=float,
        default=DEFAULT_REID_TEMPLATE_MIN_CONFIDENCE,
        help="minimum detector confidence for adding a confirmed appearance exemplar",
    )
    parser.add_argument(
        "--reid-alliance-lock-seconds",
        type=float,
        default=DEFAULT_REID_ALLIANCE_LOCK_SECONDS,
        help="trusted same-alliance observation time required to lock a robot identity",
    )
    parser.add_argument(
        "--reid-alliance-lock-margin-seconds",
        type=float,
        default=DEFAULT_REID_ALLIANCE_LOCK_MARGIN_SECONDS,
        help="required evidence-time lead over the opposite alliance before locking",
    )
    parser.add_argument(
        "--startup-position-seconds",
        type=float,
        default=DEFAULT_STARTUP_POSITION_SECONDS,
        help="opening duration where left is blue and right is red; zero disables the prior",
    )
    parser.add_argument(
        "--startup-split-x",
        type=float,
        default=DEFAULT_STARTUP_SPLIT_X,
        help="normalized image x coordinate separating the blue-left and red-right starts",
    )
    parser.add_argument(
        "--raw-output",
        help="immutable raw tracker-tracklet JSONL (default: <output stem>.raw.jsonl)",
    )
    parser.add_argument("--annotated-output", help="optional MP4 preview")
    parser.add_argument(
        "--snapshot-dir",
        help="directory for clean JPEG snapshots and one annotations.json file",
    )
    parser.add_argument(
        "--snapshot-interval",
        type=float,
        default=5.0,
        help="seconds between clean snapshots (default: 5)",
    )
    parser.add_argument(
        "--partial-output",
        help="optional JSONL file for live, partial track output",
    )
    parser.add_argument(
        "--crop",
        nargs=4,
        type=float,
        default=DEFAULT_CROP,
        metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
        help="normalized crop rectangle; defaults to the marked upper broadcast panel",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path, output = Path(args.model), Path(args.output)
    raw_output = (
        Path(args.raw_output)
        if args.raw_output
        else output.with_name(f"{output.stem}.raw.jsonl")
    )
    video_path = Path(args.video) if args.video else None
    if not model_path.is_file():
        raise SystemExit("--model must name an existing file")
    if video_path is not None and not video_path.is_file():
        raise SystemExit("--video must name an existing file")
    if output.exists():
        raise SystemExit(f"Refusing to overwrite track output: {output}")
    if raw_output.exists():
        raise SystemExit(f"Refusing to overwrite raw tracker output: {raw_output}")
    try:
        import cv2
        from ultralytics import YOLO
        from ultralytics.data.build import SourceTypes
        from ultralytics.data.loaders import LoadPilAndNumpy
    except ImportError as exc:
        raise SystemExit("Install training/requirements-yolo.txt in the dedicated vision venv") from exc
    if args.snapshot_interval <= 0:
        raise SystemExit("--snapshot-interval must be greater than zero")
    if args.reid_memory_seconds <= 0:
        raise SystemExit("--reid-memory-seconds must be greater than zero")
    if not 0.0 <= args.reid_appearance_threshold <= 1.0:
        raise SystemExit("--reid-appearance-threshold must be between zero and one")
    if args.reid_max_distance <= 0:
        raise SystemExit("--reid-max-distance must be greater than zero")
    if args.reid_score_margin < 0:
        raise SystemExit("--reid-score-margin cannot be negative")
    if args.reid_max_speed <= 0:
        raise SystemExit("--reid-max-speed must be greater than zero")
    if not 0.0 < args.reid_edge_threshold < 0.5:
        raise SystemExit("--reid-edge-threshold must be between zero and 0.5")
    if args.reid_template_gallery_size <= 0:
        raise SystemExit("--reid-template-gallery-size must be greater than zero")
    if args.reid_template_confirmation_frames <= 0:
        raise SystemExit("--reid-template-confirmation-frames must be greater than zero")
    if not 0.0 <= args.reid_template_min_confidence <= 1.0:
        raise SystemExit("--reid-template-min-confidence must be between zero and one")
    if args.reid_alliance_lock_seconds <= 0:
        raise SystemExit("--reid-alliance-lock-seconds must be greater than zero")
    if args.reid_alliance_lock_margin_seconds < 0:
        raise SystemExit("--reid-alliance-lock-margin-seconds cannot be negative")
    if args.startup_position_seconds < 0:
        raise SystemExit("--startup-position-seconds cannot be negative")
    if not 0.0 < args.startup_split_x < 1.0:
        raise SystemExit("--startup-split-x must be between zero and one")
    output.parent.mkdir(parents=True, exist_ok=True)
    homography_path = args.homography
    if args.stream_url and args.auto_homography and not homography_path:
        diagnostics_path = output.parent / "homography.diagnostics.json"
        try:
            from ingest.collection.apriltag_layout import load_layout

            layout_path = Path(__file__).resolve().parent.parent / "contracts" / "fields" / "2026-apriltags.json"
            calibration = stream_homography(args.stream_url, layout_path)
            diagnostics_path.write_text(
                json.dumps(calibration or {"solution": None}, indent=2) + "\n",
                encoding="utf-8",
            )
            if calibration and calibration.get("trustworthy") and calibration.get("matrix"):
                homography_path = str(output.parent / "homography.json")
                Path(homography_path).write_text(
                    json.dumps(calibration, indent=2) + "\n", encoding="utf-8"
                )
        except Exception as exc:
            diagnostics_path.write_text(
                json.dumps({"error": str(exc)}, indent=2) + "\n", encoding="utf-8"
            )

    mapper = None
    position_source = None
    if homography_path:
        try:
            from ingest.collection.homography import load_calibration
            mapper = load_calibration(homography_path)
        except (ImportError, OSError, ValueError) as exc:
            raise SystemExit(f"Could not load homography {homography_path}: {exc}") from exc
        if mapper is None:
            raise SystemExit(f"Homography is missing, malformed, or untrustworthy: {homography_path}")
        position_source = mapper.source

    if args.stream_url:
        fps = stream_source_fps(args.stream_url)
        total_frames = 0
        source_name = args.stream_url

        class YtFrameLoader(LoadPilAndNumpy):
            """Ultralytics-compatible lazy loader around the yt-dlp/FFmpeg frame pipe."""

            def __init__(self, frames):
                self.frames = iter(frames)
                self.mode = "stream"
                self.bs = 1
                self.count = 0
                self.source_type = SourceTypes(stream=True, screenshot=False, from_img=False, tensor=False)

            def __len__(self):
                return 0

            def __iter__(self):
                return self

            def __next__(self):
                frame = next(self.frames)
                self.count += 1
                return [f"youtube_stream_{self.count:08d}.jpg"], [frame], [""]

        source_frames = YtFrameLoader(stream_cropped_frames(args.stream_url, tuple(args.crop)))
    else:
        assert video_path is not None
        source_name = str(video_path)

        class LocalFrameLoader(LoadPilAndNumpy):
            """Lazy OpenCV source that crops each frame as YOLO requests it."""

            def __init__(self, path: Path, crop):
                self.capture = cv2.VideoCapture(str(path))
                if not self.capture.isOpened():
                    raise SystemExit(f"Could not open video: {path}")
                self.fps = float(self.capture.get(cv2.CAP_PROP_FPS) or 30.0)
                self.total_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                self.crop = tuple(crop)
                self.mode = "stream"
                self.bs = 1
                self.count = 0
                self.source_type = SourceTypes(stream=True, screenshot=False, from_img=False, tensor=False)

            def __len__(self):
                return 0

            def __iter__(self):
                return self

            def __next__(self):
                ok, frame = self.capture.read()
                if not ok:
                    self.capture.release()
                    raise StopIteration
                height, width = frame.shape[:2]
                x0, y0, x1, y1 = crop_bounds(width, height, self.crop)
                self.count += 1
                return [f"video_stream_{self.count:08d}.jpg"], [frame[y0:y1, x0:x1]], [""]

            def close(self):
                self.capture.release()

        source_frames = LocalFrameLoader(video_path, tuple(args.crop))
        fps = source_frames.fps
        total_frames = source_frames.total_frames
    model = YOLO(str(model_path))
    tracker_config = Path(__file__).resolve().parent / "trackers" / f"{args.tracker}.yaml"
    if not tracker_config.is_file():
        raise SystemExit(f"Missing persistent tracker configuration: {tracker_config}")
    results = model.track(
        source=source_frames, stream=True, persist=True, tracker=str(tracker_config),
        conf=args.confidence, imgsz=args.image_size, device=args.device, verbose=False,
    )
    tracks: dict[int, list[dict[str, object]]] = {}
    alliance_votes: dict[int, dict[str, int]] = {}
    motion_history: dict[int, list[tuple[float, float, float]]] = {}
    image_motion_windows: dict[int, list[dict[str, object]]] = {}
    last_motion_at: dict[int, float] = {}
    suppressed_stationary_ids: set[int] = set()
    reid_config = ReIDConfig(
        memory_seconds=args.reid_memory_seconds,
        appearance_threshold=args.reid_appearance_threshold,
        max_center_distance=args.reid_max_distance,
        score_margin=args.reid_score_margin,
        max_normalized_speed=args.reid_max_speed,
        edge_threshold=args.reid_edge_threshold,
        template_gallery_size=args.reid_template_gallery_size,
        template_confirmation_frames=args.reid_template_confirmation_frames,
        template_min_detection_confidence=args.reid_template_min_confidence,
        alliance_lock_seconds=args.reid_alliance_lock_seconds,
        alliance_lock_margin_seconds=args.reid_alliance_lock_margin_seconds,
        startup_position_seconds=args.startup_position_seconds,
        startup_split_x=args.startup_split_x,
    )
    track_memory = AppearanceTrackMemory(config=reid_config)
    raw_tracklets: dict[int, dict[str, object]] = {}
    raw_sessions: dict[int, tuple[int, float, int | None]] = {}
    next_raw_fragment_id = 1
    stable_to_raw: dict[int, set[int]] = {}
    writer = None
    annotated_path = Path(args.annotated_output) if args.annotated_output else None
    if annotated_path:
        annotated_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_dir = Path(args.snapshot_dir) if args.snapshot_dir else None
    if snapshot_dir:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
    next_snapshot = 0.0
    snapshot_annotations: list[dict[str, object]] = []
    partial_output = Path(args.partial_output) if args.partial_output else None
    # Publish immediately and then about once per source second. This keeps the UI useful for
    # both finite files and never-ending streams without rewriting the complete output every
    # frame.
    publish_interval = max(1, int(round(fps)))
    for frame_index, result in enumerate(results):
        timestamp = round(frame_index / fps, 6)
        stable_ids: list[int | None] = []
        if result.boxes is not None:
            xyxy = result.boxes.xyxy.detach().cpu().tolist()
            identifiers = (
                result.boxes.id.detach().cpu().tolist()
                if result.boxes.id is not None else [None] * len(xyxy)
            )
            confidences = result.boxes.conf.detach().cpu().tolist()
            height, width = result.orig_img.shape[:2]
            stable_ids = [None] * len(xyxy)
            detection_indices: list[int] = []
            reid_detections: list[ReIDDetection] = []
            frame_metadata: dict[int, tuple] = {}
            for detection_index, (raw_box, identifier, confidence) in enumerate(
                zip(xyxy, identifiers, confidences)
            ):
                left, top, right, bottom = (float(value) for value in raw_box)
                if identifier is None:
                    continue
                normalized_box = (
                    left / width,
                    top / height,
                    (right - left) / width,
                    (bottom - top) / height,
                )
                center = (
                    ((left + right) / 2.0) / width,
                    ((top + bottom) / 2.0) / height,
                )
                detected_alliance = bumper_color(result.orig_img, raw_box)
                alliance = startup_alliance(
                    detected_alliance,
                    center[0],
                    timestamp,
                    reid_config.startup_position_seconds,
                    reid_config.startup_split_x,
                )
                raw_track_id = int(identifier)
                detection_indices.append(detection_index)
                reid_detections.append(ReIDDetection(
                    raw_track_id=raw_track_id,
                    descriptor=appearance_descriptor(result.orig_img, raw_box),
                    center=center,
                    alliance=alliance,
                    edge=edge_for_box(normalized_box, reid_config.edge_threshold),
                    field_center=field_position_for_box(
                        normalized_box, mapper, tuple(args.crop), int(width), int(height)
                    ),
                    confidence=float(confidence),
                    alliance_is_authoritative=(
                        reid_config.startup_position_seconds > 0.0
                        and 0.0 <= timestamp < reid_config.startup_position_seconds
                    ),
                ))
                frame_metadata[detection_index] = (
                    raw_track_id,
                    detected_alliance,
                    alliance,
                    float(confidence),
                    left,
                    top,
                    right,
                    bottom,
                    normalized_box,
                )

            resolved_frame = track_memory.resolve_frame(timestamp, reid_detections)
            for detection_index, track_id in zip(detection_indices, resolved_frame):
                stable_ids[detection_index] = track_id
                (
                    raw_track_id,
                    detected_alliance,
                    alliance,
                    confidence,
                    left,
                    top,
                    right,
                    bottom,
                    normalized_box,
                ) = frame_metadata[detection_index]
                session = raw_sessions.get(raw_track_id)
                if (
                    session is None
                    or timestamp - session[1] > reid_config.memory_seconds
                    or (session[2] is not None and track_id != session[2])
                ):
                    fragment_id = next_raw_fragment_id
                    next_raw_fragment_id += 1
                    raw_tracklets[fragment_id] = {
                        "schema_version": 1,
                        "fragment_id": fragment_id,
                        "raw_track_id": raw_track_id,
                        "boxes": [],
                    }
                else:
                    fragment_id = session[0]
                raw_sessions[raw_track_id] = (fragment_id, timestamp, track_id)
                raw_tracklets[fragment_id]["boxes"].append({
                    "t": timestamp,
                    "x": round(normalized_box[0], 6),
                    "y": round(normalized_box[1], 6),
                    "w": round(normalized_box[2], 6),
                    "h": round(normalized_box[3], 6),
                    "confidence": round(confidence, 6),
                    "alliance": detected_alliance,
                    "effective_alliance": alliance,
                    "stable_track_id": track_id,
                })
                if track_id is None:
                    continue
                stable_to_raw.setdefault(track_id, set()).add(raw_track_id)
                if alliance:
                    counts = alliance_votes.setdefault(track_id, {"red": 0, "blue": 0})
                    counts[alliance] += 1
                sample: dict[str, object] = {
                    "t": timestamp,
                    "x": round(normalized_box[0], 6), "y": round(normalized_box[1], 6),
                    "w": round(normalized_box[2], 6),
                    "h": round(normalized_box[3], 6),
                    "track_id": track_id,
                    "bbox_normalized": {
                        "x": round(normalized_box[0], 6), "y": round(normalized_box[1], 6),
                        "width": round(normalized_box[2], 6),
                        "height": round(normalized_box[3], 6),
                    },
                }
                motion_sample = {
                    "x": sample["x"], "y": sample["y"],
                    "w": sample["w"], "h": sample["h"],
                }
                motion_window = image_motion_windows.setdefault(track_id, [])
                motion_window.append(motion_sample)
                motion_window[:] = motion_window[-max(2, int(round(fps * 1.25))):]
                last_motion_at.setdefault(track_id, timestamp)
                if image_window_has_motion(motion_window):
                    last_motion_at[track_id] = timestamp
                if (
                    timestamp - last_motion_at[track_id] >= STATIONARY_SUPPRESSION_SECONDS
                ):
                    suppressed_stationary_ids.add(track_id)
                add_field_motion(
                    sample,
                    mapper,
                    tuple(args.crop),
                    int(width),
                    int(height),
                    motion_history,
                    max(1.5 / fps, 1.0),
                )
                sample.pop("track_id", None)
                sample.pop("bbox_normalized", None)
                tracks.setdefault(track_id, []).append(sample)
        alliances = resolved_alliances(alliance_votes)
        visible_track_ids = eligible_track_ids(tracks, alliances, suppressed_stationary_ids)
        if snapshot_dir and timestamp + (0.5 / fps) >= next_snapshot:
            clean_frame = result.orig_img.copy()
            snapshot_name = f"snapshot_{int(round(next_snapshot)):06d}s.jpg"
            snapshot_path = snapshot_dir / snapshot_name
            if not cv2.imwrite(str(snapshot_path), clean_frame, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                raise RuntimeError(f"Could not write YOLO snapshot: {snapshot_path}")
            snapshot_annotations.append({
                "image": snapshot_name,
                "timestamp_seconds": round(next_snapshot, 3),
                "frame_index": frame_index,
                "width": int(clean_frame.shape[1]),
                "height": int(clean_frame.shape[0]),
                "boxes": detection_records(result, stable_ids, visible_track_ids),
            })
            next_snapshot += args.snapshot_interval
        if annotated_path:
            plotted = colored_detections(result, stable_ids, visible_track_ids)
            if writer is None:
                height, width = plotted.shape[:2]
                writer = cv2.VideoWriter(
                    str(annotated_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
                )
            writer.write(plotted)
        if frame_index == 0 or (frame_index + 1) % publish_interval == 0:
            fraction = min(1.0, (frame_index + 1) / total_frames) if total_frames else None
            if partial_output:
                write_track_records(
                    partial_output,
                    tracks,
                    fps,
                    alliances,
                    position_source,
                    visible_track_ids,
                    stable_to_raw,
                )
            print(json.dumps({
                "progress": round(0.10 + fraction * 0.80, 6) if fraction is not None else None,
                "stage": "tracking",
            }), flush=True)
    if writer is not None:
        writer.release()
    alliances = resolved_alliances(alliance_votes)
    write_track_records(
        output, tracks, fps, alliances, position_source,
        eligible_track_ids(tracks, alliances, suppressed_stationary_ids),
        stable_to_raw,
    )
    write_raw_tracklets(raw_output, raw_tracklets)
    if snapshot_dir:
        (snapshot_dir / "annotations.json").write_text(
            json.dumps({
                "schema_version": 1,
                "source_video": source_name,
                "crop": {
                    "left": args.crop[0],
                    "top": args.crop[1],
                    "right": args.crop[2],
                    "bottom": args.crop[3],
                },
                "interval_seconds": args.snapshot_interval,
                "images": snapshot_annotations,
            }, indent=2) + "\n",
            encoding="utf-8",
        )
    if hasattr(source_frames, "close"):
        source_frames.close()
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
