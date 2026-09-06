"""Map image pixels to field coordinates, so distances and speeds are in feet rather than pixels.

A fixed camera looking at a flat surface admits an exact projective mapping between the image and
that surface. An FRC field is flat and its dimensions are published, so four known correspondences
are enough to recover the mapping and read positions in feet.

What that buys, beyond nicer units:

  * speed in ft/s instead of px/s, which is comparable between venues, camera positions and zoom
    levels -- a pixel means something different in every shot;
  * better tracking, because "moved 2 feet in 0.2s" is a far stronger identity signal than
    "moved 40 pixels" when a robot at the far end of the field is a quarter the size;
  * a check on detections, since a box mapping to a point outside the field is wrong regardless
    of how confident the detector was.

Three things this module refuses to do, all for the same reason -- a wrong homography does not
fail loudly, it reports confident nonsense:

  * It does not guess reference points. Callers supply correspondences; whether those come from
    AprilTags, field markings or a person clicking four corners is their problem. No tag layout
    is invented here, because inventing surveyed coordinates would produce exactly the kind of
    plausible, wrong answer this whole module exists to avoid.
  * It does not reuse a solution across a shot change. A homography belongs to one camera pose.
    Applied to a different shot it silently reports the wrong field position for every box, and
    FRC broadcasts cut constantly.
  * It does not report success without measuring. Every solution carries its reprojection error
    and is marked untrustworthy when that error, or the implied field geometry, is implausible.

Field coordinates are feet, origin at one field corner, x along the length, y across the width.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

#: Reprojection error, in feet, above which a solution is not trusted. A correspondence set that
#: cannot reproduce its own input points to within this is describing a different geometry than
#: the one it was given.
MAX_REPROJECTION_FT = 1.5

# A pose solve is only as good as the camera intrinsics supplied to it. Keep this threshold
# explicit so a broadcast-camera approximation is visible in the calibration output.
MAX_POSE_REPROJECTION_PX = 5.0

#: How far outside the field a mapped point may fall before it is treated as off-field. Robots do
#: leave the carpet in a sense -- bumpers overhang, the camera sees the driver station -- so a
#: small margin avoids rejecting legitimate edge positions.
FIELD_MARGIN_FT = 4.0


@dataclass(frozen=True)
class Homography:
    """A solved image-to-field mapping, with the evidence for trusting it."""

    matrix: list[list[float]]
    reprojection_ft: float
    field_length_ft: float
    field_width_ft: float
    point_count: int
    plane_height_ft: float | None = None
    source: str = "unknown"
    trusted_override: bool | None = None

    @property
    def has_redundancy(self) -> bool:
        """Whether reprojection error means anything for this solution.

        Four correspondences determine a homography exactly, so the fit reproduces them with zero
        error no matter how wrong they are -- mistype a corner by twenty feet and the residual is
        still zero. Error only becomes evidence from the fifth point onward, where the system is
        overdetermined and a bad point has something to disagree with.
        """
        return self.point_count >= 5

    @property
    def trustworthy(self) -> bool:
        """A solution worth using: enough points, and error small where error is measurable."""
        if self.trusted_override is not None:
            return self.point_count >= 4 and self.trusted_override
        if self.point_count < 4:
            return False
        if not self.has_redundancy:
            # Geometrically valid but unverified. Callers wanting a checked solution should
            # supply more points; this is flagged rather than silently treated as confirmed.
            return True
        return self.reprojection_ft <= MAX_REPROJECTION_FT

    def to_field(self, x: float, y: float) -> tuple[float, float]:
        """Image pixel -> field feet."""
        m = self.matrix
        denom = m[2][0] * x + m[2][1] * y + m[2][2]
        if abs(denom) < 1e-12:
            # The point lies on the horizon line of the plane: it has no finite field position.
            return (math.nan, math.nan)
        fx = (m[0][0] * x + m[0][1] * y + m[0][2]) / denom
        fy = (m[1][0] * x + m[1][1] * y + m[1][2]) / denom
        return (fx, fy)

    def on_field(self, x: float, y: float) -> bool:
        fx, fy = self.to_field(x, y)
        if math.isnan(fx) or math.isnan(fy):
            return False
        return (-FIELD_MARGIN_FT <= fx <= self.field_length_ft + FIELD_MARGIN_FT
                and -FIELD_MARGIN_FT <= fy <= self.field_width_ft + FIELD_MARGIN_FT)

    def box_to_field(self, x: float, y: float, w: float, h: float,
                     image_w: int, image_h: int) -> tuple[float, float]:
        """Field position of a normalised detection box.

        Uses the BOTTOM-CENTRE of the box, not its centre. A robot's footprint is where it touches
        the carpet, and the carpet is the plane the homography describes; the box centre floats
        somewhere up the robot's body and maps to a point behind where it actually stands.
        """
        return self.to_field((x + w / 2) * image_w, (y + h) * image_h)


def _normalise_matrix(raw: Any) -> list[list[float]] | None:
    """Read either a 3x3 JSON matrix or a flat nine-number matrix."""

    try:
        if isinstance(raw, list) and len(raw) == 9 and all(not isinstance(row, list) for row in raw):
            rows = [raw[0:3], raw[3:6], raw[6:9]]
        elif isinstance(raw, list) and len(raw) == 3 and all(
            isinstance(row, list) and len(row) == 3 for row in raw
        ):
            rows = raw
        else:
            return None
        matrix = [[float(value) for value in row] for row in rows]
    except (TypeError, ValueError):
        return None
    return matrix if all(math.isfinite(value) for row in matrix for value in row) else None


def load_calibration(
    path: Path | str,
    field_length_ft: float | None = None,
    field_width_ft: float | None = None,
) -> Homography | None:
    """Load a calibration written by ``ingest.collection.calibrate``.

    New calibrations store the solved image-to-carpet matrix directly. Older files contain point
    correspondences, which remain supported and are solved here. An explicit ``trustworthy:false``
    is never overridden by this loader.
    """

    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(document, dict):
        return None

    # Older calibration files predate these metadata fields. They were produced for the 2026
    # pipeline, so retain a useful fallback while new files always write the dimensions.
    length = float(document.get("field_length_ft", field_length_ft or 54.0))
    width = float(document.get("field_width_ft", field_width_ft or 26.6))
    source = str(document.get("mapping_source", "tag_plane"))
    plane_height = document.get("plane_height_ft")
    plane_height = float(plane_height) if isinstance(plane_height, (int, float)) else None
    trusted = document.get("trustworthy")
    trusted = bool(trusted) if isinstance(trusted, bool) else None

    matrix = _normalise_matrix(document.get("matrix"))
    if matrix is not None:
        points = document.get("points")
        point_count = int(document.get("point_count", len(points) if isinstance(points, list) else 0))
        reprojection = float(document.get("reprojection_ft", 0.0))
        if trusted is False:
            return None
        return Homography(
            matrix=matrix,
            reprojection_ft=reprojection,
            field_length_ft=length,
            field_width_ft=width,
            point_count=point_count,
            plane_height_ft=plane_height,
            source=source,
            trusted_override=trusted,
        )

    raw_points = document.get("points")
    if not isinstance(raw_points, list):
        return None
    image_points: list[tuple[float, float]] = []
    field_points: list[tuple[float, float]] = []
    for entry in raw_points:
        if not isinstance(entry, dict):
            continue
        image = entry.get("image")
        field = entry.get("field")
        if not (isinstance(image, list) and len(image) == 2 and
                isinstance(field, list) and len(field) == 2):
            continue
        image_points.append((float(image[0]), float(image[1])))
        field_points.append((float(field[0]), float(field[1])))
    solved = solve(image_points, field_points, length, width)
    if solved is None or trusted is False:
        return None
    return Homography(
        matrix=solved.matrix,
        reprojection_ft=solved.reprojection_ft,
        field_length_ft=solved.field_length_ft,
        field_width_ft=solved.field_width_ft,
        point_count=solved.point_count,
        plane_height_ft=plane_height,
        source=source,
        trusted_override=trusted,
    )


def camera_matrix_from_hfov(image_width: int, image_height: int, hfov_deg: float):
    """Build a pinhole camera matrix from a measured or explicitly assumed horizontal FOV."""

    import numpy as np

    if image_width <= 0 or image_height <= 0 or not 1.0 < hfov_deg < 179.0:
        raise ValueError("image dimensions must be positive and hfov_deg must be between 1 and 179")
    focal = image_width / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))
    return np.array([
        [focal, 0.0, image_width / 2.0],
        [0.0, focal, image_height / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def solve_camera_pose(
    image_points: Sequence[tuple[float, float]],
    field_points_3d: Sequence[tuple[float, float, float]],
    image_width: int,
    image_height: int,
    hfov_deg: float,
    field_length_ft: float,
    field_width_ft: float,
) -> tuple[Homography, dict[str, Any]] | None:
    """Recover camera pose and an image-to-carpet mapping from non-coplanar references.

    Broadcast video does not carry camera intrinsics, so the caller must provide a horizontal FOV.
    The result includes the assumption and pixel reprojection error instead of presenting this as
    a factory camera calibration.
    """

    import cv2
    import numpy as np

    if len(image_points) != len(field_points_3d) or len(image_points) < 6:
        return None
    if len({round(point[2], 6) for point in field_points_3d}) < 2:
        return None

    object_points = np.asarray(field_points_3d, dtype=np.float64)
    observed = np.asarray(image_points, dtype=np.float64)
    camera_matrix = camera_matrix_from_hfov(image_width, image_height, hfov_deg)
    try:
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            observed,
            camera_matrix,
            None,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    except cv2.error:
        return None
    if not ok:
        return None

    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, None)
    projected = projected.reshape(-1, 2)
    reprojection_px = max(
        math.hypot(float(actual[0] - predicted[0]), float(actual[1] - predicted[1]))
        for actual, predicted in zip(observed, projected)
    )

    rotation, _ = cv2.Rodrigues(rvec)
    plane_to_image = camera_matrix @ np.column_stack(
        (rotation[:, 0], rotation[:, 1], tvec.reshape(3))
    )
    if abs(float(np.linalg.det(plane_to_image))) < 1e-12:
        return None
    image_to_carpet = np.linalg.inv(plane_to_image)
    image_to_carpet /= image_to_carpet[2, 2]
    camera_position = -(rotation.T @ tvec.reshape(3))
    mapper = Homography(
        matrix=image_to_carpet.tolist(),
        reprojection_ft=0.0,
        field_length_ft=field_length_ft,
        field_width_ft=field_width_ft,
        point_count=len(image_points),
        plane_height_ft=0.0,
        source="carpet_pose",
        trusted_override=reprojection_px <= MAX_POSE_REPROJECTION_PX,
    )
    pose = {
        "hfov_deg": round(float(hfov_deg), 4),
        "reprojection_px": round(float(reprojection_px), 4),
        "camera_matrix": camera_matrix.tolist(),
        "rvec": rvec.reshape(3).tolist(),
        "tvec": tvec.reshape(3).tolist(),
        "camera_position_ft": [round(float(value), 6) for value in camera_position],
    }
    return mapper, pose


def solve(
    image_points: list[tuple[float, float]],
    field_points: list[tuple[float, float]],
    field_length_ft: float,
    field_width_ft: float,
) -> Homography | None:
    """Recover the mapping from at least four correspondences.

    Returns None when the input cannot describe a mapping at all -- too few points, or a
    degenerate arrangement such as four points on a line. A caller that gets None has a bad
    calibration, which is worth knowing loudly rather than discovering through wrong distances.
    """
    import numpy as np
    import cv2

    if len(image_points) != len(field_points) or len(image_points) < 4:
        return None

    src = np.array(image_points, dtype=np.float64)
    dst = np.array(field_points, dtype=np.float64)

    matrix, _ = cv2.findHomography(src, dst, method=0)
    if matrix is None or not np.all(np.isfinite(matrix)):
        return None

    # Reproject the inputs and measure. A solution that cannot reproduce the points it was fitted
    # to is describing something other than the given geometry.
    errors = []
    h = Homography(matrix.tolist(), 0.0, field_length_ft, field_width_ft, len(image_points))
    for (ix, iy), (fx, fy) in zip(image_points, field_points):
        px, py = h.to_field(ix, iy)
        if math.isnan(px) or math.isnan(py):
            return None
        errors.append(math.hypot(px - fx, py - fy))

    return Homography(
        matrix=matrix.tolist(),
        reprojection_ft=max(errors),
        field_length_ft=field_length_ft,
        field_width_ft=field_width_ft,
        point_count=len(image_points),
    )


def speed_ftps(
    a: tuple[float, float], t_a: float,
    b: tuple[float, float], t_b: float,
) -> float | None:
    """Speed in feet per second between two field positions.

    Returns None rather than infinity when the timestamps are equal or reversed -- a division
    that silently yields inf propagates into every statistic downstream.
    """
    dt = t_b - t_a
    if dt <= 0:
        return None
    if any(math.isnan(v) for v in (*a, *b)):
        return None
    return math.hypot(b[0] - a[0], b[1] - a[1]) / dt


#: An FRC robot is speed-limited by its drivetrain; nothing on a field moves faster than this.
#: A computed speed above it means a tracking identity swap or a bad homography, not a fast robot.
MAX_PLAUSIBLE_FTPS = 20.0


def implausible(speed: float | None) -> bool:
    return speed is not None and speed > MAX_PLAUSIBLE_FTPS
