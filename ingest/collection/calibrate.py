"""Recover a camera's image-to-field mapping from the AprilTags already on the field.

`homography.py` deliberately refuses to guess reference points -- a caller supplies them. This is
that caller. The field's tags are surveyed to the millimetre and published by WPILib, and OpenCV
reads their family natively, so a fixed camera can be calibrated from footage alone with no tape
measure and nobody clicking corners.

Four things have to be got right, and each one fails silently if it is not:

  * **The camera must not move.** A homography belongs to one camera pose. Pooling detections
    across frames is what buys enough tags to be worth checking, and it is only valid while the
    shot is still. Every tag's spread across the sample is measured, and a wandering tag ends the
    calibration rather than being averaged into it.

  * **The calibration method must match the geometry.** The 2026 field puts its 32 tags at three
    heights, 16 at 3.68 ft, 8 at 2.92 and 8 at 1.81. The default `pose` mode uses those
    non-coplanar heights to recover camera pose and then maps the carpet. The legacy `plane` mode
    deliberately reduces observations to one height and maps that elevated tag plane instead.

  * **One camera at a time.** Some 2026 broadcasts stack two views of the same field in one frame,
    and the same physical tag can appear in both. On a real match, four usable tags were in the
    upper view and four in the lower -- fitting across that boundary mixes two cameras. Hence the
    region filter, which is not optional in that footage.

  * **Four points cannot be checked.** Any four points fit a homography exactly, so reprojection
    error is zero by construction and means nothing. Five is where it starts to. The result says
    which case it is instead of reporting a number that looks like evidence.

**What the mapping is to.** The default `pose` mode fits surveyed tag corners, camera extrinsics
and both focal axes, then derives an image-to-carpet homography. Tag-centre fitting with an
assumed or measured horizontal FOV remains a fallback when corners are inadequate. The legacy
`plane` mode maps to the selected tag height, not the carpet; `plane_height_ft` records that fact.

    python -m ingest.collection.calibrate --video data/segments/<clip>.mp4 \\
        --out analysis/config/homography.<venue>.json --region 0.0 0.68 \\
        --method pose --hfov-deg 70
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

from .apriltag_layout import correspondences_from_observations, load_layout

#: Frames to sample across the clip. Tags are occluded by robots constantly, so more samples find
#: more tags; past a point it only costs time.
SAMPLES = 24

#: The detector struggles with a broadcast-sized tag at native resolution -- 3 tags found against
#: 12 at double size on the same frame.
UPSCALE = 2

#: Pixels a tag's centre may wander across the sample and still be one fixed point. Sub-pixel is
#: what a static camera actually gives: on a real match every usable tag came in under 1 px.
MAX_DRIFT_PX = 12.0

#: Sightings below which a tag is a coincidence rather than an observation.
MIN_SIGHTINGS = 3

DEFAULT_LAYOUT = Path("contracts/fields/2026-apriltags.json")


@dataclass
class TagSighting:
    tag_id: int
    xs: list[float] = field(default_factory=list)
    ys: list[float] = field(default_factory=list)
    corners: list[list[list[float]]] = field(default_factory=list)

    def median(self) -> tuple[float, float]:
        import statistics

        return statistics.median(self.xs), statistics.median(self.ys)

    def drift(self) -> float:
        """Furthest a sighting fell from the median, in pixels."""
        import math

        mx, my = self.median()
        return max((math.hypot(x - mx, y - my) for x, y in zip(self.xs, self.ys)), default=0.0)


def build_detector():
    """An OpenCV detector tuned for tags that are small and compressed rather than printed and near."""
    import cv2

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11)
    params = cv2.aruco.DetectorParameters()
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 43
    params.adaptiveThreshWinSizeStep = 4
    params.minMarkerPerimeterRate = 0.005
    params.polygonalApproxAccuracyRate = 0.06
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(dictionary, params)


def gather_sightings(video_path, samples=SAMPLES, region=(0.0, 1.0), upscale=UPSCALE
                     ) -> tuple[dict[int, TagSighting], int]:
    """Detect tags across the clip. Returns (sightings by tag id, frames actually read)."""
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        capture.release()
        return {}, 0
    # Skip the ends: intros, replays and award cards are not the match camera.
    wanted = {int(total * (0.08 + 0.84 * i / max(1, samples - 1))) for i in range(samples)}
    detector = build_detector()

    sightings: dict[int, TagSighting] = {}
    frames = 0
    # These samples are intentionally sparse. Seeking to each one avoids decoding an entire
    # multi-minute match before analysis can start; VideoCapture seeks to the preceding keyframe
    # and decodes forward to the requested frame for ordinary MP4 inputs.
    for index in sorted(wanted):
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            continue
        frames += 1
        height = frame.shape[0]
        region_top = max(0, min(height - 1, int(round(region[0] * height))))
        region_bottom = max(region_top + 1, min(height, int(round(region[1] * height))))
        # Crop before the expensive 2x AprilTag pass. Besides reducing startup time, this keeps
        # a stacked secondary broadcast view out of the detector rather than discarding it later.
        grey = cv2.cvtColor(frame[region_top:region_bottom], cv2.COLOR_BGR2GRAY)
        image = cv2.resize(grey, None, fx=upscale, fy=upscale,
                           interpolation=cv2.INTER_CUBIC) if upscale != 1 else grey
        corners, ids, _ = detector.detectMarkers(image)
        for corner, tag_id in zip(corners, (ids.flatten() if ids is not None else [])):
            centre = corner.reshape(4, 2).mean(axis=0) / upscale
            centre[1] += region_top
            seen = sightings.setdefault(int(tag_id), TagSighting(int(tag_id)))
            seen.xs.append(float(centre[0]))
            seen.ys.append(float(centre[1]))
            pixels = corner.reshape(4, 2) / upscale
            pixels[:, 1] += region_top
            seen.corners.append(pixels.tolist())
    capture.release()
    return sightings, frames


def steady_tags(sightings: dict[int, TagSighting], max_drift=MAX_DRIFT_PX,
                min_sightings=MIN_SIGHTINGS) -> tuple[dict[int, tuple[float, float]], list[str]]:
    """Tags that held still. Returns (id -> median pixel, reasons others were dropped)."""
    kept: dict[int, tuple[float, float]] = {}
    notes: list[str] = []
    for tag_id, seen in sorted(sightings.items()):
        if len(seen.xs) < min_sightings:
            notes.append(f"tag {tag_id}: only {len(seen.xs)} sightings")
            continue
        drift = seen.drift()
        if drift > max_drift:
            # Either the camera panned or the detection is unreliable. Averaging either one
            # produces a confident point that was never there.
            notes.append(f"tag {tag_id}: drifts {drift:.1f}px, camera is not static")
            continue
        kept[tag_id] = seen.median()
    return kept, notes


def _video_size(video_path) -> tuple[int, int]:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()
    return width, height


def fit_tag_corners(sightings, used_ids, layout, image_size, region):
    """Reject inconsistent whole markers, never cherry-pick corners of a bad tag."""
    import cv2
    import numpy as np
    from .homography import solve_broadcast_corners

    ids = [i for i in used_ids if len(sightings[i].corners) >= MIN_SIGHTINGS]
    rejected = []
    budget = len(ids) // 4
    width, height = image_size
    while len(ids) >= 3:
        images = np.concatenate([np.median(sightings[i].corners, axis=0) for i in ids])
        objects = np.concatenate([layout.tags[i].corners_ft() for i in ids])
        centers = objects.reshape(-1, 4, 3).mean(axis=1)
        if (np.ptp(centers[:, 0]) < layout.length_ft * .2
                or np.ptp(centers[:, 1]) < layout.width_ft * .15):
            return None
        fit = solve_broadcast_corners(images, objects, image_size,
            (layout.length_ft, layout.width_ft), (width / 2, height * sum(region) / 2))
        if fit is None:
            return None
        mapper, pose = fit
        if mapper.trustworthy:
            pose["rejected_tags"] = rejected
            return mapper, pose, ids, images, objects
        if len(rejected) >= budget or len(ids) <= 4:
            return None
        projected, _ = cv2.projectPoints(objects, np.array(pose["rvec"]), np.array(pose["tvec"]),
                                        np.array(pose["camera_matrix"]), None)
        errors = np.linalg.norm(projected.reshape(-1, 2) - images, axis=1).reshape(-1, 4)
        # A high-leverage bad marker can drag the pose toward itself and make good
        # tags look worse. Compare whole-tag leave-one-out fits before removing one.
        trials = []
        for index in range(len(ids)):
            keep = np.ones(len(images), dtype=bool)
            keep[index * 4:index * 4 + 4] = False
            trial = solve_broadcast_corners(images[keep], objects[keep], image_size,
                (layout.length_ft, layout.width_ft), (width / 2, height * sum(region) / 2))
            if trial is not None:
                trials.append((trial[1]["reprojection_px"], trial[1]["rms_reprojection_px"], index))
        if not trials:
            return None
        worst = min(trials)[2]
        rejected.append({"tag_id": ids.pop(worst), "median_error_px": float(np.median(errors[worst]))})
    return None


def _extra_points(extra_points) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    points = []
    for point in extra_points:
        if not isinstance(point, dict) or "image" not in point or "field" not in point:
            continue
        image = point["image"]
        field = point["field"]
        if not (isinstance(image, list) and len(image) == 2 and
                isinstance(field, list) and len(field) == 2):
            continue
        points.append(((float(image[0]), float(image[1])),
                       (float(field[0]), float(field[1]))))
    return points


def calibrate(video_path, layout_path=DEFAULT_LAYOUT, samples=SAMPLES, region=(0.0, 1.0),
              extra_points=(), *, method="pose", hfov_deg=70.0,
              optimize_hfov=False) -> dict:
    """Calibrate either the carpet by camera pose or a single tag plane.

    ``pose`` prefers surveyed corners and fits both focal axes for resized broadcast panels.
    Tag centres plus optional carpet points use the supplied FOV as a fallback. ``plane`` preserves
    the old four-point fallback and maps to the dominant tag height, never silently to the carpet.
    """
    from . import homography as homography_module

    layout = load_layout(layout_path)
    sightings, frames = gather_sightings(video_path, samples=samples, region=region)
    observed, notes = steady_tags(sightings)
    extra = _extra_points(extra_points)
    used_ids = sorted(tag_id for tag_id in observed if tag_id in layout.tags)
    plane_heights = {layout.tags[tag_id].z_ft for tag_id in used_ids}
    observed_points = [
        {"tag_id": tag_id, "image": [round(observed[tag_id][0], 2), round(observed[tag_id][1], 2)]}
        for tag_id in used_ids
    ]

    result = {
        "frames_sampled": frames,
        "tags_detected": sorted(sightings),
        "tags_steady": sorted(observed),
        "tags_used": used_ids,
        "notes": notes,
        "point_count": 0,
        "plane_height_ft": 0.0 if method in ("pose", "carpet") else (round(min(plane_heights), 3) if plane_heights else None),
        "mapping_source": {"pose": "carpet_pose", "carpet": "carpet_marked"}.get(method, "tag_plane"),
        "observed_points": observed_points,
        "points": [],
        "solution": None,
    }

    if method == "carpet":
        # Tags are read only so the caller can see what was there; they take no part in the fit.
        # The carpet is its own plane and hand-marked points are already on it.
        result["mapping_source"] = "carpet_marked"
        result["plane_height_ft"] = 0.0
        result["point_count"] = len(extra)
        result["points"] = [
            {"image": [round(image[0], 2), round(image[1], 2)],
             "field": [round(field[0], 4), round(field[1], 4)]}
            for image, field in extra
        ]
        if len(extra) < 4:
            return result
        solved = homography_module.solve(
            [pair[0] for pair in extra], [pair[1] for pair in extra],
            layout.length_ft, layout.width_ft,
        )
        if solved is not None:
            result["matrix"] = solved.matrix
            result["solution"] = {
                "reprojection_ft": round(solved.reprojection_ft, 4),
                "has_redundancy": solved.has_redundancy,
                "trustworthy": solved.trustworthy,
            }
        return result

    if method == "pose":
        # Decoded corners retain both surveyed tag orientation and vertical extent.
        # Use them when available instead of discarding almost all geometric evidence.
        import numpy as np
        corner_ids = [i for i in used_ids if len(sightings[i].corners) >= MIN_SIGHTINGS]
        width, height = _video_size(video_path)
        if len(corner_ids) >= 3 and not extra and width > 0 and height > 0:
            corner_solve = fit_tag_corners(sightings, corner_ids, layout, (width, height), region)
            if corner_solve is not None and corner_solve[0].trustworthy:
                mapper, pose, corner_ids, images, objects = corner_solve
                result.update(mapping_source=mapper.source, matrix=mapper.matrix, pose=pose,
                              image_size=[width, height], point_count=len(images),
                              tags_used=corner_ids,
                              observed_corners={str(i): np.median(sightings[i].corners, axis=0).tolist()
                                                for i in corner_ids},
                              points=[{"image": a.tolist(), "field": b[:2].tolist()}
                                      for a, b in zip(images, objects)],
                              solution={"reprojection_px": pose["reprojection_px"],
                                        "has_redundancy": True, "trustworthy": True})
                return result
            notes.append("corner pose did not pass reprojection checks; trying tag centres")
        image_points = [observed[tag_id] for tag_id in used_ids]
        field_points_3d = [
            (layout.tags[tag_id].x_ft, layout.tags[tag_id].y_ft, layout.tags[tag_id].z_ft)
            for tag_id in used_ids
        ]
        image_points.extend(pair[0] for pair in extra)
        field_points_3d.extend((pair[1][0], pair[1][1], 0.0) for pair in extra)
        result["point_count"] = len(image_points)
        result["points"] = [
            {"image": [round(image[0], 2), round(image[1], 2)],
             "field": [round(field[0], 4), round(field[1], 4)]}
            for image, field in zip(image_points, field_points_3d)
        ]
        width, height = _video_size(video_path)
        solved = None
        if width > 0 and height > 0:
            candidate_hfovs = [float(hfov_deg)]
            if optimize_hfov:
                # Broadcast files omit camera intrinsics. The known 3-D tag heights let us fit
                # focal length as well as pose; first search broadly, then refine around the best
                # result. The supplied HFOV remains a candidate and a useful fallback.
                low = max(25.0, float(hfov_deg) - 30.0)
                high = min(120.0, float(hfov_deg) + 30.0)
                candidate_hfovs.extend(low + 2.0 * index for index in range(int((high - low) / 2.0) + 1))
            candidates = []
            for candidate in candidate_hfovs:
                attempt = homography_module.solve_camera_pose(
                    image_points, field_points_3d, width, height, candidate,
                    layout.length_ft, layout.width_ft,
                )
                if attempt is not None:
                    candidates.append(attempt)
            if candidates:
                solved = min(candidates, key=lambda item: item[1]["reprojection_px"])
            if optimize_hfov and solved is not None:
                best_hfov = float(solved[1]["hfov_deg"])
                refinements = []
                for offset in range(-8, 9):
                    attempt = homography_module.solve_camera_pose(
                        image_points, field_points_3d, width, height,
                        best_hfov + offset * 0.25,
                        layout.length_ft, layout.width_ft,
                    )
                    if attempt is not None:
                        refinements.append(attempt)
                if refinements:
                    solved = min(refinements, key=lambda item: item[1]["reprojection_px"])
        if solved is not None:
            mapper, pose = solved
            result["matrix"] = mapper.matrix
            result["pose"] = pose
            result["image_size"] = [width, height]
            result["solution"] = {
                "reprojection_px": pose["reprojection_px"],
                "has_redundancy": len(image_points) >= 7,
                "trustworthy": mapper.trustworthy,
            }
        return result

    pairs = correspondences_from_observations(layout, observed, require_coplanar=True)
    # Plane-mode extra points must lie on the same tag plane. Carpet points belong in pose mode;
    # mixing them here creates a surface that does not exist.
    pairs = list(pairs) + extra
    result["point_count"] = len(pairs)
    result["points"] = [
        {"image": [round(image[0], 2), round(image[1], 2)],
         "field": [round(field[0], 4), round(field[1], 4)]}
        for image, field in pairs
    ]
    if len(pairs) < 4:
        return result
    solved = homography_module.solve(
        [pair[0] for pair in pairs], [pair[1] for pair in pairs],
        layout.length_ft, layout.width_ft,
    )
    if solved is not None:
        result["matrix"] = solved.matrix
        result["solution"] = {
            "reprojection_ft": round(solved.reprojection_ft, 4),
            "has_redundancy": solved.has_redundancy,
            "trustworthy": solved.trustworthy,
        }
    return result



def write_reference_frame(video_path, out_path, region=(0.0, 1.0), observed=None) -> bool:
    """Save a frame with a pixel grid and the detected tags marked.

    Written whenever calibration cannot finish, because the fix is always the same: a person has
    to supply image points for field features they can identify. Reading a pixel coordinate off a
    gridded image is a two-minute job; guessing one is not.
    """
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    target = int(total * 0.5)
    frame = None
    for i in range(target + 1):
        ok, image = capture.read()
        if not ok:
            break
        if i == target:
            frame = image
    capture.release()
    if frame is None:
        return False

    height, width = frame.shape[:2]
    for x in range(0, width, 100):
        heavy = x % 500 == 0
        cv2.line(frame, (x, 0), (x, height), (0, 255, 255), 2 if heavy else 1)
        if heavy:
            cv2.putText(frame, str(x), (x + 4, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    for y in range(0, height, 100):
        heavy = y % 500 == 0
        cv2.line(frame, (0, y), (width, y), (0, 255, 255), 2 if heavy else 1)
        if heavy:
            cv2.putText(frame, str(y), (6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # The searched band, so it is obvious when a stacked second view was excluded.
    for edge in region:
        cv2.line(frame, (0, int(height * edge)), (width, int(height * edge)), (255, 0, 255), 3)

    for tag_id, (x, y) in (observed or {}).items():
        cv2.circle(frame, (int(x), int(y)), 14, (0, 255, 0), 3)
        cv2.putText(frame, f"tag {tag_id}", (int(x) + 16, int(y) + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), frame)
    return True


EXTRA_POINTS_TEMPLATE = """[
  {"_comment": "For --method pose, mark carpet features (z=0) such as corners. For --method plane, mark points on the same elevated tag plane. Origin is a field corner; delete this comment entry."},
  {"image": [0, 0], "field": [0.0, 0.0]},
  {"image": [0, 0], "field": [54.0, 0.0]},
  {"image": [0, 0], "field": [54.0, 26.6]},
  {"image": [0, 0], "field": [0.0, 26.6]}
]
"""

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--layout", type=Path, default=DEFAULT_LAYOUT)
    parser.add_argument("--samples", type=int, default=SAMPLES)
    parser.add_argument("--region", nargs=2, type=float, default=[0.0, 1.0],
                        metavar=("TOP", "BOTTOM"),
                        help="fraction of frame height to search; use this when a broadcast "
                             "stacks two camera views, or tags from both get mixed into one fit")
    parser.add_argument("--method", choices=("pose", "plane", "carpet"), default="pose",
                        help="carpet fits hand-marked carpet points directly; pose fits surveyed "
                             "tag corners and camera intrinsics; plane is the legacy "
                             "dominant-tag-height fallback")
    parser.add_argument("--hfov-deg", type=float, default=70.0,
                        help="horizontal camera FOV assumed by pose mode (measure it when possible)")
    parser.add_argument("--optimize-hfov", action="store_true",
                        help="fit horizontal FOV from the non-coplanar AprilTags around the supplied estimate")
    parser.add_argument("--extra-points", type=Path,
                        help="JSON list of {image:[x,y], field:[ft,ft]} measured by hand, to "
                             "reach the five points that make reprojection error meaningful")
    args = parser.parse_args(argv)

    extra = []
    if args.extra_points:
        extra = json.loads(args.extra_points.read_text(encoding="utf-8"))

    result = calibrate(args.video, args.layout, args.samples, tuple(args.region), extra,
                       method=args.method, hfov_deg=args.hfov_deg,
                       optimize_hfov=args.optimize_hfov)

    print(f"{result['frames_sampled']} frames sampled from y {args.region[0]}-{args.region[1]}")
    print(f"  tags detected : {result['tags_detected']}")
    print(f"  held still    : {result['tags_steady']}")
    print(f"  tags used     : {result['tags_used']}")
    for note in result["notes"]:
        print(f"  ! {note}")

    def rescue(reason: str) -> int:
        reference = args.out.with_suffix(".reference.png")
        template = args.out.with_suffix(".extra-points.json")
        observed = {t: tuple(p["image"]) for t, p in
                    zip(result["tags_used"], result["points"])}
        print(f"\n{reason}")
        if write_reference_frame(args.video, reference, tuple(args.region), observed):
            print(f"  reference frame: {reference}")
        if not template.exists():
            template.write_text(EXTRA_POINTS_TEMPLATE, encoding="utf-8")
            print(f"  template       : {template}")
        print("  Read pixel coordinates for field features you can identify off the grid, fill "
              "them into the template, and re-run with --extra-points.")
        return 1

    if result["point_count"] < (6 if args.method == "pose" else 4):
        minimum = 6 if args.method == "pose" else 4
        if args.method == "carpet":
            return rescue(
                f"carpet mode needs at least 4 marked points and got {result['point_count']}. "
                f"Mark them on the reference frame below; five or more make the reprojection "
                f"error mean something, because any four fit a homography exactly.")
        return rescue(f"only {result['point_count']} usable correspondences; {minimum} is the minimum for {args.method} mode.")

    height = result["plane_height_ft"]
    print(f"\n{result['point_count']} correspondences using {result['mapping_source']}")
    solution = result["solution"]
    if solution is None:
        if args.method == "pose":
            return rescue(
                "the camera pose could not be recovered; use a measured --hfov-deg, keep the "
                "camera still, and include at least two AprilTag height groups")
        if args.method == "carpet":
            return rescue(
                "the marked points are degenerate -- they lie on a line, so they cannot define a "
                "plane. Spread them across the carpet: two near corners and two far ones beat "
                "four along one edge.")
        return rescue(
            "the points are degenerate -- they lie on a line, so they cannot define a plane. "
            "On this footage all four coplanar tags sit within 0.1px of one image row, because "
            "both goal structures carry them at the same height and the camera looks down the "
            "field at them. Tags alone cannot calibrate this angle.")

    if args.method == "pose":
        print(f"  reprojection : {solution['reprojection_px']} px")
    else:
        print(f"  reprojection : {solution['reprojection_ft']} ft")
    if solution["has_redundancy"]:
        print(f"  trustworthy  : {solution['trustworthy']}")
    else:
        print("  no holdout redundancy: the solution meets the minimum correspondence count; "
              "add another known point for a stronger consistency check.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "_comment": (f"Auto-calibrated from {Path(args.video).name} using AprilTags "
                     f"{result['tags_used']}. Mapping source: {result['mapping_source']}. "
                     "One camera pose only; re-run per venue and per camera position."),
        "mapping_source": result["mapping_source"],
        "field_length_ft": load_layout(args.layout).length_ft,
        "field_width_ft": load_layout(args.layout).width_ft,
        "plane_height_ft": height,
        "point_count": result["point_count"],
        "has_redundancy": solution["has_redundancy"],
        "trustworthy": solution["trustworthy"],
        "points": result["points"],
        "matrix": result["matrix"],
        "image_size": result.get("image_size"),
        "tags_used": result.get("tags_used", []),
    }
    if args.method == "pose":
        payload["pose"] = result["pose"]
    else:
        payload["reprojection_ft"] = solution["reprojection_ft"]
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    print(f"Use it with:  $env:FRC_HOMOGRAPHY_CONFIG = "
          f'(Resolve-Path "{args.out}").Path')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
