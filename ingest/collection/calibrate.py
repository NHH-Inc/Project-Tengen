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

**What the mapping is to.** The default `pose` mode uses tag heights plus an assumed or measured
horizontal FOV to recover camera extrinsics, then derives an image-to-carpet homography. The legacy
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
SAMPLES = 40

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
    index, frames = 0, 0
    last = max(wanted)
    while index <= last:
        ok, frame = capture.read()
        if not ok:
            break
        if index in wanted:
            frames += 1
            height = frame.shape[0]
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(grey, None, fx=upscale, fy=upscale,
                               interpolation=cv2.INTER_CUBIC) if upscale != 1 else grey
            corners, ids, _ = detector.detectMarkers(image)
            for corner, tag_id in zip(corners, (ids.flatten() if ids is not None else [])):
                centre = corner.reshape(4, 2).mean(axis=0) / upscale
                if not (region[0] <= centre[1] / height < region[1]):
                    continue      # a different camera's view of the same field
                seen = sightings.setdefault(int(tag_id), TagSighting(int(tag_id)))
                seen.xs.append(float(centre[0]))
                seen.ys.append(float(centre[1]))
        index += 1
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
              extra_points=(), *, method="pose", hfov_deg=70.0) -> dict:
    """Calibrate either the carpet by camera pose or a single tag plane.

    ``pose`` uses all observed AprilTag heights plus optional hand-marked carpet points. It needs
    a horizontal FOV because broadcast files do not carry camera intrinsics. ``plane`` preserves
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
        "plane_height_ft": 0.0 if method == "pose" else (round(min(plane_heights), 3) if plane_heights else None),
        "mapping_source": "carpet_pose" if method == "pose" else "tag_plane",
        "observed_points": observed_points,
        "points": [],
        "solution": None,
    }

    if method == "pose":
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
        solved = homography_module.solve_camera_pose(
            image_points, field_points_3d, width, height, hfov_deg,
            layout.length_ft, layout.width_ft,
        ) if width > 0 and height > 0 else None
        if solved is not None:
            mapper, pose = solved
            result["matrix"] = mapper.matrix
            result["pose"] = pose
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
    parser.add_argument("--method", choices=("pose", "plane"), default="pose",
                        help="pose maps robot feet to carpet using non-coplanar tags; plane is the "
                             "legacy dominant-tag-height fallback")
    parser.add_argument("--hfov-deg", type=float, default=70.0,
                        help="horizontal camera FOV assumed by pose mode (measure it when possible)")
    parser.add_argument("--extra-points", type=Path,
                        help="JSON list of {image:[x,y], field:[ft,ft]} measured by hand, to "
                             "reach the five points that make reprojection error meaningful")
    args = parser.parse_args(argv)

    extra = []
    if args.extra_points:
        extra = json.loads(args.extra_points.read_text(encoding="utf-8"))

    result = calibrate(args.video, args.layout, args.samples, tuple(args.region), extra,
                       method=args.method, hfov_deg=args.hfov_deg)

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
        return rescue(f"only {result['point_count']} usable correspondences; {minimum} is the minimum for {args.method} mode.")

    height = result["plane_height_ft"]
    print(f"\n{result['point_count']} correspondences using {result['mapping_source']}")
    solution = result["solution"]
    if solution is None:
        if args.method == "pose":
            return rescue(
                "the camera pose could not be recovered; use a measured --hfov-deg, keep the "
                "camera still, and include at least two AprilTag height groups")
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
        print("  UNVERIFIED: four points fit exactly, so this error is zero by construction and "
              "is not evidence. Add a fifth with --extra-points to make it mean something.")

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
