"""The official 2026 REBUILT AprilTag positions, and what can honestly be built from them.

The layout in contracts/fields/2026-apriltags.json is WPILib's published file, copied verbatim.
Nothing in it is estimated here: a fabricated tag coordinate would produce a mapping that is
confidently wrong everywhere, which is the exact failure the homography work exists to avoid.

WPILib publishes positions in METRES, with the origin at a field corner. This project works in
feet, so conversion happens here, once, at the boundary.

The awkward part, stated plainly
--------------------------------
A homography maps one plane. The 32 tags sit at three different heights:

    0.5524 m   8 tags
    0.889 m    8 tags
    1.124 m   16 tags

So the tag set as a whole is not coplanar, and feeding all of it to `findHomography` asks for a
plane that does not exist. Worse, it would not fail -- it would return a best-fit surface through
points that do not share one, and every position derived from it would be quietly wrong.

Two honest options follow, and this module supports the first:

1. Use one height group. Tags sharing a z ARE coplanar, so they give a valid homography onto that
   horizontal plane. The catch is that robots stand on the carpet at z=0, not on the plane 1.1 m
   above it, so positions from such a mapping describe the wrong surface.

2. Recover the full camera pose with solvePnP, which handles non-coplanar references and can then
   project the carpet plane properly. That needs camera intrinsics, which nobody has for a
   broadcast camera at unknown zoom.

`largest_coplanar_group` therefore gives callers the best available planar subset, and
`carpet_correspondences` gives the honest answer for carpet-plane work: tag positions projected
straight down are only usable when the caller can supply the corresponding image points for those
projected positions -- which in practice means a person marking the carpet, not a tag detector.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

METRES_TO_FEET = 3.280839895013123


@dataclass(frozen=True)
class Tag:
    """One AprilTag's surveyed position, in feet, origin at a field corner."""
    id: int
    x_ft: float
    y_ft: float
    z_ft: float
    quaternion: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    def corners_ft(self, size_inches: float = 6.5):
        """OpenCV 36h11 decoded corner order in WPILib's tag coordinate frame.

        OpenCV's dictionary origin is the printed marker's bottom-right for the
        official WPILib targets: BR, BL, TL, TR as viewed facing an upright tag.
        This is decoded orientation, not sorting corners by image coordinates.
        """
        import numpy as np
        from scipy.spatial.transform import Rotation

        w, x, y, z = self.quaternion
        rotation = Rotation.from_quat([x, y, z, w]).as_matrix()
        half = size_inches / 24.0
        local = np.array([[0, half, -half], [0, -half, -half],
                          [0, -half, half], [0, half, half]])
        return local @ rotation.T + [self.x_ft, self.y_ft, self.z_ft]

    @property
    def height_key(self) -> int:
        """Height rounded to the nearest tenth of an inch, for grouping coplanar tags."""
        return round(self.z_ft * 120)


@dataclass(frozen=True)
class FieldLayout:
    name: str
    season: int
    length_ft: float
    width_ft: float
    tags: dict[int, Tag]

    def height_groups(self) -> dict[int, list[Tag]]:
        groups: dict[int, list[Tag]] = defaultdict(list)
        for tag in self.tags.values():
            groups[tag.height_key].append(tag)
        return dict(groups)

    def largest_coplanar_group(self) -> list[Tag]:
        """The biggest set of tags sharing a height, which is the best planar subset available.

        Returned sorted by id so a caller's correspondences are reproducible.
        """
        groups = self.height_groups()
        if not groups:
            return []
        best = max(groups.values(), key=len)
        return sorted(best, key=lambda t: t.id)


def load_layout(path: Path | str) -> FieldLayout:
    """Read WPILib's published layout and convert to feet."""
    document = json.loads(Path(path).read_text(encoding="utf-8-sig"))

    dims = document.get("field-dimensions", {})
    tags: dict[int, Tag] = {}
    for entry in document.get("field-tags", []):
        translation = entry["pose"]["translation"]
        tags[int(entry["ID"])] = Tag(
            id=int(entry["ID"]),
            x_ft=translation["x"] * METRES_TO_FEET,
            y_ft=translation["y"] * METRES_TO_FEET,
            z_ft=translation["z"] * METRES_TO_FEET,
            quaternion=tuple(entry["pose"]["rotation"]["quaternion"][key]
                             for key in ("W", "X", "Y", "Z")),
        )

    return FieldLayout(
        name=document.get("name", ""),
        season=int(document.get("season", 0)),
        length_ft=float(dims.get("length", 0.0)) * METRES_TO_FEET,
        width_ft=float(dims.get("width", 0.0)) * METRES_TO_FEET,
        tags=tags,
    )


def correspondences_from_observations(
    layout: FieldLayout,
    observed: dict[int, tuple[float, float]],
    require_coplanar: bool = True,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Pair observed tag image positions with their surveyed field positions.

    `observed` maps tag id to a pixel position, however the caller obtained it.

    With `require_coplanar` (the default) only tags sharing the most common height among the
    observations are returned. Mixing heights produces a fit through points that do not lie on any
    one plane, and that fit does not announce itself -- it just reports wrong positions. Callers
    who genuinely want the full non-coplanar set, for a PnP solve rather than a homography, can
    turn the restriction off.
    """
    known = {tag_id: layout.tags[tag_id] for tag_id in observed if tag_id in layout.tags}
    if not known:
        return []

    if require_coplanar:
        heights: dict[int, int] = defaultdict(int)
        for tag in known.values():
            heights[tag.height_key] += 1
        dominant = max(heights, key=lambda k: heights[k])
        known = {i: t for i, t in known.items() if t.height_key == dominant}

    return [
        (observed[tag_id], (tag.x_ft, tag.y_ft))
        for tag_id, tag in sorted(known.items())
    ]
