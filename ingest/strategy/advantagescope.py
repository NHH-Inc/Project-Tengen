"""Export a match's robot positions to a WPILOG (.wpilog) file for AdvantageScope.

PLAN-2026 section 4, decided (2026-10-30): integrate with AdvantageScope rather than build a custom
viewer. AdvantageScope ingests WPILib data logs; our tracks are already structured field-coordinate
timeseries, so this is a format bridge, not a new viewer. Each robot becomes one 2D pose entry --
`[x_m, y_m, heading_rad]` -- under `/Tengen/<alliance>/<team>`. Open the .wpilog in AdvantageScope,
drop each `/Tengen/...` field onto the 2D Field as a Robot, and colour by the alliance folder.

Coordinates: the timeline's `x`/`y` are WPILib AprilTag field coordinates in feet (same frame the
web heat map draws), so the only conversion here is feet -> metres. Heading is not measured, so it
is taken from the direction of travel (and held through a stop); a 2D field view needs a facing,
and velocity heading is the honest one when we have no gyro.

The WPILOG v1.0 binary layout is implemented inline from the WPILib spec
(wpiutil/doc/datalog.adoc): little-endian throughout. To keep the writer trivial and correct,
every record uses the maximum field widths (4-byte entry id, 4-byte payload size, 8-byte
timestamp); the header bitfield for that is 0x7F. These logs are tiny, so the few spare bytes per
record do not matter.
"""

from __future__ import annotations

import json
import math

from .timeline import samples_as_rows
import struct

#: Feet to metres. The field frame is WPILib AprilTag coordinates; AdvantageScope works in metres.
FEET_TO_M = 0.3048
#: Only update the heading once the robot has moved this far (metres), so homography jitter on a
#: parked robot does not spin the arrow.
HEADING_MOVE_M = 0.08
#: Record header bitfield for 4-byte id, 4-byte size, 8-byte timestamp (see module docstring).
_HEADER = 0x7F


class DataLogWriter:
    """A minimal WPILOG v1.0 writer: start string-keyed entries, then append timestamped values."""

    def __init__(self, extra_header: str = "") -> None:
        extra = extra_header.encode("utf-8")
        self._chunks: list[bytes] = [
            b"WPILOG",
            struct.pack("<H", 0x0100),  # version 1.0: high byte major, low byte minor
            struct.pack("<I", len(extra)),
            extra,
        ]
        self._next_id = 1

    @staticmethod
    def _lenstr(text: str) -> bytes:
        raw = text.encode("utf-8")
        return struct.pack("<I", len(raw)) + raw

    def _emit(self, entry_id: int, timestamp_us: int, payload: bytes) -> None:
        self._chunks.append(
            bytes([_HEADER])
            + struct.pack("<IIQ", entry_id, len(payload), max(0, int(timestamp_us)))
            + payload
        )

    def start_entry(self, name: str, type_str: str, metadata: str = "", timestamp_us: int = 0) -> int:
        """Declare an entry (a Start control record) and return its id for later `append`s."""
        entry_id = self._next_id
        self._next_id += 1
        payload = (b"\x00" + struct.pack("<I", entry_id)
                   + self._lenstr(name) + self._lenstr(type_str) + self._lenstr(metadata))
        self._emit(0, timestamp_us, payload)
        return entry_id

    def append(self, entry_id: int, timestamp_us: int, payload: bytes) -> None:
        self._emit(entry_id, timestamp_us, payload)

    def getvalue(self) -> bytes:
        return b"".join(self._chunks)


def _entry_name(timeline: dict) -> str:
    alliance = timeline.get("alliance") or "unknown"
    who = timeline.get("team") or timeline.get("robot_key")
    return f"/Tengen/{alliance}/{who}"


def match_to_wpilog(timelines: list[dict], *, match_id: str | None = None) -> bytes:
    """A .wpilog of every robot's 2D pose through the match, for AdvantageScope's 2D Field tab.

    One `double[]` entry per robot: `[x_m, y_m, heading_rad]`, which AdvantageScope reads as a
    Pose2d. Samples with no field position are skipped, and a gap resets the heading so a robot
    that reappears elsewhere does not draw a line across the field.
    """
    writer = DataLogWriter(extra_header=json.dumps(
        {"source": "project-tengen", "match_id": match_id, "format": "Pose2d [x_m, y_m, heading_rad]"}))
    entry_ids: dict[str, int] = {}
    for timeline in timelines:
        metadata = json.dumps({"team": timeline.get("team"), "alliance": timeline.get("alliance"),
                               "robot_key": timeline.get("robot_key")})
        entry_ids[timeline["robot_key"]] = writer.start_entry(_entry_name(timeline), "double[]", metadata)
    for timeline in timelines:
        entry_id = entry_ids[timeline["robot_key"]]
        previous: tuple[float, float] | None = None
        heading = 0.0
        for sample in samples_as_rows(timeline):  # columnar -> row dicts (timeline stores columns)
            x, y = sample.get("x"), sample.get("y")
            if x is None or y is None:
                previous = None  # a gap: do not interpolate a facing across it
                continue
            x_m, y_m = x * FEET_TO_M, y * FEET_TO_M
            if previous is not None:
                dx, dy = x_m - previous[0], y_m - previous[1]
                if dx * dx + dy * dy >= HEADING_MOVE_M * HEADING_MOVE_M:
                    heading = math.atan2(dy, dx)
            timestamp_us = int(round(float(sample["t"]) * 1_000_000))
            writer.append(entry_id, timestamp_us, struct.pack("<3d", x_m, y_m, heading))
            previous = (x_m, y_m)
    return writer.getvalue()
