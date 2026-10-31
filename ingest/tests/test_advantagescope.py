"""The AdvantageScope export: a match's robots -> a WPILOG the 2D Field tab can read.

These parse the bytes back with an independent reader, so they check the file against the WPILib
spec (wpiutil/doc/datalog.adoc), not just against the writer's own assumptions.
"""

import math
import struct
import unittest

from ingest.strategy.advantagescope import FEET_TO_M, match_to_wpilog


def parse_wpilog(data: bytes):
    """A small, general WPILOG reader: returns (version, entries-by-id, data records)."""
    assert data[:6] == b"WPILOG", "missing magic"
    version = struct.unpack_from("<H", data, 6)[0]
    extra_len = struct.unpack_from("<I", data, 8)[0]
    pos = 12 + extra_len
    entries: dict[int, dict] = {}
    records: list[tuple[int, int, bytes]] = []
    while pos < len(data):
        bitfield = data[pos]; pos += 1
        id_len = (bitfield & 0x3) + 1
        size_len = ((bitfield >> 2) & 0x3) + 1
        ts_len = ((bitfield >> 4) & 0x7) + 1
        entry_id = int.from_bytes(data[pos:pos + id_len], "little"); pos += id_len
        size = int.from_bytes(data[pos:pos + size_len], "little"); pos += size_len
        timestamp = int.from_bytes(data[pos:pos + ts_len], "little"); pos += ts_len
        payload = data[pos:pos + size]; pos += size
        if entry_id == 0 and payload and payload[0] == 0:  # Start control record
            p = 1
            eid = struct.unpack_from("<I", payload, p)[0]; p += 4
            fields = []
            for _ in range(3):  # name, type, metadata
                n = struct.unpack_from("<I", payload, p)[0]; p += 4
                fields.append(payload[p:p + n].decode("utf-8")); p += n
            entries[eid] = {"name": fields[0], "type": fields[1], "metadata": fields[2]}
        elif entry_id != 0:
            records.append((entry_id, timestamp, payload))
    return version, entries, records


def timeline(robot_key, samples, team=None, alliance=None):
    # Real build_timelines stores samples COLUMNAR (keys -> parallel lists), not as row dicts; mirror
    # that here so the export is tested against the shape it actually gets (a bug hid behind the old
    # row-dict shape). `samples` is still written as a list of dicts for readability.
    keys = ("t", "mt", "phase", "x", "y", "ax", "ay", "speed", "zone", "lane")
    columns = {key: [sample.get(key) for sample in samples] for key in keys}
    return {"robot_key": robot_key, "team": team, "alliance": alliance, "samples": columns}


class WpilogFormatTests(unittest.TestCase):
    def test_header_and_entries_match_the_spec(self):
        timelines = [
            timeline("robot1", [{"t": 0.0, "x": 10.0, "y": 5.0}], team=254, alliance="red"),
            timeline("robot2", [{"t": 0.0, "x": 40.0, "y": 20.0}], team=1678, alliance="blue"),
        ]
        version, entries, records = parse_wpilog(match_to_wpilog(timelines, match_id="2026x_qm1"))
        self.assertEqual(version, 0x0100)
        names = {e["name"] for e in entries.values()}
        self.assertEqual(names, {"/Tengen/red/254", "/Tengen/blue/1678"})
        self.assertTrue(all(e["type"] == "double[]" for e in entries.values()))
        self.assertEqual(len(records), 2)

    def test_positions_are_metres_in_the_field_frame(self):
        timelines = [timeline("robot1", [{"t": 0.0, "x": 10.0, "y": 5.0}], team=254, alliance="red")]
        _, _, records = parse_wpilog(match_to_wpilog(timelines))
        _, _, payload = records[0]
        x_m, y_m, heading = struct.unpack("<3d", payload)
        self.assertAlmostEqual(x_m, 10.0 * FEET_TO_M, places=6)
        self.assertAlmostEqual(y_m, 5.0 * FEET_TO_M, places=6)
        self.assertEqual(heading, 0.0)  # a single sample has no direction of travel yet

    def test_timestamps_are_microseconds(self):
        timelines = [timeline("r", [{"t": 0.0, "x": 1.0, "y": 1.0}, {"t": 1.5, "x": 2.0, "y": 1.0}])]
        _, _, records = parse_wpilog(match_to_wpilog(timelines))
        self.assertEqual([ts for _, ts, _ in records], [0, 1_500_000])

    def test_heading_follows_direction_of_travel(self):
        # Drives straight down the field (+x): heading should settle near 0 rad.
        samples = [{"t": i * 0.25, "x": 5.0 + i * 2.0, "y": 13.0} for i in range(4)]
        _, _, records = parse_wpilog(match_to_wpilog([timeline("r", samples)]))
        *_, last = records
        _, _, heading = struct.unpack("<3d", last[2])
        self.assertAlmostEqual(heading, 0.0, delta=1e-6)
        # And sideways (+y) settles near +pi/2.
        sideways = [{"t": i * 0.25, "x": 20.0, "y": 5.0 + i * 2.0} for i in range(4)]
        _, _, recs2 = parse_wpilog(match_to_wpilog([timeline("r", sideways)]))
        _, _, heading2 = struct.unpack("<3d", recs2[-1][2])
        self.assertAlmostEqual(heading2, math.pi / 2, delta=1e-6)

    def test_unpositioned_samples_are_skipped(self):
        samples = [{"t": 0.0, "x": 1.0, "y": 1.0},
                   {"t": 0.25, "x": None, "y": None},  # unseen: no pose written
                   {"t": 0.5, "x": 2.0, "y": 1.0}]
        _, _, records = parse_wpilog(match_to_wpilog([timeline("r", samples)]))
        self.assertEqual(len(records), 2)
        self.assertEqual([ts for _, ts, _ in records], [0, 500_000])

    def test_robot_without_team_uses_its_key(self):
        timelines = [timeline("robot5", [{"t": 0.0, "x": 1.0, "y": 1.0}], alliance="red")]
        _, entries, _ = parse_wpilog(match_to_wpilog(timelines))
        self.assertEqual(next(iter(entries.values()))["name"], "/Tengen/red/robot5")


if __name__ == "__main__":
    unittest.main()
