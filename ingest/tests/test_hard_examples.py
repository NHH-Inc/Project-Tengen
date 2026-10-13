"""The miner must point at the detector's mistakes, not at the clip's or the camera's."""

import unittest

from ingest.collection.hard_examples import Moment, find_moments, select


def boxes(start, end, x=0.5, rate=10, skip=()):
    out = []
    for i in range(int((end - start) * rate) + 1):
        t = round(start + i / rate, 3)
        if not any(a < t < b for a, b in skip):
            out.append({"t": t, "x": x - 0.03, "y": 0.47, "w": 0.06, "h": 0.06})
    return out


def track(track_id, start, end, **kwargs):
    gaps = kwargs.pop("gaps", [])
    return {"track_id": track_id, "boxes": boxes(start, end, **kwargs), "gaps": gaps}


def reasons(moments):
    return sorted(m.reason for m in moments)


class FindMomentsTests(unittest.TestCase):
    def test_a_track_spanning_the_clip_is_not_a_mistake(self):
        self.assertEqual(find_moments([track(1, 0, 60)], duration=60), [])

    def test_a_robot_vanishing_mid_field_is(self):
        moments = find_moments([track(1, 0, 30)], duration=60)
        self.assertEqual(reasons(moments), ["vanished"])
        self.assertAlmostEqual(moments[0].t, 30.2)

    def test_leaving_through_the_edge_of_the_view_is_not(self):
        self.assertEqual(find_moments([track(1, 0, 30, x=0.98)], duration=60), [])

    def test_an_unlisted_hole_is_a_miss_but_a_camera_cut_is_not(self):
        holed = track(1, 0, 60, skip=[(20, 22)])
        self.assertEqual(reasons(find_moments([holed], duration=60)), ["hole"])
        cut = track(1, 0, 60, skip=[(20, 22)], gaps=[{"start": 20, "end": 22, "reason": "shot_change"}])
        self.assertEqual(find_moments([cut], duration=60), [])

    def test_a_sub_second_track_is_a_flicker(self):
        self.assertEqual(reasons(find_moments([track(1, 0, 60), track(2, 10, 10.4)], 60)), ["flicker"])

    def test_a_tiny_persistent_box_is_not_a_robot(self):
        tiny = track(9, 0, 60, x=0.5)
        for box in tiny["boxes"]:
            box["w"] = box["h"] = 0.012
        moments = find_moments([track(i, 0, 60) for i in range(5)] + [tiny], duration=60)
        self.assertEqual(sorted({m.reason for m in moments}), ["undersized"])
        self.assertTrue(all(m.track_id == 9 for m in moments))

    def test_several_robots_missing_at_once_is_a_count_drop(self):
        tracks = [track(i, 0, 60) for i in range(6)]
        for t in tracks[:3]:
            t["boxes"] = [b for b in t["boxes"] if not 30 <= b["t"] <= 35]
            t["gaps"] = [{"start": 30, "end": 35, "reason": "occlusion"}]
        drops = [m for m in find_moments(tracks, 60) if m.reason == "count_drop"]
        self.assertTrue(drops)
        self.assertTrue(all(30 <= m.t <= 35 for m in drops))


class SelectTests(unittest.TestCase):
    def test_spread_out_and_every_reason_gets_a_turn(self):
        moments = [Moment(10.0, "flicker", 1, 0.5), Moment(10.5, "hole", 2, 1.0),
                   Moment(30.0, "vanished", 3, 1.0), Moment(31.0, "count_drop", None, 2.0)]
        chosen = select(moments, limit=10)
        self.assertEqual([(m.t, m.reason) for m in chosen], [(10.5, "hole"), (30.0, "vanished")])
        self.assertEqual(select(moments, limit=1)[0].reason, "hole")
        many = [Moment(float(t), "hole", 1, 1.0) for t in range(0, 100, 5)] + [Moment(52.0, "flicker", 2, 0.5)]
        self.assertIn("flicker", [m.reason for m in select(many, limit=3)])


if __name__ == "__main__":
    unittest.main()
