"""The scoring heatmap: where a team scores, weighted toward recent matches, practice included.

These pin the behaviour the plan asks for (docs/PLAN-2026.md section 5): trust scoring locations
even from practice, and let the latest matches count for more.
"""

import unittest

from ingest.strategy.heatmap import (
    effective_zones,
    is_practice,
    ordered_with_recency,
    scoring_heatmap,
    scoring_map,
)
from ingest.strategy.timeline import load_geometry

GEOMETRY = load_geometry(2026)


def match(match_id, points, team=254):
    """A feature dict with just what the scoring map reads: match id and shooting points (feet)."""
    return {"match_id": match_id, "team": team, "shooting": {"points": [(x, y) for x, y in points]}}


class PracticeTests(unittest.TestCase):
    def test_official_ids_are_not_practice_unknown_ones_are(self):
        self.assertFalse(is_practice(match("2026casf_qm42", [])))
        self.assertFalse(is_practice(match("2026casf_f1m2", [])))
        self.assertTrue(is_practice(match("scrimmage-tuesday", [])))
        self.assertTrue(is_practice({"match_id": "2026casf_qm1", "practice": True, "shooting": {"points": []}}))

    def test_practice_is_oldest_and_weighted_least_but_kept(self):
        matches = [match("practice-1", [(15, 13)]), match("2026x_qm1", [(15, 13)]),
                   match("2026x_qm2", [(15, 13)])]
        ordered = ordered_with_recency(matches)
        # Practice sorts first (oldest), the latest qualifier last (newest, weight 1.0).
        self.assertEqual(ordered[0][0]["match_id"], "practice-1")
        self.assertEqual(ordered[-1][1], 1.0)
        self.assertLess(ordered[0][1], ordered[-1][1])
        # But its scoring still lands on the map.
        self.assertTrue(scoring_heatmap(matches, GEOMETRY))


class HeatmapTests(unittest.TestCase):
    def test_recent_matches_count_for_more(self):
        # Two spots, each scored once, but in different matches: the newer one should be hotter.
        matches = [match("2026x_qm1", [(10.0, 4.0)]),    # old spot
                   match("2026x_qm2", [(40.0, 20.0)])]   # newest spot
        cells = scoring_heatmap(matches, GEOMETRY)
        hottest = cells[0]
        self.assertAlmostEqual(hottest["x"], 40.5, delta=3.0)
        self.assertAlmostEqual(hottest["y"], 19.5, delta=3.0)
        self.assertEqual(hottest["intensity"], 1.0)
        # The older, equal-count spot is present but cooler.
        self.assertLess(cells[1]["intensity"], 1.0)

    def test_effective_zone_is_where_they_actually_score(self):
        # A team that always scores from its own alliance zone, right lane.
        matches = [match(f"2026x_qm{i}", [(10.0, 4.0)] * 4) for i in range(1, 5)]
        zones = effective_zones(matches, GEOMETRY)
        effective = [z for z in zones if z["effective"]]
        self.assertEqual(len(effective), 1)
        self.assertEqual((effective[0]["zone"], effective[0]["lane"]), ("own_alliance_zone", "right"))
        self.assertEqual(effective[0]["share"], 1.0)

    def test_one_off_spot_is_not_called_effective(self):
        # Scored there once; real tendencies need 2+ matches.
        matches = [match("2026x_qm1", [(10.0, 4.0)]), match("2026x_qm2", [(40.0, 20.0)])]
        self.assertEqual([z for z in effective_zones(matches, GEOMETRY) if z["effective"]], [])

    def test_scoring_map_points_at_the_hot_spot_and_counts_practice(self):
        matches = [match("practice-a", [(39.0, 13.0)]),
                   match("2026x_qm1", [(39.0, 13.0)] * 3),
                   match("2026x_qm2", [(39.0, 13.0)] * 3)]
        result = scoring_map(matches, GEOMETRY)
        self.assertEqual(result["n_matches"], 3)
        self.assertEqual(result["n_practice"], 1)
        self.assertAlmostEqual(result["hot_spot"]["x"], 40.5, delta=3.0)

    def test_no_positions_gives_an_empty_map_not_a_crash(self):
        matches = [match("2026x_qm1", []), match("2026x_qm2", [])]
        result = scoring_map(matches, GEOMETRY)
        self.assertEqual(result["heatmap"], [])
        self.assertIsNone(result["hot_spot"])
        self.assertEqual(result["effective_zones"], [])


if __name__ == "__main__":
    unittest.main()
