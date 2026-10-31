"""The strategy engine (PLAN-2026 section 6): profiles -> an alliance match plan.

Synthetic profiles here, so each rule is pinned on its own: the engine is rules over the profile
fields (auto_map, scoring_map.effective_zones, defense tendency), and these check each rule fires
on the evidence it is supposed to and stays quiet without it.
"""

import unittest

from ingest.strategy.engine import (
    aggregate_zones,
    auto_conflicts,
    build_matchplan,
    defense_watch,
    predictable_autos,
)


def profile(team, *, auto=None, zones=None, defense=None):
    """A profile with only the fields the engine reads."""
    p = {"team": team}
    if auto is not None:
        p["auto_map"] = {"heatmap": [{"x": x, "y": y, "samples": s, "weight": w, "matches": m,
                                      "intensity": i} for (x, y, s, w, m, i) in auto]}
    if zones is not None:
        p["scoring_map"] = {"effective_zones": [{"zone": z, "lane": l, "weight": w, "matches": m,
                                                 "share": sh, "effective": True}
                                                for (z, l, w, m, sh) in zones]}
    if defense is not None:
        p["tendencies"] = {"defense": {"matches_played": defense[0], "of": defense[1]}}
    return p


class AutoConflictTests(unittest.TestCase):
    def test_partners_in_the_same_auto_cell_conflict(self):
        a = profile(254, auto=[(10.5, 13.5, 8, 6.0, 4, 1.0)])
        b = profile(1678, auto=[(10.5, 13.5, 7, 5.0, 4, 0.9)])
        conflicts = auto_conflicts([a, b])
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["teams"], [254, 1678])
        self.assertEqual((conflicts[0]["x"], conflicts[0]["y"]), (10.5, 13.5))

    def test_partners_in_different_cells_do_not_conflict(self):
        a = profile(254, auto=[(10.5, 13.5, 8, 6.0, 4, 1.0)])
        b = profile(1678, auto=[(40.5, 20.5, 7, 5.0, 4, 1.0)])
        self.assertEqual(auto_conflicts([a, b]), [])

    def test_a_cold_overlap_is_not_a_conflict(self):
        # Both have a sample in the cell, but one is below the hot threshold: not a real contest.
        a = profile(254, auto=[(10.5, 13.5, 8, 6.0, 4, 1.0)])
        b = profile(1678, auto=[(10.5, 13.5, 1, 0.3, 1, 0.1)])
        self.assertEqual(auto_conflicts([a, b]), [])


class PredictableAutoTests(unittest.TestCase):
    def test_a_concentrated_auto_is_flagged(self):
        p = profile(148, auto=[(40.5, 13.5, 20, 16.0, 5, 1.0), (10.5, 4.5, 2, 1.0, 1, 0.06)])
        flagged = predictable_autos([p])
        self.assertEqual(len(flagged), 1)
        self.assertEqual(flagged[0]["team"], 148)
        self.assertGreaterEqual(flagged[0]["share"], 0.4)

    def test_a_scattered_auto_is_not_flagged(self):
        p = profile(148, auto=[(40.5, 13.5, 5, 4.0, 3, 1.0), (10.5, 4.5, 5, 3.9, 3, 0.97),
                               (25.5, 13.5, 5, 3.8, 3, 0.95)])
        self.assertEqual(predictable_autos([p]), [])


class ZoneAndDefenseTests(unittest.TestCase):
    def test_zones_pool_by_weight_across_teams(self):
        a = profile(254, zones=[("opponent_alliance_zone", "centre", 10.0, 5, 0.6)])
        b = profile(1678, zones=[("opponent_alliance_zone", "centre", 6.0, 4, 0.5),
                                 ("neutral_zone", "left", 3.0, 3, 0.3)])
        pooled = aggregate_zones([a, b])
        self.assertEqual((pooled[0]["zone"], pooled[0]["lane"]), ("opponent_alliance_zone", "centre"))
        self.assertEqual(pooled[0]["weight"], 16.0)
        self.assertEqual(pooled[0]["teams"], [254, 1678])

    def test_defense_watch_fires_only_above_the_share(self):
        defender = profile(148, defense=(4, 6))     # 67% of matches
        occasional = profile(118, defense=(1, 6))   # 17%
        watched = defense_watch([defender, occasional])
        self.assertEqual([w["team"] for w in watched], [148])


class MatchPlanTests(unittest.TestCase):
    def test_a_full_plan_names_each_call_with_evidence(self):
        profiles = {
            254: profile(254, auto=[(10.5, 13.5, 8, 6.0, 4, 1.0)],
                         zones=[("own_alliance_zone", "right", 8.0, 5, 0.7)]),
            1678: profile(1678, auto=[(10.5, 13.5, 7, 5.0, 4, 0.9)]),  # conflicts with 254 in auto
            148: profile(148, auto=[(40.5, 13.5, 20, 16.0, 5, 1.0)],   # predictable opponent auto
                         zones=[("opponent_alliance_zone", "centre", 12.0, 6, 0.6)],
                         defense=(4, 6)),                               # and a defender
        }
        plan = build_matchplan([254, 1678], [148, 971], profiles)
        self.assertEqual(plan["missing"], [971])  # no profile for 971
        ids = {a["id"] for a in plan["advice"]}
        self.assertIn("auto_partner_conflict", ids)   # 254 & 1678 share an auto cell
        self.assertIn("contest_opponent_auto", ids)   # 148's auto is predictable
        self.assertIn("deny_scoring", ids)            # 148's opponent scoring zone
        self.assertIn("expect_defense", ids)          # 148 defends
        self.assertIn("play_to_strength", ids)        # our own effective zone
        self.assertTrue(all(a["advice"] and a["evidence"] for a in plan["advice"]))

    def test_no_data_yields_no_calls_not_a_crash(self):
        plan = build_matchplan([254], [148], {254: profile(254), 148: profile(148)})
        self.assertEqual(plan["advice"], [])


if __name__ == "__main__":
    unittest.main()
