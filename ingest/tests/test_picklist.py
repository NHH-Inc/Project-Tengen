"""The alliance-pick engine (PLAN-2026 section 7): profiles -> our own pick list and pick calls.

Synthetic profiles, so each piece is pinned alone: power from shots/match, conflict from shared
auto cells and scoring zones, and the ranking that trades the two off.
"""

import unittest

from ingest.strategy.picklist import (
    CONFLICT_PENALTY,
    alliance_strength,
    best_pick,
    conflict,
    offensive_power,
    rank_partners,
    should_accept,
)


def profile(team, *, shots=0.0, auto=None, zones=None):
    p = {"team": team, "launches_per_match": shots}
    p["auto_map"] = {"heatmap": [{"x": x, "y": y, "intensity": i} for (x, y, i) in (auto or [])]}
    p["scoring_map"] = {"effective_zones": [{"zone": z, "lane": l, "weight": w}
                                            for (z, l, w) in (zones or [])]}
    return p


class PowerTests(unittest.TestCase):
    def test_power_is_shots_per_match(self):
        self.assertEqual(offensive_power(profile(254, shots=12)), 12.0)

    def test_unknown_power_is_zero(self):
        self.assertEqual(offensive_power({"team": 1}), 0.0)


class ConflictTests(unittest.TestCase):
    def test_shared_auto_cell_and_zone_both_count(self):
        a = profile(254, auto=[(10.5, 13.5, 1.0)], zones=[("opp", "centre", 6.0)])
        b = profile(1678, auto=[(10.5, 13.5, 0.8)], zones=[("opp", "centre", 4.0)])
        c = conflict(a, b)
        self.assertAlmostEqual(c["auto"], 0.8)   # min(1.0, 0.8)
        self.assertAlmostEqual(c["zones"], 4.0)  # min(6, 4)
        self.assertAlmostEqual(c["total"], 4.8)

    def test_disjoint_robots_do_not_conflict(self):
        a = profile(254, auto=[(10.5, 13.5, 1.0)], zones=[("own", "right", 5.0)])
        b = profile(1678, auto=[(40.5, 20.5, 1.0)], zones=[("opp", "left", 5.0)])
        self.assertEqual(conflict(a, b)["total"], 0.0)


class RankTests(unittest.TestCase):
    def test_a_high_scorer_that_does_not_conflict_ranks_first(self):
        us = profile(254, shots=12, zones=[("opp", "centre", 8.0)])
        profiles = {
            254: us,
            111: profile(111, shots=10, zones=[("own", "right", 6.0)]),      # no conflict, strong
            222: profile(222, shots=10, zones=[("opp", "centre", 8.0)]),     # same strong zone as us
            333: profile(333, shots=3),                                       # weak
        }
        ranked = rank_partners([254], [111, 222, 333], profiles)
        self.assertEqual([r["team"] for r in ranked][0], 111)
        # 222 is penalised for contesting our own best zone, so it falls behind 111.
        self.assertLess(next(r["score"] for r in ranked if r["team"] == 222),
                        next(r["score"] for r in ranked if r["team"] == 111))

    def test_our_own_team_and_unknown_teams_are_excluded(self):
        profiles = {254: profile(254, shots=12), 111: profile(111, shots=8)}
        ranked = rank_partners([254], [254, 111, 999], profiles)  # 999 has no profile
        self.assertEqual([r["team"] for r in ranked], [111])

    def test_conflict_penalty_can_outweigh_raw_power(self):
        us = profile(254, shots=10, auto=[(10.5, 13.5, 1.0)], zones=[("opp", "centre", 10.0)])
        # A slightly stronger scorer that stomps all over our auto and zone...
        clash = profile(2, shots=11, auto=[(10.5, 13.5, 1.0)], zones=[("opp", "centre", 10.0)])
        # ...versus a hair weaker one that complements us.
        clean = profile(3, shots=10, zones=[("own", "left", 5.0)])
        profiles = {254: us, 2: clash, 3: clean}
        best = best_pick([254], [2, 3], profiles)
        self.assertEqual(best["team"], 3)


class AllianceAndAcceptTests(unittest.TestCase):
    def test_alliance_strength_sums_power_and_subtracts_internal_conflict(self):
        profiles = {
            1: profile(1, shots=10, zones=[("opp", "centre", 5.0)]),
            2: profile(2, shots=8, zones=[("opp", "centre", 4.0)]),  # conflicts with 1
        }
        s = alliance_strength([1, 2], profiles)
        self.assertEqual(s["power"], 18.0)
        self.assertEqual(s["internal_conflict"], 4.0)  # min(5,4)
        self.assertAlmostEqual(s["score"], 18.0 - CONFLICT_PENALTY * 4.0)

    def test_accept_when_the_offered_alliance_is_stronger(self):
        profiles = {
            254: profile(254, shots=9),
            9: profile(9, shots=14, zones=[("opp", "centre", 9.0)]),   # strong captain offering us
            8: profile(8, shots=13),                                    # their existing pick
            5: profile(5, shots=4),                                     # the best we could pick alone
        }
        decision = should_accept(254, [9, 8], [5], profiles)
        self.assertEqual(decision["recommendation"], "accept")
        self.assertEqual(decision["decline_best_pick"], 5)

    def test_decline_when_we_can_build_better_ourselves(self):
        profiles = {
            254: profile(254, shots=12),
            9: profile(9, shots=3),       # a weak alliance is offering
            8: profile(8, shots=3),
            7: profile(7, shots=13),      # but a powerhouse is still available to us
        }
        decision = should_accept(254, [9, 8], [7], profiles)
        self.assertEqual(decision["recommendation"], "decline")


if __name__ == "__main__":
    unittest.main()
