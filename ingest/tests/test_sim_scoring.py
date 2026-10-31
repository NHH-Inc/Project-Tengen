"""T3: the game model and scoring, driven by the season config.

Checks the config-driven loader (placeholder zeros load cleanly), the phase/range scoring, and the
payoff that ties the tiers together: a defended attacker scores less because T2 keeps it out of range.
"""

import unittest

from ingest.sim.game import DEFAULT_SHOT_RANGE_FT, Goal, GameModel, load_game_model
from ingest.sim.match import SimRobot, simulate_match
from ingest.sim.model import Pose
from ingest.sim.scoring import Shooter, score_tracks, shooter_from_profile
from ingest.strategy.timeline import load_geometry

GEOMETRY = load_geometry(2026)

SEASON_2026 = {
    "auto_seconds": 20, "teleop_seconds": 140, "endgame_seconds": 30,
    "game_pieces": ["ball"], "goals": ["high", "low"],
    "point_values": {"auto": {"shot_made_high": 0, "shot_made_low": 0},
                     "teleop": {"shot_made_high": 0, "shot_made_low": 0}, "endgame": {}},
}


class GameModelTests(unittest.TestCase):
    def test_it_loads_the_placeholder_config_without_error(self):
        game = load_game_model(SEASON_2026, GEOMETRY)
        self.assertEqual(game.pieces, ("ball",))
        self.assertEqual({g.name for g in game.goals}, {"high", "low"})
        self.assertEqual(game.goal("high").points["teleop"], 0.0)  # placeholder
        # Goals sit at the opponent hub (we score in the opponent's, in the alliance frame).
        self.assertAlmostEqual(game.goal("high").x, GEOMETRY["hubs"]["opponent_hub"]["x"], places=3)
        self.assertEqual(game.goal("high").effective_range_ft, DEFAULT_SHOT_RANGE_FT)

    def test_phase_boundaries(self):
        game = load_game_model(SEASON_2026, GEOMETRY)
        self.assertEqual(game.phase_at(5), "auto")          # < 20
        self.assertEqual(game.phase_at(60), "teleop")       # 20 .. 130
        self.assertEqual(game.phase_at(135), "endgame")     # last 30 s of 20+140
        self.assertEqual(game.match_seconds, 160)


def _game_with_points(teleop_high: float, range_ft: float = DEFAULT_SHOT_RANGE_FT) -> GameModel:
    hub = GEOMETRY["hubs"]["opponent_hub"]
    goal = Goal("high", hub["x"], hub["y"], range_ft,
                {"auto": 0.0, "teleop": teleop_high, "endgame": 0.0})
    return GameModel(("ball",), (goal,), 20.0, 140.0, 30.0)


class ScoringTests(unittest.TestCase):
    def test_a_shooter_in_range_scores_its_expected_value(self):
        game = _game_with_points(teleop_high=2.0)
        hub = GEOMETRY["hubs"]["opponent_hub"]
        # Parked on the hub for 40 s of teleop, firing every 10 s, 3 shots/volley, 50% accuracy.
        poses = [Pose(round(30 + i * 0.5, 1), hub["x"], hub["y"]) for i in range(80)]  # t=30..69.5
        shooter = Shooter("254", "red", "high", volley_interval_s=10.0, shots_per_volley=3.0, accuracy=0.5)
        report = score_tracks({"254": poses}, [shooter], game, dt=0.5)
        # ~40 s in range / 10 s = 4 volleys * 3 * 0.5 made * 2 pts = ~12 pts.
        self.assertAlmostEqual(report["by_alliance"]["red"], 12.0, delta=1.5)

    def test_out_of_range_scores_nothing(self):
        game = _game_with_points(teleop_high=2.0)
        far = [Pose(round(30 + i * 0.5, 1), 1.0, 1.0) for i in range(40)]  # nowhere near the hub
        shooter = Shooter("254", "red", "high", 10.0, 3.0, 0.5)
        self.assertEqual(score_tracks({"254": far}, [shooter], game, dt=0.5)["total"], 0.0)

    def test_placeholder_values_score_zero(self):
        game = load_game_model(SEASON_2026, GEOMETRY)
        hub = GEOMETRY["hubs"]["opponent_hub"]
        poses = [Pose(round(30 + i * 0.5, 1), hub["x"], hub["y"]) for i in range(80)]
        shooter = Shooter("254", "red", "high", 10.0, 3.0, 0.6)
        self.assertEqual(score_tracks({"254": poses}, [shooter], game, dt=0.5)["total"], 0.0)

    def test_shooter_from_profile_reads_cadence(self):
        profile = {"team": 254, "timing": {"median_launches_per_volley": 4},
                   "tendencies": {"shooting": {"median_volley_interval_s": 6.0}}}
        s = shooter_from_profile(profile, "red", "high")
        self.assertEqual(s.volley_interval_s, 6.0)
        self.assertEqual(s.shots_per_volley, 4.0)


class DefenseScoringTests(unittest.TestCase):
    def test_defense_keeps_the_attacker_out_of_range_longer(self):
        # The payoff: run T2, then score. An attacker drives to the hub to shoot at close range; a
        # defender camps on the goal and body-blocks it. The scorer needs no notion of defense -- it
        # just sees the attacker spend less time in range, which is the mechanism points flow from.
        # (A determined attacker eventually shoves through, so the clean, robust signal is the
        # reduced time in range; point totals follow once the game carries non-zero values and the
        # block holds into teleop.)
        game = _game_with_points(teleop_high=2.0, range_ft=2.5)
        hub = (GEOMETRY["hubs"]["opponent_hub"]["x"], GEOMETRY["hubs"]["opponent_hub"]["y"])
        shooter = Shooter("att", "red", "high", volley_interval_s=4.0, shots_per_volley=3.0, accuracy=0.6)

        solo = simulate_match([SimRobot("att", (hub[0] - 20, hub[1]), waypoints=[hub])], duration_s=15.0)
        defended = simulate_match([
            SimRobot("att", (hub[0] - 20, hub[1]), waypoints=[hub]),
            SimRobot("def", hub, waypoints=[hub]),  # a defender camped on the goal
        ], duration_s=15.0)

        solo_t = score_tracks(solo, [shooter], game)["shooters"][0]["time_in_range_s"]
        defended_rows = score_tracks(defended, [shooter], game)["shooters"]
        defended_t = defended_rows[0]["time_in_range_s"] if defended_rows else 0.0

        self.assertGreater(solo_t, 0.0)
        self.assertLess(defended_t, solo_t)


if __name__ == "__main__":
    unittest.main()
