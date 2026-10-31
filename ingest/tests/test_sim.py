"""The sim core (PLAN-2026 section 2): the motion model and the validation harness.

The model tests pin the physics that a constant-speed replay cannot give -- acceleration, a speed
cap, and braking onto the last waypoint. The harness test is self-consistency: a path the model
produced, fed back as 'actual', must be retraced within a fraction of a foot.
"""

import math
import unittest

from ingest.sim.model import (
    DEFAULT_MAX_SPEED_FTPS,
    Pose,
    RobotParams,
    params_from_profile,
    pose_at,
    simulate_path,
)
from ingest.sim.validate import auto_waypoints, trajectory_error, validate_auto


def auto_timeline(points, team=254, match_id="2026x_qm1"):
    """A timeline whose columnar samples are `points` = [(t, phase, x, y), ...]."""
    keys = ("t", "mt", "phase", "x", "y", "ax", "ay", "speed", "zone", "lane")
    cols = {k: [] for k in keys}
    for t, phase, x, y in points:
        cols["t"].append(t); cols["mt"].append(t); cols["phase"].append(phase)
        cols["x"].append(x); cols["y"].append(y); cols["ax"].append(x); cols["ay"].append(y)
        cols["speed"].append(0.0); cols["zone"].append(None); cols["lane"].append(None)
    return {"team": team, "match_id": match_id, "robot_key": "robot1", "samples": cols}


class MotionModelTests(unittest.TestCase):
    def test_it_reaches_the_final_waypoint(self):
        poses = simulate_path((5.0, 13.0), [(10.0, 13.0), (20.0, 13.0)])
        last = poses[-1]
        self.assertAlmostEqual(last.x, 20.0, delta=1.0)
        self.assertAlmostEqual(last.y, 13.0, delta=1.0)

    def test_it_never_exceeds_top_speed(self):
        params = RobotParams(max_speed_ftps=10.0)
        dt = 0.02
        poses = simulate_path((0.0, 0.0), [(50.0, 0.0)], params, dt=dt)
        for a, b in zip(poses, poses[1:]):
            step_speed = math.hypot(b.x - a.x, b.y - a.y) / dt
            self.assertLessEqual(step_speed, params.max_speed_ftps + 0.3)

    def test_it_accelerates_from_rest(self):
        # A constant-speed replay would jump to top speed on step one; a real robot ramps up.
        dt = 0.02
        poses = simulate_path((0.0, 0.0), [(50.0, 0.0)], RobotParams(max_speed_ftps=12.0), dt=dt)
        first_step_speed = math.hypot(poses[1].x - poses[0].x, poses[1].y - poses[0].y) / dt
        self.assertLess(first_step_speed, 2.0)  # still slow right after the start

    def test_pose_at_interpolates(self):
        poses = [Pose(0.0, 0.0, 0.0), Pose(1.0, 10.0, 0.0)]
        self.assertEqual(pose_at(poses, 0.5), (5.0, 0.0))
        self.assertIsNone(pose_at(poses, 2.0))

    def test_params_from_profile_uses_measured_top_speed(self):
        profile = {"tendencies": {"movement": {"top_speed_ftps": 16.5}}}
        self.assertEqual(params_from_profile(profile).max_speed_ftps, 16.5)
        self.assertEqual(params_from_profile({}).max_speed_ftps, DEFAULT_MAX_SPEED_FTPS)


class HarnessTests(unittest.TestCase):
    def test_trajectory_error_measures_position_gap(self):
        sim = [Pose(0.0, 0.0, 0.0), Pose(1.0, 10.0, 0.0)]
        actual = [(0.0, 0.0, 0.0), (1.0, 10.0, 3.0)]  # 3 ft off at t=1
        err = trajectory_error(sim, actual)
        self.assertEqual(err["n"], 2)
        self.assertAlmostEqual(err["max_ft"], 3.0)

    def test_auto_waypoints_ignores_non_auto_and_unseen(self):
        tl = auto_timeline([
            (0.0, "auto", 5.0, 13.0), (0.1, "auto", 6.0, 13.0), (0.2, "auto", 7.0, 13.0),
            (0.3, "auto", 8.0, 13.0), (0.4, "auto", None, None),  # unseen: dropped
            (3.0, "teleop", 30.0, 13.0),                           # not auto: dropped
        ])
        found = auto_waypoints(tl)
        self.assertIsNotNone(found)
        start, waypoints, actual = found
        self.assertEqual(start, (5.0, 13.0))
        self.assertTrue(all(y == 13.0 for _, _, y in actual))  # only the auto, positioned samples

    def test_model_retraces_a_straight_auto_closely(self):
        # Self-consistency on a straight path (no corner to cut): a model-produced path, fed back as
        # "actual", must be retraced within a fraction of a foot -- the only slack is downsampling.
        truth = simulate_path((5.0, 13.0), [(25.0, 13.0)])
        points = [(p.t, "auto", p.x, p.y) for p in truth if abs(p.t * 10 - round(p.t * 10)) < 1e-6]
        report = validate_auto(auto_timeline(points))
        self.assertIsNotNone(report)
        self.assertEqual(report["team"], 254)
        self.assertLess(report["error"]["mean_ft"], 1.0)
        self.assertAlmostEqual(report["sim_duration_s"], report["actual_duration_s"], delta=1.0)

    def test_a_cornered_auto_reproduces_within_a_few_feet(self):
        # With a corner the model cuts it (it brakes only for the final waypoint), so expect a few
        # feet, not inches. This documents the T1 limit that the real-match validation will measure.
        truth = simulate_path((5.0, 13.0), [(15.0, 13.0), (15.0, 20.0)])
        points = [(p.t, "auto", p.x, p.y) for p in truth if abs(p.t * 10 - round(p.t * 10)) < 1e-6]
        report = validate_auto(auto_timeline(points))
        self.assertIsNotNone(report)
        self.assertLess(report["error"]["mean_ft"], 4.0)

    def test_an_idle_robot_has_nothing_to_validate(self):
        self.assertIsNone(validate_auto(auto_timeline([(0.0, "teleop", 5.0, 13.0)])))


if __name__ == "__main__":
    unittest.main()
