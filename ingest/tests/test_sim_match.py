"""T2: multi-robot simulation with blocking, and the turn-rate limit.

These pin the two things T2 adds over T1: robots can't drive through each other, and a defender
that shadows an attacker actually slows it down -- the whole reason to simulate defense.
"""

import math
import unittest

from ingest.sim.match import SimRobot, reached, simulate_match
from ingest.sim.model import ROBOT_RADIUS_FT, RobotParams, _apply_limits, simulate_path


class TurnRateTests(unittest.TestCase):
    def test_the_velocity_heading_cannot_rotate_faster_than_the_limit(self):
        # Moving +x at 10 ft/s, asked to go +y (a 90 deg turn), with acceleration not the binding
        # constraint: the heading may only swing by max_turn_radps * dt this step.
        params = RobotParams(max_accel_ftps2=1000.0, max_turn_radps=1.0)
        nvx, nvy = _apply_limits(10.0, 0.0, 0.0, 10.0, params, dt=0.1)
        self.assertLessEqual(abs(math.atan2(nvy, nvx)), 1.0 * 0.1 + 1e-6)  # <= 0.1 rad

    def test_a_turn_limited_robot_overshoots_a_corner(self):
        # Fast down +x, then a 90 deg corner to +y. A low turn rate means it keeps going +x while it
        # swings around, overshooting well past the corner's x -- what a tank drive really does.
        tight = RobotParams(max_turn_radps=1.0)
        poses = simulate_path((0.0, 0.0), [(20.0, 0.0), (20.0, 25.0)], tight, max_time_s=8.0)
        self.assertGreater(max(p.x for p in poses), 23.0)

    def test_the_default_stays_holonomic(self):
        # With the large default turn rate, a straight drive stays on the axis (T1 unchanged).
        poses = simulate_path((0.0, 0.0), [(20.0, 0.0)])
        self.assertLess(max(abs(p.y) for p in poses), 0.1)


class BlockingTests(unittest.TestCase):
    def test_robots_do_not_overlap(self):
        # Two robots aimed at each other's start must not pass through -- they jam at contact.
        a = SimRobot("a", (0.0, 0.0), waypoints=[(20.0, 0.0)])
        b = SimRobot("b", (20.0, 0.0), waypoints=[(0.0, 0.0)])
        tracks = simulate_match([a, b], duration_s=6.0)
        min_sep = 2 * ROBOT_RADIUS_FT
        closest = min(math.hypot(pa.x - pb.x, pa.y - pb.y)
                      for pa, pb in zip(tracks["a"], tracks["b"]))
        self.assertGreaterEqual(closest, min_sep - 0.3)

    def test_a_defender_slows_the_attacker(self):
        goal = (24.0, 0.0)
        attacker_path = [goal]
        # Unopposed: how fast does the attacker reach the goal?
        solo = simulate_match([SimRobot("att", (0.0, 0.0), waypoints=attacker_path)], duration_s=8.0)
        solo_t = reached(solo["att"], goal)
        self.assertIsNotNone(solo_t)
        # Now put a defender in the way, shadowing the attacker from just ahead of the goal.
        defended = simulate_match([
            SimRobot("att", (0.0, 0.0), waypoints=attacker_path),
            SimRobot("def", (12.0, 0.0), shadow="att"),
        ], duration_s=8.0)
        def_t = reached(defended["att"], goal)
        # Either the attacker is delayed, or it never gets there in the window.
        self.assertTrue(def_t is None or def_t > solo_t + 0.3)

    def test_a_shadow_defender_chases_its_target(self):
        tracks = simulate_match([
            SimRobot("att", (0.0, 0.0), waypoints=[(30.0, 0.0)]),
            SimRobot("def", (30.0, 10.0), shadow="att"),
        ], duration_s=6.0)
        # The defender should end up near the attacker, not stuck at its own start.
        last_att, last_def = tracks["att"][-1], tracks["def"][-1]
        start_gap = math.hypot(30.0 - 0.0, 10.0 - 0.0)
        end_gap = math.hypot(last_def.x - last_att.x, last_def.y - last_att.y)
        self.assertLess(end_gap, start_gap)


if __name__ == "__main__":
    unittest.main()
