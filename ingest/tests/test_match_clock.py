"""Match time must come from the clock the referees ran, and survive the clock being misread.

The readings below are modelled on two real 2026 matches: the clock frozen at 0:20 before auto
(and once misread as 29), frozen at 2:20 before teleop (misread as 22 every time), single-digit
seconds misread at random, and nothing readable during the title card.
"""

import random
import unittest

from ingest.strategy.clock import MatchClock, fit_clock, fixed_clock

SEASON = {"auto_seconds": 20, "teleop_seconds": 110, "endgame_seconds": 30}
AUTO_START, PAUSE, TELEOP_START = 7.3, 4.2, 31.5  # teleop starts after auto's 20 s and a pause


def true_clock(t):
    if t < AUTO_START:
        return 20
    if t < AUTO_START + 20:
        return int(AUTO_START + 20 - t + 0.999)
    if t < TELEOP_START:
        return 140
    return max(0, int(TELEOP_START + 140 - t + 0.999))


def readings(start=0, end=215, misread=0.1, seed=3):
    rng = random.Random(seed)
    out = []
    for t in range(start, end):
        value = None if t < 3 else true_clock(t)
        if value == 140:
            value = 22          # "2:20" loses its zero, as it did on every real frame
        elif value == 20 and t < AUTO_START:
            value = 29 if t < 5 else 20
        elif value is not None and value < 10 and rng.random() < 0.5:
            value = rng.randint(0, 9)   # single digits are the least reliable reads
        elif value is not None and rng.random() < misread:
            value = rng.randint(0, 199)
        out.append((float(t), value))
    return out


class FitTests(unittest.TestCase):
    def assertFits(self, clock, auto=AUTO_START, teleop=TELEOP_START):
        self.assertAlmostEqual(clock.match_start_t, auto, delta=1.0)
        self.assertAlmostEqual(clock.teleop_start_t, teleop, delta=1.0)
        self.assertEqual(clock.auto_seconds, 20)
        self.assertEqual(clock.teleop_seconds, 140)

    def test_a_whole_broadcast_clip(self):
        clock = fit_clock(readings(), SEASON)
        self.assertFits(clock)
        self.assertEqual(clock.source, "scoreboard_clock")
        self.assertEqual(clock.notes, [])

    def test_survives_heavy_misreads(self):
        for seed in range(20):
            self.assertFits(fit_clock(readings(misread=0.3, seed=seed), SEASON))

    def test_a_clip_starting_mid_teleop_still_lands(self):
        clock = fit_clock(readings(start=60), SEASON)
        self.assertAlmostEqual(clock.teleop_start_t, TELEOP_START, delta=1.0)
        # Auto is not in the clip, so the pause before teleop cannot be measured: none is assumed.
        self.assertAlmostEqual(clock.match_start_t, TELEOP_START - 20, delta=1.0)
        self.assertTrue(any("assumed" in note for note in clock.notes))
        self.assertTrue(any("auto not found" in note for note in clock.notes))

    def test_the_clock_wins_over_a_wrong_season_file_and_says_so(self):
        clock = fit_clock(readings(), {"auto_seconds": 15, "teleop_seconds": 135, "endgame_seconds": 20})
        self.assertEqual(clock.auto_seconds, 20)
        self.assertEqual(clock.teleop_seconds, 140)
        self.assertEqual(len(clock.notes), 2)

    def test_no_readable_clock_is_said_plainly(self):
        clock = fit_clock([(float(t), None) for t in range(100)], SEASON)
        self.assertEqual(clock.source, "none")
        self.assertIn("no period found on the clock", clock.notes)


class PhaseTests(unittest.TestCase):
    clock = MatchClock(auto_start_t=7.0, auto_seconds=20, teleop_start_t=31.0, teleop_seconds=140,
                       endgame_seconds=30, source="scoreboard_clock")

    def test_phases_and_match_time(self):
        at = self.clock.at
        self.assertEqual(at(5.0), (-2.0, "pre_match"))
        self.assertEqual(at(17.0), (10.0, "auto"))
        self.assertEqual(at(29.0), (20.0, "transition"))   # the pause holds match time
        self.assertEqual(at(41.0), (30.0, "teleop"))
        self.assertEqual(at(31.0 + 111), (131.0, "endgame"))  # last 30 s of 140
        self.assertEqual(at(200.0)[1], "post_match")

    def test_without_a_clock_the_start_is_fixed_and_there_is_no_pause(self):
        clock = fixed_clock(0.0, SEASON, "clip_start")
        self.assertEqual(clock.at(25.0), (25.0, "teleop"))
        self.assertEqual(clock.source, "clip_start")


if __name__ == "__main__":
    unittest.main()
