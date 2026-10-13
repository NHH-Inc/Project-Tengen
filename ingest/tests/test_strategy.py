"""The strategy layer must report behaviour it saw, and say so when it saw nothing.

Synthetic matches with known answers: robots parked, driving, shooting from a spot, marking an
opponent. Each test pins one way the layer could quietly mislead a drive team.
"""

import json
import math
import tempfile
import unittest
from pathlib import Path

from ingest.strategy.features import match_features
from ingest.strategy.profile import build_profile, match_order, next_match_test
from ingest.strategy.timeline import (
    build_timelines,
    load_geometry,
    read_json,
    resample_track,
    SEASONS_DIR,
    to_alliance_frame,
)

GEOMETRY = load_geometry(2026)
SEASON = read_json(SEASONS_DIR / "2026.json")
LENGTH, WIDTH = GEOMETRY["field_length_ft"], GEOMETRY["field_width_ft"]


def boxes(path, start=0.0, end=60.0, rate=10.0):
    """Boxes along `path(t) -> (field_x, field_y)` in field feet."""
    out = []
    for i in range(int((end - start) * rate) + 1):
        t = round(start + i / rate, 3)
        fx, fy = path(t)
        out.append({"t": t, "x": 0.5, "y": 0.5, "w": 0.05, "h": 0.05, "field_x": fx, "field_y": fy})
    return out


def track(track_id, team, alliance, path=None, start=0.0, end=60.0, gaps=(), positioned=True):
    b = boxes(path or (lambda t: (10.0, 10.0)), start, end)
    if not positioned:
        for box in b:
            box["field_x"] = box["field_y"] = None
    return {"schema_version": 3, "track_id": track_id, "team": team, "alliance": alliance,
            "team_confidence": None, "boxes": b, "gaps": list(gaps)}


def blue_to_field(ax, ay, alliance):
    """Inverse of the alliance frame, so tests can be written from the driver's point of view."""
    return (ax, ay) if alliance == "blue" else (LENGTH - ax, WIDTH - ay)


class Match:
    def __init__(self, match_id="2026test_qm1"):
        self.dir = Path(tempfile.mkdtemp(prefix="strategy-test-"))
        self.match_id = match_id
        self.tracks, self.shots, self.events = [], [], []

    def write(self):
        (self.dir / "job.json").write_text(json.dumps(
            {"job_id": "j-" + self.match_id, "match_id": self.match_id, "season": 2026}))
        for name, rows in (("tracks", self.tracks), ("shots", self.shots), ("events", self.events)):
            (self.dir / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        return self.dir

    def shoot(self, track_id, t):
        self.shots.append({"shot_id": f"s{len(self.shots)}", "launch_t_seconds": t,
                           "robot_track_id": track_id, "confidence": 0.9, "outcome": "unknown"})


def features_of(directory):
    return {f["robot_key"]: f for f in match_features(build_timelines(directory), GEOMETRY, SEASON)}


class TimelineTests(unittest.TestCase):
    def test_a_hole_is_never_bridged(self):
        t = track(1, None, "blue", lambda t: (t, 5.0), end=10)
        # Remove 3-6 s: a detection hole longer than any sampling interval.
        t["boxes"] = [b for b in t["boxes"] if not 3.0 < b["t"] < 6.0]
        times = [s["t"] for s in resample_track(t)]
        self.assertFalse(any(3.0 < x < 6.0 for x in times))
        self.assertIn(2.9, times)
        self.assertIn(6.0, times)

    def test_a_listed_gap_is_never_bridged_even_when_short(self):
        t = track(1, None, "blue", lambda t: (t, 5.0), end=4, gaps=[{"start": 1.0, "end": 1.4,
                                                                     "reason": "shot_change"}])
        t["boxes"] = [b for b in t["boxes"] if not 1.0 < b["t"] < 1.4] + []
        t["boxes"] = [b for b in t["boxes"] if b["t"] not in (1.1, 1.2, 1.3)]
        times = [s["t"] for s in resample_track(t)]
        self.assertNotIn(1.2, times)

    def test_short_spans_are_interpolated(self):
        t = track(1, None, "blue", lambda t: (t, 5.0), end=4)
        t["boxes"] = t["boxes"][::3]  # 3.3 Hz analysis
        samples = {s["t"]: s for s in resample_track(t)}
        self.assertAlmostEqual(samples[0.1]["x"], 0.1, places=6)

    def test_red_is_rotated_into_the_alliance_frame(self):
        self.assertEqual(to_alliance_frame(5.0, 3.0, "blue", GEOMETRY), (5.0, 3.0))
        ax, ay = to_alliance_frame(LENGTH - 5.0, WIDTH - 3.0, "red", GEOMETRY)
        self.assertAlmostEqual(ax, 5.0)
        self.assertAlmostEqual(ay, 3.0)
        self.assertEqual(to_alliance_frame(5.0, 3.0, None, GEOMETRY), (None, None))

    def test_team_corrections_apply_and_fragments_merge(self):
        match = Match()
        match.tracks = [track(1, None, "red", end=20), track(2, None, "red", start=25, end=40)]
        directory = match.write()
        (directory / "desktop-corrections.json").write_text(json.dumps([
            {"scope": "track", "target": 1, "fields": {"team": 254}, "action": "edit"},
            {"scope": "track", "target": 2, "fields": {"team": 254}, "action": "edit"}]))
        timelines = build_timelines(directory)
        self.assertEqual(len(timelines), 1)
        self.assertEqual(timelines[0]["track_ids"], [1, 2])
        self.assertEqual(timelines[0]["team"], 254)

    def test_two_simultaneous_tracks_claiming_one_team_are_kept_apart(self):
        match = Match()
        match.tracks = [track(1, 254, "red"), track(2, 254, "red")]
        timelines = build_timelines(match.write())
        self.assertEqual(len(timelines), 2)
        self.assertTrue(any("overlapping_team_claim" in t["identity_issues"] for t in timelines))


class FeatureTests(unittest.TestCase):
    def test_no_positions_means_unavailable_not_zero(self):
        match = Match()
        match.tracks = [track(1, 254, "red", positioned=False)]
        f = features_of(match.write())["team:254"]
        self.assertIsNone(f["positions"])
        self.assertNotIn("movement", f)
        self.assertGreater(f["coverage"]["observed_s"], 0)
        self.assertEqual(f["coverage"]["positioned_s"], 0)

    def test_a_parked_robot_with_jitter_did_not_drive(self):
        match = Match()
        jitter = lambda t: (10.0 + 0.15 * math.sin(t * 17), 10.0 + 0.15 * math.cos(t * 13))
        match.tracks = [track(1, 254, "blue", jitter)]
        movement = features_of(match.write())["team:254"]["movement"]
        self.assertLess(movement["distance_ft"], 5.0)
        self.assertGreater(movement["stationary_share"], 0.9)

    def test_a_driving_robot_covers_its_distance(self):
        match = Match()
        # 2 ft/s for 15 s, then parked: 30 ft. (Creeping below 1 ft/s is deliberately counted as
        # stopped, because that is the speed homography jitter fakes on a parked robot.)
        match.tracks = [track(1, 254, "blue", lambda t: (5.0 + 2.0 * min(t, 15.0), 10.0), end=60)]
        movement = features_of(match.write())["team:254"]["movement"]
        self.assertAlmostEqual(movement["distance_ft"], 30.0, delta=2.0)
        self.assertAlmostEqual(movement["median_moving_speed_ftps"], 2.0, delta=0.1)

    def test_shooting_spot_dwell_and_volley_interval(self):
        match = Match()

        def path(t):
            # Drives out, parks at (12, 20) in the blue frame for 4 s every 10 s, and drives back.
            phase = t % 10
            if phase < 3:
                return (5.0 + 2.33 * phase, 20.0)
            if phase < 7:
                return (12.0, 20.0)
            return (12.0 - 2.33 * (phase - 7), 20.0)

        match.tracks = [track(1, 254, "blue", path, end=60)]
        for visit in range(2, 6):
            for k in range(3):
                match.shoot(1, visit * 10 + 4 + k * 0.5)
        shooting = features_of(match.write())["team:254"]["shooting"]
        self.assertEqual(shooting["launches"], 12)
        self.assertEqual(shooting["volleys"], 4)
        self.assertEqual(shooting["median_volley_interval_s"], 10.0)
        spot = shooting["spots"][0]
        self.assertAlmostEqual(spot["ax"], 12.0, delta=0.5)
        self.assertAlmostEqual(spot["ay"], 20.0, delta=0.5)
        self.assertGreaterEqual(shooting["median_dwell_s"], 4.0)
        self.assertEqual(shooting["on_the_move_share"], 0.0)

    def test_red_and_blue_launches_from_mirrored_spots_are_one_spot(self):
        spots = []
        for alliance in ("blue", "red"):
            match = Match()
            fx, fy = blue_to_field(12.0, 20.0, alliance)
            match.tracks = [track(1, 254, alliance, lambda t: (fx, fy))]
            match.shoot(1, 30.0)
            spots.append(features_of(match.write())["team:254"]["shooting"]["spots"][0])
        self.assertAlmostEqual(spots[0]["ax"], spots[1]["ax"], places=2)
        self.assertAlmostEqual(spots[0]["ay"], spots[1]["ay"], places=2)

    def test_marking_an_opponent_in_their_half_is_defense(self):
        match = Match()
        # A blue scorer sits in its own zone; a red robot sits 3 ft away. In the red robot's
        # alliance frame that is deep in the opponent's half.
        match.tracks = [track(1, 118, "blue", lambda t: (10.0, 10.0)),
                        track(2, 254, "red", lambda t: (13.0, 10.0))]
        features = features_of(match.write())
        defense = features["team:254"]["defense"]
        self.assertGreater(defense["near_opponent_s"], 30)
        self.assertEqual(defense["marked"], "118")
        self.assertEqual(features["team:118"]["defense"]["near_opponent_s"], 0)


def team_match(match_id, start_ay, spot, defends=False):
    """One match's features for team 254, built through the real pipeline."""
    match = Match(match_id)
    path = lambda t: blue_to_field(3.0 + min(t, 10) * 0.8, start_ay, "blue")
    match.tracks = [track(1, 254, "blue", path)]
    if defends:
        match.tracks = [track(1, 254, "blue", lambda t: (45.0, 10.0)),
                        track(2, 118, "red", lambda t: (47.0, 10.0))]
    for k in range(3):
        match.shoot(1, 20 + k)
    if spot and not defends:
        sx, sy = spot
        match.tracks[0]["boxes"] = [dict(b, field_x=sx, field_y=sy) if 19 <= b["t"] <= 23 else b
                                    for b in match.tracks[0]["boxes"]]
    return features_of(match.write())["team:254"]


class ProfileTests(unittest.TestCase):
    def test_match_order_is_numeric(self):
        ids = ["2026x_qm10", "2026x_sf1m1", "2026x_qm2"]
        self.assertEqual(sorted(ids, key=lambda m: match_order({"match_id": m})),
                         ["2026x_qm2", "2026x_qm10", "2026x_sf1m1"])

    def test_a_repeated_auto_and_spot_produce_counters_with_evidence(self):
        matches = [team_match(f"2026x_qm{i}", 22.0, (11.0, 22.0)) for i in range(1, 5)]
        profile = build_profile(matches)
        self.assertEqual(profile["n_with_positions"], 4)
        self.assertEqual(profile["prediction"]["start_lane"], "left")
        ids = {c["id"] for c in profile["counters"]}
        self.assertIn("predictable_auto", ids)
        self.assertIn("favourite_spot", ids)
        for counter in profile["counters"]:
            self.assertIn("of 4", counter["evidence"] + " of 4")

    def test_too_few_matches_gives_tendencies_but_no_counters(self):
        profile = build_profile([team_match("2026x_qm1", 22.0, (11.0, 22.0))])
        self.assertIsNotNone(profile["tendencies"])
        self.assertEqual(profile["counters"], [])

    def test_next_match_test_scores_predictions(self):
        matches = [team_match(f"2026x_qm{i}", 22.0, (11.0, 22.0)) for i in range(1, 6)]
        report = next_match_test({254: matches})
        self.assertEqual(report["start_lane"]["n"], 3)
        self.assertEqual(report["start_lane"]["accuracy"], 1.0)
        self.assertEqual(report["primary_spot"]["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()


class CommandLineTests(unittest.TestCase):
    """Nothing else imports the CLI, so a broken line there would only show up when someone ran it."""

    def test_every_command_runs_on_an_analysed_match(self):
        from contextlib import redirect_stdout
        from io import StringIO

        from ingest.strategy.__main__ import main

        match = Match("2026x_qm1")
        match.tracks = [track(1, 254, "blue"), track(2, 118, "red")]
        match.shoot(1, 20.0)
        directory = match.write()
        for command in ("clock", "timeline", "features", "profile", "evaluate"):
            with self.subTest(command=command), redirect_stdout(StringIO()) as out:
                self.assertEqual(main([command, str(directory)]), 0)
            if command == "clock":
                self.assertIn("video not found", out.getvalue())
        self.assertTrue((directory / "timeline.jsonl").exists())
        self.assertTrue((directory / "strategy_features.json").exists())
