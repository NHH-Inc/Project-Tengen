import json
import io
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from ingest.yolo_orchestrator import YoloAnalysisOrchestrator, shot_events
from training.ball_scouting import (
    BallDetectorConfig,
    BallScoutingConfig,
    BallShotAnalyzer,
    BlobTrackingConfig,
    DebugConfig,
    DirectedBoundary,
    GoalGeometry,
    RobotObservation,
    ShotDetectionConfig,
    YellowBallDetector,
    crosses_directed_boundary,
    load_ball_scouting_config,
    shot_statistics,
)


WIDTH = 200
HEIGHT = 100
FPS = 30.0


def frame_with_balls(*centres):
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    for center in centres:
        cv2.circle(frame, tuple(int(value) for value in center), 5, (0, 255, 255), -1)
    return frame


def analyzer_config(*, goal=None):
    return BallScoutingConfig(
        detector=BallDetectorConfig(
            blur_kernel=0,
            morph_open_iterations=0,
            morph_close_iterations=0,
            min_area_px=20,
            max_area_px=200,
            min_radius_px=3,
            max_radius_px=9,
        ),
        tracking=BlobTrackingConfig(
            minimum_confirmed_hits=2,
            maximum_missed_frames=3,
            base_link_distance_ratio=0.004,
            acceleration_allowance_ratio_per_second=2.2,
            maximum_speed_ratio_per_second=2.2,
        ),
        shot=ShotDetectionConfig(
            carry_confirmation_frames=3,
            source_memory_frames=8,
            robot_padding_ratio=0.08,
            maximum_carried_relative_speed_ratio_per_second=0.12,
            minimum_launch_relative_speed_ratio_per_second=0.30,
            minimum_radial_speed_ratio_per_second=0.10,
            minimum_departure_distance_ratio=0.04,
            confirmation_frames=3,
            maximum_candidate_frames=10,
            minimum_direction_consistency=0.75,
        ),
        debug=DebugConfig(),
        goals=(goal,) if goal else (),
    )


def run_path(analyzer, ball_positions, robot_boxes):
    for frame_index, ball_position in enumerate(ball_positions):
        boxes = robot_boxes[frame_index]
        analyzer.process_frame(
            frame_with_balls(ball_position),
            frame_index,
            frame_index / FPS,
            [RobotObservation(track_id, box) for track_id, box in boxes],
        )


class YellowDetectorTests(unittest.TestCase):
    def test_size_and_shape_filters_keep_ball_and_drop_yellow_decoration(self):
        frame = frame_with_balls((40, 50))
        cv2.rectangle(frame, (90, 45), (150, 51), (0, 255, 255), -1)
        detector = YellowBallDetector(analyzer_config().detector)
        detections = detector.detect(frame)
        self.assertEqual(len(detections), 1)
        self.assertAlmostEqual(detections[0].center[0], 40, delta=1)

    def test_configuration_is_strict_and_loads_example(self):
        loaded = load_ball_scouting_config(
            Path(__file__).parents[2] / "analysis" / "config" / "ball_scouting.example.json"
        )
        self.assertTrue(loaded.enabled)
        self.assertEqual(loaded.detector.hsv_lower, (20, 90, 110))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps({
                "schema_version": 1,
                "detector": {"hsv_lower": [40, 0, 0], "hsv_upper": [20, 255, 255]},
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "cannot exceed"):
                load_ball_scouting_config(path)


class ShotStateMachineTests(unittest.TestCase):
    def test_visible_ball_carried_by_moving_robot_is_not_a_shot(self):
        analyzer = BallShotAnalyzer(analyzer_config())
        balls = [(48 + frame * 2, 50) for frame in range(14)]
        boxes = [
            [(12, (35 + frame * 2, 30, 65 + frame * 2, 70))]
            for frame in range(14)
        ]
        run_path(analyzer, balls, boxes)
        self.assertEqual(analyzer.shots, [])

    def test_ball_must_leave_robot_and_sustain_relative_motion(self):
        analyzer = BallShotAnalyzer(analyzer_config())
        balls = [(52, 50)] * 5 + [(68, 50), (80, 50), (92, 50)]
        boxes = [[(12, (35, 30, 65, 70))] for _ in balls]
        run_path(analyzer, balls, boxes)
        self.assertEqual(len(analyzer.shots), 1)
        shot = analyzer.shots[0]
        self.assertEqual(shot.launch_frame, 5)
        self.assertEqual(shot.robot_track_id, 12)
        self.assertEqual(shot.outcome, "unknown")
        self.assertGreater(shot.confidence, 0.8)

    def test_ambiguous_close_robots_leave_shot_unassigned(self):
        analyzer = BallShotAnalyzer(analyzer_config())
        balls = [(53, 50)] * 5 + [(70, 50), (82, 50), (94, 50)]
        boxes = [
            [(1, (30, 30, 62, 70)), (2, (44, 30, 66, 70))]
            for _ in balls
        ]
        run_path(analyzer, balls, boxes)
        self.assertEqual(len(analyzer.shots), 1)
        self.assertIsNone(analyzer.shots[0].robot_track_id)
        self.assertEqual(analyzer.shots[0].attribution_confidence, 0.0)

    def test_two_balls_launched_in_quick_succession_get_distinct_shots(self):
        analyzer = BallShotAnalyzer(analyzer_config())
        positions = [
            [(52, 40), (52, 60)],
            [(52, 40), (52, 60)],
            [(52, 40), (52, 60)],
            [(52, 40), (52, 60)],
            [(52, 40), (52, 60)],
            [(68, 40), (52, 60)],
            [(80, 40), (52, 60)],
            [(92, 40), (52, 60)],
            [(104, 40), (68, 60)],
            [(116, 40), (80, 60)],
            [(128, 40), (92, 60)],
        ]
        for frame_index, balls in enumerate(positions):
            analyzer.process_frame(
                frame_with_balls(*balls),
                frame_index,
                frame_index / FPS,
                [RobotObservation(12, (30, 20, 65, 80))],
            )
        self.assertEqual(len(analyzer.shots), 2)
        self.assertEqual(len({shot.shot_id for shot in analyzer.shots}), 2)
        self.assertEqual(len({shot.ball_track_id for shot in analyzer.shots}), 2)
        self.assertTrue(all(shot.robot_track_id == 12 for shot in analyzer.shots))

    def test_rapid_fire_tracks_first_seen_at_robot_edge_are_counted(self):
        analyzer = BallShotAnalyzer(analyzer_config())
        positions = [
            [(68, 38)],
            [(80, 38)],
            [(92, 38), (68, 62)],
            [(104, 38), (80, 62)],
            [(116, 38), (92, 62)],
            [(128, 38), (104, 62)],
        ]
        for frame_index, balls in enumerate(positions):
            analyzer.process_frame(
                frame_with_balls(*balls),
                frame_index,
                frame_index / FPS,
                [RobotObservation(12, (30, 20, 65, 80))],
            )

        self.assertEqual(len(analyzer.shots), 2)
        self.assertEqual([shot.launch_frame for shot in analyzer.shots], [0, 2])
        self.assertTrue(all(shot.robot_track_id == 12 for shot in analyzer.shots))

    def test_fast_new_track_far_from_robot_is_not_an_edge_launch(self):
        analyzer = BallShotAnalyzer(analyzer_config())
        balls = [(120, 50), (132, 50), (144, 50), (156, 50)]
        boxes = [[(12, (30, 20, 65, 80))] for _ in balls]
        run_path(analyzer, balls, boxes)
        self.assertEqual(analyzer.shots, [])

    def test_floor_ball_pushed_from_robot_bottom_is_not_an_edge_launch(self):
        analyzer = BallShotAnalyzer(analyzer_config())
        balls = [(68, 78), (80, 78), (92, 78), (104, 78)]
        boxes = [[(12, (30, 20, 65, 80))] for _ in balls]
        run_path(analyzer, balls, boxes)
        self.assertEqual(analyzer.shots, [])

    def test_shot_path_stops_after_configured_flight_window(self):
        config = analyzer_config()
        config = replace(
            config,
            shot=replace(
                config.shot,
                maximum_shot_track_seconds=0.20,
                minimum_shot_track_speed_ratio_per_second=0.0,
            ),
        )
        analyzer = BallShotAnalyzer(config)
        balls = [(52, 50)] * 5 + [
            (68, 50), (80, 50), (92, 50), (104, 50), (116, 50),
            (128, 50), (140, 50), (152, 50), (164, 50),
        ]
        boxes = [[(12, (35, 30, 65, 70))] for _ in balls]
        run_path(analyzer, balls, boxes)

        self.assertEqual(len(analyzer.shots), 1)
        shot = analyzer.shots[0]
        self.assertLessEqual(
            shot.ball_track[-1]["t_seconds"] - shot.launch_t_seconds,
            config.shot.maximum_shot_track_seconds + 1e-6,
        )

    def test_directed_goal_crossing_is_counted_once(self):
        goal = GoalGeometry(
            goal_id="high",
            polygon=None,
            entry_direction=(1.0, 0.0),
            made_line=DirectedBoundary(((0.50, 0.35), (0.50, 0.65)), (1.0, 0.0)),
        )
        analyzer = BallShotAnalyzer(analyzer_config(goal=goal))
        balls = [(52, 50)] * 5 + [(68, 50), (80, 50), (92, 50), (104, 50), (116, 50)]
        boxes = [[(12, (35, 30, 65, 70))] for _ in balls]
        run_path(analyzer, balls, boxes)
        self.assertEqual(len(analyzer.shots), 1)
        shot = analyzer.shots[0]
        self.assertEqual(shot.outcome, "made")
        self.assertEqual(shot.goal, "high")
        self.assertEqual(shot.outcome_frame, 8)
        self.assertFalse(crosses_directed_boundary(
            (0.55, 0.5), (0.45, 0.5), goal.made_line
        ))

    def test_explicit_miss_boundary_is_required_for_a_miss(self):
        goal = GoalGeometry(
            goal_id="high",
            polygon=None,
            entry_direction=(1.0, 0.0),
            made_line=DirectedBoundary(((0.50, 0.05), (0.50, 0.30)), (1.0, 0.0)),
            miss_boundaries=(
                DirectedBoundary(((0.50, 0.40), (0.50, 0.70)), (1.0, 0.0)),
            ),
        )
        analyzer = BallShotAnalyzer(analyzer_config(goal=goal))
        balls = [(52, 50)] * 5 + [(68, 50), (80, 50), (92, 50), (104, 50)]
        boxes = [[(12, (35, 30, 65, 70))] for _ in balls]
        run_path(analyzer, balls, boxes)
        self.assertEqual(analyzer.shots[0].outcome, "missed")
        self.assertEqual(analyzer.shots[0].goal, "high")

    def test_statistics_do_not_turn_unknown_into_missed(self):
        rows = [
            {"robot_track_id": 12, "outcome": "made"},
            {"robot_track_id": 12, "outcome": "unknown"},
            {"robot_track_id": None, "outcome": "missed"},
        ]
        statistics = shot_statistics(rows)
        self.assertEqual(statistics["attempted"], 3)
        self.assertEqual(statistics["made"], 1)
        self.assertEqual(statistics["missed"], 1)
        self.assertEqual(statistics["unknown"], 1)
        self.assertEqual(statistics["per_robot"]["12"]["unknown"], 1)
        self.assertEqual(statistics["per_robot"]["unassigned"]["missed"], 1)


class ShotEventProjectionTests(unittest.TestCase):
    def test_attempt_and_made_share_shot_evidence_without_inventing_unknown_make(self):
        shots = [
            {
                "shot_id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
                "launch_frame": 30,
                "launch_t_seconds": 1.0,
                "robot_track_id": 2,
                "ball_track_id": 9,
                "confidence": 0.9,
                "outcome": "made",
                "outcome_t_seconds": 1.2,
                "outcome_frame": 36,
                "goal": "high",
            },
            {
                "shot_id": "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
                "launch_frame": 60,
                "launch_t_seconds": 2.0,
                "robot_track_id": None,
                "ball_track_id": 10,
                "confidence": 0.6,
                "outcome": "unknown",
                "goal": None,
            },
        ]
        events = shot_events(
            {"job_id": "job", "match_id": "2026test_qm1"},
            shots,
            [{"track_id": 2, "team": 123}],
            {"season": 2026, "auto_seconds": 15, "teleop_seconds": 135, "goals": ["high"]},
        )
        self.assertEqual(
            [event["event_type"] for event in events],
            ["shot_attempt", "shot_made", "shot_attempt"],
        )
        self.assertEqual(events[0]["team"], 123)
        self.assertIsNone(events[2]["track_id"])
        self.assertEqual(events[0]["shot_id"], events[1]["shot_id"])

    def test_orchestrator_passes_snapshot_config_and_imports_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            python = root / "python.exe"
            model = root / "model.pt"
            video = root / "match.mp4"
            config = root / "balls.json"
            for path in (python, model, video):
                path.write_bytes(b"")
            config.write_text(json.dumps({
                "schema_version": 1,
                "enabled": True,
                "goals": [],
            }), encoding="utf-8")
            commands = []

            class FakeProcess:
                returncode = 0

                def __init__(self, command, **_kwargs):
                    commands.append(command)
                    tracks_path = Path(command[command.index("--output") + 1])
                    shots_path = Path(command[command.index("--shots-output") + 1])
                    tracks_path.write_text(json.dumps({
                        "track_id": 2,
                        "robot_name": "robot2",
                        "team": 123,
                        "alliance": "blue",
                        "boxes": [{"t": 0, "x": 0.1, "y": 0.2, "w": 0.1, "h": 0.1}],
                        "gaps": [],
                    }) + "\n", encoding="utf-8")
                    shots_path.write_text(json.dumps({
                        "shot_id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
                        "launch_frame": 30,
                        "launch_t_seconds": 1.0,
                        "robot_track_id": 2,
                        "ball_track_id": 9,
                        "confidence": 0.9,
                        "outcome": "unknown",
                        "goal": None,
                    }) + "\n", encoding="utf-8")
                    self.stdout = io.StringIO('{"stage":"tracking"}\n')
                    self.stderr = io.StringIO("")

                def wait(self):
                    return 0

            adapter = YoloAnalysisOrchestrator(
                repo_root=Path(__file__).parents[2],
                python_path=python,
                model_path=model,
                output_base_dir=root / "jobs",
                ball_config_path=config,
                auto_homography=False,
            )
            with patch("ingest.yolo_orchestrator.subprocess.Popen", FakeProcess):
                result = adapter.run_job(
                    {
                        "job_id": "11111111-1111-4111-8111-111111111111",
                        "video_id": "abcdefghijk",
                        "match_id": "2026test_qm1",
                        "season": 2026,
                        "local_path": str(video),
                        "duration": 2.0,
                        "fps": 30.0,
                    },
                    str(Path(__file__).parents[2] / "contracts" / "seasons" / "2026.json"),
                )

            command = commands[0]
            snapshot = Path(command[command.index("--ball-config") + 1])
            self.assertEqual(snapshot.name, "ball_scouting.config.json")
            self.assertTrue(snapshot.is_file())
            self.assertEqual(result["result"]["shots_emitted"], 1)
            events = [
                json.loads(line)
                for line in Path(result["events_path"]).read_text(encoding="utf-8").splitlines()
            ]
            attempts = [event for event in events if event["event_type"] == "shot_attempt"]
            self.assertEqual(len(attempts), 1)
            self.assertEqual(attempts[0]["team"], 123)


if __name__ == "__main__":
    unittest.main()
