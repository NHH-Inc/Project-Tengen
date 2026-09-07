import json
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ingest.yolo_orchestrator import YoloAnalysisOrchestrator, match_events, phase_at
from ingest.serializers import normalize_public_track_labels
from training.track_yolo import (
    AppearanceTrackMemory,
    ReIDConfig,
    ReIDDetection,
    add_field_motion,
    contract_track_records,
    crop_box_to_source,
    eligible_track_ids,
    image_window_has_motion,
    resolved_alliances,
    startup_alliance,
    track_has_motion,
)


class YoloOrchestratorTests(unittest.TestCase):
    def test_legacy_public_track_view_keeps_boxes_and_caps_label_namespace(self):
        raw = [
            {
                "track_id": track_id,
                "robot_name": f"robot{track_id}",
                "alliance": "red" if track_id % 2 else None,
                "boxes": [{}] * track_id,
            }
            for track_id in range(1, 10)
        ]
        public = normalize_public_track_labels(raw)
        self.assertEqual(len(public), 9)
        self.assertEqual(len(raw), 9)
        self.assertEqual(sum(len(track["boxes"]) for track in public), 45)
        self.assertEqual(raw[6]["robot_name"], "robot7")
        self.assertEqual(public[6]["robot_name"], "robot1")
        self.assertEqual(public[8]["robot_name"], "robot3")
        self.assertTrue(
            all(track["robot_name"] in {f"robot{i}" for i in range(1, 7)} for track in public)
        )

    def test_phase_boundaries_are_season_configured(self):
        season = {"auto_seconds": 15, "teleop_seconds": 135}
        self.assertEqual(phase_at(15, season), "auto")
        self.assertEqual(phase_at(15.01, season), "teleop")
        self.assertEqual(phase_at(150, season), "teleop")
        self.assertEqual(phase_at(150.01, season), "endgame")

    def test_match_events_are_only_emitted_for_known_matches(self):
        season = {"auto_seconds": 15, "teleop_seconds": 135}
        self.assertEqual(match_events({"job_id": "job"}, 12, season), [])
        events = match_events({"job_id": "job", "match_id": "2026galileo_qm1"}, 12, season)
        self.assertEqual([event["event_type"] for event in events], ["match_start", "match_end"])
        self.assertEqual(events[1]["t_seconds"], 12)

    def test_health_reports_byte_track_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            python = root / "python.exe"
            model = root / "model" / "weights" / "best.pt"
            model.parent.mkdir(parents=True)
            python.write_bytes(b"")
            model.write_bytes(b"")
            adapter = YoloAnalysisOrchestrator(
                repo_root=root,
                python_path=python,
                model_path=model,
                output_base_dir=root / "jobs",
            )
            health = adapter.health()
            self.assertTrue(health["available"])
            self.assertEqual(health["tracker"], "bytetrack")
            self.assertEqual(adapter.model_version, "model+bytetrack")
            self.assertEqual(health["reid_memory_seconds"], 5.0)
            self.assertEqual(health["reid_alliance_lock_seconds"], 5.0)
            self.assertEqual(health["reid_alliance_lock_margin_seconds"], 2.0)
            self.assertEqual(health["startup_position_seconds"], 2.0)
            self.assertEqual(health["startup_split_x"], 0.5)
            self.assertTrue(health["auto_homography"])

    def test_stream_job_invokes_yolo_without_a_local_video(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            python = root / "python.exe"
            model = root / "model.pt"
            output_base = root / "jobs"
            python.write_bytes(b"")
            model.write_bytes(b"")
            command_seen = []

            class FakeProcess:
                returncode = 0

                def __init__(self, command, **_kwargs):
                    command_seen.append(command)
                    output = Path(command[command.index("--output") + 1])
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_text(json.dumps({
                        "track_id": 1,
                        "robot_name": "robot1",
                        "alliance": "blue",
                        "boxes": [{"t": 0.0, "x": 0.1, "y": 0.2, "w": 0.1, "h": 0.1}],
                        "gaps": [],
                    }) + "\n", encoding="utf-8")
                    self.stdout = io.StringIO('{"stage":"tracking"}\n')
                    self.stderr = io.StringIO("")

                def wait(self):
                    return 0

            adapter = YoloAnalysisOrchestrator(
                repo_root=Path.cwd(),
                python_path=python,
                model_path=model,
                output_base_dir=output_base,
                auto_homography=False,
            )
            with patch("ingest.yolo_orchestrator.subprocess.Popen", FakeProcess):
                result = adapter.run_job(
                    {
                        "job_id": "11111111-1111-4111-8111-111111111111",
                        "video_id": "abcdefghijk",
                        "match_id": None,
                        "season": 2026,
                        "duration": 2.0,
                        "fps": 30.0,
                        "stream_url": "https://www.youtube.com/watch?v=abcdefghijk",
                    },
                    season_path=str(Path.cwd() / "contracts" / "seasons" / "2026.json"),
                )

            command = command_seen[0]
            self.assertIn("--stream-url", command)
            self.assertNotIn("--video", command)
            self.assertIn("--reid-alliance-lock-seconds", command)
            self.assertIn("--reid-alliance-lock-margin-seconds", command)
            self.assertIn("--startup-position-seconds", command)
            self.assertIn("--startup-split-x", command)
            self.assertEqual(result["result"]["duration"], 2.0)

    def test_crop_coordinates_can_be_mapped_back_to_the_source(self):
        box = crop_box_to_source((0.0, 0.0, 1.0, 1.0), (0.02, 0.035, 0.98, 0.66), 960, 625)
        self.assertEqual(box, (0.02, 0.035, 0.96, 0.625))

    def test_startup_position_overrides_bad_bumper_colour(self):
        self.assertEqual(startup_alliance("red", 0.20, 0.0), "blue")
        self.assertEqual(startup_alliance("blue", 0.80, 1.0), "red")

    def test_startup_position_prior_expires_and_can_be_disabled(self):
        self.assertEqual(startup_alliance("red", 0.20, 2.0), "red")
        self.assertEqual(startup_alliance("red", 0.20, 0.0, startup_seconds=0.0), "red")

    def test_track_records_keep_position_fields_and_mapping_source(self):
        records = contract_track_records(
            {3: [{"t": 0.0, "x": 0.1, "y": 0.2, "w": 0.1, "h": 0.2,
                  "field_x": 4.0, "field_y": 5.0, "speed_ftps": 2.0}]},
            5.0,
            position_source="carpet_pose",
        )
        self.assertEqual(records[0]["position_source"], "carpet_pose")
        self.assertEqual(records[0]["robot_name"], "robot3")
        self.assertEqual(records[0]["boxes"][0]["field_x"], 4.0)

    def test_contract_exposes_only_robot1_through_robot6(self):
        boxes = [{"t": 0.0, "x": 0.1, "y": 0.2, "w": 0.1, "h": 0.2}]
        records = contract_track_records(
            {track_id: boxes for track_id in range(1, 8)},
            30.0,
            {track_id: "red" for track_id in range(1, 8)},
            visible_track_ids=set(range(1, 8)),
        )
        self.assertEqual(
            [record["robot_name"] for record in records],
            ["robot1", "robot2", "robot3", "robot4", "robot5", "robot6"],
        )

    def test_field_motion_rejects_identity_swap_speed(self):
        class Mapper:
            def box_to_field(self, x, y, w, h, image_w, image_h):
                return ((x + w / 2) * image_w / 100, (y + h) * image_h / 100)

            def on_field(self, x, y):
                return True

        history = {}
        first = {"t": 0.0, "track_id": 1,
                 "bbox_normalized": {"x": 0.0, "y": 0.0, "width": 0.1, "height": 0.1}}
        second = {"t": 0.01, "track_id": 1,
                  "bbox_normalized": {"x": 0.9, "y": 0.0, "width": 0.1, "height": 0.1}}
        add_field_motion(first, Mapper(), (0.0, 0.0, 1.0, 1.0), 100, 100, history, 1.0)
        add_field_motion(second, Mapper(), (0.0, 0.0, 1.0, 1.0), 100, 100, history, 1.0)
        self.assertIn("field_x", second)
        self.assertNotIn("speed_ftps", second)

    def test_auto_homography_writes_a_carpet_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            python = root / "python.exe"
            model = root / "best.pt"
            video = root / "match.mp4"
            jobs = root / "jobs"
            jobs.mkdir()
            for path in (python, model, video):
                path.write_bytes(b"")
            adapter = YoloAnalysisOrchestrator(
                repo_root=Path.cwd(),
                python_path=python,
                model_path=model,
                output_base_dir=jobs,
            )
            calibration_result = {
                "mapping_source": "carpet_pose",
                "tags_used": [1, 2, 3, 4, 5, 6],
                "point_count": 6,
                "points": [],
                "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "pose": {"reprojection_px": 0.5},
                "solution": {"trustworthy": True, "has_redundancy": False},
            }
            with patch("ingest.collection.calibrate.calibrate", return_value=calibration_result):
                path = adapter._job_homography(video, jobs)
            self.assertIsNotNone(path)
            document = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(document["mapping_source"], "carpet_pose")
            self.assertTrue(document["trustworthy"])


class StationaryNeutralFilterTests(unittest.TestCase):
    def test_stationary_colorless_track_is_not_published(self):
        boxes = [
            {"t": i / 10, "x": 0.20 + (i % 2) * 0.001, "y": 0.30,
             "w": 0.10, "h": 0.14}
            for i in range(20)
        ]
        self.assertFalse(track_has_motion(boxes))
        self.assertEqual(eligible_track_ids({1: boxes}, {1: None}), set())
        self.assertEqual(contract_track_records({1: boxes}, 10.0, {1: None}), [])

    def test_coloured_stationary_robot_is_kept(self):
        boxes = [{"t": i / 10, "x": 0.20, "y": 0.30, "w": 0.10, "h": 0.14}
                 for i in range(10)]
        self.assertEqual(eligible_track_ids({1: boxes}, {1: "red"}), {1})

    def test_moving_colorless_robot_is_kept(self):
        boxes = [{"t": i / 10, "x": 0.20 + i * 0.01, "y": 0.30,
                  "w": 0.10, "h": 0.14} for i in range(10)]
        self.assertTrue(track_has_motion(boxes))
        self.assertEqual(eligible_track_ids({1: boxes}, {1: None}), {1})

    def test_stationary_timer_ignores_one_bad_box(self):
        boxes = [
            {"t": i / 10, "x": 0.20, "y": 0.30, "w": 0.10, "h": 0.14}
            for i in range(12)
        ]
        boxes[5]["x"] = 0.24
        self.assertFalse(image_window_has_motion(boxes))

    def test_stationary_timer_sees_sustained_motion(self):
        boxes = [
            {"t": i / 10, "x": 0.20 + i * 0.01, "y": 0.30, "w": 0.10, "h": 0.14}
            for i in range(12)
        ]
        self.assertTrue(image_window_has_motion(boxes))

    def test_suppressed_stationary_false_positive_stays_hidden(self):
        boxes = [{"t": i / 10, "x": 0.96, "y": 0.30, "w": 0.04, "h": 0.14}
                 for i in range(80)]
        self.assertEqual(
            eligible_track_ids({1: boxes}, {1: None}, suppressed_track_ids={1}),
            set(),
        )
        self.assertEqual(
            contract_track_records(
                {1: boxes}, 10.0, {1: None}, visible_track_ids=set()
            ),
            [],
        )

    def test_stationary_coloured_robot_survives_suppression_timer(self):
        boxes = [{"t": i / 10, "x": 0.20, "y": 0.30, "w": 0.10, "h": 0.14}
                 for i in range(80)]
        self.assertEqual(
            eligible_track_ids({1: boxes}, {1: "blue"}, suppressed_track_ids={1}),
            {1},
        )

    def test_repeated_strict_colour_majority_keeps_robot_identified(self):
        self.assertEqual(
            resolved_alliances({1: {"red": 3, "blue": 2}, 2: {"red": 2, "blue": 2}}),
            {1: "red", 2: None},
        )


class AppearanceMemoryTests(unittest.TestCase):
    @staticmethod
    def detection(
        raw_id, descriptor, center, alliance="red", edge=None, alliance_is_authoritative=False
    ):
        return ReIDDetection(
            raw_id,
            descriptor,
            center,
            alliance,
            edge=edge,
            alliance_is_authoritative=alliance_is_authoritative,
        )

    def memory(self, **changes):
        values = {
            "memory_seconds": 2.0,
            "appearance_threshold": 0.75,
            "score_margin": 0.06,
            "max_center_distance": 0.65,
        }
        values.update(changes)
        return AppearanceTrackMemory(config=ReIDConfig(**values))

    def test_one_robot_disappears_and_returns_with_new_raw_id(self):
        memory = self.memory()
        first = memory.resolve_frame(0.0, [self.detection(10, [1.0, 0.0], (0.20, 0.5))])[0]
        memory.resolve_frame(0.2, [self.detection(10, [0.99, 0.01], (0.24, 0.5))])
        returned = memory.resolve_frame(
            1.0, [self.detection(99, [0.98, 0.02], (0.30, 0.5))]
        )[0]
        self.assertEqual(returned, first)

    def test_two_same_colour_robots_do_not_merge_when_one_disappears(self):
        memory = self.memory()
        first, second = memory.resolve_frame(0.0, [
            self.detection(10, [1.0, 0.0], (0.20, 0.5)),
            self.detection(20, [0.0, 1.0], (0.75, 0.5)),
        ])
        memory.resolve_frame(0.2, [self.detection(20, [0.01, 0.99], (0.73, 0.5))])
        returned = memory.resolve_frame(
            1.0, [self.detection(99, [0.99, 0.01], (0.28, 0.5))]
        )[0]
        self.assertEqual(returned, first)
        self.assertNotEqual(returned, second)

    def test_alliance_colour_alone_is_not_identity_evidence(self):
        memory = self.memory()
        first = memory.resolve_frame(0.0, [
            self.detection(10, [1.0, 0.0], (0.20, 0.5), "red")
        ])[0]
        returned = memory.resolve_frame(0.5, [
            self.detection(99, [0.0, 1.0], (0.21, 0.5), "red")
        ])[0]
        self.assertNotEqual(returned, first)

    def test_alliance_becomes_permanent_after_sustained_clear_evidence(self):
        memory = self.memory(
            alliance_lock_seconds=1.0,
            alliance_lock_margin_seconds=0.5,
            alliance_evidence_max_gap_seconds=0.5,
        )
        first = memory.resolve_frame(0.0, [
            self.detection(10, [1.0, 0.0], (0.20, 0.5), "blue")
        ])[0]
        memory.resolve_frame(0.5, [
            self.detection(10, [1.0, 0.0], (0.21, 0.5), "blue")
        ])
        memory.resolve_frame(1.0, [
            self.detection(10, [1.0, 0.0], (0.22, 0.5), "blue")
        ])
        self.assertEqual(memory.states[first].alliance, "blue")

        conflicting = memory.resolve_frame(1.2, [
            self.detection(99, [1.0, 0.0], (0.23, 0.5), "red")
        ])[0]
        self.assertNotEqual(conflicting, first)
        self.assertEqual(memory.states[first].alliance, "blue")

    def test_one_wrong_colour_read_does_not_lock_an_identity(self):
        memory = self.memory(
            alliance_lock_seconds=1.0,
            alliance_lock_margin_seconds=0.5,
            alliance_evidence_max_gap_seconds=0.5,
        )
        first = memory.resolve_frame(0.0, [
            self.detection(10, [1.0, 0.0], (0.20, 0.5), "blue")
        ])[0]
        memory.resolve_frame(0.1, [
            self.detection(10, [1.0, 0.0], (0.20, 0.5), "red")
        ])
        self.assertIsNone(memory.states[first].alliance)
        memory.resolve_frame(0.6, [
            self.detection(10, [1.0, 0.0], (0.21, 0.5), "blue")
        ])
        memory.resolve_frame(1.1, [
            self.detection(10, [1.0, 0.0], (0.22, 0.5), "blue")
        ])
        self.assertEqual(memory.states[first].alliance, "blue")

    def test_starting_side_immediately_locks_identity_alliance(self):
        memory = self.memory()
        first = memory.resolve_frame(0.0, [
            self.detection(
                10,
                [1.0, 0.0],
                (0.20, 0.5),
                "blue",
                alliance_is_authoritative=True,
            )
        ])[0]
        self.assertEqual(memory.states[first].alliance, "blue")
        conflicting = memory.resolve_frame(0.2, [
            self.detection(99, [1.0, 0.0], (0.21, 0.5), "red")
        ])[0]
        self.assertNotEqual(conflicting, first)

    def test_two_same_colour_robots_returning_simultaneously_use_global_assignment(self):
        memory = self.memory()
        first, second = memory.resolve_frame(0.0, [
            self.detection(10, [1.0, 0.0, 0.0], (0.20, 0.5)),
            self.detection(20, [0.0, 1.0, 0.0], (0.70, 0.5)),
        ])
        # Detector order is deliberately reversed.
        resolved = memory.resolve_frame(0.8, [
            self.detection(91, [0.02, 0.99, 0.0], (0.66, 0.5)),
            self.detection(92, [0.99, 0.02, 0.0], (0.24, 0.5)),
        ])
        self.assertEqual(resolved, [second, first])

    def test_crossing_robots_keep_identity_from_appearance_and_raw_continuity(self):
        memory = self.memory()
        first, second = memory.resolve_frame(0.0, [
            self.detection(10, [1.0, 0.0], (0.30, 0.5)),
            self.detection(20, [0.0, 1.0], (0.70, 0.5)),
        ])
        resolved = memory.resolve_frame(0.5, [
            self.detection(10, [0.99, 0.01], (0.58, 0.5)),
            self.detection(20, [0.01, 0.99], (0.42, 0.5)),
        ])
        self.assertEqual(resolved, [first, second])

    def test_return_from_wrong_edge_does_not_merge(self):
        memory = self.memory(max_center_distance=1.0, max_normalized_speed=2.0)
        first = memory.resolve_frame(
            0.0, [self.detection(10, [1.0, 0.0], (0.04, 0.5), edge="left")]
        )[0]
        returned = memory.resolve_frame(
            0.8, [self.detection(99, [1.0, 0.0], (0.96, 0.5), edge="right")]
        )[0]
        self.assertNotEqual(returned, first)

    def test_impossible_speed_reappearance_does_not_merge(self):
        memory = self.memory(max_center_distance=0.6, max_normalized_speed=0.5)
        first = memory.resolve_frame(0.0, [self.detection(10, [1.0, 0.0], (0.10, 0.5))])[0]
        returned = memory.resolve_frame(
            0.1, [self.detection(99, [1.0, 0.0], (0.90, 0.5))]
        )[0]
        self.assertNotEqual(returned, first)

    def test_supplied_camera_motion_compensates_image_space_prediction(self):
        memory = self.memory(max_normalized_speed=0.2)
        first = memory.resolve_frame(0.0, [
            self.detection(10, [1.0, 0.0], (0.20, 0.5))
        ])[0]
        returned = memory.resolve_frame(
            0.2,
            [self.detection(99, [1.0, 0.0], (0.50, 0.5))],
            camera_motion=(0.30, 0.0),
        )[0]
        self.assertEqual(returned, first)

    def test_ambiguous_candidates_are_left_tentative(self):
        memory = self.memory(score_margin=0.08)
        memory.resolve_frame(0.0, [
            self.detection(10, [1.0, 0.0], (0.30, 0.5)),
            self.detection(20, [1.0, 0.0], (0.70, 0.5)),
        ])
        resolved = memory.resolve_frame(0.5, [
            self.detection(91, [1.0, 0.0], (0.50, 0.5)),
            self.detection(92, [1.0, 0.0], (0.50, 0.5)),
        ])
        self.assertEqual(resolved, [None, None])

    def test_template_is_not_blended_and_new_view_requires_confirmation(self):
        memory = self.memory(template_confirmation_frames=2)
        first = memory.resolve_frame(
            0.0, [self.detection(10, [1.0, 0.0], (0.20, 0.5))]
        )[0]
        original = list(memory.states[first].templates[0])
        memory.resolve_frame(0.2, [self.detection(99, [0.8, 0.6], (0.21, 0.5))])
        self.assertEqual(memory.states[first].templates, [original])
        memory.resolve_frame(0.3, [self.detection(99, [0.8, 0.6], (0.22, 0.5))])
        self.assertEqual(len(memory.states[first].templates), 2)
        self.assertEqual(memory.states[first].templates[0], original)

    def test_recycled_raw_tracker_id_is_revalidated(self):
        memory = self.memory()
        first = memory.resolve_frame(
            0.0, [self.detection(10, [1.0, 0.0], (0.20, 0.5), "red")]
        )[0]
        recycled = memory.resolve_frame(
            0.2, [self.detection(10, [0.0, 1.0], (0.22, 0.5), "red")]
        )[0]
        self.assertNotEqual(recycled, first)

    def test_public_identity_pool_is_capped_at_six(self):
        memory = self.memory(max_robots=12)
        resolved = memory.resolve_frame(0.0, [
            self.detection(raw_id, [float(raw_id), 1.0], (0.1 * raw_id, 0.5), None)
            for raw_id in range(1, 8)
        ])
        self.assertEqual(resolved[:6], [1, 2, 3, 4, 5, 6])
        self.assertIsNone(resolved[6])


if __name__ == "__main__":
    unittest.main()
