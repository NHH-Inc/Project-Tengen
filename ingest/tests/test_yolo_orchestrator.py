import json
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ingest.yolo_orchestrator import YoloAnalysisOrchestrator, match_events, phase_at
from training.track_yolo import (
    AppearanceTrackMemory,
    add_field_motion,
    contract_track_records,
    crop_box_to_source,
    eligible_track_ids,
    image_window_has_motion,
    resolved_alliances,
    track_has_motion,
)


class YoloOrchestratorTests(unittest.TestCase):
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
            self.assertEqual(result["result"]["duration"], 2.0)

    def test_crop_coordinates_can_be_mapped_back_to_the_source(self):
        box = crop_box_to_source((0.0, 0.0, 1.0, 1.0), (0.02, 0.035, 0.98, 0.66), 960, 625)
        self.assertEqual(box, (0.02, 0.035, 0.96, 0.625))

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
    def test_new_raw_id_reuses_recent_matching_appearance(self):
        memory = AppearanceTrackMemory(memory_seconds=2.0, appearance_threshold=0.8)
        first = memory.resolve(10, 0.0, [1.0, 0.0], (0.2, 0.5), "red", set(), {10})
        second = memory.resolve(99, 1.0, [0.98, 0.02], (0.25, 0.5), "red", set(), {99})
        self.assertEqual(second, first)

    def test_same_colour_reuses_identity_even_when_appearance_changes(self):
        memory = AppearanceTrackMemory(memory_seconds=2.0, appearance_threshold=0.8)
        first = memory.resolve(10, 0.0, [1.0, 0.0], (0.2, 0.5), "red", set(), {10})
        second = memory.resolve(99, 1.0, [0.0, 1.0], (0.25, 0.5), "red", set(), {99})
        self.assertEqual(second, first)

    def test_unambiguous_same_colour_handoff_can_cross_the_distance_gate(self):
        memory = AppearanceTrackMemory(
            memory_seconds=2.0, appearance_threshold=0.8, max_center_distance=0.2
        )
        first = memory.resolve(10, 0.0, [1.0, 0.0], (0.1, 0.5), "red", set(), {10})
        second = memory.resolve(99, 1.0, [0.0, 1.0], (0.9, 0.5), "red", set(), {99})
        self.assertEqual(second, first)

    def test_opposite_alliance_never_reuses_identity(self):
        memory = AppearanceTrackMemory(memory_seconds=2.0, appearance_threshold=0.8)
        first = memory.resolve(10, 0.0, [1.0, 0.0], (0.2, 0.5), "red", set(), {10})
        second = memory.resolve(99, 1.0, [1.0, 0.0], (0.25, 0.5), "blue", set(), {99})
        self.assertNotEqual(second, first)

    def test_match_expires_after_memory_window(self):
        memory = AppearanceTrackMemory(memory_seconds=1.0, appearance_threshold=0.8)
        first = memory.resolve(10, 0.0, [1.0, 0.0], (0.2, 0.5), None, set(), {10})
        second = memory.resolve(99, 1.5, [1.0, 0.0], (0.25, 0.5), None, set(), {99})
        self.assertNotEqual(second, first)


if __name__ == "__main__":
    unittest.main()
