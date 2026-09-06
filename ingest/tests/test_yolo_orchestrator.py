import tempfile
import unittest
from pathlib import Path

from ingest.yolo_orchestrator import YoloAnalysisOrchestrator, match_events, phase_at
from training.track_yolo import add_field_motion, contract_track_records, crop_box_to_source


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


if __name__ == "__main__":
    unittest.main()
