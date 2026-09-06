import tempfile
import unittest
from pathlib import Path

from ingest.yolo_orchestrator import YoloAnalysisOrchestrator, match_events, phase_at


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


if __name__ == "__main__":
    unittest.main()
