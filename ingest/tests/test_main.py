import os
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("DATABASE_URL", "sqlite://")
_test_data_dir = None
if "FRC_DATA_DIR" not in os.environ:
    _test_data_dir = tempfile.mkdtemp(prefix="frc-scouting-tests-")
    os.environ["FRC_DATA_DIR"] = _test_data_dir

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from ingest import main, models


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        models.Base.metadata.create_all(cls.engine)
        cls.sessions = sessionmaker(bind=cls.engine)

        def override_db():
            db = cls.sessions()
            try:
                yield db
            finally:
                db.close()

        main.app.dependency_overrides[main.get_db] = override_db
        cls.client = TestClient(main.app)

    @classmethod
    def tearDownClass(cls):
        main.app.dependency_overrides.clear()
        cls.engine.dispose()
        if _test_data_dir:
            shutil.rmtree(_test_data_dir)

    def setUp(self):
        with self.sessions() as db:
            for table in reversed(models.Base.metadata.sorted_tables):
                db.execute(table.delete())
            db.commit()

    def test_goal_entries_api_keeps_unassigned_makes_and_legacy_jobs(self):
        job_id = "33333333-3333-4333-8333-333333333333"
        with self.sessions() as db:
            db.add(models.Job(job_id=job_id, video_id="abcdefghijk", status="complete"))
            db.commit()
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / job_id
            folder.mkdir()
            (folder / "shots.jsonl").write_text("")
            with patch.object(main.analysis_orchestrator, "output_base_dir", Path(directory)):
                legacy = self.client.get(f"/api/jobs/{job_id}/shots")
                self.assertEqual(legacy.status_code, 200)
                self.assertEqual(legacy.json()["goal_entries"], [])
                (folder / "goal_entries.jsonl").write_text(json.dumps({
                    "entry_id": "entry", "region_id": "blue_hub", "goal": "high",
                    "robot_track_id": None, "t_seconds": 1.5,
                }) + "\n")
                result = self.client.get(f"/api/jobs/{job_id}/shots")
                self.assertEqual(result.status_code, 200)
                self.assertEqual(result.json()["statistics"]["attempted"], 0)
                self.assertEqual(result.json()["goal_statistics"]["made"], 1)
                self.assertEqual(result.json()["goal_statistics"]["unassigned"], 1)
                (folder / "goal_entries.jsonl").write_text("broken json")
                self.assertEqual(self.client.get(f"/api/jobs/{job_id}/shots").status_code, 500)

    def test_create_job_preserves_youtube_timestamp(self):
        info = {
            "id": "abcdefghijk",
            "duration": 300,
            "fps": 30,
            "width": 1280,
            "height": 720,
        }
        with (
            patch.object(main.video_downloader, "get_video_info", return_value=info),
            patch.object(main, "process_job"),
        ):
            response = self.client.post(
                "/api/jobs",
                json={"url": "https://youtube.com/watch?v=abcdefghijk&t=2m", "match_id": None},
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["video_id"], "abcdefghijk")
        self.assertEqual(body["start_offset"], 120)
        self.assertEqual(body["duration"], 180)
        self.assertEqual(body["status"], "queued")

    def test_currently_live_link_becomes_a_capture_job(self):
        info = {"id": "abcdefghijk", "is_live": True, "width": 1280, "height": 720}
        with (
            patch.object(main.video_downloader, "get_video_info", return_value=info),
            patch.object(main, "process_job"),
        ):
            response = self.client.post(
                "/api/jobs", json={"url": "https://youtube.com/watch?v=abcdefghijk", "live_capture": True}
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["capture_mode"], "live")
        self.assertIsNone(body["duration"])
        self.assertEqual(body["status"], "queued")

    def test_live_capture_rejects_a_finished_video(self):
        info = {"id": "abcdefghijk", "duration": 300, "fps": 30, "width": 1280, "height": 720}
        with patch.object(main.video_downloader, "get_video_info", return_value=info):
            response = self.client.post(
                "/api/jobs", json={"url": "https://youtube.com/watch?v=abcdefghijk", "live_capture": True}
            )
        self.assertEqual(response.status_code, 400)
        self.assertIn("not currently live", response.json()["error"])

    def test_errors_use_contract_shape(self):
        with patch.object(
            main.video_downloader, "get_video_info", side_effect=RuntimeError("unavailable")
        ):
            response = self.client.post(
                "/api/jobs", json={"url": "https://youtube.com/watch?v=abcdefghijk"}
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn("unavailable", response.json()["error"])
        self.assertNotIn("detail", response.json())

    def test_local_video_supports_byte_ranges(self):
        with tempfile.TemporaryDirectory() as directory:
            media = Path(directory) / "segment.mp4"
            media.write_bytes(bytes(range(256)) * 8)
            with self.sessions() as db:
                db.add(
                    models.Job(
                        job_id="11111111-1111-4111-8111-111111111111",
                        video_id="abcdefghijk",
                        local_path=str(media),
                        duration=10,
                        fps=30,
                        width=1280,
                        height=720,
                        status="downloaded",
                    )
                )
                db.commit()

            response = self.client.get(
                "/api/video/11111111-1111-4111-8111-111111111111",
                headers={"Range": "bytes=10-19"},
            )

        self.assertEqual(response.status_code, 206)
        self.assertEqual(response.content, (bytes(range(256)) * 8)[10:20])
        self.assertEqual(response.headers["accept-ranges"], "bytes")
        self.assertEqual(response.headers["content-range"], "bytes 10-19/2048")

    def test_completed_job_can_be_rerun_and_old_output_directory_is_replaced(self):
        job_id = "22222222-2222-4222-8222-222222222222"
        video_id = "abcdefghijk"
        with self.sessions() as db:
            db.add(
                models.Job(
                    job_id=job_id,
                    video_id=video_id,
                    status="complete",
                    attempt=3,
                )
            )
            db.commit()

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / job_id
            output_dir.mkdir(parents=True)
            (output_dir / "tracks.jsonl").write_text("old", encoding="utf-8")
            with (
                patch.object(main.analysis_orchestrator, "output_base_dir", Path(directory)),
                patch.object(main, "process_job") as worker,
            ):
                response = self.client.post(f"/api/jobs/{job_id}/retry")

            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertEqual(body["status"], "queued")
            self.assertEqual(body["attempt"], 4)
            self.assertFalse(output_dir.exists())
            worker.assert_called_once_with(job_id, f"https://www.youtube.com/watch?v={video_id}", False)


if __name__ == "__main__":
    unittest.main()
