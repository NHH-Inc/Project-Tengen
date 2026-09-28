"""Recordings made at an event -- a phone or a webcam -- go in through an upload, not YouTube.

The failures worth pinning are the silent ones: a job that quietly tries to fetch its made-up
video_id from YouTube, a completed job whose recording is unlinked, and a retention sweep that
treats someone's only copy of a match as a disposable cache.
"""

import datetime
import os
import re
import shutil
import subprocess
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

from ingest import main, models, retention

VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")
PROBED = {"duration": 150.0, "fps": 30.0, "width": 1920, "height": 1080}


class LocalUploadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = create_engine(
            "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
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
            shutil.rmtree(_test_data_dir, ignore_errors=True)

    def setUp(self):
        with self.sessions() as db:
            for table in reversed(models.Base.metadata.sorted_tables):
                db.execute(table.delete())
            db.commit()
        self.scratch = Path(tempfile.mkdtemp(prefix="frc-upload-"))

    def tearDown(self):
        shutil.rmtree(self.scratch, ignore_errors=True)

    def _upload(self, name="match01.mp4", body=b"not really a video", probe=PROBED, **form):
        probe_patch = (
            patch.object(main.video_downloader, "probe_media", side_effect=probe)
            if isinstance(probe, Exception)
            else patch.object(main.video_downloader, "probe_media", return_value=probe)
        )
        with probe_patch, patch.object(main, "process_job") as worker:
            response = self.client.post(
                "/api/jobs/upload", files={"file": (name, body, "video/mp4")}, data=form
            )
        return response, worker

    def _local_job(self, status="downloaded", **overrides):
        recording = self.scratch / "recording.mp4"
        recording.write_bytes(b"frames")
        fields = dict(
            job_id="33333333-3333-4333-8333-333333333333",
            video_id=main._local_video_id("33333333-3333-4333-8333-333333333333"),
            capture_mode=main.LOCAL_CAPTURE,
            local_path=str(recording),
            status=status,
            duration=150.0,
            fps=30.0,
            width=1920,
            height=1080,
        )
        fields.update(overrides)
        with self.sessions() as db:
            db.add(models.Job(**fields))
            db.commit()
        return fields["job_id"], recording

    # --------------------------------------------------------------------- the upload itself

    def test_upload_creates_a_local_job_ready_for_analysis(self):
        response, worker = self._upload(match_id="2026flroc_qm1")
        self.assertEqual(response.status_code, 200, response.text)
        job = response.json()

        self.assertEqual(job["capture_mode"], "local")
        self.assertEqual(job["status"], "downloaded")
        self.assertEqual(job["match_id"], "2026flroc_qm1")
        self.assertRegex(job["video_id"], VIDEO_ID)
        self.assertEqual(
            (job["duration"], job["fps"], job["width"], job["height"]), (150.0, 30.0, 1920, 1080)
        )
        stored = Path(job["local_path"])
        self.assertTrue(stored.is_file())
        self.assertEqual(stored.parent.name, "uploads")
        # An empty url is the signal to analyze the file; anything else would go to YouTube.
        worker.assert_called_once_with(job["job_id"], "", False)
        stored.unlink()

    def test_a_non_video_file_is_refused_and_nothing_is_created(self):
        response, worker = self._upload(name="notes.txt")
        self.assertEqual(response.status_code, 400)
        worker.assert_not_called()
        with self.sessions() as db:
            self.assertEqual(db.query(models.Job).count(), 0)

    def test_an_unreadable_video_leaves_no_file_and_no_job(self):
        uploads = Path(main.data_dir) / "uploads"
        before = set(uploads.iterdir()) if uploads.is_dir() else set()
        response, worker = self._upload(probe=RuntimeError("no video track"))
        self.assertEqual(response.status_code, 400)
        self.assertIn("no video track", response.json()["error"])
        worker.assert_not_called()
        after = set(uploads.iterdir()) if uploads.is_dir() else set()
        self.assertEqual(after, before)
        with self.sessions() as db:
            self.assertEqual(db.query(models.Job).count(), 0)

    def test_the_generated_video_id_satisfies_the_contract_and_is_stable(self):
        job_id = "44444444-4444-4444-8444-444444444444"
        self.assertRegex(main._local_video_id(job_id), VIDEO_ID)
        self.assertEqual(main._local_video_id(job_id), main._local_video_id(job_id))
        self.assertNotEqual(
            main._local_video_id(job_id),
            main._local_video_id("55555555-5555-4555-8555-555555555555"),
        )

    # ------------------------------------------------------------------ container handling

    def test_an_mp4_is_left_alone(self):
        path = self.scratch / "clip.mp4"
        path.write_bytes(b"x")
        with patch.object(main.subprocess, "run") as run:
            self.assertEqual(main._browser_playable(path), path)
        run.assert_not_called()

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "needs ffmpeg")
    def test_an_mkv_recording_is_remuxed_to_mp4_for_the_browser(self):
        source = self.scratch / "match.mkv"
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=30",
             "-t", "1", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source)],
            check=True,
        )
        result = main._browser_playable(source)
        self.assertEqual(result.suffix, ".mp4")
        self.assertFalse(source.exists(), "the MKV should be replaced, not duplicated")
        media = main.video_downloader.probe_media(str(result))
        self.assertEqual((media["width"], media["height"]), (320, 240))

    # ------------------------------------------------------------------- pipeline and retry

    def test_a_local_job_is_analyzed_from_its_file_and_keeps_it(self):
        job_id, recording = self._local_job()
        seen = {}

        def fake_run(job_data, season_path, on_progress=None):
            seen.update(job_data)
            return {"result": {"box_sample_rate": 30.0}}

        def session():
            db = self.sessions()
            try:
                yield db
            finally:
                db.close()

        with (
            patch.object(main.database, "get_db", session),
            patch.object(main.analysis_orchestrator, "output_base_dir", self.scratch / "jobs"),
            patch.object(main.analysis_orchestrator, "run_job", side_effect=fake_run),
            patch.object(main, "import_results"),
            patch.object(main.video_downloader, "get_video_info") as youtube,
        ):
            main.process_job(job_id, "", False)

        youtube.assert_not_called()
        self.assertNotIn("stream_url", seen)
        self.assertEqual(seen["local_path"], str(recording))
        with self.sessions() as db:
            job = db.get(models.Job, job_id)
            self.assertEqual(job.status, "complete", job.error)
            self.assertEqual(job.local_path, str(recording), "the recording must stay linked")
        self.assertTrue(recording.exists())

    def test_a_local_job_whose_file_is_gone_fails_clearly(self):
        job_id, recording = self._local_job()
        recording.unlink()

        def session():
            db = self.sessions()
            try:
                yield db
            finally:
                db.close()

        with (
            patch.object(main.database, "get_db", session),
            patch.object(main.analysis_orchestrator, "output_base_dir", self.scratch / "jobs"),
            patch.object(main.analysis_orchestrator, "run_job") as run,
        ):
            main.process_job(job_id, "", False)

        run.assert_not_called()
        with self.sessions() as db:
            job = db.get(models.Job, job_id)
            self.assertEqual(job.status, "failed")
            self.assertEqual(job.error_code, "video_unavailable")

    def test_retrying_a_local_job_never_goes_to_youtube(self):
        job_id, _ = self._local_job(status="failed")
        with (
            patch.object(main.analysis_orchestrator, "output_base_dir", self.scratch / "jobs"),
            patch.object(main, "process_job") as worker,
        ):
            response = self.client.post(f"/api/jobs/{job_id}/retry")
        self.assertEqual(response.status_code, 200, response.text)
        worker.assert_called_once_with(job_id, "", False)

    # ------------------------------------------------------------------------- retention

    def test_retention_never_deletes_an_uploaded_recording(self):
        old = datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc)
        job_id, recording = self._local_job(status="complete", updated_at=old)

        cached = self.scratch / "segment.mp4"
        cached.write_bytes(b"re-downloadable")
        with self.sessions() as db:
            db.add(models.Job(
                job_id="66666666-6666-4666-8666-666666666666", video_id="abcdefghijk",
                local_path=str(cached), status="complete", updated_at=old,
            ))
            db.commit()
            retention.sweep(db, grace_days=0)
            db.commit()

        self.assertTrue(recording.exists(), "an upload is the only copy of that match")
        self.assertFalse(cached.exists(), "a YouTube segment is still a cache")
        with self.sessions() as db:
            self.assertEqual(db.get(models.Job, job_id).local_path, str(recording))


if __name__ == "__main__":
    unittest.main()
