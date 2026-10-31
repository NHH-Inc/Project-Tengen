"""A hosted Tengen: keys keep strangers out, and a published run is served like an analysed one."""

import json
import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from ingest import hosted, main, models

JOB_ID = "44444444-4444-4444-8444-444444444444"


def boxes(x, start=0.0, end=60.0, rate=30.0):
    return [{"t": round(start + i / rate, 4), "x": x, "y": 0.4, "w": 0.06, "h": 0.06,
             "field_x": 10.0 + i / rate * 0.5, "field_y": 8.0}
            for i in range(int((end - start) * rate) + 1)]


def run_files():
    tracks = [{"schema_version": 3, "track_id": 1, "team": 254, "alliance": "blue", "team_confidence": None,
               "boxes": boxes(0.2), "gaps": []},
              {"schema_version": 3, "track_id": 2, "team": None, "alliance": "red", "team_confidence": None,
               "boxes": boxes(0.7), "gaps": []}]
    events = [{"schema_version": 3, "event_id": "e1", "job_id": JOB_ID, "match_id": "2026x_qm1", "team": None,
               "track_id": None, "t_seconds": 0.0, "phase": "auto", "event_type": "match_start",
               "confidence": 1.0, "field_x": None, "field_y": None, "source": "scoreboard_ocr"}]
    shots = [{"shot_id": f"s{i}", "launch_t_seconds": 25.0 + i, "robot_track_id": 1, "confidence": 0.9,
              "outcome": "unknown"} for i in range(3)]
    return {
        "tracks.jsonl": "".join(json.dumps(t) + "\n" for t in tracks).encode(),
        "events.jsonl": "".join(json.dumps(e) + "\n" for e in events).encode(),
        "shots.jsonl": "".join(json.dumps(s) + "\n" for s in shots).encode(),
        "result.json": json.dumps({"box_sample_rate": 30.0, "frames_processed": 1800}).encode(),
    }


class HostedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
        models.Base.metadata.create_all(cls.engine)
        cls.sessions = sessionmaker(bind=cls.engine)

        def override():
            db = cls.sessions()
            try:
                yield db
            finally:
                db.close()

        main.app.dependency_overrides[main.get_db] = override
        cls.client = TestClient(main.app)

    @classmethod
    def tearDownClass(cls):
        main.app.dependency_overrides.clear()

    def setUp(self):
        with self.sessions() as db:
            for table in reversed(models.Base.metadata.sorted_tables):
                db.execute(table.delete())
            db.commit()
        hosted._feature_cache.clear()
        hosted._rate_hits.clear()
        self.env = patch.dict(os.environ, {"TENGEN_WRITE_KEY": "write-secret", "TENGEN_READ_KEY": "read-secret"})
        self.env.start()
        # _read_result opens its own session; point it at the same in-memory database.
        self.db_patch = patch.object(main.database, "get_db", lambda: iter([self.sessions()]))
        self.db_patch.start()

    def tearDown(self):
        self.db_patch.stop()
        self.env.stop()

    def publish(self, key="write-secret", files=None):
        files = files or run_files()
        job = {"job_id": JOB_ID, "match_id": "2026x_qm1", "season": 2026, "youtube_id": "refM5LLkuJ8",
               "start_offset": 0.0, "duration": 60.0, "fps": 30.0}
        return self.client.post("/api/publish", headers={"X-Tengen-Key": key}, data={"job": json.dumps(job)},
                                files=[("files", (name, data)) for name, data in files.items()])

    # ------------------------------------------------------------------------- keys

    def test_reading_needs_a_key_when_one_is_set(self):
        self.assertEqual(self.client.get("/api/jobs").status_code, 401)
        self.assertEqual(self.client.get("/api/jobs", headers={"X-Tengen-Key": "read-secret"}).status_code, 200)
        # Keys never travel in URLs, where request logs keep them.
        self.assertEqual(self.client.get("/api/jobs?key=read-secret").status_code, 401)
        self.assertEqual(self.client.get("/api/health").status_code, 200, "health stays open for monitors")

    def test_the_read_key_cannot_write(self):
        self.assertEqual(self.publish(key="read-secret").status_code, 401)
        self.assertEqual(self.publish(key="nope").status_code, 401)

    def test_a_hosted_server_without_keys_is_closed_not_open(self):
        with patch.dict(os.environ, {"TENGEN_WRITE_KEY": "", "TENGEN_READ_KEY": "", "TENGEN_REQUIRE_KEYS": "1"}):
            self.assertEqual(self.client.get("/api/jobs").status_code, 503)
            self.assertEqual(self.client.get("/api/health").status_code, 200)

    def test_a_rate_limited_server_returns_429_after_the_budget(self):
        read = {"X-Tengen-Key": "read-secret"}
        with patch.dict(os.environ, {"TENGEN_RATE_LIMIT": "3"}):
            for _ in range(3):
                self.assertEqual(self.client.get("/api/jobs", headers=read).status_code, 200)
            over = self.client.get("/api/jobs", headers=read)
            self.assertEqual(over.status_code, 429)
            self.assertIn("retry-after", {k.lower() for k in over.headers})
            # Health is exempt, so a monitor is never throttled.
            self.assertEqual(self.client.get("/api/health").status_code, 200)

    def test_rate_limiting_is_off_for_a_local_install(self):
        # No keys required and no explicit limit: a local desktop app is not throttled.
        with patch.dict(os.environ, {"TENGEN_WRITE_KEY": "", "TENGEN_READ_KEY": ""}):
            self.assertEqual(hosted._rate_limit(), 0)
            for _ in range(50):
                self.assertEqual(self.client.get("/api/jobs").status_code, 200)

    def test_a_publish_only_server_refuses_to_analyse(self):
        with patch.dict(os.environ, {"TENGEN_PUBLISH_ONLY": "1"}):
            response = self.client.post("/api/jobs", headers={"X-Tengen-Key": "write-secret"},
                                        json={"url": "https://www.youtube.com/watch?v=refM5LLkuJ8"})
            self.assertEqual(response.status_code, 409)
            self.assertEqual(self.publish().status_code, 200, "publishing still works")

    def test_without_keys_a_local_install_is_open_as_before(self):
        with patch.dict(os.environ, {"TENGEN_WRITE_KEY": "", "TENGEN_READ_KEY": ""}):
            self.assertEqual(self.client.get("/api/jobs").status_code, 200)

    # ------------------------------------------------------------------------- publishing

    def test_a_published_run_is_served_like_an_analysed_one(self):
        response = self.publish()
        self.assertEqual(response.status_code, 200, response.text)
        read = {"X-Tengen-Key": "read-secret"}
        job = self.client.get(f"/api/jobs/{JOB_ID}", headers=read).json()
        self.assertEqual(job["status"], "complete")
        self.assertEqual(job["video_id"], "refM5LLkuJ8")
        tracks = self.client.get(f"/api/jobs/{JOB_ID}/tracks", headers=read).json()
        self.assertEqual(len(tracks["tracks"]), 2)
        self.assertLessEqual(tracks["box_sample_rate"], 10.0)
        # 60 s at 30 fps thinned to 10 Hz: about 601 boxes, not 1801.
        self.assertLess(len(tracks["tracks"][0]["boxes"]), 650)
        shots = self.client.get(f"/api/jobs/{JOB_ID}/shots", headers=read).json()
        self.assertEqual(shots["statistics"]["attempted"], 3)
        self.assertEqual(self.client.get(f"/api/jobs/{JOB_ID}/result", headers=read).json()["frames_processed"], 1800)

    def test_republishing_replaces_rather_than_duplicates(self):
        self.publish()
        self.publish()
        read = {"X-Tengen-Key": "read-secret"}
        self.assertEqual(len(self.client.get(f"/api/jobs/{JOB_ID}/tracks", headers=read).json()["tracks"]), 2)

    def test_unknown_files_are_refused(self):
        files = run_files() | {"notes.txt": b"hello"}
        self.assertEqual(self.publish(files=files).status_code, 422)

    # ------------------------------------------------------------------------- video

    def test_media_urls_use_a_token_for_one_job_not_the_key(self):
        self.publish()
        token = self.client.get(f"/api/jobs/{JOB_ID}/media-token",
                                headers={"X-Tengen-Key": "read-secret"}).json()["token"]
        self.assertNotIn("read-secret", token)
        self.assertTrue(hosted.check_media_token(JOB_ID, token))
        self.assertFalse(hosted.check_media_token("another-job", token))
        self.assertFalse(hosted.check_media_token(JOB_ID, token, now=10 ** 11), "tokens expire")
        # The middleware lets the token through to the media route (404 here: no file), not 401.
        self.assertEqual(self.client.get(f"/api/video/{JOB_ID}?t={token}").status_code, 404)
        self.assertEqual(self.client.get(f"/api/video/{JOB_ID}?t=1.bad").status_code, 401)
        self.assertEqual(self.client.get(f"/api/jobs?t={token}").status_code, 401, "only media routes")

    def test_a_published_video_copy_is_stored_and_served_by_redirect(self):
        stored = {}
        with patch.object(hosted, "store_video", side_effect=lambda job_id, data: stored.update({job_id: data}) or "tengen-video/x.mp4"),              patch.object(hosted, "video_redirect", side_effect=lambda job: __import__("fastapi").responses.RedirectResponse("https://storage.example/signed", 307)):
            response = self.publish(files=run_files() | {"preview.mp4": b"fake mp4"})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertTrue(response.json()["video"])
            self.assertEqual(stored[JOB_ID], b"fake mp4")
            job = self.client.get(f"/api/jobs/{JOB_ID}", headers={"X-Tengen-Key": "read-secret"}).json()
            self.assertTrue(job["hosted_video"])
            moved = self.client.get(f"/api/video/{JOB_ID}", headers={"X-Tengen-Key": "read-secret"},
                                    follow_redirects=False)
            self.assertEqual(moved.status_code, 307)
            self.assertEqual(moved.headers["location"], "https://storage.example/signed")

    def test_deleting_a_published_match_removes_its_stored_files(self):
        self.publish()
        with patch.object(hosted, "delete_video") as deleted:
            response = self.client.delete(f"/api/jobs/{JOB_ID}", headers={"X-Tengen-Key": "write-secret"})
        self.assertEqual(response.status_code, 204, response.text)
        deleted.assert_called_once()
        with self.sessions() as db:
            self.assertEqual(db.query(models.RunArtifact).count(), 0)

    # ------------------------------------------------------------------------- strategy

    def test_strategy_comes_from_published_runs_and_web_corrections(self):
        self.publish()
        read, write = {"X-Tengen-Key": "read-secret"}, {"X-Tengen-Key": "write-secret"}
        robots = self.client.get(f"/api/jobs/{JOB_ID}/strategy", headers=read).json()["robots"]
        by_key = {r["robot_key"]: r for r in robots}
        self.assertIn("team:254", by_key)
        self.assertEqual(by_key["team:254"]["launches"]["total"], 3)
        self.assertEqual(self.client.get("/api/strategy/teams", headers=read).json()["teams"][0]["team"], 254)

        # Assigning track 2 a team on the website reaches the strategy layer.
        response = self.client.patch(f"/api/jobs/{JOB_ID}/tracks/2", headers=write, json={"team": 1678})
        self.assertIn(response.status_code, (200, 201), response.text)
        teams = [t["team"] for t in self.client.get("/api/strategy/teams", headers=read).json()["teams"]]
        self.assertEqual(teams, [254, 1678])
        profile = self.client.get("/api/strategy/teams/1678", headers=read).json()["profile"]
        self.assertEqual(profile["n_matches"], 1)


class ThinTests(unittest.TestCase):
    def test_ten_hz_and_holes_keep_both_sides(self):
        b = boxes(0.5, end=10.0)
        b = [x for x in b if not 4.0 < x["t"] < 6.0]
        kept = hosted.thin_boxes(b)
        times = [x["t"] for x in kept]
        self.assertLess(len(kept), 130)
        self.assertIn(max(t for t in times if t <= 4.0), [x["t"] for x in b if x["t"] <= 4.0][-1:])
        self.assertIn(min(t for t in times if t >= 6.0), [x["t"] for x in b if x["t"] >= 6.0][:1])


if __name__ == "__main__":
    unittest.main()
