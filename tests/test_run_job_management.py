from __future__ import annotations

import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import run


class RunJobManagementTests(unittest.TestCase):
    def setUp(self) -> None:
        with run.JOB_STORE_LOCK:
            self.previous_jobs = dict(run.JOB_STORE)
            run.JOB_STORE.clear()

    def tearDown(self) -> None:
        with run.JOB_STORE_LOCK:
            run.JOB_STORE.clear()
            run.JOB_STORE.update(self.previous_jobs)

    def test_prune_finished_jobs_removes_old_terminal_artifacts(self) -> None:
        with TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            old_job_dir = temp_root / "job_old"
            new_job_dir = temp_root / "job_new"
            old_upload = temp_root / "old_upload.mp4"
            new_upload = temp_root / "new_upload.mp4"
            old_job_dir.mkdir()
            new_job_dir.mkdir()
            old_upload.write_bytes(b"old")
            new_upload.write_bytes(b"new")

            with run.JOB_STORE_LOCK:
                run.JOB_STORE["old"] = {
                    "status": "completed",
                    "created_at": 100.0,
                    "finished_at": 100.0,
                    "job_dir": str(old_job_dir),
                    "source_is_upload": True,
                    "uploaded_source_path": str(old_upload),
                    "lock": threading.Lock(),
                }
                run.JOB_STORE["new"] = {
                    "status": "completed",
                    "created_at": 200.0,
                    "finished_at": 200.0,
                    "job_dir": str(new_job_dir),
                    "source_is_upload": True,
                    "uploaded_source_path": str(new_upload),
                    "lock": threading.Lock(),
                }

            run._prune_finished_jobs(max_terminal_jobs=1)

            with run.JOB_STORE_LOCK:
                self.assertNotIn("old", run.JOB_STORE)
                self.assertIn("new", run.JOB_STORE)
            self.assertFalse(old_job_dir.exists())
            self.assertFalse(old_upload.exists())
            self.assertTrue(new_job_dir.exists())
            self.assertTrue(new_upload.exists())

    def test_get_active_job_views_only_returns_active_jobs(self) -> None:
        with run.JOB_STORE_LOCK:
            run.JOB_STORE["active"] = {
                "source_name": "active.mp4",
                "device": "cpu",
                "status": "streaming",
                "total_alerts": 3,
                "unique_track_count": 1,
                "crop_dir": "/tmp/active",
                "created_at": 2.0,
                "lock": threading.Lock(),
            }
            run.JOB_STORE["done"] = {
                "source_name": "done.mp4",
                "device": "cpu",
                "status": "completed",
                "total_alerts": 1,
                "unique_track_count": 1,
                "crop_dir": "/tmp/done",
                "created_at": 1.0,
                "lock": threading.Lock(),
            }

        active_jobs = run._get_active_job_views()

        self.assertEqual(len(active_jobs), 1)
        self.assertEqual(active_jobs[0]["job_id"], "active")
