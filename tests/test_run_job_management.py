from __future__ import annotations

import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from main_pipeline.types import StageRunResult
from person_detection.config import PersonInstance
from web_app import app as web_app


class RunJobManagementTests(unittest.TestCase):
    def setUp(self) -> None:
        with web_app.JOB_STORE_LOCK:
            self.previous_jobs = dict(web_app.JOB_STORE)
            web_app.JOB_STORE.clear()

    def tearDown(self) -> None:
        with web_app.JOB_STORE_LOCK:
            web_app.JOB_STORE.clear()
            web_app.JOB_STORE.update(self.previous_jobs)

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

            with web_app.JOB_STORE_LOCK:
                web_app.JOB_STORE["old"] = {
                    "status": "completed",
                    "created_at": 100.0,
                    "finished_at": 100.0,
                    "job_dir": str(old_job_dir),
                    "source_is_upload": True,
                    "uploaded_source_path": str(old_upload),
                    "lock": threading.Lock(),
                }
                web_app.JOB_STORE["new"] = {
                    "status": "completed",
                    "created_at": 200.0,
                    "finished_at": 200.0,
                    "job_dir": str(new_job_dir),
                    "source_is_upload": True,
                    "uploaded_source_path": str(new_upload),
                    "lock": threading.Lock(),
                }

            web_app._prune_finished_jobs(max_terminal_jobs=1)

            with web_app.JOB_STORE_LOCK:
                self.assertNotIn("old", web_app.JOB_STORE)
                self.assertIn("new", web_app.JOB_STORE)
            self.assertFalse(old_job_dir.exists())
            self.assertFalse(old_upload.exists())
            self.assertTrue(new_job_dir.exists())
            self.assertTrue(new_upload.exists())

    def test_get_active_job_views_only_returns_active_jobs(self) -> None:
        with web_app.JOB_STORE_LOCK:
            web_app.JOB_STORE["active"] = {
                "source_name": "active.mp4",
                "device": "cpu",
                "status": "streaming",
                "total_alerts": 3,
                "unique_track_count": 1,
                "crop_dir": "/tmp/active",
                "created_at": 2.0,
                "lock": threading.Lock(),
            }
            web_app.JOB_STORE["done"] = {
                "source_name": "done.mp4",
                "device": "cpu",
                "status": "completed",
                "total_alerts": 1,
                "unique_track_count": 1,
                "crop_dir": "/tmp/done",
                "created_at": 1.0,
                "lock": threading.Lock(),
            }

        active_jobs = web_app._get_active_job_views()

        self.assertEqual(len(active_jobs), 1)
        self.assertEqual(active_jobs[0]["job_id"], "active")

    def test_normalize_stage_selection_auto_enables_dependencies(self) -> None:
        stage_flags = web_app._normalize_stage_selection(
            run_person_detection=False,
            run_region_extraction=False,
            run_facemask_detection=True,
            run_glove_detection=False,
            run_behavior_detection=False,
        )

        self.assertTrue(stage_flags["run_person_detection"])
        self.assertTrue(stage_flags["run_region_extraction"])
        self.assertTrue(stage_flags["run_facemask_detection"])
        self.assertFalse(stage_flags["run_glove_detection"])
        self.assertFalse(stage_flags["run_behavior_detection"])

    def test_normalize_stage_selection_requires_at_least_one_stage(self) -> None:
        with self.assertRaisesRegex(ValueError, "ít nhất 1 stage"):
            web_app._normalize_stage_selection(
                run_person_detection=False,
                run_region_extraction=False,
                run_facemask_detection=False,
                run_glove_detection=False,
                run_behavior_detection=False,
            )

    def test_resolve_source_input_keeps_rtsp_url_as_string(self) -> None:
        source, source_name = web_app._resolve_source_input(
            "rtsp://user:pass@192.168.1.10:554/stream1"
        )

        self.assertIsInstance(source, str)
        self.assertEqual(source, "rtsp://user:pass@192.168.1.10:554/stream1")
        self.assertEqual(source_name, "stream1")

    def test_run_job_worker_executes_main_pipeline_and_updates_job_view(self) -> None:
        class FakePipeline:
            last_config = None

            def __init__(
                self,
                config,
                stage_callback=None,
                frame_callback=None,
            ) -> None:
                self.config = config
                FakePipeline.last_config = config
                self.stage_callback = stage_callback
                self.frame_callback = frame_callback

            def run(self):
                assert self.stage_callback is not None
                self.stage_callback("person_detection", "running", None)
                if self.frame_callback is not None:
                    self.frame_callback(
                        0,
                        1,
                        np.zeros((10, 10, 3), dtype=np.uint8),
                        [
                            PersonInstance(
                                tracker_id=3,
                                confidence=0.92,
                                bbox_xyxy=(0, 0, 10, 10),
                                crop=np.zeros((5, 5, 3), dtype=np.uint8),
                                should_alert=True,
                            )
                        ],
                    )
                self.stage_callback(
                    "person_detection",
                    "completed",
                    StageRunResult(
                        name="person_detection",
                        enabled=True,
                        input_path="/tmp/input.mp4",
                        output_path="/tmp/crops",
                        summary={"crop_file_count": 1},
                    ),
                )
                self.stage_callback(
                    "person_region_extraction",
                    "completed",
                    StageRunResult(
                        name="person_region_extraction",
                        enabled=True,
                        input_path="/tmp/crops",
                        output_path="/tmp/regions",
                        summary={"processed_count": 1},
                    ),
                )
                self.stage_callback(
                    "facemask_detection",
                    "completed",
                    StageRunResult(
                        name="facemask_detection",
                        enabled=True,
                        input_path="/tmp/regions",
                        output_path="/tmp/facemask",
                        summary={"mask_count": 1},
                    ),
                )
                self.stage_callback(
                    "glove_detection",
                    "completed",
                    StageRunResult(
                        name="glove_detection",
                        enabled=True,
                        input_path="/tmp/regions",
                        output_path="/tmp/glove",
                        summary={"glove_count": 1},
                    ),
                )
                self.stage_callback(
                    "behavior_detection",
                    "disabled",
                    StageRunResult(name="behavior_detection", enabled=False),
                )
                return {
                    "status": "completed",
                    "output_root": "/tmp/job_output",
                    "paths": {
                        "crops_dir": "/tmp/job_output/crops",
                        "summary_path": "/tmp/job_output/summary.json",
                    },
                }

        with TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            with web_app.JOB_STORE_LOCK:
                web_app.JOB_STORE["job123"] = {
                    "source": "./vid_test/vid_1.mp4",
                    "source_name": "vid_1.mp4",
                    "model_path": "./weights/yolo11n.pt",
                    "device": "cpu",
                    "confidence_threshold": 0.5,
                    "iou_threshold": 0.45,
                    "crop_padding": 8,
                    "input_resize_scale": 0.75,
                    "frame_skip": 0,
                    "crop_save_interval": 10,
                    "track_ttl_frames": 60,
                    "alert_cooldown_frames": None,
                    "max_frames": 5,
                    "crop_dir": str(temp_root / "crops"),
                    "job_dir": str(temp_root),
                    "pipeline_output_root": str(temp_root),
                    "summary_path": str(temp_root / "summary.json"),
                    "run_person_detection": True,
                    "run_region_extraction": True,
                    "run_facemask_detection": True,
                    "run_glove_detection": True,
                    "run_behavior_detection": False,
                    "source_is_upload": False,
                    "uploaded_source_path": None,
                    "status": "ready",
                    "current_stage": None,
                    "total_alerts": 0,
                    "unique_track_count": 0,
                    "last_frame_index": -1,
                    "unique_ids": set(),
                    "latest_frame_jpeg": None,
                    "latest_frame_seq": -1,
                    "latest_summary": None,
                    "stage_states": web_app._initial_stage_states(
                        {
                            "run_person_detection": True,
                            "run_region_extraction": True,
                            "run_facemask_detection": True,
                            "run_glove_detection": True,
                            "run_behavior_detection": False,
                        }
                    ),
                    "created_at": 1.0,
                    "finished_at": None,
                    "lock": threading.Lock(),
                }

            original_pipeline = web_app.StreamPipeline
            original_prune = web_app._prune_finished_jobs
            try:
                web_app.StreamPipeline = FakePipeline
                web_app._prune_finished_jobs = lambda max_terminal_jobs=web_app.MAX_RETAINED_TERMINAL_JOBS: None
                web_app.run_job_worker("job123")
            finally:
                web_app.StreamPipeline = original_pipeline
                web_app._prune_finished_jobs = original_prune

            job_view = web_app._job_to_view("job123", web_app.JOB_STORE["job123"])
            self.assertEqual(job_view["status"], "completed")
            self.assertEqual(job_view["current_stage"], "completed")
            self.assertEqual(job_view["total_alerts"], 1)
            self.assertEqual(job_view["unique_track_count"], 1)
            self.assertEqual(job_view["summary_path"], "/tmp/job_output/summary.json")
            self.assertEqual(job_view["crop_dir"], "/tmp/job_output/crops")
            stage_names = [stage["name"] for stage in job_view["stage_views"] if stage["status"] == "completed"]
            self.assertIn("person_detection", stage_names)
            self.assertIn("facemask_detection", stage_names)
            self.assertTrue(FakePipeline.last_config.run_person_detection)
            self.assertTrue(FakePipeline.last_config.run_region_extraction)
            self.assertTrue(FakePipeline.last_config.run_facemask_detection)
            self.assertTrue(FakePipeline.last_config.run_glove_detection)
            self.assertFalse(FakePipeline.last_config.run_behavior_detection)
