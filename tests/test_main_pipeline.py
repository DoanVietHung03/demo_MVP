from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from main_pipeline.config import MainPipelineConfig
from main_pipeline.pipeline import MainPipeline
from person_detection.config import PersonInstance


class FakeOfflineVideoRecorder:
    last_instance = None

    def __init__(self, source, output_path, codec, fps=None) -> None:
        self.source = source
        self.output_path = output_path
        self.codec = codec
        self.requested_fps = fps
        self.resolved_fps = fps or 24.0
        self.frame_count = 0
        self.closed = False
        FakeOfflineVideoRecorder.last_instance = self

    def write_frame(self, frame) -> None:
        self.frame_count += 1
        self.last_shape = tuple(frame.shape)

    def close(self) -> None:
        self.closed = True


class FakeStreamPipeline:
    last_config = None

    def __init__(self, config, stage_callback=None, frame_callback=None) -> None:
        self.config = config
        self.stage_callback = stage_callback
        self.frame_callback = frame_callback
        FakeStreamPipeline.last_config = config

    def run(self):
        if self.stage_callback is not None:
            self.stage_callback("person_detection", "running", None)
            self.stage_callback("ppe_detection", "running", None)

        if self.frame_callback is not None:
            self.frame_callback(
                0,
                1,
                np.full((12, 16, 3), 90, dtype=np.uint8),
                [
                    PersonInstance(
                        tracker_id=7,
                        confidence=0.91,
                        bbox_xyxy=(1, 2, 10, 11),
                        crop=np.zeros((6, 6, 3), dtype=np.uint8),
                        should_alert=False,
                    )
                ],
            )

        summary_path = (
            Path(self.config.output_root) / str(self.config.job_name) / "summary.json"
        )
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "job_name": str(self.config.job_name),
            "source": self.config.source,
            "output_root": str(summary_path.parent),
            "status": "completed",
            "paths": {
                "run_root": str(summary_path.parent),
                "crops_dir": str(summary_path.parent / "crops"),
                "ppe_output_root": str(summary_path.parent / "ppe_detection"),
                "region_output_root": str(summary_path.parent / "person_region_extraction"),
                "facemask_output_root": str(summary_path.parent / "facemask_detection"),
                "glove_output_root": str(summary_path.parent / "glove_detection"),
                "behavior_output_root": str(summary_path.parent / "behavior_detection"),
                "summary_path": str(summary_path),
            },
            "stages": {
                "person_detection": {
                    "name": "person_detection",
                    "enabled": True,
                    "success": True,
                    "input_path": self.config.source,
                    "output_path": str(summary_path.parent / "crops"),
                    "summary": {},
                    "error": None,
                },
                "ppe_detection": {
                    "name": "ppe_detection",
                    "enabled": True,
                    "success": True,
                    "input_path": self.config.source,
                    "output_path": str(summary_path.parent / "ppe_detection"),
                    "summary": {},
                    "error": None,
                },
            },
            "error": None,
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        if self.stage_callback is not None:
            self.stage_callback("person_detection", "completed", None)
            self.stage_callback("ppe_detection", "completed", None)

        return summary


class MainPipelineTests(unittest.TestCase):
    @patch("main_pipeline.pipeline.OfflineVideoRecorder", side_effect=FakeOfflineVideoRecorder)
    @patch("main_pipeline.pipeline.StreamPipeline", side_effect=FakeStreamPipeline)
    def test_run_uses_realtime_engine_and_writes_annotated_video_summary(
        self,
        _stream_pipeline_cls,
        _video_recorder_cls,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            config = MainPipelineConfig(
                source="./vid_test/vid_1.mp4",
                output_root=temp_dir,
                job_name="offline_job",
                run_person_detection=False,
                run_ppe_detection=True,
                run_region_extraction=False,
                run_facemask_detection=True,
                run_glove_detection=False,
                run_behavior_detection=True,
                save_annotated_video=True,
                annotated_video_fps=18.0,
            )

            stage_events = []
            frame_events = []

            summary = MainPipeline(
                config,
                stage_callback=lambda stage_name, stage_status, _stage_result: stage_events.append(
                    (stage_name, stage_status)
                ),
                person_detection_frame_callback=lambda frame_index, total_frames, _annotated_frame, persons: frame_events.append(
                    (frame_index, total_frames, len(persons))
                ),
            ).run()

            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["pipeline_mode"], "offline")
            self.assertEqual(frame_events, [(0, 1, 1)])
            self.assertIn(("person_detection", "running"), stage_events)
            self.assertIn(("ppe_detection", "completed"), stage_events)

            effective_flags = summary["effective_stage_flags"]
            self.assertTrue(effective_flags["run_person_detection"])
            self.assertTrue(effective_flags["run_region_extraction"])
            self.assertTrue(effective_flags["run_facemask_detection"])
            self.assertTrue(effective_flags["run_ppe_detection"])
            self.assertTrue(effective_flags["run_behavior_detection"])
            self.assertFalse(effective_flags["run_glove_detection"])

            self.assertTrue(FakeStreamPipeline.last_config.run_person_detection)
            self.assertTrue(FakeStreamPipeline.last_config.run_region_extraction)
            self.assertTrue(FakeStreamPipeline.last_config.run_facemask_detection)
            self.assertTrue(FakeStreamPipeline.last_config.run_ppe_detection)
            self.assertFalse(FakeStreamPipeline.last_config.run_glove_detection)

            self.assertIsNotNone(summary["annotated_video"])
            self.assertEqual(summary["annotated_video"]["frame_count"], 1)
            self.assertEqual(summary["annotated_video"]["fps"], 18.0)
            self.assertIn("video_output_root", summary["paths"])

            recorder = FakeOfflineVideoRecorder.last_instance
            self.assertIsNotNone(recorder)
            self.assertTrue(recorder.closed)
            self.assertEqual(recorder.frame_count, 1)
            self.assertEqual(recorder.last_shape, (12, 16, 3))

            summary_path = Path(summary["paths"]["summary_path"])
            self.assertTrue(summary_path.exists())
            loaded_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded_summary["pipeline_mode"], "offline")
            self.assertIn("annotated_video", loaded_summary)

    @patch("main_pipeline.pipeline.StreamPipeline", side_effect=FakeStreamPipeline)
    def test_run_supports_disabling_annotated_video_export(
        self,
        _stream_pipeline_cls,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            config = MainPipelineConfig(
                source="./vid_test/vid_2.mp4",
                output_root=temp_dir,
                job_name="no_video_job",
                run_person_detection=False,
                run_ppe_detection=True,
                run_region_extraction=False,
                run_facemask_detection=False,
                run_glove_detection=False,
                run_behavior_detection=False,
                save_annotated_video=False,
            )

            summary = MainPipeline(config).run()

            self.assertEqual(summary["status"], "completed")
            self.assertIsNone(summary["annotated_video"])

    def test_run_requires_at_least_one_enabled_stage(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = MainPipelineConfig(
                source="./vid_test/vid_3.mp4",
                output_root=temp_dir,
                job_name="empty_job",
                run_person_detection=False,
                run_ppe_detection=False,
                run_region_extraction=False,
                run_facemask_detection=False,
                run_glove_detection=False,
                run_behavior_detection=False,
            )

            with self.assertRaisesRegex(ValueError, "At least one offline stage must be enabled"):
                MainPipeline(config).run()
