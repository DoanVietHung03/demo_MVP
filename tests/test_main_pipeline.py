from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import numpy as np

from main_pipeline.config import MainPipelineConfig
from main_pipeline.pipeline import MainPipeline, build_pipeline_paths
from person_detection.config import PersonInstance


class MainPipelineTests(unittest.TestCase):
    @patch("main_pipeline.pipeline.log_queue")
    @patch("main_pipeline.pipeline.BehaviorDetectionPipeline")
    @patch("main_pipeline.pipeline.process_hands_tree")
    @patch("main_pipeline.pipeline.process_head_face_tree")
    @patch("main_pipeline.pipeline.process_crop_tree")
    @patch("main_pipeline.pipeline.PersonDetectionPipeline")
    def test_run_links_stage_outputs_in_sequence(
        self,
        person_pipeline_cls,
        process_crop_tree_mock,
        process_head_face_tree_mock,
        process_hands_tree_mock,
        behavior_pipeline_cls,
        log_queue_mock,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            config = MainPipelineConfig(
                source="./vid_test/vid_1.mp4",
                output_root=temp_dir,
                job_name="demo_job",
                run_behavior_detection=True,
            )
            paths = build_pipeline_paths(temp_dir, "demo_job")
            expected_region_root = paths.region_output_root / "demo_job"

            person_pipeline_instance = Mock()
            person_pipeline_cls.return_value = person_pipeline_instance
            process_crop_tree_mock.return_value = {
                "job_name": "demo_job",
                "input_crops_dir": str(paths.crops_dir),
                "output_root": str(expected_region_root),
                "processed_count": 2,
                "skipped_count": 0,
            }
            process_head_face_tree_mock.return_value = {
                "job_name": "demo_job",
                "input_root": str(expected_region_root),
                "output_root": str(paths.facemask_output_root / "demo_job"),
                "processed_count": 2,
                "skipped_count": 0,
                "mask_count": 1,
                "no_mask_count": 1,
                "no_detection_count": 0,
            }
            process_hands_tree_mock.return_value = {
                "job_name": "demo_job",
                "input_root": str(expected_region_root),
                "output_root": str(paths.glove_output_root / "demo_job"),
                "processed_count": 2,
                "skipped_count": 0,
                "glove_count": 1,
                "no_glove_count": 1,
                "uncertain_count": 0,
            }
            behavior_pipeline_instance = Mock()
            behavior_pipeline_instance.process_video.return_value = {
                "source": config.source,
                "output_root": str(paths.behavior_output_root),
                "events_path": str(paths.behavior_output_root / "events.jsonl"),
                "processed_frames": 3,
                "emitted_events": 0,
            }
            behavior_pipeline_cls.return_value = behavior_pipeline_instance

            summary = MainPipeline(config).run()

            person_pipeline_instance.process_video.assert_called_once()
            process_crop_tree_mock.assert_called_once()
            region_config = process_crop_tree_mock.call_args.args[0]
            self.assertEqual(region_config.input_crops_dir, str(paths.crops_dir))
            self.assertEqual(region_config.output_root, str(paths.region_output_root))

            facemask_config = process_head_face_tree_mock.call_args.args[0]
            self.assertEqual(facemask_config.input_head_face_dir, str(expected_region_root))
            self.assertEqual(facemask_config.output_root, str(paths.facemask_output_root))

            glove_config = process_hands_tree_mock.call_args.args[0]
            self.assertEqual(glove_config.input_hands_dir, str(expected_region_root))
            self.assertEqual(glove_config.output_root, str(paths.glove_output_root))

            behavior_config = behavior_pipeline_cls.call_args.args[0]
            self.assertEqual(behavior_config.source, config.source)
            self.assertEqual(behavior_config.output_root, str(paths.behavior_output_root))

            log_queue_mock.join.assert_called_once()
            self.assertEqual(summary["status"], "completed")
            summary_path = Path(summary["paths"]["summary_path"])
            self.assertTrue(summary_path.exists())
            loaded_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded_summary["job_name"], "demo_job")

    @patch("main_pipeline.pipeline.log_queue")
    @patch("main_pipeline.pipeline.process_crop_tree")
    @patch("main_pipeline.pipeline.PersonDetectionPipeline")
    def test_run_exposes_stage_and_frame_callbacks(
        self,
        person_pipeline_cls,
        process_crop_tree_mock,
        log_queue_mock,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            config = MainPipelineConfig(
                source="./vid_test/vid_1.mp4",
                output_root=temp_dir,
                job_name="callback_job",
                run_facemask_detection=False,
                run_glove_detection=False,
                run_behavior_detection=False,
            )
            paths = build_pipeline_paths(temp_dir, "callback_job")
            process_crop_tree_mock.return_value = {
                "job_name": "callback_job",
                "input_crops_dir": str(paths.crops_dir),
                "output_root": str(paths.region_output_root / "callback_job"),
                "processed_count": 1,
                "skipped_count": 0,
            }

            person_pipeline_instance = Mock()

            def process_video_side_effect(*, frame_callback=None, **_kwargs):
                self.assertIsNotNone(frame_callback)
                frame_callback(
                    0,
                    1,
                    np.zeros((8, 8, 3), dtype=np.uint8),
                    [
                        PersonInstance(
                            tracker_id=7,
                            confidence=0.9,
                            bbox_xyxy=(0, 0, 8, 8),
                            crop=np.zeros((4, 4, 3), dtype=np.uint8),
                            should_alert=True,
                        )
                    ],
                )

            person_pipeline_instance.process_video.side_effect = process_video_side_effect
            person_pipeline_cls.return_value = person_pipeline_instance

            stage_events = []
            frame_events = []

            pipeline = MainPipeline(
                config,
                stage_callback=lambda stage_name, stage_status, _stage_result: stage_events.append(
                    (stage_name, stage_status)
                ),
                person_detection_frame_callback=lambda frame_index, total_frames, _annotated_frame, persons: frame_events.append(
                    (frame_index, total_frames, len(persons))
                ),
            )

            summary = pipeline.run()

            self.assertEqual(summary["status"], "completed")
            self.assertIn(("person_detection", "running"), stage_events)
            self.assertIn(("person_detection", "completed"), stage_events)
            self.assertIn(("person_region_extraction", "completed"), stage_events)
            self.assertIn(("facemask_detection", "disabled"), stage_events)
            self.assertEqual(frame_events, [(0, 1, 1)])
            log_queue_mock.join.assert_called_once()
