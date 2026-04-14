from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import cv2
import numpy as np

from behavior_detection.pipeline import DetectionMatch, TrackedPersonBehavior
from facemask_detection.inference import FaceMaskDetection
from glove_detection.inference import GloveClassification
from person_detection.config import PersonInstance
from person_region_extraction.pipeline import RegionResult
from stream_pipeline.config import StreamPipelineConfig
from stream_pipeline.pipeline import StreamPipeline


class FakeCapture:
    def __init__(self, frames):
        self.frames = [frame.copy() for frame in frames]
        self.index = 0
        self.released = False

    def isOpened(self) -> bool:
        return True

    def read(self):
        if self.index >= len(self.frames):
            return False, None
        frame = self.frames[self.index]
        self.index += 1
        return True, frame.copy()

    def get(self, prop_id):
        if prop_id == cv2.CAP_PROP_FRAME_COUNT:
            return len(self.frames)
        return 0

    def release(self) -> None:
        self.released = True


class FakePersonPipeline:
    def __init__(self, **_kwargs) -> None:
        self.device = "cpu"

    def runtime_backend_label(self) -> str:
        return "onnx-cpu"

    def process_frame(self, frame: np.ndarray, frame_index: int):
        person = PersonInstance(
            tracker_id=11,
            confidence=0.93,
            bbox_xyxy=(4, 4, 28, 28),
            crop=np.full((32, 32, 3), 120 + frame_index, dtype=np.uint8),
            should_alert=frame_index == 0,
        )
        return frame.copy(), [person]


class FakeRegionExtractor:
    def __init__(self, _config) -> None:
        self.device = "cpu"

    def runtime_backend_label(self) -> str:
        return "onnx-cpu"

    def split_regions(self, image: np.ndarray) -> RegionResult:
        height, width = image.shape[:2]
        head_face = np.full((24, 24, 3), 180, dtype=np.uint8)
        hands = np.full((24, 40, 3), 90, dtype=np.uint8)
        torso = np.full((height, width, 3), 60, dtype=np.uint8)
        return RegionResult(
            head_face=head_face,
            hands=hands,
            torso=torso,
            face_bbox=(0, 0, 24, 24),
            hand_bboxes=[(2, 12, 14, 24)],
        )


class FakeBehaviorPipeline:
    def __init__(self, _config) -> None:
        pass

    def runtime_backend_summary(self):
        return {
            "person_phone": "onnx-cpu",
            "pose": "onnx-cpu",
            "smoke": "onnx-cpu",
        }

    def process_frame(self, frame: np.ndarray, frame_index: int):
        annotated = frame.copy()
        annotated[:, :, 1] = 255
        behaviors = [
            TrackedPersonBehavior(
                tracker_id=11,
                bbox_xyxy=(4, 4, 28, 28),
                label="id:11 0.93 | PHONE_USE[1] SMOKE[1]",
                smoking_candidate=True,
                phone_present=True,
                phone_in_hand=True,
                phone_usage=True,
                events=[],
                phone_match=DetectionMatch(
                    bbox_xyxy=(10, 10, 14, 18),
                    confidence=0.81,
                    class_name="cell phone",
                ),
                smoke_match=DetectionMatch(
                    bbox_xyxy=(16, 10, 20, 16),
                    confidence=0.72,
                    class_name="cigarette",
                ),
            )
        ]
        events = []
        if frame_index == 0:
            events = [
                type(
                    "Event",
                    (),
                    {
                        "frame_index": 0,
                        "tracker_id": 7,
                        "event_type": "phone_present",
                        "confidence": 0.7,
                        "details": {"phone_present_streak": 1.0},
                    },
                )()
            ]
        return annotated, behaviors, events


class FakePPEDetectionPipeline:
    def __init__(self, _config) -> None:
        self.inference_engine = type("Engine", (), {"device": "cpu"})()

    def runtime_backend_label(self) -> str:
        return "torch-cpu"

    def predict_records(self, _frame: np.ndarray):
        return [
            type(
                "PPEDetectionRecord",
                (),
                {
                    "class_name": "hardhat",
                    "confidence": 0.88,
                    "bbox_xyxy": (6, 6, 20, 20),
                },
            )()
        ]


class StreamPipelineTests(unittest.TestCase):
    @patch("stream_pipeline.pipeline.log_queue")
    @patch("stream_pipeline.pipeline.enqueue_log_item", return_value=True)
    @patch("stream_pipeline.pipeline.BehaviorDetectionPipeline", side_effect=FakeBehaviorPipeline)
    @patch("stream_pipeline.pipeline.classify_glove_image")
    @patch("stream_pipeline.pipeline.load_tflite_glove_classifier")
    @patch("stream_pipeline.pipeline.run_facemask_inference")
    @patch("stream_pipeline.pipeline.load_facemask_inference_engine")
    @patch("stream_pipeline.pipeline.PPEDetectionPipeline", side_effect=FakePPEDetectionPipeline)
    @patch("stream_pipeline.pipeline.PersonRegionExtractor", side_effect=FakeRegionExtractor)
    @patch("stream_pipeline.pipeline.PersonDetectionPipeline", side_effect=FakePersonPipeline)
    @patch("stream_pipeline.pipeline.cv2.VideoCapture")
    def test_run_stream_pipeline_processes_enabled_stages_realtime(
        self,
        video_capture_mock,
        _person_pipeline_cls,
        _region_extractor_cls,
        _ppe_pipeline_cls,
        _load_facemask_engine_mock,
        run_facemask_inference_mock,
        load_glove_classifier_mock,
        classify_glove_image_mock,
        _behavior_pipeline_cls,
        _enqueue_log_item_mock,
        log_queue_mock,
    ) -> None:
        frames = [
            np.zeros((48, 48, 3), dtype=np.uint8),
            np.full((48, 48, 3), 40, dtype=np.uint8),
        ]
        video_capture_mock.return_value = FakeCapture(frames)
        _load_facemask_engine_mock.return_value = type(
            "FaceEngine",
            (),
            {"device": "cpu", "backend": "onnx", "onnx_session": None},
        )()
        run_facemask_inference_mock.return_value = [
            FaceMaskDetection(
                class_id=0,
                class_name="Mask",
                confidence=0.91,
                bbox_xyxy=(0, 0, 12, 12),
            )
        ]
        load_glove_classifier_mock.return_value = object()
        classify_glove_image_mock.return_value = GloveClassification(
            class_index=0,
            raw_label="glove",
            normalized_label="Glove",
            confidence=0.87,
        )

        stage_events = []
        frame_events = []

        with TemporaryDirectory() as temp_dir:
            config = StreamPipelineConfig(
                source="./vid_test/vid_1.mp4",
                output_root=temp_dir,
                job_name="stream_job",
                run_ppe_detection=True,
                run_behavior_detection=True,
                attribute_refresh_interval_frames=1,
                summary_flush_interval_frames=1,
            )
            config.person_detection.device = "cpu"
            config.person_detection.model_path = "./weights/onnx/yolo11n.onnx"
            config.person_detection.max_frames = 2
            config.person_detection.crop_save_interval = 1
            config.ppe_detection.device = "cpu"
            config.ppe_detection.max_frames = 2
            config.person_region_extraction.device = "cpu"
            config.facemask_detection.device = "cpu"

            summary = StreamPipeline(
                config,
                stage_callback=lambda stage_name, stage_status, _stage_result: stage_events.append(
                    (stage_name, stage_status)
                ),
                frame_callback=lambda frame_index, total_frames, annotated_frame, persons: frame_events.append(
                    (
                        frame_index,
                        total_frames,
                        len(persons),
                        int(np.max(annotated_frame)),
                        annotated_frame[10, 10].tolist(),
                    )
                ),
            ).run()

            self.assertEqual(summary["status"], "completed")
            self.assertEqual(frame_events[0][0:4], (0, 2, 1, 255))
            self.assertEqual(frame_events[1][0:4], (1, 2, 1, 255))
            self.assertNotEqual(frame_events[0][4], [0, 0, 0])
            self.assertIn(("person_detection", "running"), stage_events)
            self.assertIn(("ppe_detection", "completed"), stage_events)
            self.assertIn(("glove_detection", "completed"), stage_events)
            self.assertIn(("behavior_detection", "completed"), stage_events)
            self.assertEqual(
                summary["stages"]["facemask_detection"]["summary"]["mask_count"],
                2,
            )
            self.assertEqual(
                summary["stages"]["ppe_detection"]["summary"]["class_counts"]["hardhat"],
                2,
            )
            self.assertEqual(
                summary["stages"]["glove_detection"]["summary"]["glove_count"],
                2,
            )
            self.assertEqual(
                summary["stages"]["person_detection"]["summary"]["crop_file_count"],
                2,
            )

            summary_path = Path(summary["paths"]["summary_path"])
            self.assertTrue(summary_path.exists())
            loaded_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded_summary["job_name"], "stream_job")

        log_queue_mock.join.assert_called_once()

    @patch("stream_pipeline.pipeline.log_queue")
    @patch("stream_pipeline.pipeline.BehaviorDetectionPipeline", side_effect=FakeBehaviorPipeline)
    @patch("stream_pipeline.pipeline.cv2.VideoCapture")
    def test_run_stream_pipeline_supports_behavior_only_mode(
        self,
        video_capture_mock,
        _behavior_pipeline_cls,
        log_queue_mock,
    ) -> None:
        frames = [np.zeros((32, 32, 3), dtype=np.uint8)]
        video_capture_mock.return_value = FakeCapture(frames)
        frame_events = []

        with TemporaryDirectory() as temp_dir:
            config = StreamPipelineConfig(
                source="./vid_test/vid_2.mp4",
                output_root=temp_dir,
                job_name="behavior_only",
                run_person_detection=False,
                run_region_extraction=False,
                run_facemask_detection=False,
                run_glove_detection=False,
                run_behavior_detection=True,
            )
            config.behavior_detection.device = "cpu"
            config.behavior_detection.max_frames = 1

            summary = StreamPipeline(
                config,
                frame_callback=lambda frame_index, total_frames, annotated_frame, persons: frame_events.append(
                    (frame_index, total_frames, int(np.max(annotated_frame[:, :, 1])), len(persons))
                ),
            ).run()

            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["stages"]["person_detection"]["enabled"], False)
            self.assertEqual(
                summary["stages"]["behavior_detection"]["summary"]["emitted_events"],
                1,
            )
            self.assertEqual(frame_events, [(0, 1, 255, 0)])

        log_queue_mock.join.assert_called_once()
