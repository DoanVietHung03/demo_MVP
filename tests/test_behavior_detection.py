from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import supervision as sv

from behavior_detection.config import BehaviorDetectionConfig
from behavior_detection.inference import filter_detections_by_class_names, infer_detections
from behavior_detection.pipeline import (
    BehaviorDetectionPipeline,
    DetectionMatch,
    TrackedPersonBehavior,
    _should_keep_smoke_match,
    annotate_behavior_evidence,
)


class BehaviorDetectionInferenceTests(unittest.TestCase):
    def test_filter_detections_by_class_names_respects_custom_threshold(self) -> None:
        engine = SimpleNamespace(class_name_to_id={"person": 0, "cell phone": 67})
        detections = sv.Detections(
            xyxy=np.asarray(
                [
                    [0, 0, 10, 10],
                    [10, 10, 20, 20],
                    [20, 20, 30, 30],
                    [30, 30, 40, 40],
                ],
                dtype=np.float32,
            ),
            confidence=np.asarray([0.30, 0.40, 0.24, 0.26], dtype=np.float32),
            class_id=np.asarray([0, 0, 67, 67], dtype=np.int32),
        )

        person_detections = filter_detections_by_class_names(
            engine,
            detections,
            ["person"],
            confidence_threshold=0.35,
        )
        phone_detections = filter_detections_by_class_names(
            engine,
            detections,
            ["cell phone"],
            confidence_threshold=0.25,
        )

        self.assertEqual(len(person_detections), 1)
        self.assertEqual(len(phone_detections), 1)
        self.assertAlmostEqual(float(person_detections.confidence[0]), 0.40, places=6)
        self.assertAlmostEqual(float(phone_detections.confidence[0]), 0.26, places=6)

    @patch("behavior_detection.pipeline.load_yolo_pose_inference")
    @patch("behavior_detection.pipeline.load_yolo_detection_inference")
    def test_person_and_phone_are_split_from_one_shared_inference(
        self,
        load_detection_inference_mock,
        load_pose_inference_mock,
    ) -> None:
        shared_engine = SimpleNamespace(class_name_to_id={"person": 0, "cell phone": 67})
        smoke_engine = SimpleNamespace(class_name_to_id={"cigarette": 0, "vape": 1})
        pose_engine = SimpleNamespace()
        load_detection_inference_mock.side_effect = [shared_engine, smoke_engine]
        load_pose_inference_mock.return_value = pose_engine

        pipeline = BehaviorDetectionPipeline(
            BehaviorDetectionConfig(
                device="cpu",
                person_confidence_threshold=0.35,
                phone_confidence_threshold=0.25,
            )
        )

        frame = np.zeros((12, 12, 3), dtype=np.uint8)
        shared_detections = object()
        person_detections = object()
        phone_detections = object()

        with patch(
            "behavior_detection.pipeline.infer_detections",
            return_value=shared_detections,
        ) as infer_detections_mock, patch(
            "behavior_detection.pipeline.filter_detections_by_class_names",
            side_effect=[person_detections, phone_detections],
        ) as filter_mock:
            resolved_person_detections, resolved_phone_detections = (
                pipeline._infer_person_and_phone_detections(frame)
            )

        self.assertIs(resolved_person_detections, person_detections)
        self.assertIs(resolved_phone_detections, phone_detections)
        infer_detections_mock.assert_called_once_with(shared_engine, frame)
        self.assertEqual(filter_mock.call_count, 2)
        self.assertEqual(
            filter_mock.call_args_list[0].kwargs["confidence_threshold"],
            0.35,
        )
        self.assertEqual(
            filter_mock.call_args_list[1].kwargs["confidence_threshold"],
            0.25,
        )

    def test_annotate_behavior_evidence_draws_phone_and_smoke_boxes(self) -> None:
        frame = np.zeros((48, 48, 3), dtype=np.uint8)
        behaviors = [
            TrackedPersonBehavior(
                tracker_id=1,
                bbox_xyxy=(4, 4, 32, 32),
                label="id:1 0.90",
                smoking_candidate=True,
                phone_present=True,
                phone_in_hand=True,
                phone_usage=True,
                events=[],
                phone_match=DetectionMatch(
                    bbox_xyxy=(10, 10, 18, 20),
                    confidence=0.82,
                    class_name="cell phone",
                ),
                smoke_match=DetectionMatch(
                    bbox_xyxy=(20, 12, 26, 18),
                    confidence=0.76,
                    class_name="cigarette",
                ),
            )
        ]

        annotated = annotate_behavior_evidence(frame, behaviors)

        self.assertEqual(int(frame.max()), 0)
        self.assertGreater(int(annotated.max()), 0)
        self.assertFalse(np.array_equal(annotated[10, 10], frame[10, 10]))
        self.assertFalse(np.array_equal(annotated[12, 20], frame[12, 20]))

    def test_should_keep_smoke_match_rejects_overlap_with_phone(self) -> None:
        config = BehaviorDetectionConfig(
            device="cpu",
            smoke_phone_overlap_iou_threshold=0.1,
            smoke_max_box_area_ratio=0.05,
        )
        smoke_match = DetectionMatch(
            bbox_xyxy=(40, 30, 54, 42),
            confidence=0.71,
            class_name="cigarette",
        )
        phone_match = DetectionMatch(
            bbox_xyxy=(38, 28, 56, 44),
            confidence=0.83,
            class_name="cell phone",
        )

        keep_smoke_match, smoke_near_mouth, smoke_near_hand, hand_to_mouth = (
            _should_keep_smoke_match(
                smoke_match,
                phone_match,
                person_box=(0, 0, 100, 200),
                mouth_point=(46.0, -10.0),
                wrist_points=[(46.0, 70.0)],
                config=config,
            )
        )

        self.assertFalse(keep_smoke_match)
        self.assertTrue(smoke_near_hand)
        self.assertFalse(hand_to_mouth)
        self.assertFalse(smoke_near_mouth)

    @patch("behavior_detection.inference.run_yolo_onnx_detections")
    def test_infer_detections_attaches_class_name_metadata_for_onnx(
        self,
        run_yolo_onnx_detections_mock,
    ) -> None:
        run_yolo_onnx_detections_mock.return_value = (
            np.asarray([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32),
            np.asarray([0.9, 0.8], dtype=np.float32),
            np.asarray([0, 1], dtype=np.int32),
        )
        engine = SimpleNamespace(
            backend="onnx",
            onnx_session=object(),
            input_resize_scale=1.0,
            confidence_threshold=0.25,
            iou_threshold=0.45,
            class_name_to_id={"cigarette": 0, "vape": 1},
        )

        detections = infer_detections(engine, np.zeros((16, 16, 3), dtype=np.uint8))

        self.assertEqual(detections.data["class_name"].tolist(), ["cigarette", "vape"])
