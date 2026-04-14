from __future__ import annotations

import unittest

import numpy as np
import supervision as sv

from ppe_detection.inference import filter_detections_by_class_names
from ppe_detection.pipeline import PPEDetectionRecord, annotate_image, detections_to_records


class _FakeInferenceEngine:
    def __init__(self) -> None:
        self.class_name_to_id = {"hardhat": 0, "vest": 1, "person": 2}
        self.class_names = {0: "hardhat", 1: "vest", 2: "person"}


class PPEDetectionTests(unittest.TestCase):
    def test_filter_detections_by_class_names_keeps_hardhat_and_vest(self) -> None:
        inference_engine = _FakeInferenceEngine()
        detections = sv.Detections(
            xyxy=np.asarray(
                [
                    [0, 0, 10, 10],
                    [10, 10, 20, 20],
                    [20, 20, 30, 30],
                ],
                dtype=np.float32,
            ),
            confidence=np.asarray([0.9, 0.8, 0.7], dtype=np.float32),
            class_id=np.asarray([0, 1, 2], dtype=np.int32),
        )

        filtered = filter_detections_by_class_names(
            inference_engine,
            detections,
            ["hardhat", "vest"],
        )
        records = detections_to_records(inference_engine, filtered)

        self.assertEqual([record.class_name for record in records], ["hardhat", "vest"])

    def test_annotate_image_draws_ppe_boxes(self) -> None:
        image = np.zeros((40, 40, 3), dtype=np.uint8)
        records = [
            PPEDetectionRecord(
                class_name="hardhat",
                confidence=0.92,
                bbox_xyxy=(5, 5, 20, 20),
            )
        ]

        annotated = annotate_image(image, records)

        self.assertGreater(int(np.max(annotated)), 0)
