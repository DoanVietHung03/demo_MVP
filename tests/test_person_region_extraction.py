from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from person_region_extraction.config import RegionExtractionConfig
from person_region_extraction.pipeline import (
    PersonRegionExtractor,
    RegionResult,
    process_crop_tree,
    _select_primary_person_index,
)


class PersonRegionExtractionTests(unittest.TestCase):
    def test_select_primary_person_index_prefers_centered_larger_person(self) -> None:
        image_shape = (60, 40, 3)
        person_indices = np.asarray([0, 1], dtype=np.int32)
        boxes_xyxy = np.asarray(
            [
                [1, 1, 10, 12],
                [7, 4, 34, 57],
            ],
            dtype=np.float32,
        )
        confidences = np.asarray([0.98, 0.60], dtype=np.float32)

        chosen_index = _select_primary_person_index(
            person_indices,
            boxes_xyxy,
            confidences,
            image_shape,
        )

        self.assertEqual(chosen_index, 1)

    def test_split_regions_applies_min_face_size_to_bbox_and_crop(self) -> None:
        extractor = PersonRegionExtractor.__new__(PersonRegionExtractor)
        extractor.config = RegionExtractionConfig(min_face_size=24)
        extractor._predict_pose = lambda image: None
        extractor._estimate_head_face_bbox = lambda image, pose: (10, 10, 17, 20)
        extractor._estimate_torso_bbox = lambda image, pose, head_face_bbox: (5, 20, 25, 50)
        extractor._estimate_hands_crop = lambda image, pose: np.zeros((8, 8, 3), dtype=np.uint8)

        image = np.zeros((60, 40, 3), dtype=np.uint8)
        regions = extractor.split_regions(image)

        self.assertEqual(regions.face_bbox, (10, 10, 17, 20))
        self.assertGreaterEqual(regions.head_face.shape[0], 24)
        self.assertGreaterEqual(regions.head_face.shape[1], 24)

    @patch("person_region_extraction.pipeline.PersonRegionExtractor")
    def test_process_crop_tree_persists_regions_metadata(
        self,
        extractor_cls,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            crops_dir = Path(temp_dir) / "job_demo" / "crops"
            image_path = crops_dir / "track_1" / "frame_000001.jpg"
            image_path.parent.mkdir(parents=True)

            image = np.full((32, 24, 3), 100, dtype=np.uint8)
            import cv2

            cv2.imwrite(str(image_path), image)

            extractor_instance = extractor_cls.return_value
            extractor_instance.runtime_backend_label.return_value = "onnx-cpu"
            extractor_instance.device = "cpu"
            extractor_instance.split_regions.return_value = RegionResult(
                head_face=np.zeros((24, 24, 3), dtype=np.uint8),
                hands=np.zeros((24, 40, 3), dtype=np.uint8),
                torso=np.zeros((30, 20, 3), dtype=np.uint8),
                face_bbox=(1, 2, 20, 24),
                hand_bboxes=[(0, 10, 8, 20), (16, 10, 24, 20)],
                hand_crop_boxes=[(0, 0, 16, 24), (24, 0, 40, 24)],
            )

            config = RegionExtractionConfig(
                input_crops_dir=str(crops_dir),
                output_root=str(Path(temp_dir) / "region_out"),
                process_latest_job=False,
                device="cpu",
            )

            summary = process_crop_tree(config)
            metadata_path = (
                Path(summary["output_root"])
                / "track_1"
                / "frame_000001"
                / "regions.json"
            )
            self.assertTrue(metadata_path.exists())
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["face_bbox"], [1, 2, 20, 24])
            self.assertEqual(metadata["hand_bboxes"], [[0, 10, 8, 20], [16, 10, 24, 20]])
            self.assertEqual(metadata["hand_crop_boxes"], [[0, 0, 16, 24], [24, 0, 40, 24]])
