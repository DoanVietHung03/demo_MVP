from __future__ import annotations

import unittest

import numpy as np

from person_region_extraction.config import RegionExtractionConfig
from person_region_extraction.pipeline import (
    PersonRegionExtractor,
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
