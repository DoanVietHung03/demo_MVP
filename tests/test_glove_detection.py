from __future__ import annotations

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from glove_detection.config import GloveDetectionConfig
from glove_detection.inference import load_labels, normalize_glove_label
from glove_detection.pipeline import resolve_final_label, resolve_input_hands_dir


class GloveDetectionInferenceTests(unittest.TestCase):
    def test_load_labels_supports_indexed_format(self) -> None:
        with TemporaryDirectory() as temp_dir:
            labels_path = Path(temp_dir) / "labels.txt"
            labels_path.write_text("0 glove\n1 hand\n", encoding="utf-8")

            labels = load_labels(str(labels_path))

            self.assertEqual(labels, ["glove", "hand"])

    def test_normalize_glove_label_maps_hand_to_no_glove(self) -> None:
        self.assertEqual(normalize_glove_label("glove"), "Glove")
        self.assertEqual(normalize_glove_label("hand"), "NoGlove")

    def test_resolve_final_label_marks_low_confidence_as_uncertain(self) -> None:
        from glove_detection.inference import GloveClassification

        classification = GloveClassification(
            class_index=0,
            raw_label="glove",
            normalized_label="Glove",
            confidence=0.42,
        )

        self.assertEqual(resolve_final_label(classification, 0.5), "Uncertain")
        self.assertEqual(resolve_final_label(classification, 0.4), "Glove")


class ResolveInputHandsDirTests(unittest.TestCase):
    def test_explicit_input_dir_is_used(self) -> None:
        with TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "job_a"
            (input_dir / "track_1" / "frame_000001").mkdir(parents=True)

            config = GloveDetectionConfig(
                input_hands_dir=str(input_dir),
                process_latest_job=False,
            )

            resolved_dir, job_name = resolve_input_hands_dir(config)

            self.assertEqual(resolved_dir, input_dir)
            self.assertEqual(job_name, "job_a")

    def test_latest_hands_job_is_used_when_enabled(self) -> None:
        with TemporaryDirectory() as temp_dir:
            results_root = Path(temp_dir)
            older = results_root / "job_old" / "track_1" / "frame_000001"
            newer = results_root / "job_new" / "track_1" / "frame_000001"
            older.mkdir(parents=True)
            newer.mkdir(parents=True)
            (older / "hands.jpg").write_bytes(b"old")
            (newer / "hands.jpg").write_bytes(b"new")
            os.utime(results_root / "job_old", (1_000, 1_000))
            os.utime(results_root / "job_new", (2_000, 2_000))

            config = GloveDetectionConfig(
                results_root=str(results_root),
                process_latest_job=True,
            )

            resolved_dir, job_name = resolve_input_hands_dir(config)

            self.assertEqual(resolved_dir, results_root / "job_new")
            self.assertEqual(job_name, "job_new")

    def test_explicit_input_is_required_when_latest_job_is_disabled(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = GloveDetectionConfig(
                results_root=str(Path(temp_dir)),
                process_latest_job=False,
            )

            with self.assertRaisesRegex(
                ValueError,
                "input_hands_dir must be set when process_latest_job is False",
            ):
                resolve_input_hands_dir(config)
