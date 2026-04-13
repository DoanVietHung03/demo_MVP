from __future__ import annotations

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from facemask_detection.config import FaceMaskConfig
from facemask_detection.pipeline import resolve_input_head_face_dir
from person_region_extraction.config import RegionExtractionConfig
from person_region_extraction.pipeline import resolve_input_crops_dir


class ResolveInputCropsDirTests(unittest.TestCase):
    def test_explicit_input_dir_is_used(self) -> None:
        with TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "job_a" / "crops"
            input_dir.mkdir(parents=True)

            config = RegionExtractionConfig(
                input_crops_dir=str(input_dir),
                process_latest_job=False,
            )

            resolved_dir, job_name = resolve_input_crops_dir(config)

            self.assertEqual(resolved_dir, input_dir)
            self.assertEqual(job_name, "job_a")

    def test_latest_crop_job_is_used_when_enabled(self) -> None:
        with TemporaryDirectory() as temp_dir:
            results_root = Path(temp_dir)
            older = results_root / "job_old" / "crops"
            newer = results_root / "job_new" / "crops"
            older.mkdir(parents=True)
            newer.mkdir(parents=True)
            (older / "track_1").mkdir()
            (newer / "track_1").mkdir()
            (older / "track_1" / "frame_000001.jpg").write_bytes(b"old")
            (newer / "track_1" / "frame_000001.jpg").write_bytes(b"new")
            os.utime(results_root / "job_old", (1_000, 1_000))
            os.utime(results_root / "job_new", (2_000, 2_000))

            config = RegionExtractionConfig(
                results_root=str(results_root),
                process_latest_job=True,
            )

            resolved_dir, job_name = resolve_input_crops_dir(config)

            self.assertEqual(resolved_dir, newer)
            self.assertEqual(job_name, "job_new")

    def test_explicit_input_is_required_when_latest_job_is_disabled(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = RegionExtractionConfig(
                results_root=str(Path(temp_dir)),
                process_latest_job=False,
            )

            with self.assertRaisesRegex(
                ValueError,
                "input_crops_dir must be set when process_latest_job is False",
            ):
                resolve_input_crops_dir(config)


class ResolveInputHeadFaceDirTests(unittest.TestCase):
    def test_explicit_input_dir_is_used(self) -> None:
        with TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "job_a"
            (input_dir / "track_1" / "frame_000001").mkdir(parents=True)

            config = FaceMaskConfig(
                input_head_face_dir=str(input_dir),
                process_latest_job=False,
            )

            resolved_dir, job_name = resolve_input_head_face_dir(config)

            self.assertEqual(resolved_dir, input_dir)
            self.assertEqual(job_name, "job_a")

    def test_latest_region_job_is_used_when_enabled(self) -> None:
        with TemporaryDirectory() as temp_dir:
            results_root = Path(temp_dir)
            older = results_root / "job_old" / "track_1" / "frame_000001"
            newer = results_root / "job_new" / "track_1" / "frame_000001"
            older.mkdir(parents=True)
            newer.mkdir(parents=True)
            (older / "head_face.jpg").write_bytes(b"old")
            (newer / "head_face.jpg").write_bytes(b"new")
            os.utime(results_root / "job_old", (1_000, 1_000))
            os.utime(results_root / "job_new", (2_000, 2_000))

            config = FaceMaskConfig(
                results_root=str(results_root),
                process_latest_job=True,
            )

            resolved_dir, job_name = resolve_input_head_face_dir(config)

            self.assertEqual(resolved_dir, results_root / "job_new")
            self.assertEqual(job_name, "job_new")

    def test_explicit_input_is_required_when_latest_job_is_disabled(self) -> None:
        with TemporaryDirectory() as temp_dir:
            config = FaceMaskConfig(
                results_root=str(Path(temp_dir)),
                process_latest_job=False,
            )

            with self.assertRaisesRegex(
                ValueError,
                "input_head_face_dir must be set when process_latest_job is False",
            ):
                resolve_input_head_face_dir(config)
