from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np

try:
    from facemask_detection.config import FaceMaskConfig
    from facemask_detection.inference import (
        FaceMaskDetection,
        detections_to_json,
        load_facemask_inference_engine,
        run_facemask_inference,
    )
except ModuleNotFoundError:  # pragma: no cover - convenience for direct script execution
    from config import FaceMaskConfig
    from inference import (
        FaceMaskDetection,
        detections_to_json,
        load_facemask_inference_engine,
        run_facemask_inference,
    )


CLASS_TO_COLOR = {
    "Mask": (0, 200, 0),
    "NoMask": (0, 0, 255),
}


def resolve_input_head_face_dir(config: FaceMaskConfig) -> Tuple[Path, str]:
    if config.input_head_face_dir:
        input_dir = Path(config.input_head_face_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"Input head_face directory not found: {input_dir}")
        return input_dir, input_dir.parent.name

    results_root = Path(config.results_root)
    if not results_root.exists():
        raise FileNotFoundError(
            f"Results root not found: {results_root}. "
            "Run person_region_extraction/main.py first, or set input_head_face_dir explicitly."
        )

    candidate_jobs = [
        job_dir
        for job_dir in results_root.iterdir()
        if job_dir.is_dir() and list(job_dir.glob("track_*/*/head_face.jpg"))
    ]
    if not candidate_jobs:
        raise FileNotFoundError(
            f"No head_face crops found inside: {results_root}. "
            "Run person_region_extraction/main.py first, or set input_head_face_dir explicitly."
        )

    latest_job_dir = max(candidate_jobs, key=lambda path: path.stat().st_mtime)
    return latest_job_dir, latest_job_dir.name


def iter_head_face_images(input_root: Path) -> Iterable[Path]:
    for image_path in sorted(input_root.glob("track_*/*/head_face.jpg")):
        if image_path.is_file():
            yield image_path


def annotate_image(image: np.ndarray, detections: List[FaceMaskDetection]) -> np.ndarray:
    annotated = image.copy()
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox_xyxy
        color = CLASS_TO_COLOR.get(detection.class_name, (255, 255, 0))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            f"{detection.class_name}: {detection.confidence:.2f}",
            (x1, max(14, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return annotated


def process_head_face_tree(config: FaceMaskConfig) -> Dict[str, object]:
    input_root, job_name = resolve_input_head_face_dir(config)
    output_root = Path(config.output_root) / job_name
    output_root.mkdir(parents=True, exist_ok=True)

    engine = load_facemask_inference_engine(config)
    processed_count = 0
    skipped_count = 0
    mask_count = 0
    no_mask_count = 0
    no_detection_count = 0

    for image_path in iter_head_face_images(input_root):
        image = cv2.imread(str(image_path))
        if image is None or image.size == 0:
            skipped_count += 1
            continue

        detections = run_facemask_inference(engine, image)
        relative_parent = image_path.relative_to(input_root).parent
        frame_output_dir = output_root / relative_parent
        frame_output_dir.mkdir(parents=True, exist_ok=True)

        annotated_image = annotate_image(image, detections)
        cv2.imwrite(str(frame_output_dir / "annotated.jpg"), annotated_image)
        (frame_output_dir / "detections.json").write_text(
            detections_to_json(detections),
            encoding="utf-8",
        )

        if detections:
            best_detection = detections[0]
            x1, y1, x2, y2 = best_detection.bbox_xyxy
            best_crop = image[y1:y2, x1:x2]
            if best_crop.size > 0:
                cv2.imwrite(str(frame_output_dir / "best_detection.jpg"), best_crop)
            if best_detection.class_name == "Mask":
                mask_count += 1
            elif best_detection.class_name == "NoMask":
                no_mask_count += 1
        else:
            no_detection_count += 1

        processed_count += 1

    summary = {
        "job_name": job_name,
        "input_root": str(input_root),
        "output_root": str(output_root),
        "processed_count": processed_count,
        "skipped_count": skipped_count,
        "mask_count": mask_count,
        "no_mask_count": no_mask_count,
        "no_detection_count": no_detection_count,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary
