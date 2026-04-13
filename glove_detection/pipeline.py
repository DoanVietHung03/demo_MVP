from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import cv2
import numpy as np

try:
    from glove_detection.config import GloveDetectionConfig
    from glove_detection.inference import (
        GloveClassification,
        classify_glove_image,
        classify_glove_to_json,
        load_tflite_glove_classifier,
    )
except ModuleNotFoundError:  # pragma: no cover - convenience for direct script execution
    from config import GloveDetectionConfig
    from inference import (
        GloveClassification,
        classify_glove_image,
        classify_glove_to_json,
        load_tflite_glove_classifier,
    )


CLASS_TO_COLOR = {
    "Glove": (0, 200, 0),
    "NoGlove": (0, 0, 255),
    "Uncertain": (0, 215, 255),
}


def resolve_input_hands_dir(config: GloveDetectionConfig) -> Tuple[Path, str]:
    if config.input_hands_dir:
        input_dir = Path(config.input_hands_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"Input hands directory not found: {input_dir}")
        return input_dir, input_dir.name

    if not config.process_latest_job:
        raise ValueError(
            "input_hands_dir must be set when process_latest_job is False."
        )

    results_root = Path(config.results_root)
    if not results_root.exists():
        raise FileNotFoundError(
            f"Results root not found: {results_root}. "
            "Run person_region_extraction/main.py first, or set input_hands_dir explicitly."
        )

    candidate_jobs = [
        job_dir
        for job_dir in results_root.iterdir()
        if job_dir.is_dir() and list(job_dir.glob("track_*/*/hands.jpg"))
    ]
    if not candidate_jobs:
        raise FileNotFoundError(
            f"No hands crops found inside: {results_root}. "
            "Run person_region_extraction/main.py first, or set input_hands_dir explicitly."
        )

    latest_job_dir = max(candidate_jobs, key=lambda path: path.stat().st_mtime)
    return latest_job_dir, latest_job_dir.name


def iter_hands_images(input_root: Path) -> Iterable[Path]:
    for image_path in sorted(input_root.glob("track_*/*/hands.jpg")):
        if image_path.is_file():
            yield image_path


def resolve_final_label(
    classification: GloveClassification,
    confidence_threshold: float,
) -> str:
    if classification.confidence < confidence_threshold:
        return "Uncertain"
    return classification.normalized_label


def annotate_image(
    image: np.ndarray,
    classification: GloveClassification,
    final_label: str,
) -> np.ndarray:
    annotated = image.copy()
    color = CLASS_TO_COLOR.get(final_label, (255, 255, 0))
    cv2.rectangle(
        annotated,
        (1, 1),
        (max(1, annotated.shape[1] - 2), max(1, annotated.shape[0] - 2)),
        color,
        2,
    )
    cv2.putText(
        annotated,
        f"{final_label}: {classification.confidence:.2f}",
        (8, max(20, annotated.shape[0] - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )
    return annotated


def process_hands_tree(config: GloveDetectionConfig) -> Dict[str, object]:
    input_root, job_name = resolve_input_hands_dir(config)
    output_root = Path(config.output_root) / job_name
    output_root.mkdir(parents=True, exist_ok=True)

    classifier = load_tflite_glove_classifier(config)
    processed_count = 0
    skipped_count = 0
    glove_count = 0
    no_glove_count = 0
    uncertain_count = 0

    for image_path in iter_hands_images(input_root):
        image = cv2.imread(str(image_path))
        if image is None or image.size == 0:
            skipped_count += 1
            continue

        classification = classify_glove_image(classifier, image)
        final_label = resolve_final_label(
            classification,
            float(config.confidence_threshold),
        )

        relative_parent = image_path.relative_to(input_root).parent
        frame_output_dir = output_root / relative_parent
        frame_output_dir.mkdir(parents=True, exist_ok=True)

        annotated_image = annotate_image(image, classification, final_label)
        cv2.imwrite(str(frame_output_dir / "annotated.jpg"), annotated_image)

        prediction_payload = json.loads(classify_glove_to_json(classification))
        prediction_payload["final_label"] = final_label
        prediction_payload["confidence_threshold"] = float(config.confidence_threshold)
        (frame_output_dir / "prediction.json").write_text(
            json.dumps(prediction_payload, indent=2),
            encoding="utf-8",
        )

        if final_label == "Glove":
            glove_count += 1
        elif final_label == "NoGlove":
            no_glove_count += 1
        else:
            uncertain_count += 1

        processed_count += 1

    summary = {
        "job_name": job_name,
        "input_root": str(input_root),
        "output_root": str(output_root),
        "processed_count": processed_count,
        "skipped_count": skipped_count,
        "glove_count": glove_count,
        "no_glove_count": no_glove_count,
        "uncertain_count": uncertain_count,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary
