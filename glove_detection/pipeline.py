from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

try:
    from glove_detection.config import GloveDetectionConfig
    from glove_detection.inference import (
        GloveClassification,
        classify_glove_image,
        describe_glove_classifier_backend,
        load_tflite_glove_classifier,
    )
except ModuleNotFoundError:  # pragma: no cover - convenience for direct script execution
    from config import GloveDetectionConfig
    from inference import (
        GloveClassification,
        classify_glove_image,
        describe_glove_classifier_backend,
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


def _load_regions_metadata(image_path: Path) -> Optional[Dict[str, object]]:
    metadata_path = image_path.parent / "regions.json"
    if not metadata_path.exists():
        return None
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _clip_box_to_image(
    box: Tuple[int, int, int, int],
    image: np.ndarray,
) -> Tuple[int, int, int, int]:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box
    clipped_x1 = max(0, min(int(x1), width - 1))
    clipped_y1 = max(0, min(int(y1), height - 1))
    clipped_x2 = max(clipped_x1 + 1, min(int(x2), width))
    clipped_y2 = max(clipped_y1 + 1, min(int(y2), height))
    return clipped_x1, clipped_y1, clipped_x2, clipped_y2


def _iter_hand_crops(
    image: np.ndarray,
    metadata: Optional[Dict[str, object]],
) -> List[Tuple[Tuple[int, int, int, int], np.ndarray]]:
    raw_boxes = metadata.get("hand_crop_boxes") if isinstance(metadata, dict) else None
    if isinstance(raw_boxes, list) and raw_boxes:
        hand_regions: List[Tuple[Tuple[int, int, int, int], np.ndarray]] = []
        for raw_box in raw_boxes:
            if not isinstance(raw_box, (list, tuple)) or len(raw_box) != 4:
                continue
            box = _clip_box_to_image(
                tuple(int(value) for value in raw_box),
                image,
            )
            x1, y1, x2, y2 = box
            crop = image[y1:y2, x1:x2].copy()
            if crop.size == 0:
                continue
            hand_regions.append((box, crop))
        if hand_regions:
            return hand_regions

    full_box = (0, 0, image.shape[1], image.shape[0])
    return [(full_box, image.copy())]


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


def annotate_hand_predictions(
    image: np.ndarray,
    hand_predictions: List[Dict[str, object]],
) -> np.ndarray:
    if not hand_predictions:
        return image.copy()

    annotated = image.copy()
    for hand_prediction in hand_predictions:
        label = str(hand_prediction["label"])
        confidence = float(hand_prediction["confidence"])
        x1, y1, x2, y2 = [int(value) for value in hand_prediction["bbox_xyxy"]]
        color = CLASS_TO_COLOR.get(label, (255, 255, 0))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            f"{label}: {confidence:.2f}",
            (x1 + 2, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
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
    runtime_backend = describe_glove_classifier_backend(classifier)
    print(
        "[Stage Runtime] "
        f"glove_detection: {runtime_backend} | model={config.model_path}"
    )
    processed_count = 0
    processed_image_count = 0
    skipped_count = 0
    glove_count = 0
    no_glove_count = 0
    uncertain_count = 0

    for image_path in iter_hands_images(input_root):
        image = cv2.imread(str(image_path))
        if image is None or image.size == 0:
            skipped_count += 1
            continue

        metadata = _load_regions_metadata(image_path)
        hand_regions = _iter_hand_crops(image, metadata)
        hand_predictions: List[Dict[str, object]] = []
        best_label = "Uncertain"
        best_confidence = 0.0

        for hand_index, (hand_box, hand_crop) in enumerate(hand_regions):
            classification = classify_glove_image(classifier, hand_crop)
            final_label = resolve_final_label(
                classification,
                float(config.confidence_threshold),
            )
            confidence_value = float(classification.confidence)
            hand_predictions.append(
                {
                    "index": hand_index,
                    "bbox_xyxy": list(hand_box),
                    "label": final_label,
                    "confidence": round(confidence_value, 6),
                    "raw_label": classification.raw_label,
                    "normalized_label": classification.normalized_label,
                    "class_index": int(classification.class_index),
                }
            )
            processed_count += 1

            if final_label == "Glove":
                glove_count += 1
            elif final_label == "NoGlove":
                no_glove_count += 1
            else:
                uncertain_count += 1

            if confidence_value >= best_confidence:
                best_confidence = confidence_value
                best_label = final_label

        processed_image_count += 1

        relative_parent = image_path.relative_to(input_root).parent
        frame_output_dir = output_root / relative_parent
        frame_output_dir.mkdir(parents=True, exist_ok=True)

        if len(hand_predictions) == 1 and hand_predictions[0]["bbox_xyxy"] == [
            0,
            0,
            image.shape[1],
            image.shape[0],
        ]:
            classification = GloveClassification(
                class_index=int(hand_predictions[0]["class_index"]),
                raw_label=str(hand_predictions[0]["raw_label"]),
                normalized_label=str(hand_predictions[0]["normalized_label"]),
                confidence=float(hand_predictions[0]["confidence"]),
            )
            annotated_image = annotate_image(
                image,
                classification,
                str(hand_predictions[0]["label"]),
            )
        else:
            annotated_image = annotate_hand_predictions(image, hand_predictions)
        cv2.imwrite(str(frame_output_dir / "annotated.jpg"), annotated_image)

        prediction_payload = {
            "final_label": best_label,
            "confidence": round(best_confidence, 6),
            "confidence_threshold": float(config.confidence_threshold),
            "hands": hand_predictions,
        }
        if metadata is not None:
            prediction_payload["regions_metadata_path"] = str(
                image_path.parent / "regions.json"
            )
        (frame_output_dir / "prediction.json").write_text(
            json.dumps(prediction_payload, indent=2),
            encoding="utf-8",
        )

    summary = {
        "job_name": job_name,
        "input_root": str(input_root),
        "output_root": str(output_root),
        "runtime_backend": runtime_backend,
        "model_path": config.model_path,
        "processed_count": processed_count,
        "processed_image_count": processed_image_count,
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
