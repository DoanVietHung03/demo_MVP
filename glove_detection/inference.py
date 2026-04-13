from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    from glove_detection.config import GloveDetectionConfig
except ModuleNotFoundError:  # pragma: no cover - convenience for direct script execution
    from config import GloveDetectionConfig


def _resolve_tflite_interpreter_class() -> Optional[type]:
    try:
        from tflite_runtime.interpreter import Interpreter  # type: ignore

        return Interpreter
    except ImportError:
        pass

    try:
        import tensorflow as tf  # type: ignore

        return tf.lite.Interpreter
    except ImportError:
        pass

    return None


@dataclass
class GloveClassification:
    class_index: int
    raw_label: str
    normalized_label: str
    confidence: float


@dataclass
class TFLiteGloveClassifier:
    interpreter: Any
    input_index: int
    output_index: int
    input_height: int
    input_width: int
    input_channels: int
    input_dtype: np.dtype
    output_dtype: np.dtype
    input_quantization: Tuple[float, int]
    output_quantization: Tuple[float, int]
    labels: List[str]
    input_color_space: str
    normalize_to_unit: bool
    input_mean: float
    input_std: float


def load_labels(labels_path: str) -> List[str]:
    normalized_labels_path = labels_path.strip()
    if not normalized_labels_path:
        raise ValueError("Labels path must not be empty.")

    path = Path(normalized_labels_path)
    if not path.exists():
        raise FileNotFoundError(f"Labels file not found: {path}")

    labels: List[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) == 2 and parts[0].isdigit():
            labels.append(parts[1].strip())
        else:
            labels.append(line)

    if not labels:
        raise ValueError(f"Labels file is empty: {path}")
    return labels


def normalize_glove_label(raw_label: str) -> str:
    normalized_raw = raw_label.strip()
    lowered = normalized_raw.lower().replace("-", " ").replace("_", " ")
    if "glove" in lowered:
        return "Glove"
    if "hand" in lowered or "bare" in lowered:
        return "NoGlove"
    return normalized_raw or "Unknown"


def _extract_quantization(
    tensor_details: Dict[str, Any],
) -> Tuple[float, int]:
    scale, zero_point = tensor_details.get("quantization", (0.0, 0))
    if scale not in (None, 0, 0.0):
        return float(scale), int(zero_point)

    quantization_parameters = tensor_details.get("quantization_parameters") or {}
    scales = quantization_parameters.get("scales")
    zero_points = quantization_parameters.get("zero_points")
    if scales is not None and len(scales) > 0:
        scale_value = float(scales[0])
        zero_point_value = (
            int(zero_points[0]) if zero_points is not None and len(zero_points) > 0 else 0
        )
        return scale_value, zero_point_value

    return 0.0, 0


def _prepare_float_input(
    image: np.ndarray,
    normalize_to_unit: bool,
    input_mean: float,
    input_std: float,
    dtype: np.dtype,
) -> np.ndarray:
    input_tensor = image.astype(np.float32)
    if normalize_to_unit:
        input_tensor = input_tensor / 255.0
    safe_input_std = input_std if abs(input_std) > 1e-6 else 1.0
    input_tensor = (input_tensor - input_mean) / safe_input_std
    return input_tensor.astype(dtype)


def _prepare_quantized_input(
    image: np.ndarray,
    dtype: np.dtype,
    quantization: Tuple[float, int],
) -> np.ndarray:
    if np.issubdtype(dtype, np.integer):
        dtype_info = np.iinfo(dtype)
        clipped = np.clip(image, dtype_info.min, dtype_info.max)
        return clipped.astype(dtype)
    return image.astype(dtype)


def _prepare_input_tensor(
    classifier: TFLiteGloveClassifier,
    image: np.ndarray,
) -> np.ndarray:
    resized = cv2.resize(image, (classifier.input_width, classifier.input_height))
    color_space = classifier.input_color_space.strip().lower()
    if color_space == "rgb":
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    elif color_space != "bgr":
        raise ValueError(f"Unsupported input_color_space: {classifier.input_color_space}")

    if np.issubdtype(classifier.input_dtype, np.floating):
        input_tensor = _prepare_float_input(
            resized,
            normalize_to_unit=classifier.normalize_to_unit,
            input_mean=classifier.input_mean,
            input_std=classifier.input_std,
            dtype=classifier.input_dtype,
        )
    else:
        input_tensor = _prepare_quantized_input(
            resized,
            dtype=classifier.input_dtype,
            quantization=classifier.input_quantization,
        )

    return np.expand_dims(input_tensor, axis=0)


def _decode_output_scores(
    classifier: TFLiteGloveClassifier,
    output_tensor: np.ndarray,
) -> np.ndarray:
    flattened = output_tensor.reshape(-1).astype(np.float32)
    scale, zero_point = classifier.output_quantization
    if not np.issubdtype(classifier.output_dtype, np.floating) and scale > 0:
        return (flattened - zero_point) * scale
    return flattened


def _scores_to_probabilities(raw_scores: np.ndarray) -> np.ndarray:
    if raw_scores.size == 0:
        raise ValueError("Model output scores are empty.")

    if raw_scores.size == 1:
        positive_probability = 1.0 / (1.0 + np.exp(-raw_scores[0]))
        return np.asarray([1.0 - positive_probability, positive_probability], dtype=np.float32)

    if np.all(raw_scores >= 0.0):
        score_sum = float(np.sum(raw_scores))
        if 0.99 <= score_sum <= 1.01:
            return raw_scores / max(score_sum, 1e-6)

    shifted = raw_scores - np.max(raw_scores)
    exp_scores = np.exp(shifted)
    return exp_scores / max(float(np.sum(exp_scores)), 1e-6)


def load_tflite_glove_classifier(
    config: GloveDetectionConfig,
) -> TFLiteGloveClassifier:
    model_path = config.model_path.strip()
    if not model_path:
        raise ValueError("Model path must not be empty.")
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Glove model not found: {model_path}")

    interpreter_class = _resolve_tflite_interpreter_class()
    if interpreter_class is None:
        raise ImportError(
            "Missing TFLite runtime. Install `tflite-runtime` or `tensorflow` to run glove_detection."
        )

    interpreter = interpreter_class(
        model_path=model_path,
        num_threads=max(1, int(config.num_threads)),
    )
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    input_shape = tuple(int(value) for value in input_details["shape"])
    if len(input_shape) != 4:
        raise ValueError(
            "Expected a 4D image classifier input tensor, "
            f"but got shape={input_shape}."
        )

    batch_size, input_height, input_width, input_channels = input_shape
    if batch_size not in (1, -1):
        raise ValueError(f"Unsupported batch size for glove classifier: {batch_size}")

    labels = load_labels(config.labels_path)
    return TFLiteGloveClassifier(
        interpreter=interpreter,
        input_index=int(input_details["index"]),
        output_index=int(output_details["index"]),
        input_height=input_height,
        input_width=input_width,
        input_channels=input_channels,
        input_dtype=np.dtype(input_details["dtype"]),
        output_dtype=np.dtype(output_details["dtype"]),
        input_quantization=_extract_quantization(input_details),
        output_quantization=_extract_quantization(output_details),
        labels=labels,
        input_color_space=config.input_color_space,
        normalize_to_unit=bool(config.normalize_to_unit),
        input_mean=float(config.input_mean),
        input_std=float(config.input_std),
    )


def classify_glove_image(
    classifier: TFLiteGloveClassifier,
    image: np.ndarray,
) -> GloveClassification:
    if image is None or image.size == 0:
        raise ValueError("Input image must not be empty.")

    input_tensor = _prepare_input_tensor(classifier, image)
    classifier.interpreter.set_tensor(classifier.input_index, input_tensor)
    classifier.interpreter.invoke()
    output_tensor = classifier.interpreter.get_tensor(classifier.output_index)

    raw_scores = _decode_output_scores(classifier, output_tensor)
    probabilities = _scores_to_probabilities(raw_scores)
    if len(classifier.labels) != probabilities.size:
        if len(classifier.labels) < probabilities.size:
            labels = classifier.labels + [
                f"class_{index}" for index in range(len(classifier.labels), probabilities.size)
            ]
        else:
            labels = classifier.labels[: probabilities.size]
    else:
        labels = classifier.labels

    class_index = int(np.argmax(probabilities))
    raw_label = labels[class_index]
    return GloveClassification(
        class_index=class_index,
        raw_label=raw_label,
        normalized_label=normalize_glove_label(raw_label),
        confidence=float(probabilities[class_index]),
    )


def classify_glove_to_json(classification: GloveClassification) -> str:
    payload: Dict[str, object] = {
        "class_index": classification.class_index,
        "raw_label": classification.raw_label,
        "normalized_label": classification.normalized_label,
        "confidence": round(classification.confidence, 6),
    }
    return json.dumps(payload, indent=2)
