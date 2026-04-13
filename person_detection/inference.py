from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import supervision as sv
import torch

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover - import error is environment-specific
    raise ImportError(
        "Missing dependency 'ultralytics'. Install it with `pip install ultralytics`."
    ) from exc


@dataclass
class YOLOPersonInference:
    model: YOLO
    device: str
    confidence_threshold: float
    iou_threshold: float
    person_class_id: int
    input_resize_scale: float


def validate_device(device: str) -> str:
    normalized_device = device.strip()
    if not normalized_device:
        raise ValueError("Device must not be empty.")
    if normalized_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is not available. "
            "Please check your NVIDIA driver, CUDA runtime, and PyTorch installation."
        )
    return normalized_device


def resolve_person_class_id(model: YOLO) -> int:
    names = getattr(model.model, "names", None)
    if isinstance(names, dict):
        for class_id, class_name in names.items():
            if str(class_name).lower() == "person":
                return int(class_id)
    if isinstance(names, list):
        for class_id, class_name in enumerate(names):
            if str(class_name).lower() == "person":
                return class_id
    return 0


def load_yolo_person_inference(
    model_path: str,
    device: str,
    confidence_threshold: float,
    iou_threshold: float,
    input_resize_scale: float,
) -> YOLOPersonInference:
    normalized_model_path = model_path.strip()
    if not normalized_model_path:
        raise ValueError("Model path must not be empty.")
    if input_resize_scale <= 0:
        raise ValueError("Input resize scale must be greater than 0.")

    normalized_device = validate_device(device)
    model = YOLO(normalized_model_path).to(normalized_device)
    return YOLOPersonInference(
        model=model,
        device=normalized_device,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        person_class_id=resolve_person_class_id(model),
        input_resize_scale=input_resize_scale,
    )


def infer_person_detections(
    inference_engine: YOLOPersonInference,
    frame: np.ndarray,
) -> sv.Detections:
    inference_frame = frame
    resize_scale = float(inference_engine.input_resize_scale)
    if resize_scale != 1.0:
        resized_width = max(1, int(frame.shape[1] * resize_scale))
        resized_height = max(1, int(frame.shape[0] * resize_scale))
        inference_frame = cv2.resize(frame, (resized_width, resized_height))

    result = inference_engine.model.predict(
        source=inference_frame,
        device=inference_engine.device,
        conf=inference_engine.confidence_threshold,
        iou=inference_engine.iou_threshold,
        verbose=False,
    )[0]
    detections = sv.Detections.from_ultralytics(result)
    if len(detections) == 0 or detections.class_id is None:
        return detections

    if resize_scale != 1.0:
        detections.xyxy = detections.xyxy / resize_scale
    person_mask = detections.class_id == inference_engine.person_class_id
    return detections[person_mask]
