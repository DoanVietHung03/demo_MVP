from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

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

from runtime_support.onnx import (
    ONNXSessionInfo,
    format_runtime_backend_label,
    is_onnx_model_path,
    load_onnx_session,
    run_yolo_onnx_detections,
)


@dataclass
class YOLOPPEInference:
    model: object
    device: str
    confidence_threshold: float
    iou_threshold: float
    input_resize_scale: float
    class_name_to_id: Dict[str, int]
    class_names: Dict[int, str] = field(default_factory=dict)
    backend: str = "torch"
    onnx_session: ONNXSessionInfo | None = None


def describe_ppe_inference_backend(inference_engine: YOLOPPEInference) -> str:
    return format_runtime_backend_label(
        inference_engine.backend,
        inference_engine.device,
        inference_engine.onnx_session,
    )


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


def resolve_class_name_to_id_from_names(names: object) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    if isinstance(names, dict):
        for class_id, class_name in names.items():
            mapping[str(class_name).lower()] = int(class_id)
    elif isinstance(names, list):
        for class_id, class_name in enumerate(names):
            mapping[str(class_name).lower()] = class_id
    return mapping


def load_yolo_ppe_inference(
    model_path: str,
    device: str,
    confidence_threshold: float,
    iou_threshold: float,
    input_resize_scale: float,
) -> YOLOPPEInference:
    normalized_model_path = model_path.strip()
    if not normalized_model_path:
        raise ValueError("Model path must not be empty.")
    if input_resize_scale <= 0:
        raise ValueError("Input resize scale must be greater than 0.")

    normalized_device = validate_device(device)
    if is_onnx_model_path(normalized_model_path):
        session_info = load_onnx_session(normalized_model_path, normalized_device)
        class_names = dict(session_info.class_names)
        return YOLOPPEInference(
            model=session_info.session,
            device=normalized_device,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            input_resize_scale=input_resize_scale,
            class_name_to_id=resolve_class_name_to_id_from_names(class_names),
            class_names=class_names,
            backend="onnx",
            onnx_session=session_info,
        )

    model = YOLO(normalized_model_path).to(normalized_device)
    raw_names = getattr(model.model, "names", None)
    class_names = (
        {
            int(class_id): str(class_name)
            for class_id, class_name in raw_names.items()
        }
        if isinstance(raw_names, dict)
        else {
            class_id: str(class_name)
            for class_id, class_name in enumerate(raw_names or [])
        }
    )
    return YOLOPPEInference(
        model=model,
        device=normalized_device,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        input_resize_scale=input_resize_scale,
        class_name_to_id=resolve_class_name_to_id_from_names(raw_names),
        class_names=class_names,
    )


def _resize_frame_for_inference(
    frame: np.ndarray,
    resize_scale: float,
) -> Tuple[np.ndarray, float]:
    if resize_scale == 1.0:
        return frame, 1.0

    resized_width = max(1, int(frame.shape[1] * resize_scale))
    resized_height = max(1, int(frame.shape[0] * resize_scale))
    resized_frame = cv2.resize(frame, (resized_width, resized_height))
    return resized_frame, resize_scale


def infer_ppe_detections(
    inference_engine: YOLOPPEInference,
    frame: np.ndarray,
) -> sv.Detections:
    if inference_engine.backend == "onnx":
        if inference_engine.onnx_session is None:
            raise RuntimeError("ONNX PPE inference session is not initialized.")
        xyxy, confidence, class_id = run_yolo_onnx_detections(
            inference_engine.onnx_session,
            frame,
            resize_scale=float(inference_engine.input_resize_scale),
            confidence_threshold=float(inference_engine.confidence_threshold),
            iou_threshold=float(inference_engine.iou_threshold),
        )
        return sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
        )

    inference_frame, resize_scale = _resize_frame_for_inference(
        frame,
        float(inference_engine.input_resize_scale),
    )
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
    return detections


def filter_detections_by_class_names(
    inference_engine: YOLOPPEInference,
    detections: sv.Detections,
    target_class_names: Sequence[str],
) -> sv.Detections:
    if len(detections) == 0 or detections.class_id is None:
        return detections

    allowed_ids: List[int] = []
    for class_name in target_class_names:
        class_id = inference_engine.class_name_to_id.get(class_name.lower())
        if class_id is not None:
            allowed_ids.append(class_id)
    if not allowed_ids:
        return detections[np.array([], dtype=int)]

    class_mask = np.isin(detections.class_id, np.asarray(allowed_ids))
    return detections[class_mask]

