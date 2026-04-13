from __future__ import annotations

import ast
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover - dependency is environment-specific
    ort = None  # type: ignore[assignment]

from ultralytics.data.augment import LetterBox
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import scale_boxes, scale_coords


DEFAULT_ONNX_INPUT_SIZE = 640


@dataclass
class ONNXSessionInfo:
    session: Any
    input_name: str
    output_names: Tuple[str, ...]
    input_size: int
    class_names: Dict[int, str]
    task: str
    keypoint_shape: Optional[Tuple[int, int]]
    active_provider: str
    requested_device: str


def is_onnx_model_path(model_path: str) -> bool:
    return Path(model_path.strip()).suffix.lower() == ".onnx"


def _require_onnxruntime() -> Any:
    if ort is None:
        raise ImportError(
            "Missing dependency 'onnxruntime'. Install `onnxruntime-gpu` or `onnxruntime` "
            "to run ONNX models."
        )
    return ort


def _parse_metadata_literal(raw_value: Optional[str]) -> object:
    if not raw_value:
        return None
    try:
        return ast.literal_eval(raw_value)
    except (ValueError, SyntaxError):
        return None


def parse_names_metadata(raw_value: Optional[str]) -> Dict[int, str]:
    parsed_value = _parse_metadata_literal(raw_value)
    if isinstance(parsed_value, dict):
        names: Dict[int, str] = {}
        for key, value in parsed_value.items():
            try:
                names[int(key)] = str(value)
            except (TypeError, ValueError):
                continue
        return names
    return {}


def resolve_onnx_input_size(
    metadata: Dict[str, str],
    fallback: int = DEFAULT_ONNX_INPUT_SIZE,
) -> int:
    parsed_value = _parse_metadata_literal(metadata.get("imgsz"))
    if isinstance(parsed_value, (list, tuple)) and parsed_value:
        try:
            return int(parsed_value[0])
        except (TypeError, ValueError):
            return fallback
    if isinstance(parsed_value, int):
        return parsed_value
    return fallback


def parse_keypoint_shape(metadata: Dict[str, str]) -> Optional[Tuple[int, int]]:
    parsed_value = _parse_metadata_literal(metadata.get("kpt_shape"))
    if isinstance(parsed_value, (list, tuple)) and len(parsed_value) == 2:
        try:
            return int(parsed_value[0]), int(parsed_value[1])
        except (TypeError, ValueError):
            return None
    return None


def resolve_requested_providers(device: str) -> Tuple[str, ...]:
    normalized_device = device.strip()
    if normalized_device.startswith("cuda"):
        return ("CUDAExecutionProvider", "CPUExecutionProvider")
    return ("CPUExecutionProvider",)


def format_runtime_backend_label(
    backend: str,
    device: str,
    onnx_session: Optional[ONNXSessionInfo] = None,
) -> str:
    normalized_backend = backend.strip().lower()
    normalized_device = device.strip().lower()
    device_suffix = "cuda" if normalized_device.startswith("cuda") else "cpu"

    if normalized_backend == "onnx":
        if onnx_session is not None and onnx_session.active_provider == "CUDAExecutionProvider":
            return "onnx-cuda"
        return "onnx-cpu"
    if normalized_backend == "torch":
        return f"torch-{device_suffix}"
    if normalized_backend == "tflite":
        return "tflite-cpu"
    return f"{normalized_backend}-{device_suffix}"


def _preload_onnxruntime_gpu_dependencies(
    ort_module: Any,
    device: str,
) -> None:
    normalized_device = device.strip()
    if not normalized_device.startswith("cuda"):
        return
    if hasattr(ort_module, "preload_dlls"):
        ort_module.preload_dlls(cuda=True, cudnn=True, msvc=False)


def load_onnx_session(
    model_path: str,
    device: str,
) -> ONNXSessionInfo:
    normalized_model_path = model_path.strip()
    if not normalized_model_path:
        raise ValueError("Model path must not be empty.")
    if not Path(normalized_model_path).exists():
        raise FileNotFoundError(f"ONNX model not found: {normalized_model_path}")

    ort_module = _require_onnxruntime()
    _preload_onnxruntime_gpu_dependencies(ort_module, device)
    requested_providers = resolve_requested_providers(device)
    session_options = ort_module.SessionOptions()
    session_options.graph_optimization_level = ort_module.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.log_severity_level = 3

    session = ort_module.InferenceSession(
        normalized_model_path,
        sess_options=session_options,
        providers=list(requested_providers),
    )
    active_provider_list = session.get_providers()
    active_provider = active_provider_list[0] if active_provider_list else "UnknownExecutionProvider"

    normalized_device = device.strip()
    if normalized_device.startswith("cuda") and active_provider != "CUDAExecutionProvider":
        warnings.warn(
            "ONNX Runtime could not initialize CUDAExecutionProvider and fell back to "
            f"{active_provider}. Please verify the CUDA/cuDNN runtime required by onnxruntime-gpu.",
            RuntimeWarning,
            stacklevel=2,
        )

    metadata = dict(session.get_modelmeta().custom_metadata_map)
    input_details = session.get_inputs()[0]
    output_names = tuple(output.name for output in session.get_outputs())
    return ONNXSessionInfo(
        session=session,
        input_name=str(input_details.name),
        output_names=output_names,
        input_size=resolve_onnx_input_size(metadata),
        class_names=parse_names_metadata(metadata.get("names")),
        task=str(metadata.get("task", "")),
        keypoint_shape=parse_keypoint_shape(metadata),
        active_provider=active_provider,
        requested_device=normalized_device,
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


def _preprocess_yolo_frame(
    frame: np.ndarray,
    input_size: int,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    letterbox = LetterBox(new_shape=(input_size, input_size), auto=False, stride=32)
    prepared_frame = letterbox(image=frame)
    rgb_frame = cv2.cvtColor(prepared_frame, cv2.COLOR_BGR2RGB)
    nchw = np.transpose(rgb_frame, (2, 0, 1)).astype(np.float32) / 255.0
    return np.ascontiguousarray(nchw[None, ...]), prepared_frame.shape[:2]


def _run_onnx_output(
    session_info: ONNXSessionInfo,
    input_tensor: np.ndarray,
) -> np.ndarray:
    outputs = session_info.session.run(
        list(session_info.output_names),
        {session_info.input_name: input_tensor},
    )
    if not outputs:
        raise RuntimeError("ONNX session returned no outputs.")
    return outputs[0]


def run_yolo_onnx_detections(
    session_info: ONNXSessionInfo,
    frame: np.ndarray,
    resize_scale: float,
    confidence_threshold: float,
    iou_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    resized_frame, applied_resize_scale = _resize_frame_for_inference(frame, resize_scale)
    input_tensor, inference_shape = _preprocess_yolo_frame(
        resized_frame,
        session_info.input_size,
    )
    raw_output = _run_onnx_output(session_info, input_tensor)
    predictions = torch.from_numpy(raw_output)
    class_count = max(session_info.class_names.keys(), default=-1) + 1
    nms_results = non_max_suppression(
        predictions,
        conf_thres=float(confidence_threshold),
        iou_thres=float(iou_threshold),
        nc=class_count,
    )
    if not nms_results or len(nms_results[0]) == 0:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )

    detections = nms_results[0].clone()
    detections[:, :4] = scale_boxes(
        inference_shape,
        detections[:, :4],
        resized_frame.shape,
    )
    if applied_resize_scale != 1.0:
        detections[:, :4] = detections[:, :4] / applied_resize_scale

    xyxy = detections[:, :4].detach().cpu().numpy().astype(np.float32)
    confidence = detections[:, 4].detach().cpu().numpy().astype(np.float32)
    class_id = detections[:, 5].detach().cpu().numpy().astype(np.int32)
    return xyxy, confidence, class_id


def run_yolo_onnx_pose(
    session_info: ONNXSessionInfo,
    frame: np.ndarray,
    resize_scale: float,
    confidence_threshold: float,
    iou_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if session_info.keypoint_shape is None:
        raise ValueError("The ONNX model does not expose pose keypoint metadata.")

    resized_frame, applied_resize_scale = _resize_frame_for_inference(frame, resize_scale)
    input_tensor, inference_shape = _preprocess_yolo_frame(
        resized_frame,
        session_info.input_size,
    )
    raw_output = _run_onnx_output(session_info, input_tensor)
    predictions = torch.from_numpy(raw_output)
    class_count = max(session_info.class_names.keys(), default=-1) + 1
    nms_results = non_max_suppression(
        predictions,
        conf_thres=float(confidence_threshold),
        iou_thres=float(iou_threshold),
        nc=class_count,
    )
    if not nms_results or len(nms_results[0]) == 0:
        keypoint_count, keypoint_dims = session_info.keypoint_shape
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
            np.empty((0, keypoint_count, keypoint_dims), dtype=np.float32),
        )

    detections = nms_results[0].clone()
    detections[:, :4] = scale_boxes(
        inference_shape,
        detections[:, :4],
        resized_frame.shape,
    )
    keypoint_count, keypoint_dims = session_info.keypoint_shape
    keypoints = detections[:, 6:].view(len(detections), keypoint_count, keypoint_dims)
    keypoints = scale_coords(inference_shape, keypoints, resized_frame.shape)

    if applied_resize_scale != 1.0:
        detections[:, :4] = detections[:, :4] / applied_resize_scale
        keypoints[..., :2] = keypoints[..., :2] / applied_resize_scale

    xyxy = detections[:, :4].detach().cpu().numpy().astype(np.float32)
    confidence = detections[:, 4].detach().cpu().numpy().astype(np.float32)
    class_id = detections[:, 5].detach().cpu().numpy().astype(np.int32)
    keypoints_array = keypoints.detach().cpu().numpy().astype(np.float32)
    return xyxy, confidence, class_id, keypoints_array
