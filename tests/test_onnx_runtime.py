from __future__ import annotations

import unittest
from unittest.mock import Mock

from runtime_support.onnx import (
    _preload_onnxruntime_gpu_dependencies,
    ONNXSessionInfo,
    format_runtime_backend_label,
    parse_keypoint_shape,
    parse_names_metadata,
    resolve_onnx_input_size,
    resolve_requested_providers,
)


class ONNXRuntimeHelperTests(unittest.TestCase):
    def test_parse_names_metadata_handles_ultralytics_string_dict(self) -> None:
        metadata_value = "{0: 'person', 67: 'cell phone'}"

        parsed = parse_names_metadata(metadata_value)

        self.assertEqual(parsed, {0: "person", 67: "cell phone"})

    def test_parse_keypoint_shape_reads_exported_pose_metadata(self) -> None:
        metadata = {"kpt_shape": "[17, 3]"}

        parsed = parse_keypoint_shape(metadata)

        self.assertEqual(parsed, (17, 3))

    def test_resolve_onnx_input_size_uses_imgsz_metadata(self) -> None:
        metadata = {"imgsz": "[640, 640]"}

        self.assertEqual(resolve_onnx_input_size(metadata), 640)

    def test_resolve_requested_providers_prefers_cuda_then_cpu(self) -> None:
        self.assertEqual(
            resolve_requested_providers("cuda:0"),
            ("CUDAExecutionProvider", "CPUExecutionProvider"),
        )
        self.assertEqual(
            resolve_requested_providers("cpu"),
            ("CPUExecutionProvider",),
        )

    def test_preload_onnxruntime_gpu_dependencies_only_runs_for_cuda(self) -> None:
        ort_module = Mock()

        _preload_onnxruntime_gpu_dependencies(ort_module, "cuda:0")
        ort_module.preload_dlls.assert_called_once_with(
            cuda=True,
            cudnn=True,
            msvc=False,
        )

        ort_module.reset_mock()
        _preload_onnxruntime_gpu_dependencies(ort_module, "cpu")
        ort_module.preload_dlls.assert_not_called()

    def test_format_runtime_backend_label_reports_onnx_cuda(self) -> None:
        session_info = ONNXSessionInfo(
            session=None,
            input_name="images",
            output_names=("output0",),
            input_size=640,
            class_names={0: "person"},
            task="detect",
            keypoint_shape=None,
            active_provider="CUDAExecutionProvider",
            requested_device="cuda:0",
        )

        self.assertEqual(
            format_runtime_backend_label("onnx", "cuda:0", session_info),
            "onnx-cuda",
        )

    def test_format_runtime_backend_label_reports_torch_and_tflite_cpu(self) -> None:
        self.assertEqual(format_runtime_backend_label("torch", "cpu"), "torch-cpu")
        self.assertEqual(format_runtime_backend_label("tflite", "cpu"), "tflite-cpu")
