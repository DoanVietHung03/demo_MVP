from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict

import torch
from ultralytics import YOLO

from facemask_detection.inference import _register_legacy_module

try:
    import onnx
except ImportError as exc:  # pragma: no cover - dependency is environment-specific
    raise ImportError(
        "Missing dependency 'onnx'. Install it before exporting models to ONNX."
    ) from exc


DEFAULT_YOLO_MODELS: Dict[str, str] = {
    "person_detection": "weights/yolo11n.pt",
    "person_region_extraction": "weights/yolo11n-pose.pt",
    "behavior_smoke": "weights/smoke.pt",
}
DEFAULT_FACEMASK_MODEL = "weights/facemask.pth"


def export_ultralytics_model(
    model_path: str,
    imgsz: int,
    opset: int,
    dynamic: bool,
    device: str,
) -> Path:
    normalized_model_path = Path(model_path)
    if not normalized_model_path.exists():
        raise FileNotFoundError(f"Model not found: {normalized_model_path}")

    model = YOLO(str(normalized_model_path))
    exported_path = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        simplify=False,
        dynamic=dynamic,
        device=device,
    )
    exported_path_obj = Path(str(exported_path))
    onnx.checker.check_model(str(exported_path_obj))
    return exported_path_obj


def export_facemask_model(
    model_path: str,
    input_size: int,
    opset: int,
    output_path: str | None = None,
) -> Path:
    normalized_model_path = Path(model_path)
    if not normalized_model_path.exists():
        raise FileNotFoundError(f"Facemask model not found: {normalized_model_path}")

    output_path_obj = (
        Path(output_path)
        if output_path
        else normalized_model_path.with_suffix(".onnx")
    )

    _register_legacy_module()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = torch.load(str(normalized_model_path), map_location="cpu")
    if not isinstance(model, torch.nn.Module):
        raise TypeError("Loaded facemask checkpoint is not a torch.nn.Module.")

    model.eval()
    dummy_input = torch.randn(1, 3, input_size, input_size, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path_obj),
        input_names=["images"],
        output_names=["boxes", "scores"],
        opset_version=opset,
        dynamic_axes={
            "images": {0: "batch"},
            "boxes": {0: "batch"},
            "scores": {0: "batch"},
        },
    )
    onnx.checker.check_model(str(output_path_obj))
    return output_path_obj


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export the repo's PyTorch YOLO/facemask weights to ONNX.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Export device for Ultralytics models. Example: cpu or cuda:0",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference/export image size used for Ultralytics models.",
    )
    parser.add_argument(
        "--facemask-input-size",
        type=int,
        default=360,
        help="Input size used to export the facemask model.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version.",
    )
    parser.add_argument(
        "--static",
        action="store_true",
        help="Disable dynamic input axes for Ultralytics exports.",
    )
    parser.add_argument(
        "--skip-yolo",
        action="store_true",
        help="Skip exporting the Ultralytics YOLO models.",
    )
    parser.add_argument(
        "--skip-facemask",
        action="store_true",
        help="Skip exporting facemask.pth.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    exported_models: Dict[str, str] = {}

    if not args.skip_yolo:
        for model_name, model_path in DEFAULT_YOLO_MODELS.items():
            exported_path = export_ultralytics_model(
                model_path=model_path,
                imgsz=int(args.imgsz),
                opset=int(args.opset),
                dynamic=not bool(args.static),
                device=str(args.device),
            )
            exported_models[model_name] = str(exported_path)
            print(f"[OK] {model_name}: {model_path} -> {exported_path}")

    if not args.skip_facemask:
        exported_path = export_facemask_model(
            model_path=DEFAULT_FACEMASK_MODEL,
            input_size=int(args.facemask_input_size),
            opset=int(args.opset),
        )
        exported_models["facemask_detection"] = str(exported_path)
        print(f"[OK] facemask_detection: {DEFAULT_FACEMASK_MODEL} -> {exported_path}")

    if not exported_models:
        print("No models were exported.")
        return

    print("\nExport summary:")
    for model_name, exported_path in exported_models.items():
        print(f"- {model_name}: {exported_path}")


if __name__ == "__main__":
    main()
