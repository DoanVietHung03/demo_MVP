from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class FaceMaskConfig:
    results_root: str = "person_region_extraction/output"
    input_head_face_dir: Optional[str] = None
    output_root: str = "facemask_detection/output"
    process_latest_job: bool = True
    model_path: str = "./weights/facemask.onnx"
    device: str = "cpu"
    input_size: int = 360
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.4


CONFIG = FaceMaskConfig()
