from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class GloveDetectionConfig:
    results_root: str = "person_region_extraction/output"
    input_hands_dir: Optional[str] = None
    output_root: str = "glove_detection/output"
    process_latest_job: bool = True
    model_path: str = "./weights/converted_tflite_quantized/model.tflite"
    labels_path: str = "./weights/converted_tflite_quantized/labels.txt"
    num_threads: int = 1
    input_color_space: str = "rgb"
    normalize_to_unit: bool = True
    input_mean: float = 0.0
    input_std: float = 1.0
    confidence_threshold: float = 0.5


CONFIG = GloveDetectionConfig()
