from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class RegionExtractionConfig:
    results_root: str = "web_output/results"
    input_crops_dir: Optional[str] = None
    output_root: str = "person_region_extraction/output"
    process_latest_job: bool = True
    model_path: str = "./weights/yolo11n-pose.pt"
    device: str = "cuda:0"
    person_confidence_threshold: float = 0.25
    keypoint_confidence_threshold: float = 0.25
    face_padding_ratio: float = 0.15
    head_height_ratio: float = 0.32
    torso_top_ratio: float = 0.28
    torso_bottom_ratio: float = 0.78
    torso_width_ratio: float = 0.56
    hands_top_ratio: float = 0.34
    hands_bottom_ratio: float = 0.82
    hand_side_width_ratio: float = 0.28
    hand_box_size_ratio: float = 0.18
    min_face_size: int = 24


CONFIG = RegionExtractionConfig()
