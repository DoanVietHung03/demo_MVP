from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np


@dataclass
class PersonInstance:
    tracker_id: int
    confidence: float
    bbox_xyxy: Tuple[int, int, int, int]
    crop: np.ndarray
    should_alert: bool


@dataclass
class AppConfig:
    source: str = "./vid_test/vid_1.mp4"
    model_path: str = "yolo11n.pt"
    device: str = "cuda:0"
    output_dir: Optional[str] = "person_detection/output/crops"
    output_video_path: Optional[str] = "person_detection/output/annotated.mp4"
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    crop_padding: int = 8
    input_resize_scale: float = 0.75
    frame_skip: int = 0
    track_ttl_frames: int = 60
    alert_cooldown_frames: Optional[int] = None
    max_frames: Optional[int] = None


CONFIG = AppConfig()
