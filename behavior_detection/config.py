from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class BehaviorDetectionConfig:
    source: str = "./vid_test/vid_1.mp4"
    person_phone_model_path: str = "./weights/onnx/yolo11n.onnx"
    pose_model_path: str = "./weights/onnx/yolo11n-pose.onnx"
    smoke_model_path: str = "./weights/onnx/smoke.onnx"
    device: str = "cuda:0"
    output_root: str = "behavior_detection/output/default_run"
    person_confidence_threshold: float = 0.35
    phone_confidence_threshold: float = 0.25
    smoke_confidence_threshold: float = 0.35
    pose_confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    input_resize_scale: float = 1.0
    crop_padding: int = 8
    frame_skip: int = 0
    snapshot_interval_frames: int = 10
    track_ttl_frames: int = 45
    alert_cooldown_frames: int = 90
    smoking_min_consecutive_frames: int = 4
    phone_present_min_consecutive_frames: int = 3
    phone_in_hand_min_consecutive_frames: int = 3
    phone_usage_min_consecutive_frames: int = 4
    smoking_mouth_distance_ratio: float = 0.18
    smoking_hand_distance_ratio: float = 0.24
    hand_to_mouth_distance_ratio: float = 0.22
    phone_hand_distance_ratio: float = 0.26
    phone_face_distance_ratio: float = 0.24
    smoke_phone_overlap_iou_threshold: float = 0.1
    smoke_max_box_area_ratio: float = 0.015
    max_frames: Optional[int] = None


CONFIG = BehaviorDetectionConfig()
