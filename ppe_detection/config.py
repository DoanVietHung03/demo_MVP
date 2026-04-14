from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PPEDetectionConfig:
    source: str = "./vid_test/vid_1.mp4"
    model_path: str = "./weights/pytorch/PPE.pt"
    device: str = "cuda:0"
    output_root: str = "ppe_detection/output/default_run"
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    input_resize_scale: float = 0.75
    frame_skip: int = 0
    snapshot_interval_frames: int = 10
    max_frames: int | None = None
    target_class_names: tuple[str, ...] = field(
        default_factory=lambda: ("hardhat", "vest")
    )


CONFIG = PPEDetectionConfig()

