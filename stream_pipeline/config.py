from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

from behavior_detection.config import (
    CONFIG as BEHAVIOR_CONFIG,
    BehaviorDetectionConfig,
)
from facemask_detection.config import CONFIG as FACEMASK_CONFIG, FaceMaskConfig
from glove_detection.config import CONFIG as GLOVE_CONFIG, GloveDetectionConfig
from ppe_detection.config import CONFIG as PPE_CONFIG, PPEDetectionConfig
from person_detection.config import CONFIG as PERSON_CONFIG, AppConfig
from person_region_extraction.config import (
    CONFIG as REGION_CONFIG,
    RegionExtractionConfig,
)


@dataclass
class StreamPipelineConfig:
    source: str = PERSON_CONFIG.source
    output_root: str = "stream_pipeline/output"
    job_name: Optional[str] = None
    run_person_detection: bool = True
    run_ppe_detection: bool = False
    run_region_extraction: bool = True
    run_facemask_detection: bool = True
    run_glove_detection: bool = True
    run_behavior_detection: bool = False
    attribute_refresh_interval_frames: int = 6
    summary_flush_interval_frames: int = 15
    save_artifacts: bool = True
    use_latest_frame_reader_for_streams: bool = True
    capture_buffer_size: Optional[int] = 1
    capture_poll_timeout_seconds: float = 0.2
    person_detection: AppConfig = field(
        default_factory=lambda: replace(PERSON_CONFIG)
    )
    ppe_detection: PPEDetectionConfig = field(
        default_factory=lambda: replace(PPE_CONFIG)
    )
    person_region_extraction: RegionExtractionConfig = field(
        default_factory=lambda: replace(REGION_CONFIG)
    )
    facemask_detection: FaceMaskConfig = field(
        default_factory=lambda: replace(FACEMASK_CONFIG)
    )
    glove_detection: GloveDetectionConfig = field(
        default_factory=lambda: replace(GLOVE_CONFIG)
    )
    behavior_detection: BehaviorDetectionConfig = field(
        default_factory=lambda: replace(BEHAVIOR_CONFIG)
    )


CONFIG = StreamPipelineConfig()
