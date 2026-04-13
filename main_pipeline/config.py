from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

from behavior_detection.config import (
    CONFIG as BEHAVIOR_CONFIG,
    BehaviorDetectionConfig,
)
from facemask_detection.config import CONFIG as FACEMASK_CONFIG, FaceMaskConfig
from glove_detection.config import CONFIG as GLOVE_CONFIG, GloveDetectionConfig
from person_detection.config import CONFIG as PERSON_CONFIG, AppConfig
from person_region_extraction.config import (
    CONFIG as REGION_CONFIG,
    RegionExtractionConfig,
)


@dataclass
class MainPipelineConfig:
    source: str = PERSON_CONFIG.source
    output_root: str = "main_pipeline/output"
    job_name: Optional[str] = None
    run_person_detection: bool = True
    run_region_extraction: bool = True
    run_facemask_detection: bool = True
    run_glove_detection: bool = True
    run_behavior_detection: bool = False
    person_detection: AppConfig = field(
        default_factory=lambda: replace(PERSON_CONFIG)
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


CONFIG = MainPipelineConfig()
