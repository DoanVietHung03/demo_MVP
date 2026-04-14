from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


@dataclass
class StageRunResult:
    name: str
    enabled: bool
    success: bool = True
    input_path: Optional[str] = None
    output_path: Optional[str] = None
    summary: Dict[str, object] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "enabled": self.enabled,
            "success": self.success,
            "input_path": self.input_path,
            "output_path": self.output_path,
            "summary": self.summary,
            "error": self.error,
        }


@dataclass
class PipelinePaths:
    run_root: Path
    crops_dir: Path
    region_output_root: Path
    facemask_output_root: Path
    glove_output_root: Path
    behavior_output_root: Path
    summary_path: Path

    def to_dict(self) -> Dict[str, str]:
        return {
            "run_root": str(self.run_root),
            "crops_dir": str(self.crops_dir),
            "region_output_root": str(self.region_output_root),
            "facemask_output_root": str(self.facemask_output_root),
            "glove_output_root": str(self.glove_output_root),
            "behavior_output_root": str(self.behavior_output_root),
            "summary_path": str(self.summary_path),
        }


@dataclass
class PipelineRunSummary:
    job_name: str
    source: str
    output_root: str
    status: str
    paths: PipelinePaths
    stages: Dict[str, StageRunResult] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "job_name": self.job_name,
            "source": self.source,
            "output_root": self.output_root,
            "status": self.status,
            "paths": self.paths.to_dict(),
            "stages": {
                stage_name: stage_result.to_dict()
                for stage_name, stage_result in self.stages.items()
            },
            "error": self.error,
        }

