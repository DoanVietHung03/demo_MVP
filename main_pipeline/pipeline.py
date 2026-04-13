from __future__ import annotations

import json
import re
import time
from dataclasses import replace
from pathlib import Path
from typing import Callable, Dict, Optional, Union

from behavior_detection.pipeline import BehaviorDetectionPipeline
from facemask_detection.pipeline import process_head_face_tree
from glove_detection.pipeline import process_hands_tree
from main_pipeline.config import MainPipelineConfig
from main_pipeline.types import PipelinePaths, PipelineRunSummary, StageRunResult
from person_detection.config import PersonInstance
from person_detection.pipeline import PersonDetectionPipeline, log_queue
from person_region_extraction.pipeline import process_crop_tree

StageCallback = Callable[[str, str, Optional[StageRunResult]], None]
PersonFrameCallback = Callable[[int, Optional[int], object, list[PersonInstance]], None]


def sanitize_job_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    normalized = normalized.strip("._-")
    return normalized or "job"


def build_job_name(source: str, override: Optional[str] = None) -> str:
    if override:
        return sanitize_job_name(override)

    source_name = Path(str(source)).stem or "source"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return f"{sanitize_job_name(source_name)}_{timestamp}"


def build_pipeline_paths(
    output_root: Union[str, Path],
    job_name: str,
) -> PipelinePaths:
    run_root = Path(output_root) / job_name
    return PipelinePaths(
        run_root=run_root,
        crops_dir=run_root / "crops",
        region_output_root=run_root / "person_region_extraction",
        facemask_output_root=run_root / "facemask_detection",
        glove_output_root=run_root / "glove_detection",
        behavior_output_root=run_root / "behavior_detection",
        summary_path=run_root / "summary.json",
    )


class MainPipeline:
    def __init__(
        self,
        config: MainPipelineConfig,
        stage_callback: Optional[StageCallback] = None,
        person_detection_frame_callback: Optional[PersonFrameCallback] = None,
    ) -> None:
        self.config = config
        self.stage_callback = stage_callback
        self.person_detection_frame_callback = person_detection_frame_callback

    def _notify_stage(
        self,
        stage_name: str,
        status: str,
        stage_result: Optional[StageRunResult] = None,
    ) -> None:
        if self.stage_callback is not None:
            self.stage_callback(stage_name, status, stage_result)

    def _write_summary(
        self,
        summary: PipelineRunSummary,
    ) -> None:
        summary.paths.run_root.mkdir(parents=True, exist_ok=True)
        summary.paths.summary_path.write_text(
            json.dumps(summary.to_dict(), indent=2),
            encoding="utf-8",
        )

    def _run_person_detection_stage(
        self,
        paths: PipelinePaths,
    ) -> StageRunResult:
        config = replace(self.config.person_detection)
        config.source = self.config.source

        pipeline = PersonDetectionPipeline(
            model_path=config.model_path,
            device=config.device,
            confidence_threshold=config.confidence_threshold,
            iou_threshold=config.iou_threshold,
            crop_padding=config.crop_padding,
            input_resize_scale=config.input_resize_scale,
            track_ttl_frames=config.track_ttl_frames,
            alert_cooldown_frames=config.alert_cooldown_frames,
        )
        pipeline.process_video(
            source=config.source,
            output_dir=paths.crops_dir,
            frame_skip=config.frame_skip,
            crop_save_interval=config.crop_save_interval,
            max_frames=config.max_frames,
            frame_callback=self.person_detection_frame_callback,
        )
        log_queue.join()

        track_dirs = [
            path for path in paths.crops_dir.glob("track_*") if path.is_dir()
        ]
        crop_files = [
            path for path in paths.crops_dir.glob("track_*/*.jpg") if path.is_file()
        ]
        summary: Dict[str, object] = {
            "source": str(config.source),
            "output_dir": str(paths.crops_dir),
            "track_dir_count": len(track_dirs),
            "crop_file_count": len(crop_files),
        }
        return StageRunResult(
            name="person_detection",
            enabled=True,
            input_path=str(config.source),
            output_path=str(paths.crops_dir),
            summary=summary,
        )

    def _run_region_extraction_stage(
        self,
        paths: PipelinePaths,
        person_result: Optional[StageRunResult],
    ) -> StageRunResult:
        config = replace(self.config.person_region_extraction)
        config.output_root = str(paths.region_output_root)
        if person_result is not None and person_result.output_path:
            config.input_crops_dir = person_result.output_path
            config.process_latest_job = False

        summary = process_crop_tree(config)
        return StageRunResult(
            name="person_region_extraction",
            enabled=True,
            input_path=str(summary["input_crops_dir"]),
            output_path=str(summary["output_root"]),
            summary=summary,
        )

    def _run_facemask_stage(
        self,
        paths: PipelinePaths,
        region_result: Optional[StageRunResult],
    ) -> StageRunResult:
        config = replace(self.config.facemask_detection)
        config.output_root = str(paths.facemask_output_root)
        if region_result is not None and region_result.output_path:
            config.input_head_face_dir = region_result.output_path
            config.process_latest_job = False

        summary = process_head_face_tree(config)
        return StageRunResult(
            name="facemask_detection",
            enabled=True,
            input_path=str(summary["input_root"]),
            output_path=str(summary["output_root"]),
            summary=summary,
        )

    def _run_glove_stage(
        self,
        paths: PipelinePaths,
        region_result: Optional[StageRunResult],
    ) -> StageRunResult:
        config = replace(self.config.glove_detection)
        config.output_root = str(paths.glove_output_root)
        if region_result is not None and region_result.output_path:
            config.input_hands_dir = region_result.output_path
            config.process_latest_job = False

        summary = process_hands_tree(config)
        return StageRunResult(
            name="glove_detection",
            enabled=True,
            input_path=str(summary["input_root"]),
            output_path=str(summary["output_root"]),
            summary=summary,
        )

    def _run_behavior_stage(
        self,
        paths: PipelinePaths,
    ) -> StageRunResult:
        config = replace(self.config.behavior_detection)
        config.source = self.config.source
        config.output_root = str(paths.behavior_output_root)

        summary = BehaviorDetectionPipeline(config).process_video(
            source=config.source,
            output_root=config.output_root,
        )
        return StageRunResult(
            name="behavior_detection",
            enabled=True,
            input_path=str(config.source),
            output_path=str(summary["output_root"]),
            summary=summary,
        )

    def run(self) -> Dict[str, object]:
        job_name = build_job_name(self.config.source, self.config.job_name)
        paths = build_pipeline_paths(self.config.output_root, job_name)
        paths.run_root.mkdir(parents=True, exist_ok=True)

        summary = PipelineRunSummary(
            job_name=job_name,
            source=str(self.config.source),
            output_root=str(paths.run_root),
            status="running",
            paths=paths,
        )
        self._write_summary(summary)

        try:
            person_result: Optional[StageRunResult] = None
            region_result: Optional[StageRunResult] = None

            if self.config.run_person_detection:
                self._notify_stage("person_detection", "running")
                try:
                    person_result = self._run_person_detection_stage(paths)
                except Exception as exc:
                    failed_result = StageRunResult(
                        name="person_detection",
                        enabled=True,
                        success=False,
                        input_path=str(self.config.source),
                        output_path=str(paths.crops_dir),
                        error=str(exc),
                    )
                    summary.stages["person_detection"] = failed_result
                    self._notify_stage("person_detection", "failed", failed_result)
                    raise
                summary.stages[person_result.name] = person_result
                self._notify_stage("person_detection", "completed", person_result)
                self._write_summary(summary)
            else:
                summary.stages["person_detection"] = StageRunResult(
                    name="person_detection",
                    enabled=False,
                )
                self._notify_stage(
                    "person_detection",
                    "disabled",
                    summary.stages["person_detection"],
                )

            if self.config.run_region_extraction:
                self._notify_stage("person_region_extraction", "running")
                try:
                    region_result = self._run_region_extraction_stage(paths, person_result)
                except Exception as exc:
                    failed_result = StageRunResult(
                        name="person_region_extraction",
                        enabled=True,
                        success=False,
                        input_path=person_result.output_path if person_result else None,
                        output_path=str(paths.region_output_root),
                        error=str(exc),
                    )
                    summary.stages["person_region_extraction"] = failed_result
                    self._notify_stage(
                        "person_region_extraction",
                        "failed",
                        failed_result,
                    )
                    raise
                summary.stages[region_result.name] = region_result
                self._notify_stage(
                    "person_region_extraction",
                    "completed",
                    region_result,
                )
                self._write_summary(summary)
            else:
                summary.stages["person_region_extraction"] = StageRunResult(
                    name="person_region_extraction",
                    enabled=False,
                )
                self._notify_stage(
                    "person_region_extraction",
                    "disabled",
                    summary.stages["person_region_extraction"],
                )

            if self.config.run_facemask_detection:
                self._notify_stage("facemask_detection", "running")
                try:
                    facemask_result = self._run_facemask_stage(paths, region_result)
                except Exception as exc:
                    failed_result = StageRunResult(
                        name="facemask_detection",
                        enabled=True,
                        success=False,
                        input_path=region_result.output_path if region_result else None,
                        output_path=str(paths.facemask_output_root),
                        error=str(exc),
                    )
                    summary.stages["facemask_detection"] = failed_result
                    self._notify_stage(
                        "facemask_detection",
                        "failed",
                        failed_result,
                    )
                    raise
                summary.stages[facemask_result.name] = facemask_result
                self._notify_stage(
                    "facemask_detection",
                    "completed",
                    facemask_result,
                )
                self._write_summary(summary)
            else:
                summary.stages["facemask_detection"] = StageRunResult(
                    name="facemask_detection",
                    enabled=False,
                )
                self._notify_stage(
                    "facemask_detection",
                    "disabled",
                    summary.stages["facemask_detection"],
                )

            if self.config.run_glove_detection:
                self._notify_stage("glove_detection", "running")
                try:
                    glove_result = self._run_glove_stage(paths, region_result)
                except Exception as exc:
                    failed_result = StageRunResult(
                        name="glove_detection",
                        enabled=True,
                        success=False,
                        input_path=region_result.output_path if region_result else None,
                        output_path=str(paths.glove_output_root),
                        error=str(exc),
                    )
                    summary.stages["glove_detection"] = failed_result
                    self._notify_stage("glove_detection", "failed", failed_result)
                    raise
                summary.stages[glove_result.name] = glove_result
                self._notify_stage("glove_detection", "completed", glove_result)
                self._write_summary(summary)
            else:
                summary.stages["glove_detection"] = StageRunResult(
                    name="glove_detection",
                    enabled=False,
                )
                self._notify_stage(
                    "glove_detection",
                    "disabled",
                    summary.stages["glove_detection"],
                )

            if self.config.run_behavior_detection:
                self._notify_stage("behavior_detection", "running")
                try:
                    behavior_result = self._run_behavior_stage(paths)
                except Exception as exc:
                    failed_result = StageRunResult(
                        name="behavior_detection",
                        enabled=True,
                        success=False,
                        input_path=str(self.config.source),
                        output_path=str(paths.behavior_output_root),
                        error=str(exc),
                    )
                    summary.stages["behavior_detection"] = failed_result
                    self._notify_stage("behavior_detection", "failed", failed_result)
                    raise
                summary.stages[behavior_result.name] = behavior_result
                self._notify_stage(
                    "behavior_detection",
                    "completed",
                    behavior_result,
                )
                self._write_summary(summary)
            else:
                summary.stages["behavior_detection"] = StageRunResult(
                    name="behavior_detection",
                    enabled=False,
                )
                self._notify_stage(
                    "behavior_detection",
                    "disabled",
                    summary.stages["behavior_detection"],
                )

            summary.status = "completed"
        except Exception as exc:
            summary.status = "failed"
            summary.error = str(exc)
            self._write_summary(summary)
            raise

        self._write_summary(summary)
        return summary.to_dict()
