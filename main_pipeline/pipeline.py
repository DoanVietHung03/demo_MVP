from __future__ import annotations

import json
import re
import time
from dataclasses import replace
from pathlib import Path
from typing import Callable, Dict, Optional, Union

import cv2
import numpy as np

from main_pipeline.config import MainPipelineConfig
from main_pipeline.types import PipelinePaths
from person_detection.config import PersonInstance
from stream_pipeline.config import StreamPipelineConfig
from stream_pipeline.pipeline import StreamPipeline

StageCallback = Callable[[str, str, Optional[object]], None]
FrameCallback = Callable[[int, Optional[int], np.ndarray, list[PersonInstance]], None]


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
        ppe_output_root=run_root / "ppe_detection",
        region_output_root=run_root / "person_region_extraction",
        facemask_output_root=run_root / "facemask_detection",
        glove_output_root=run_root / "glove_detection",
        behavior_output_root=run_root / "behavior_detection",
        video_output_root=run_root / "videos",
        summary_path=run_root / "summary.json",
    )


def _normalize_stage_selection(config: MainPipelineConfig) -> Dict[str, bool]:
    stage_flags = {
        "run_person_detection": bool(config.run_person_detection),
        "run_ppe_detection": bool(config.run_ppe_detection),
        "run_region_extraction": bool(config.run_region_extraction),
        "run_facemask_detection": bool(config.run_facemask_detection),
        "run_glove_detection": bool(config.run_glove_detection),
        "run_behavior_detection": bool(config.run_behavior_detection),
    }

    if stage_flags["run_facemask_detection"] or stage_flags["run_glove_detection"]:
        stage_flags["run_region_extraction"] = True
    if stage_flags["run_region_extraction"]:
        stage_flags["run_person_detection"] = True

    if not any(stage_flags.values()):
        raise ValueError("At least one offline stage must be enabled.")

    return stage_flags


class OfflineVideoRecorder:
    def __init__(
        self,
        source: Union[str, int],
        output_path: Path,
        codec: str,
        fps: Optional[float] = None,
    ) -> None:
        self.source = source
        self.output_path = output_path
        self.codec = (codec or "mp4v")[:4]
        self.requested_fps = fps
        self.writer: Optional[cv2.VideoWriter] = None
        self.resolved_fps: Optional[float] = None
        self.frame_count = 0

    def _probe_source_fps(self) -> float:
        cap = cv2.VideoCapture(self.source)
        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS)) if cap.isOpened() else 0.0
        finally:
            cap.release()
        if fps <= 1.0:
            return 25.0
        return fps

    def _ensure_writer(self, frame: np.ndarray) -> None:
        if self.writer is not None:
            return

        frame_height, frame_width = frame.shape[:2]
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        fps = float(self.requested_fps) if self.requested_fps else self._probe_source_fps()
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            fps,
            (frame_width, frame_height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Cannot open video writer: {self.output_path}")
        self.writer = writer
        self.resolved_fps = fps

    def write_frame(self, frame: np.ndarray) -> None:
        self._ensure_writer(frame)
        assert self.writer is not None
        self.writer.write(frame)
        self.frame_count += 1

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None


class MainPipeline:
    def __init__(
        self,
        config: MainPipelineConfig,
        stage_callback: Optional[StageCallback] = None,
        person_detection_frame_callback: Optional[FrameCallback] = None,
    ) -> None:
        self.config = config
        self.stage_callback = stage_callback
        self.frame_callback = person_detection_frame_callback

    def _build_stream_config(self) -> tuple[StreamPipelineConfig, Dict[str, bool]]:
        stage_flags = _normalize_stage_selection(self.config)
        stream_config = StreamPipelineConfig(
            source=self.config.source,
            output_root=self.config.output_root,
            job_name=self.config.job_name,
            run_person_detection=stage_flags["run_person_detection"],
            run_ppe_detection=stage_flags["run_ppe_detection"],
            run_region_extraction=stage_flags["run_region_extraction"],
            run_facemask_detection=stage_flags["run_facemask_detection"],
            run_glove_detection=stage_flags["run_glove_detection"],
            run_behavior_detection=stage_flags["run_behavior_detection"],
            attribute_refresh_interval_frames=int(
                self.config.attribute_refresh_interval_frames
            ),
            summary_flush_interval_frames=int(self.config.summary_flush_interval_frames),
            person_detection=replace(self.config.person_detection),
            ppe_detection=replace(self.config.ppe_detection),
            person_region_extraction=replace(self.config.person_region_extraction),
            facemask_detection=replace(self.config.facemask_detection),
            glove_detection=replace(self.config.glove_detection),
            behavior_detection=replace(self.config.behavior_detection),
        )
        return stream_config, stage_flags

    def run(self) -> Dict[str, object]:
        if not str(self.config.source).strip():
            raise ValueError("Offline pipeline source must not be empty.")

        stream_config, stage_flags = self._build_stream_config()
        job_name = build_job_name(stream_config.source, stream_config.job_name)
        paths = build_pipeline_paths(stream_config.output_root, job_name)

        video_recorder: Optional[OfflineVideoRecorder] = None
        if self.config.save_annotated_video:
            annotated_video_path = (
                paths.video_output_root / self.config.annotated_video_filename
            )
            video_recorder = OfflineVideoRecorder(
                source=stream_config.source,
                output_path=annotated_video_path,
                codec=self.config.annotated_video_codec,
                fps=self.config.annotated_video_fps,
            )

        def combined_frame_callback(
            frame_index: int,
            total_frames: Optional[int],
            annotated_frame: np.ndarray,
            persons: list[PersonInstance],
        ) -> None:
            if video_recorder is not None:
                video_recorder.write_frame(annotated_frame)
            if self.frame_callback is not None:
                self.frame_callback(frame_index, total_frames, annotated_frame, persons)

        summary: Optional[Dict[str, object]] = None
        try:
            summary = StreamPipeline(
                stream_config,
                stage_callback=self.stage_callback,
                frame_callback=combined_frame_callback,
            ).run()
        finally:
            if video_recorder is not None:
                video_recorder.close()

        paths_payload = dict(summary.get("paths", {}))
        paths_payload["video_output_root"] = str(paths.video_output_root)
        summary["paths"] = paths_payload
        summary["pipeline_mode"] = "offline"
        summary["requested_stage_flags"] = {
            "run_person_detection": bool(self.config.run_person_detection),
            "run_ppe_detection": bool(self.config.run_ppe_detection),
            "run_region_extraction": bool(self.config.run_region_extraction),
            "run_facemask_detection": bool(self.config.run_facemask_detection),
            "run_glove_detection": bool(self.config.run_glove_detection),
            "run_behavior_detection": bool(self.config.run_behavior_detection),
        }
        summary["effective_stage_flags"] = dict(stage_flags)

        if video_recorder is not None and video_recorder.frame_count > 0:
            summary["annotated_video"] = {
                "path": str(paths.video_output_root / self.config.annotated_video_filename),
                "frame_count": int(video_recorder.frame_count),
                "fps": float(video_recorder.resolved_fps or 25.0),
                "codec": self.config.annotated_video_codec,
            }
        else:
            summary["annotated_video"] = None

        summary_path = Path(str(summary["paths"]["summary_path"]))
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary
