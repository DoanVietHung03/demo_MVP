from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from behavior_detection.pipeline import (
    BehaviorDetectionPipeline,
    TrackedPersonBehavior,
    annotate_behavior_evidence,
)
from facemask_detection.inference import (
    describe_facemask_inference_backend,
    load_facemask_inference_engine,
    run_facemask_inference,
)
from facemask_detection.pipeline import annotate_image as annotate_facemask_image
from glove_detection.inference import (
    GloveClassification,
    classify_glove_image,
    describe_glove_classifier_backend,
    load_tflite_glove_classifier,
)
from glove_detection.pipeline import (
    annotate_image as annotate_glove_image,
    resolve_final_label,
)
from ppe_detection.pipeline import (
    PPEDetectionPipeline,
    annotate_image as annotate_ppe_image,
)
from person_detection.config import PersonInstance
from person_detection.pipeline import PersonDetectionPipeline, enqueue_log_item, log_queue
from person_region_extraction.pipeline import PersonRegionExtractor
from stream_pipeline.config import StreamPipelineConfig
from stream_pipeline.types import PipelinePaths, PipelineRunSummary, StageRunResult


StageCallback = Callable[[str, str, Optional[StageRunResult]], None]
FrameCallback = Callable[[int, Optional[int], np.ndarray, List[PersonInstance]], None]
ControlCallback = Callable[[], bool]


class StreamPipelineCancelled(Exception):
    pass


@dataclass
class HandPrediction:
    bbox_xyxy: Tuple[int, int, int, int]
    label: str
    confidence: float


@dataclass
class TrackAttributeSnapshot:
    tracker_id: int
    last_updated_frame_index: int
    mask_label: Optional[str] = None
    mask_confidence: float = 0.0
    mask_bbox_xyxy: Optional[Tuple[int, int, int, int]] = None
    glove_label: Optional[str] = None
    glove_confidence: float = 0.0
    hand_predictions: List[HandPrediction] = field(default_factory=list)


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
        summary_path=run_root / "summary.json",
    )


def _looks_like_stream_source(source: Union[str, Path, int]) -> bool:
    return isinstance(source, str) and "://" in source


def _configure_capture_buffer(
    cap: cv2.VideoCapture,
    buffer_size: Optional[int],
) -> None:
    if buffer_size is None or not hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
        return
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, int(buffer_size))
    except Exception:
        return


class DirectFrameReader:
    def __init__(
        self,
        source: Union[str, Path, int],
        buffer_size: Optional[int] = None,
    ) -> None:
        self.cap = cv2.VideoCapture(source)
        _configure_capture_buffer(self.cap, buffer_size)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")
        total_frames_raw = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.total_frames = total_frames_raw if total_frames_raw > 0 else None
        self._frame_index = -1

    def read_next(
        self,
        timeout_seconds: float = 0.0,
    ) -> Tuple[str, Optional[int], Optional[np.ndarray]]:
        _ = timeout_seconds
        ok, frame = self.cap.read()
        if not ok:
            return "eof", None, None
        self._frame_index += 1
        return "frame", self._frame_index, frame

    def close(self) -> None:
        self.cap.release()


class LatestFrameReader:
    def __init__(
        self,
        source: Union[str, Path, int],
        buffer_size: Optional[int] = 1,
    ) -> None:
        self.cap = cv2.VideoCapture(source)
        _configure_capture_buffer(self.cap, buffer_size)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")
        total_frames_raw = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.total_frames = total_frames_raw if total_frames_raw > 0 else None

        self._condition = threading.Condition()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_frame_index = -1
        self._last_delivered_index = -1
        self._finished = False
        self._stop_requested = False
        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            daemon=True,
        )
        self._reader_thread.start()

    def _reader_loop(self) -> None:
        frame_index = -1
        try:
            while True:
                with self._condition:
                    if self._stop_requested:
                        self._finished = True
                        self._condition.notify_all()
                        return

                ok, frame = self.cap.read()
                if not ok:
                    with self._condition:
                        self._finished = True
                        self._condition.notify_all()
                    return

                frame_index += 1
                with self._condition:
                    self._latest_frame = frame
                    self._latest_frame_index = frame_index
                    self._condition.notify_all()
        finally:
            self.cap.release()

    def read_next(
        self,
        timeout_seconds: float = 0.2,
    ) -> Tuple[str, Optional[int], Optional[np.ndarray]]:
        deadline = time.time() + max(0.0, float(timeout_seconds))
        with self._condition:
            while True:
                if (
                    self._latest_frame is not None
                    and self._latest_frame_index > self._last_delivered_index
                ):
                    self._last_delivered_index = self._latest_frame_index
                    return (
                        "frame",
                        self._latest_frame_index,
                        self._latest_frame.copy(),
                    )
                if self._finished:
                    return "eof", None, None
                remaining = deadline - time.time()
                if remaining <= 0:
                    return "timeout", None, None
                self._condition.wait(timeout=remaining)

    def close(self) -> None:
        with self._condition:
            self._stop_requested = True
            self._condition.notify_all()
        try:
            self.cap.release()
        except Exception:
            pass
        if self._reader_thread.is_alive():
            self._reader_thread.join(timeout=1.0)


def _draw_text_chip(
    image: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    text_color: Tuple[int, int, int] = (255, 255, 255),
    background_color: Tuple[int, int, int] = (24, 24, 24),
) -> None:
    x_coord, y_coord = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        font,
        scale,
        thickness,
    )
    top_left = (
        max(0, x_coord - 4),
        max(0, y_coord - text_height - baseline - 4),
    )
    bottom_right = (
        min(image.shape[1], x_coord + text_width + 4),
        min(image.shape[0], y_coord + baseline + 4),
    )
    cv2.rectangle(image, top_left, bottom_right, background_color, thickness=-1)
    cv2.putText(
        image,
        text,
        (x_coord, min(image.shape[0] - 4, y_coord)),
        font,
        scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )


def _annotate_person_attributes(
    image: np.ndarray,
    persons: List[PersonInstance],
    track_snapshots: Dict[int, TrackAttributeSnapshot],
) -> np.ndarray:
    annotated = image.copy()
    frame_height = annotated.shape[0]

    for person in persons:
        snapshot = track_snapshots.get(person.tracker_id)
        if snapshot is None:
            continue

        tokens: List[str] = []
        if snapshot.mask_label:
            if snapshot.mask_label == "NoDetection":
                tokens.append("Face:NoDet")
            else:
                tokens.append(f"Face:{snapshot.mask_label} {snapshot.mask_confidence:.2f}")
        if snapshot.glove_label:
            tokens.append(f"Glove:{snapshot.glove_label} {snapshot.glove_confidence:.2f}")
        if not tokens:
            continue

        text = " | ".join(tokens)
        x1, y1, _, _ = person.bbox_xyxy
        text_y = min(frame_height - 6, max(26, y1 + 34))
        _draw_text_chip(
            annotated,
            text,
            (max(4, x1 + 4), text_y),
            background_color=(0, 70, 44),
        )

        if snapshot.mask_bbox_xyxy is not None:
            fx1, fy1, fx2, fy2 = snapshot.mask_bbox_xyxy
            color = (0, 200, 0) if snapshot.mask_label == "Mask" else (0, 0, 255)
            cv2.rectangle(annotated, (fx1, fy1), (fx2, fy2), color, 2)
            if snapshot.mask_label:
                _draw_text_chip(
                    annotated,
                    f"{snapshot.mask_label} {snapshot.mask_confidence:.2f}",
                    (fx1 + 2, max(18, fy1 - 4)),
                    background_color=color,
                )

        for hand_prediction in snapshot.hand_predictions:
            hx1, hy1, hx2, hy2 = hand_prediction.bbox_xyxy
            if hand_prediction.label == "Glove":
                color = (0, 200, 0)
            elif hand_prediction.label == "NoGlove":
                color = (0, 0, 255)
            else:
                color = (0, 215, 255)
            cv2.rectangle(annotated, (hx1, hy1), (hx2, hy2), color, 2)
            _draw_text_chip(
                annotated,
                f"Hand:{hand_prediction.label} {hand_prediction.confidence:.2f}",
                (hx1 + 2, min(frame_height - 8, hy2 + 16)),
                background_color=color,
            )

    return annotated


def _summarize_behavior_lines(
    behaviors: List[TrackedPersonBehavior],
    emitted_event_count: int,
) -> List[str]:
    lines: List[str] = []
    if emitted_event_count > 0:
        lines.append(f"Behavior events: {emitted_event_count}")

    for behavior in behaviors:
        tokens: List[str] = []
        if behavior.smoking_candidate:
            tokens.append("SMOKE")
        if behavior.phone_usage:
            tokens.append("PHONE_USE")
        elif behavior.phone_in_hand:
            tokens.append("PHONE_HAND")
        elif behavior.phone_present:
            tokens.append("PHONE")
        if not tokens:
            continue
        lines.append(f"Track {behavior.tracker_id}: {'/'.join(tokens)}")
        if len(lines) >= 4:
            break
    return lines


def _overlay_behavior_summary(
    image: np.ndarray,
    behaviors: List[TrackedPersonBehavior],
    emitted_event_count: int,
) -> np.ndarray:
    annotated = image.copy()
    lines = _summarize_behavior_lines(behaviors, emitted_event_count)
    if not lines:
        return annotated

    origin_y = 24
    for line in lines:
        _draw_text_chip(
            annotated,
            line,
            (12, origin_y),
            background_color=(66, 40, 5),
        )
        origin_y += 24
    return annotated


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )


def _offset_box(
    box: Tuple[int, int, int, int],
    offset_x: int,
    offset_y: int,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        int(round(offset_x + x1 * scale_x)),
        int(round(offset_y + y1 * scale_y)),
        int(round(offset_x + x2 * scale_x)),
        int(round(offset_y + y2 * scale_y)),
    )


class StreamPipeline:
    def __init__(
        self,
        config: StreamPipelineConfig,
        stage_callback: Optional[StageCallback] = None,
        frame_callback: Optional[FrameCallback] = None,
        should_pause: Optional[ControlCallback] = None,
        should_cancel: Optional[ControlCallback] = None,
    ) -> None:
        self.config = config
        self.stage_callback = stage_callback
        self.frame_callback = frame_callback
        self.should_pause = should_pause
        self.should_cancel = should_cancel

    def _validate_stage_dependencies(self) -> None:
        if (
            self.config.run_region_extraction
            or self.config.run_facemask_detection
            or self.config.run_glove_detection
        ) and not self.config.run_person_detection:
            raise ValueError(
                "Realtime stream pipeline requires person_detection before region/facemask/glove stages."
            )
        if (
            self.config.run_facemask_detection
            or self.config.run_glove_detection
        ) and not self.config.run_region_extraction:
            raise ValueError(
                "Realtime facemask/glove stages require person_region_extraction to be enabled."
            )
        if not any(
            [
                self.config.run_person_detection,
                self.config.run_ppe_detection,
                self.config.run_region_extraction,
                self.config.run_facemask_detection,
                self.config.run_glove_detection,
                self.config.run_behavior_detection,
            ]
        ):
            raise ValueError("At least one realtime stage must be enabled.")

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

    def _update_stage_summary(
        self,
        summary: PipelineRunSummary,
        stage_name: str,
        stage_summary: Dict[str, object],
    ) -> None:
        stage_result = summary.stages.get(stage_name)
        if stage_result is None:
            return
        stage_result.summary = dict(stage_summary)

    def _cancel_requested(self) -> bool:
        return bool(self.should_cancel is not None and self.should_cancel())

    def _wait_while_paused(self) -> None:
        while self.should_pause is not None and self.should_pause():
            if self._cancel_requested():
                raise StreamPipelineCancelled("Job cancelled by user.")
            time.sleep(0.05)

    def run(self) -> Dict[str, object]:
        self._validate_stage_dependencies()

        job_name = build_job_name(self.config.source, self.config.job_name)
        paths = build_pipeline_paths(self.config.output_root, job_name)
        paths.run_root.mkdir(parents=True, exist_ok=True)

        summary = PipelineRunSummary(
            job_name=job_name,
            source=str(self.config.source),
            output_root=str(paths.run_root),
            status="initializing",
            paths=paths,
        )
        self._write_summary(summary)

        attribute_refresh_interval = max(
            1,
            int(self.config.attribute_refresh_interval_frames),
        )
        summary_flush_interval = max(
            1,
            int(self.config.summary_flush_interval_frames),
        )
        crop_save_interval = max(
            1,
            int(self.config.person_detection.crop_save_interval),
        )
        save_artifacts = bool(self.config.save_artifacts)
        use_latest_frame_reader = bool(
            self.config.use_latest_frame_reader_for_streams
            and _looks_like_stream_source(self.config.source)
        )
        capture_mode = "latest-frame-thread" if use_latest_frame_reader else "direct"

        person_pipeline: Optional[PersonDetectionPipeline] = None
        ppe_pipeline: Optional[PPEDetectionPipeline] = None
        region_extractor: Optional[PersonRegionExtractor] = None
        facemask_engine = None
        glove_classifier = None
        behavior_pipeline: Optional[BehaviorDetectionPipeline] = None

        person_summary: Dict[str, object] = {}
        ppe_summary: Dict[str, object] = {}
        region_summary: Dict[str, object] = {}
        facemask_summary: Dict[str, object] = {}
        glove_summary: Dict[str, object] = {}
        behavior_summary: Dict[str, object] = {}

        started_stage_names: List[str] = []
        track_snapshots: Dict[int, TrackAttributeSnapshot] = {}
        track_last_seen: Dict[int, int] = {}
        seen_track_ids: set[int] = set()

        frame_reader: Optional[Union[DirectFrameReader, LatestFrameReader]] = None
        ppe_detections_file = None
        behavior_events_file = None
        try:
            if self.config.run_person_detection:
                person_pipeline = PersonDetectionPipeline(
                    model_path=self.config.person_detection.model_path,
                    device=self.config.person_detection.device,
                    confidence_threshold=self.config.person_detection.confidence_threshold,
                    iou_threshold=self.config.person_detection.iou_threshold,
                    crop_padding=self.config.person_detection.crop_padding,
                    input_resize_scale=self.config.person_detection.input_resize_scale,
                    track_ttl_frames=self.config.person_detection.track_ttl_frames,
                    alert_cooldown_frames=self.config.person_detection.alert_cooldown_frames,
                )
                person_summary = {
                    "source": str(self.config.source),
                    "output_dir": str(paths.crops_dir),
                    "runtime_backend": person_pipeline.runtime_backend_label(),
                    "model_path": self.config.person_detection.model_path,
                    "capture_mode": capture_mode,
                    "save_artifacts": save_artifacts,
                    "processed_frames": 0,
                    "crop_file_count": 0,
                    "track_dir_count": 0,
                    "alert_count": 0,
                }
                summary.stages["person_detection"] = StageRunResult(
                    name="person_detection",
                    enabled=True,
                    input_path=str(self.config.source),
                    output_path=str(paths.crops_dir),
                    summary=dict(person_summary),
                )
                print(
                    "[Stage Runtime] "
                    f"person_detection(stream): {person_summary['runtime_backend']} | "
                    f"device={person_pipeline.device} | "
                    f"model={self.config.person_detection.model_path}"
                )
                self._notify_stage(
                    "person_detection",
                    "running",
                    summary.stages["person_detection"],
                )
                started_stage_names.append("person_detection")
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

            if self.config.run_ppe_detection:
                if save_artifacts:
                    paths.ppe_output_root.mkdir(parents=True, exist_ok=True)
                ppe_pipeline = PPEDetectionPipeline(self.config.ppe_detection)
                ppe_summary = {
                    "source": str(self.config.source),
                    "output_root": str(paths.ppe_output_root),
                    "detections_path": str(paths.ppe_output_root / "detections.jsonl"),
                    "runtime_backend": ppe_pipeline.runtime_backend_label(),
                    "model_path": self.config.ppe_detection.model_path,
                    "capture_mode": capture_mode,
                    "save_artifacts": save_artifacts,
                    "processed_frames": 0,
                    "emitted_detections": 0,
                    "class_counts": {
                        class_name: 0
                        for class_name in self.config.ppe_detection.target_class_names
                    },
                }
                summary.stages["ppe_detection"] = StageRunResult(
                    name="ppe_detection",
                    enabled=True,
                    input_path=str(self.config.source),
                    output_path=str(paths.ppe_output_root),
                    summary=dict(ppe_summary),
                )
                print(
                    "[Stage Runtime] "
                    f"ppe_detection(stream): {ppe_summary['runtime_backend']} | "
                    f"device={ppe_pipeline.inference_engine.device} | "
                    f"model={self.config.ppe_detection.model_path}"
                )
                self._notify_stage(
                    "ppe_detection",
                    "running",
                    summary.stages["ppe_detection"],
                )
                started_stage_names.append("ppe_detection")
                if save_artifacts:
                    ppe_detections_file = (
                        paths.ppe_output_root / "detections.jsonl"
                    ).open("w", encoding="utf-8")
            else:
                summary.stages["ppe_detection"] = StageRunResult(
                    name="ppe_detection",
                    enabled=False,
                )
                self._notify_stage(
                    "ppe_detection",
                    "disabled",
                    summary.stages["ppe_detection"],
                )

            if self.config.run_region_extraction:
                region_extractor = PersonRegionExtractor(self.config.person_region_extraction)
                if save_artifacts:
                    paths.region_output_root.mkdir(parents=True, exist_ok=True)
                region_summary = {
                    "input_stream": str(self.config.source),
                    "output_root": str(paths.region_output_root),
                    "runtime_backend": region_extractor.runtime_backend_label(),
                    "model_path": self.config.person_region_extraction.model_path,
                    "capture_mode": capture_mode,
                    "save_artifacts": save_artifacts,
                    "processed_count": 0,
                    "saved_count": 0,
                    "attribute_refresh_interval_frames": attribute_refresh_interval,
                }
                summary.stages["person_region_extraction"] = StageRunResult(
                    name="person_region_extraction",
                    enabled=True,
                    input_path=str(paths.crops_dir),
                    output_path=str(paths.region_output_root),
                    summary=dict(region_summary),
                )
                print(
                    "[Stage Runtime] "
                    f"person_region_extraction(stream): {region_summary['runtime_backend']} | "
                    f"device={region_extractor.device} | "
                    f"model={self.config.person_region_extraction.model_path}"
                )
                self._notify_stage(
                    "person_region_extraction",
                    "running",
                    summary.stages["person_region_extraction"],
                )
                started_stage_names.append("person_region_extraction")
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
                if save_artifacts:
                    paths.facemask_output_root.mkdir(parents=True, exist_ok=True)
                facemask_engine = load_facemask_inference_engine(
                    self.config.facemask_detection
                )
                facemask_summary = {
                    "input_stream": str(self.config.source),
                    "output_root": str(paths.facemask_output_root),
                    "runtime_backend": describe_facemask_inference_backend(
                        facemask_engine
                    ),
                    "model_path": self.config.facemask_detection.model_path,
                    "capture_mode": capture_mode,
                    "save_artifacts": save_artifacts,
                    "processed_count": 0,
                    "mask_count": 0,
                    "no_mask_count": 0,
                    "no_detection_count": 0,
                    "attribute_refresh_interval_frames": attribute_refresh_interval,
                }
                summary.stages["facemask_detection"] = StageRunResult(
                    name="facemask_detection",
                    enabled=True,
                    input_path=str(paths.region_output_root),
                    output_path=str(paths.facemask_output_root),
                    summary=dict(facemask_summary),
                )
                print(
                    "[Stage Runtime] "
                    f"facemask_detection(stream): {facemask_summary['runtime_backend']} | "
                    f"device={facemask_engine.device} | "
                    f"model={self.config.facemask_detection.model_path}"
                )
                self._notify_stage(
                    "facemask_detection",
                    "running",
                    summary.stages["facemask_detection"],
                )
                started_stage_names.append("facemask_detection")
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
                if save_artifacts:
                    paths.glove_output_root.mkdir(parents=True, exist_ok=True)
                glove_classifier = load_tflite_glove_classifier(
                    self.config.glove_detection
                )
                glove_summary = {
                    "input_stream": str(self.config.source),
                    "output_root": str(paths.glove_output_root),
                    "runtime_backend": describe_glove_classifier_backend(
                        glove_classifier
                    ),
                    "model_path": self.config.glove_detection.model_path,
                    "capture_mode": capture_mode,
                    "save_artifacts": save_artifacts,
                    "processed_count": 0,
                    "glove_count": 0,
                    "no_glove_count": 0,
                    "uncertain_count": 0,
                    "attribute_refresh_interval_frames": attribute_refresh_interval,
                }
                summary.stages["glove_detection"] = StageRunResult(
                    name="glove_detection",
                    enabled=True,
                    input_path=str(paths.region_output_root),
                    output_path=str(paths.glove_output_root),
                    summary=dict(glove_summary),
                )
                print(
                    "[Stage Runtime] "
                    f"glove_detection(stream): {glove_summary['runtime_backend']} | "
                    f"model={self.config.glove_detection.model_path}"
                )
                self._notify_stage(
                    "glove_detection",
                    "running",
                    summary.stages["glove_detection"],
                )
                started_stage_names.append("glove_detection")
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
                if save_artifacts:
                    paths.behavior_output_root.mkdir(parents=True, exist_ok=True)
                behavior_pipeline = BehaviorDetectionPipeline(self.config.behavior_detection)
                behavior_summary = {
                    "source": str(self.config.source),
                    "output_root": str(paths.behavior_output_root),
                    "events_path": str(paths.behavior_output_root / "events.jsonl"),
                    "runtime_backends": behavior_pipeline.runtime_backend_summary(),
                    "capture_mode": capture_mode,
                    "save_artifacts": save_artifacts,
                    "model_paths": {
                        "person_phone": self.config.behavior_detection.person_phone_model_path,
                        "pose": self.config.behavior_detection.pose_model_path,
                        "smoke": self.config.behavior_detection.smoke_model_path,
                    },
                    "processed_frames": 0,
                    "emitted_events": 0,
                }
                summary.stages["behavior_detection"] = StageRunResult(
                    name="behavior_detection",
                    enabled=True,
                    input_path=str(self.config.source),
                    output_path=str(paths.behavior_output_root),
                    summary=dict(behavior_summary),
                )
                print(
                    "[Stage Runtime] "
                    "behavior_detection(stream): "
                    f"shared={behavior_summary['runtime_backends']['person_phone']} | "
                    f"pose={behavior_summary['runtime_backends']['pose']} | "
                    f"smoke={behavior_summary['runtime_backends']['smoke']} | "
                    f"device={self.config.behavior_detection.device}"
                )
                self._notify_stage(
                    "behavior_detection",
                    "running",
                    summary.stages["behavior_detection"],
                )
                started_stage_names.append("behavior_detection")
                if save_artifacts:
                    behavior_events_file = (
                        paths.behavior_output_root / "events.jsonl"
                    ).open("w", encoding="utf-8")
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

            summary.status = "running"
            self._write_summary(summary)

            if self.config.run_person_detection:
                primary_frame_skip = int(self.config.person_detection.frame_skip)
                primary_max_frames = self.config.person_detection.max_frames
            elif self.config.run_ppe_detection:
                primary_frame_skip = int(self.config.ppe_detection.frame_skip)
                primary_max_frames = self.config.ppe_detection.max_frames
            else:
                primary_frame_skip = int(self.config.behavior_detection.frame_skip)
                primary_max_frames = self.config.behavior_detection.max_frames

            if use_latest_frame_reader:
                frame_reader = LatestFrameReader(
                    self.config.source,
                    buffer_size=self.config.capture_buffer_size,
                )
            else:
                frame_reader = DirectFrameReader(
                    self.config.source,
                    buffer_size=self.config.capture_buffer_size,
                )

            total_frames = frame_reader.total_frames

            while True:
                self._wait_while_paused()
                if self._cancel_requested():
                    raise StreamPipelineCancelled("Job cancelled by user.")
                read_status, frame_index_raw, frame = frame_reader.read_next(
                    timeout_seconds=float(self.config.capture_poll_timeout_seconds),
                )
                if read_status == "timeout":
                    continue
                if read_status == "eof":
                    break
                if frame_index_raw is None or frame is None:
                    continue
                frame_index = int(frame_index_raw)
                if primary_max_frames is not None and frame_index >= primary_max_frames:
                    break
                if primary_frame_skip > 0 and frame_index % (primary_frame_skip + 1) != 0:
                    continue

                annotated_frame = frame.copy()
                persons: List[PersonInstance] = []

                if person_pipeline is not None:
                    annotated_frame, persons = person_pipeline.process_frame(frame, frame_index)
                    person_summary["processed_frames"] = int(person_summary["processed_frames"]) + 1
                    alert_increment = sum(1 for person in persons if person.should_alert)
                    person_summary["alert_count"] = int(person_summary["alert_count"]) + alert_increment

                    for person in persons:
                        seen_track_ids.add(int(person.tracker_id))
                        track_last_seen[int(person.tracker_id)] = frame_index
                        if save_artifacts and frame_index % crop_save_interval == 0:
                            crop_path = (
                                paths.crops_dir
                                / f"track_{person.tracker_id}"
                                / f"frame_{frame_index:06d}.jpg"
                            )
                            if enqueue_log_item("crop", crop_path, image=person.crop.copy()):
                                person_summary["crop_file_count"] = (
                                    int(person_summary["crop_file_count"]) + 1
                                )
                    person_summary["track_dir_count"] = len(seen_track_ids)
                    person_summary["last_frame_index"] = frame_index

                    if region_extractor is not None:
                        for person in persons:
                            tracker_id = int(person.tracker_id)
                            snapshot = track_snapshots.get(tracker_id)
                            should_refresh_attributes = (
                                snapshot is None
                                or frame_index - snapshot.last_updated_frame_index
                                >= attribute_refresh_interval
                            )
                            if not should_refresh_attributes:
                                continue

                            regions = region_extractor.split_regions(person.crop)
                            region_summary["processed_count"] = (
                                int(region_summary["processed_count"]) + 1
                            )
                            region_summary["last_frame_index"] = frame_index

                            save_stage_artifacts = frame_index % crop_save_interval == 0
                            save_stage_artifacts = save_artifacts and save_stage_artifacts
                            frame_output_dir = (
                                paths.region_output_root
                                / f"track_{tracker_id}"
                                / f"frame_{frame_index:06d}"
                            )
                            if save_stage_artifacts:
                                frame_output_dir.mkdir(parents=True, exist_ok=True)
                                cv2.imwrite(str(frame_output_dir / "head_face.jpg"), regions.head_face)
                                cv2.imwrite(str(frame_output_dir / "hands.jpg"), regions.hands)
                                cv2.imwrite(str(frame_output_dir / "torso.jpg"), regions.torso)
                                region_summary["saved_count"] = (
                                    int(region_summary["saved_count"]) + 1
                                )

                            mask_label: Optional[str] = None
                            mask_confidence = 0.0
                            mask_bbox_xyxy: Optional[Tuple[int, int, int, int]] = None
                            if facemask_engine is not None:
                                detections = run_facemask_inference(
                                    facemask_engine,
                                    regions.head_face,
                                )
                                facemask_summary["processed_count"] = (
                                    int(facemask_summary["processed_count"]) + 1
                                )
                                facemask_summary["last_frame_index"] = frame_index

                                if detections:
                                    best_detection = detections[0]
                                    mask_label = best_detection.class_name
                                    mask_confidence = float(best_detection.confidence)
                                    if regions.face_bbox is not None:
                                        face_x1, face_y1, face_x2, face_y2 = regions.face_bbox
                                        face_width = max(1, face_x2 - face_x1)
                                        face_height = max(1, face_y2 - face_y1)
                                        scale_x = face_width / max(1, regions.head_face.shape[1])
                                        scale_y = face_height / max(1, regions.head_face.shape[0])
                                        person_x1, person_y1, _, _ = person.bbox_xyxy
                                        mask_bbox_xyxy = _offset_box(
                                            best_detection.bbox_xyxy,
                                            person_x1 + face_x1,
                                            person_y1 + face_y1,
                                            scale_x=scale_x,
                                            scale_y=scale_y,
                                        )
                                    if best_detection.class_name == "Mask":
                                        facemask_summary["mask_count"] = (
                                            int(facemask_summary["mask_count"]) + 1
                                        )
                                    elif best_detection.class_name == "NoMask":
                                        facemask_summary["no_mask_count"] = (
                                            int(facemask_summary["no_mask_count"]) + 1
                                        )
                                else:
                                    mask_label = "NoDetection"
                                    facemask_summary["no_detection_count"] = (
                                        int(facemask_summary["no_detection_count"]) + 1
                                    )

                                if save_stage_artifacts:
                                    facemask_output_dir = (
                                        paths.facemask_output_root
                                        / f"track_{tracker_id}"
                                        / f"frame_{frame_index:06d}"
                                    )
                                    facemask_output_dir.mkdir(parents=True, exist_ok=True)
                                    annotated_face = annotate_facemask_image(
                                        regions.head_face,
                                        detections,
                                    )
                                    cv2.imwrite(
                                        str(facemask_output_dir / "annotated.jpg"),
                                        annotated_face,
                                    )
                                    _write_json(
                                        facemask_output_dir / "detections.json",
                                        [
                                            {
                                                "class_id": detection.class_id,
                                                "class_name": detection.class_name,
                                                "confidence": round(
                                                    detection.confidence,
                                                    6,
                                                ),
                                                "bbox_xyxy": list(detection.bbox_xyxy),
                                            }
                                            for detection in detections
                                        ],
                                    )
                                    if detections:
                                        x1, y1, x2, y2 = detections[0].bbox_xyxy
                                        best_crop = regions.head_face[y1:y2, x1:x2]
                                        if best_crop.size > 0:
                                            cv2.imwrite(
                                                str(facemask_output_dir / "best_detection.jpg"),
                                                best_crop,
                                            )

                            glove_label: Optional[str] = None
                            glove_confidence = 0.0
                            hand_predictions: List[HandPrediction] = []
                            if glove_classifier is not None:
                                person_x1, person_y1, _, _ = person.bbox_xyxy
                                for hand_box in regions.hand_bboxes:
                                    hx1, hy1, hx2, hy2 = hand_box
                                    hand_crop = person.crop[hy1:hy2, hx1:hx2]
                                    if hand_crop.size == 0:
                                        classification = GloveClassification(
                                            class_index=-1,
                                            raw_label="unknown",
                                            normalized_label="Uncertain",
                                            confidence=0.0,
                                        )
                                    else:
                                        classification = classify_glove_image(
                                            glove_classifier,
                                            hand_crop,
                                        )

                                    final_label = resolve_final_label(
                                        classification,
                                        float(self.config.glove_detection.confidence_threshold),
                                    )
                                    global_hand_box = _offset_box(
                                        hand_box,
                                        person_x1,
                                        person_y1,
                                    )
                                    hand_predictions.append(
                                        HandPrediction(
                                            bbox_xyxy=global_hand_box,
                                            label=final_label,
                                            confidence=float(classification.confidence),
                                        )
                                    )
                                    glove_summary["processed_count"] = (
                                        int(glove_summary["processed_count"]) + 1
                                    )
                                    glove_summary["last_frame_index"] = frame_index
                                    if final_label == "Glove":
                                        glove_summary["glove_count"] = (
                                            int(glove_summary["glove_count"]) + 1
                                        )
                                    elif final_label == "NoGlove":
                                        glove_summary["no_glove_count"] = (
                                            int(glove_summary["no_glove_count"]) + 1
                                        )
                                    else:
                                        glove_summary["uncertain_count"] = (
                                            int(glove_summary["uncertain_count"]) + 1
                                        )

                                if hand_predictions:
                                    best_hand_prediction = max(
                                        hand_predictions,
                                        key=lambda item: item.confidence,
                                    )
                                    glove_label = best_hand_prediction.label
                                    glove_confidence = best_hand_prediction.confidence

                                if save_stage_artifacts:
                                    glove_output_dir = (
                                        paths.glove_output_root
                                        / f"track_{tracker_id}"
                                        / f"frame_{frame_index:06d}"
                                    )
                                    glove_output_dir.mkdir(parents=True, exist_ok=True)
                                    combined_classification = GloveClassification(
                                        class_index=-1,
                                        raw_label="combined",
                                        normalized_label=glove_label or "Uncertain",
                                        confidence=float(glove_confidence),
                                    )
                                    annotated_hands = annotate_glove_image(
                                        regions.hands,
                                        combined_classification,
                                        glove_label or "Uncertain",
                                    )
                                    cv2.imwrite(
                                        str(glove_output_dir / "annotated.jpg"),
                                        annotated_hands,
                                    )
                                    prediction_payload = {
                                        "final_label": glove_label or "Uncertain",
                                        "confidence": round(float(glove_confidence), 6),
                                        "confidence_threshold": float(
                                            self.config.glove_detection.confidence_threshold
                                        ),
                                        "hands": [
                                            {
                                                "bbox_xyxy": list(hand_prediction.bbox_xyxy),
                                                "label": hand_prediction.label,
                                                "confidence": round(
                                                    hand_prediction.confidence,
                                                    6,
                                                ),
                                            }
                                            for hand_prediction in hand_predictions
                                        ],
                                    }
                                    prediction_payload["confidence_threshold"] = float(
                                        self.config.glove_detection.confidence_threshold
                                    )
                                    _write_json(
                                        glove_output_dir / "prediction.json",
                                        prediction_payload,
                                    )

                            track_snapshots[tracker_id] = TrackAttributeSnapshot(
                                tracker_id=tracker_id,
                                last_updated_frame_index=frame_index,
                                mask_label=mask_label,
                                mask_confidence=mask_confidence,
                                mask_bbox_xyxy=mask_bbox_xyxy,
                                glove_label=glove_label,
                                glove_confidence=glove_confidence,
                                hand_predictions=hand_predictions,
                            )

                        stale_track_ids = [
                            tracker_id
                            for tracker_id, last_seen in track_last_seen.items()
                            if frame_index - last_seen
                            > self.config.person_detection.track_ttl_frames
                        ]
                        for tracker_id in stale_track_ids:
                            track_last_seen.pop(tracker_id, None)
                            track_snapshots.pop(tracker_id, None)

                        annotated_frame = _annotate_person_attributes(
                            annotated_frame,
                            persons,
                            track_snapshots,
                        )

                if ppe_pipeline is not None:
                    ppe_records = ppe_pipeline.predict_records(frame)
                    ppe_summary["processed_frames"] = int(ppe_summary["processed_frames"]) + 1
                    ppe_summary["emitted_detections"] = (
                        int(ppe_summary["emitted_detections"]) + len(ppe_records)
                    )
                    ppe_summary["last_frame_index"] = frame_index
                    class_counts = ppe_summary.get("class_counts")
                    if not isinstance(class_counts, dict):
                        class_counts = {}
                    for record in ppe_records:
                        normalized_class_name = record.class_name.lower()
                        class_counts[normalized_class_name] = int(
                            class_counts.get(normalized_class_name, 0)
                        ) + 1
                    ppe_summary["class_counts"] = class_counts
                    if ppe_detections_file is not None and ppe_records:
                        ppe_detections_file.write(
                            json.dumps(
                                {
                                    "frame_index": frame_index,
                                    "detections": [
                                        {
                                            "class_name": record.class_name,
                                            "confidence": round(record.confidence, 6),
                                            "bbox_xyxy": list(record.bbox_xyxy),
                                        }
                                        for record in ppe_records
                                    ],
                                },
                                ensure_ascii=True,
                            )
                            + "\n"
                        )
                    annotated_frame = annotate_ppe_image(annotated_frame, ppe_records)

                if behavior_pipeline is not None:
                    behavior_annotated_frame, behaviors, events = behavior_pipeline.process_frame(
                        frame,
                        frame_index,
                    )
                    behavior_summary["processed_frames"] = (
                        int(behavior_summary["processed_frames"]) + 1
                    )
                    behavior_summary["emitted_events"] = (
                        int(behavior_summary["emitted_events"]) + len(events)
                    )
                    if behavior_events_file is not None:
                        for event in events:
                            behavior_events_file.write(
                                json.dumps(
                                    {
                                        "frame_index": event.frame_index,
                                        "tracker_id": event.tracker_id,
                                        "event_type": event.event_type,
                                        "confidence": event.confidence,
                                        "details": event.details,
                                    },
                                    ensure_ascii=True,
                                )
                                + "\n"
                            )

                    if person_pipeline is None and ppe_pipeline is None:
                        annotated_frame = behavior_annotated_frame
                    else:
                        annotated_frame = annotate_behavior_evidence(
                            annotated_frame,
                            behaviors,
                        )
                        annotated_frame = _overlay_behavior_summary(
                            annotated_frame,
                            behaviors,
                            len(events),
                        )

                if self.frame_callback is not None:
                    self.frame_callback(
                        frame_index,
                        total_frames,
                        annotated_frame,
                        persons,
                    )

                self._update_stage_summary(summary, "person_detection", person_summary)
                self._update_stage_summary(summary, "ppe_detection", ppe_summary)
                self._update_stage_summary(
                    summary,
                    "person_region_extraction",
                    region_summary,
                )
                self._update_stage_summary(summary, "facemask_detection", facemask_summary)
                self._update_stage_summary(summary, "glove_detection", glove_summary)
                self._update_stage_summary(summary, "behavior_detection", behavior_summary)

                if frame_index % summary_flush_interval == 0:
                    self._write_summary(summary)

                if self._cancel_requested():
                    raise StreamPipelineCancelled("Job cancelled by user.")

            if ppe_detections_file is not None:
                ppe_detections_file.flush()
            if behavior_events_file is not None:
                behavior_events_file.flush()
            log_queue.join()

            for stage_name in started_stage_names:
                stage_result = summary.stages[stage_name]
                stage_result.success = True
                self._notify_stage(stage_name, "completed", stage_result)

            summary.status = "completed"
            self._write_summary(summary)
            return summary.to_dict()
        except StreamPipelineCancelled as exc:
            summary.status = "cancelled"
            summary.error = str(exc)
            for stage_name in started_stage_names:
                stage_result = summary.stages.get(stage_name)
                if stage_result is None:
                    continue
                stage_result.success = False
                stage_result.error = str(exc)
                self._notify_stage(stage_name, "cancelled", stage_result)
            self._write_summary(summary)
            return summary.to_dict()
        except Exception as exc:
            summary.status = "failed"
            summary.error = str(exc)
            for stage_name in started_stage_names:
                stage_result = summary.stages.get(stage_name)
                if stage_result is None:
                    continue
                stage_result.success = False
                stage_result.error = str(exc)
                self._notify_stage(stage_name, "failed", stage_result)
            self._write_summary(summary)
            raise
        finally:
            if frame_reader is not None:
                frame_reader.close()
            if ppe_detections_file is not None:
                ppe_detections_file.close()
            if behavior_events_file is not None:
                behavior_events_file.close()
