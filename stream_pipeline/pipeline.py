from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from behavior_detection.pipeline import BehaviorDetectionPipeline, TrackedPersonBehavior
from facemask_detection.inference import (
    describe_facemask_inference_backend,
    load_facemask_inference_engine,
    run_facemask_inference,
)
from facemask_detection.pipeline import annotate_image as annotate_facemask_image
from glove_detection.inference import (
    GloveClassification,
    classify_glove_image,
    classify_glove_to_json,
    describe_glove_classifier_backend,
    load_tflite_glove_classifier,
)
from glove_detection.pipeline import (
    annotate_image as annotate_glove_image,
    resolve_final_label,
)
from person_detection.config import PersonInstance
from person_detection.pipeline import PersonDetectionPipeline, enqueue_log_item, log_queue
from person_region_extraction.pipeline import PersonRegionExtractor
from stream_pipeline.config import StreamPipelineConfig
from stream_pipeline.types import PipelinePaths, PipelineRunSummary, StageRunResult


StageCallback = Callable[[str, str, Optional[StageRunResult]], None]
FrameCallback = Callable[[int, Optional[int], np.ndarray, List[PersonInstance]], None]


@dataclass
class TrackAttributeSnapshot:
    tracker_id: int
    last_updated_frame_index: int
    mask_label: Optional[str] = None
    mask_confidence: float = 0.0
    glove_label: Optional[str] = None
    glove_confidence: float = 0.0


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


class StreamPipeline:
    def __init__(
        self,
        config: StreamPipelineConfig,
        stage_callback: Optional[StageCallback] = None,
        frame_callback: Optional[FrameCallback] = None,
    ) -> None:
        self.config = config
        self.stage_callback = stage_callback
        self.frame_callback = frame_callback

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

        person_pipeline: Optional[PersonDetectionPipeline] = None
        region_extractor: Optional[PersonRegionExtractor] = None
        facemask_engine = None
        glove_classifier = None
        behavior_pipeline: Optional[BehaviorDetectionPipeline] = None

        person_summary: Dict[str, object] = {}
        region_summary: Dict[str, object] = {}
        facemask_summary: Dict[str, object] = {}
        glove_summary: Dict[str, object] = {}
        behavior_summary: Dict[str, object] = {}

        started_stage_names: List[str] = []
        track_snapshots: Dict[int, TrackAttributeSnapshot] = {}
        track_last_seen: Dict[int, int] = {}
        seen_track_ids: set[int] = set()

        cap = None
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

            if self.config.run_region_extraction:
                region_extractor = PersonRegionExtractor(self.config.person_region_extraction)
                paths.region_output_root.mkdir(parents=True, exist_ok=True)
                region_summary = {
                    "input_stream": str(self.config.source),
                    "output_root": str(paths.region_output_root),
                    "runtime_backend": region_extractor.runtime_backend_label(),
                    "model_path": self.config.person_region_extraction.model_path,
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
                paths.behavior_output_root.mkdir(parents=True, exist_ok=True)
                behavior_pipeline = BehaviorDetectionPipeline(self.config.behavior_detection)
                behavior_summary = {
                    "source": str(self.config.source),
                    "output_root": str(paths.behavior_output_root),
                    "events_path": str(paths.behavior_output_root / "events.jsonl"),
                    "runtime_backends": behavior_pipeline.runtime_backend_summary(),
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

            primary_frame_skip = (
                int(self.config.person_detection.frame_skip)
                if self.config.run_person_detection
                else int(self.config.behavior_detection.frame_skip)
            )
            primary_max_frames = (
                self.config.person_detection.max_frames
                if self.config.run_person_detection
                else self.config.behavior_detection.max_frames
            )

            cap = cv2.VideoCapture(self.config.source)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video source: {self.config.source}")

            total_frames_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            total_frames = total_frames_raw if total_frames_raw > 0 else None
            frame_index = 0

            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if primary_max_frames is not None and frame_index >= primary_max_frames:
                    break
                if primary_frame_skip > 0 and frame_index % (primary_frame_skip + 1) != 0:
                    frame_index += 1
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
                        if frame_index % crop_save_interval == 0:
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
                            if glove_classifier is not None:
                                if regions.hands is None or regions.hands.size == 0:
                                    classification = GloveClassification(
                                        class_index=-1,
                                        raw_label="unknown",
                                        normalized_label="Uncertain",
                                        confidence=0.0,
                                    )
                                else:
                                    classification = classify_glove_image(
                                        glove_classifier,
                                        regions.hands,
                                    )

                                glove_label = resolve_final_label(
                                    classification,
                                    float(self.config.glove_detection.confidence_threshold),
                                )
                                glove_confidence = float(classification.confidence)
                                glove_summary["processed_count"] = (
                                    int(glove_summary["processed_count"]) + 1
                                )
                                glove_summary["last_frame_index"] = frame_index
                                if glove_label == "Glove":
                                    glove_summary["glove_count"] = (
                                        int(glove_summary["glove_count"]) + 1
                                    )
                                elif glove_label == "NoGlove":
                                    glove_summary["no_glove_count"] = (
                                        int(glove_summary["no_glove_count"]) + 1
                                    )
                                else:
                                    glove_summary["uncertain_count"] = (
                                        int(glove_summary["uncertain_count"]) + 1
                                    )

                                if save_stage_artifacts:
                                    glove_output_dir = (
                                        paths.glove_output_root
                                        / f"track_{tracker_id}"
                                        / f"frame_{frame_index:06d}"
                                    )
                                    glove_output_dir.mkdir(parents=True, exist_ok=True)
                                    annotated_hands = annotate_glove_image(
                                        regions.hands,
                                        classification,
                                        glove_label,
                                    )
                                    cv2.imwrite(
                                        str(glove_output_dir / "annotated.jpg"),
                                        annotated_hands,
                                    )
                                    prediction_payload = json.loads(
                                        classify_glove_to_json(classification)
                                    )
                                    prediction_payload["final_label"] = glove_label
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
                                glove_label=glove_label,
                                glove_confidence=glove_confidence,
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

                    if person_pipeline is None:
                        annotated_frame = behavior_annotated_frame
                    else:
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

                frame_index += 1

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
            if cap is not None:
                cap.release()
            if behavior_events_file is not None:
                behavior_events_file.close()
