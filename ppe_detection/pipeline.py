from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import supervision as sv

try:
    from ppe_detection.config import PPEDetectionConfig
    from ppe_detection.inference import (
        YOLOPPEInference,
        describe_ppe_inference_backend,
        filter_detections_by_class_names,
        infer_ppe_detections,
        load_yolo_ppe_inference,
    )
except ModuleNotFoundError:  # pragma: no cover - convenience for direct script execution
    from config import PPEDetectionConfig
    from inference import (
        YOLOPPEInference,
        describe_ppe_inference_backend,
        filter_detections_by_class_names,
        infer_ppe_detections,
        load_yolo_ppe_inference,
    )


@dataclass
class PPEDetectionRecord:
    class_name: str
    confidence: float
    bbox_xyxy: Tuple[int, int, int, int]


CLASS_TO_COLOR = {
    "hardhat": (0, 215, 255),
    "vest": (0, 140, 255),
    "person": (255, 255, 0),
}


def _class_names_to_counts(target_class_names: Sequence[str]) -> Dict[str, int]:
    return {class_name: 0 for class_name in target_class_names}


def detections_to_records(
    inference_engine: YOLOPPEInference,
    detections: sv.Detections,
) -> List[PPEDetectionRecord]:
    if len(detections) == 0:
        return []

    confidence_array = detections.confidence
    class_id_array = detections.class_id
    if confidence_array is None:
        confidence_array = np.zeros(len(detections), dtype=float)
    if class_id_array is None:
        class_id_array = np.full(len(detections), -1, dtype=int)

    records: List[PPEDetectionRecord] = []
    for xyxy, confidence, class_id in zip(
        detections.xyxy,
        confidence_array,
        class_id_array,
    ):
        normalized_class_id = int(class_id)
        class_name = inference_engine.class_names.get(
            normalized_class_id,
            f"class_{normalized_class_id}",
        )
        records.append(
            PPEDetectionRecord(
                class_name=class_name,
                confidence=float(confidence),
                bbox_xyxy=tuple(int(value) for value in xyxy.astype(int)),
            )
        )

    records.sort(key=lambda item: item.confidence, reverse=True)
    return records


def annotate_image(
    image: np.ndarray,
    records: Sequence[PPEDetectionRecord],
) -> np.ndarray:
    annotated = image.copy()
    for record in records:
        x1, y1, x2, y2 = record.bbox_xyxy
        color = CLASS_TO_COLOR.get(record.class_name.lower(), (255, 255, 0))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            f"{record.class_name} {record.confidence:.2f}",
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA,
        )
    return annotated


class PPEDetectionPipeline:
    def __init__(self, config: PPEDetectionConfig) -> None:
        self.config = config
        self.inference_engine = load_yolo_ppe_inference(
            model_path=config.model_path,
            device=config.device,
            confidence_threshold=config.confidence_threshold,
            iou_threshold=config.iou_threshold,
            input_resize_scale=config.input_resize_scale,
        )

    def runtime_backend_label(self) -> str:
        return describe_ppe_inference_backend(self.inference_engine)

    def predict_records(self, frame: np.ndarray) -> List[PPEDetectionRecord]:
        detections = infer_ppe_detections(self.inference_engine, frame)
        filtered = filter_detections_by_class_names(
            self.inference_engine,
            detections,
            self.config.target_class_names,
        )
        return detections_to_records(self.inference_engine, filtered)

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[PPEDetectionRecord]]:
        records = self.predict_records(frame)
        return annotate_image(frame, records), records

    def process_video(
        self,
        source: Union[str, int],
        output_root: Optional[Union[str, Path]] = None,
        frame_callback: Optional[
            Callable[[int, Optional[int], np.ndarray, List[PPEDetectionRecord]], None]
        ] = None,
    ) -> Dict[str, object]:
        print(
            "[Stage Runtime] "
            f"ppe_detection: {self.runtime_backend_label()} | "
            f"device={self.inference_engine.device} | "
            f"model={self.config.model_path}"
        )
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

        total_frames_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_frames = total_frames_raw if total_frames_raw > 0 else None
        destination_root = Path(output_root) if output_root else Path(self.config.output_root)
        destination_root.mkdir(parents=True, exist_ok=True)
        detections_path = destination_root / "detections.jsonl"
        snapshots_root = destination_root / "snapshots"
        snapshot_interval = max(1, int(self.config.snapshot_interval_frames))

        processed_frames = 0
        emitted_detections = 0
        class_counts = _class_names_to_counts(self.config.target_class_names)
        frame_index = 0
        try:
            with detections_path.open("w", encoding="utf-8") as detections_file:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if self.config.max_frames is not None and frame_index >= self.config.max_frames:
                        break
                    if self.config.frame_skip > 0 and frame_index % (self.config.frame_skip + 1) != 0:
                        frame_index += 1
                        continue

                    annotated_frame, records = self.process_frame(frame)
                    processed_frames += 1
                    if records:
                        emitted_detections += len(records)
                        frame_payload = {
                            "frame_index": frame_index,
                            "detections": [
                                {
                                    "class_name": record.class_name,
                                    "confidence": round(record.confidence, 6),
                                    "bbox_xyxy": list(record.bbox_xyxy),
                                }
                                for record in records
                            ],
                        }
                        detections_file.write(
                            json.dumps(frame_payload, ensure_ascii=True) + "\n"
                        )
                        if frame_index % snapshot_interval == 0:
                            snapshot_path = snapshots_root / f"frame_{frame_index:06d}.jpg"
                            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
                            cv2.imwrite(str(snapshot_path), annotated_frame)
                    for record in records:
                        normalized_class_name = record.class_name.lower()
                        if normalized_class_name in class_counts:
                            class_counts[normalized_class_name] += 1

                    if frame_callback is not None:
                        frame_callback(frame_index, total_frames, annotated_frame, records)

                    frame_index += 1
        finally:
            cap.release()

        return {
            "source": str(source),
            "output_root": str(destination_root),
            "detections_path": str(detections_path),
            "runtime_backend": self.runtime_backend_label(),
            "model_path": self.config.model_path,
            "processed_frames": processed_frames,
            "emitted_detections": emitted_detections,
            "class_counts": class_counts,
        }

