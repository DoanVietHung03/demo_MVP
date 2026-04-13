from __future__ import annotations

from pathlib import Path
import queue
import threading
from typing import Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import supervision as sv

from person_detection.config import PersonInstance
from person_detection.inference import YOLOPersonInference, infer_person_detections, load_yolo_person_inference


log_queue: "queue.Queue[Tuple[str, str, Optional[np.ndarray], Optional[str]]]" = queue.Queue(maxsize=4096)
log_queue_drop_count = 0
log_queue_drop_lock = threading.Lock()


def _warn_log_queue_drop(
    item_type: str,
    path: Union[str, Path],
) -> None:
    global log_queue_drop_count

    with log_queue_drop_lock:
        log_queue_drop_count += 1
        dropped_count = log_queue_drop_count

    if dropped_count == 1 or dropped_count % 100 == 0:
        print(
            "[Async Logger Warning] "
            f"Queue full, dropped {dropped_count} item(s). "
            f"Last item_type={item_type} path={path}"
        )


def _write_log_item(
    item: Tuple[str, str, Optional[np.ndarray], Optional[str]],
) -> None:
    item_type, path_str, image, text = item
    output_path = Path(path_str)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if item_type == "crop":
        if image is not None and image.size > 0:
            cv2.imwrite(str(output_path), image)
        return

    if item_type == "alert":
        with output_path.open("a", encoding="utf-8") as file_obj:
            file_obj.write((text or "") + "\n")


def async_logger_worker() -> None:
    while True:
        item = log_queue.get()
        if item is None:
            break
        try:
            _write_log_item(item)
        except Exception as exc:  # pragma: no cover - depends on filesystem/runtime
            print(f"[Async Logger Error] {exc}")
        finally:
            log_queue.task_done()


def enqueue_log_item(
    item_type: str,
    path: Union[str, Path],
    image: Optional[np.ndarray] = None,
    text: Optional[str] = None,
) -> bool:
    try:
        log_queue.put_nowait((item_type, str(path), image, text))
        return True
    except queue.Full:
        _warn_log_queue_drop(item_type, path)
        return False


threading.Thread(target=async_logger_worker, daemon=True).start()


class TrackAlertMemory:
    def __init__(
        self,
        track_ttl_frames: int = 60,
        cooldown_frames: Optional[int] = None,
    ) -> None:
        self.track_ttl_frames = track_ttl_frames
        self.cooldown_frames = cooldown_frames
        self.last_seen_frame: Dict[int, int] = {}
        self.last_alert_frame: Dict[int, int] = {}

    def mark_seen(self, tracker_id: int, frame_index: int) -> None:
        self.last_seen_frame[tracker_id] = frame_index

    def should_alert(self, tracker_id: int, frame_index: int) -> bool:
        self.mark_seen(tracker_id, frame_index)
        last_alert = self.last_alert_frame.get(tracker_id)
        if last_alert is None:
            self.last_alert_frame[tracker_id] = frame_index
            return True
        if self.cooldown_frames is None:
            return False
        if frame_index - last_alert >= self.cooldown_frames:
            self.last_alert_frame[tracker_id] = frame_index
            return True
        return False

    def evict_stale_tracks(self, frame_index: int) -> None:
        stale_track_ids = [
            track_id
            for track_id, last_seen in self.last_seen_frame.items()
            if frame_index - last_seen > self.track_ttl_frames
        ]
        for track_id in stale_track_ids:
            self.last_seen_frame.pop(track_id, None)
            self.last_alert_frame.pop(track_id, None)


class PersonDetectionPipeline:
    def __init__(
        self,
        model_path: str = "./weights/yolo11n.pt",
        device: str = "cuda:0",
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        crop_padding: int = 8,
        input_resize_scale: float = 1.0,
        track_ttl_frames: int = 60,
        alert_cooldown_frames: Optional[int] = None,
    ) -> None:
        self.inference_engine: YOLOPersonInference = load_yolo_person_inference(
            model_path=model_path,
            device=device,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            input_resize_scale=input_resize_scale,
        )
        self.device = self.inference_engine.device
        self.model_path = model_path.strip()
        self.crop_padding = crop_padding
        self.tracker = sv.ByteTrack()
        self.alert_memory = TrackAlertMemory(
            track_ttl_frames=track_ttl_frames,
            cooldown_frames=alert_cooldown_frames,
        )
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_LEFT)

    def _clip_bbox(
        self,
        xyxy: np.ndarray,
        frame_shape: Tuple[int, int, int],
    ) -> Tuple[int, int, int, int]:
        height, width = frame_shape[:2]
        x1, y1, x2, y2 = xyxy.astype(int)
        x1 = int(max(0, x1 - self.crop_padding))
        y1 = int(max(0, y1 - self.crop_padding))
        x2 = int(min(width, x2 + self.crop_padding))
        y2 = int(min(height, y2 + self.crop_padding))
        return x1, y1, x2, y2

    def _crop_person(
        self,
        frame: np.ndarray,
        bbox_xyxy: Tuple[int, int, int, int],
    ) -> np.ndarray:
        x1, y1, x2, y2 = bbox_xyxy
        if x2 <= x1 or y2 <= y1:
            return np.empty((0, 0, 3), dtype=frame.dtype)
        return frame[y1:y2, x1:x2].copy()

    def process_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
    ) -> Tuple[np.ndarray, List[PersonInstance]]:
        detections = infer_person_detections(self.inference_engine, frame)
        tracked_detections = self.tracker.update_with_detections(detections)
        persons: List[PersonInstance] = []
        labels: List[str] = []
        valid_indices: List[int] = []

        if len(tracked_detections) == 0:
            self.alert_memory.evict_stale_tracks(frame_index)
            return frame.copy(), persons

        confidences = tracked_detections.confidence
        tracker_ids = tracked_detections.tracker_id
        if confidences is None:
            confidences = np.zeros(len(tracked_detections), dtype=float)
        if tracker_ids is None:
            tracker_ids = np.full(len(tracked_detections), -1, dtype=int)

        for index, (xyxy, confidence, tracker_id) in enumerate(
            zip(
                tracked_detections.xyxy,
                confidences,
                tracker_ids,
            )
        ):
            if tracker_id is None or int(tracker_id) < 0:
                continue

            clipped_bbox = self._clip_bbox(xyxy, frame.shape)
            crop = self._crop_person(frame, clipped_bbox)
            if crop.size == 0:
                continue

            tracker_id_int = int(tracker_id)
            should_alert = self.alert_memory.should_alert(tracker_id_int, frame_index)
            valid_indices.append(index)
            persons.append(
                PersonInstance(
                    tracker_id=tracker_id_int,
                    confidence=float(confidence),
                    bbox_xyxy=clipped_bbox,
                    crop=crop,
                    should_alert=should_alert,
                )
            )
            labels.append(f"id:{tracker_id_int} person {float(confidence):.2f}")

        self.alert_memory.evict_stale_tracks(frame_index)

        valid_tracked_detections = (
            tracked_detections[np.array(valid_indices)]
            if valid_indices
            else tracked_detections[np.array([], dtype=int)]
        )
        annotated = self.box_annotator.annotate(
            scene=frame.copy(),
            detections=valid_tracked_detections,
        )
        annotated = self.label_annotator.annotate(
            scene=annotated,
            detections=valid_tracked_detections,
            labels=labels,
        )
        return annotated, persons

    def process_video(
        self,
        source: Union[str, int],
        output_dir: Optional[Union[str, Path]] = None,
        frame_skip: int = 0,
        crop_save_interval: int = 10,
        max_frames: Optional[int] = None,
        frame_callback: Optional[
            Callable[[int, Optional[int], np.ndarray, List[PersonInstance]], None]
        ] = None,
    ) -> None:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

        total_frames_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_frames = total_frames_raw if total_frames_raw > 0 else None
        output_root = Path(output_dir) if output_dir else None
        if output_root is not None:
            output_root.mkdir(parents=True, exist_ok=True)
        crop_interval = max(1, int(crop_save_interval))

        frame_index = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if max_frames is not None and frame_index >= max_frames:
                    break
                if frame_skip > 0 and frame_index % (frame_skip + 1) != 0:
                    frame_index += 1
                    continue

                annotated_frame, persons = self.process_frame(frame, frame_index)

                for person in persons:
                    if output_root is not None and frame_index % crop_interval == 0:
                        person_dir = output_root / f"track_{person.tracker_id}"
                        person_dir.mkdir(parents=True, exist_ok=True)
                        crop_path = person_dir / f"frame_{frame_index:06d}.jpg"
                        enqueue_log_item("crop", crop_path, image=person.crop.copy())

                    if person.should_alert:
                        print(
                            f"[ALERT] frame={frame_index} "
                            f"track_id={person.tracker_id} "
                            f"bbox={person.bbox_xyxy} "
                            f"conf={person.confidence:.2f}"
                        )

                if frame_callback is not None:
                    frame_callback(frame_index, total_frames, annotated_frame, persons)

                frame_index += 1
        finally:
            cap.release()
