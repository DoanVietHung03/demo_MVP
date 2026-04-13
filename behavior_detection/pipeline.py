from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import supervision as sv

try:
    from behavior_detection.config import BehaviorDetectionConfig
    from behavior_detection.inference import (
        PosePerson,
        YOLODetectionInference,
        YOLOPoseInference,
        infer_filtered_detections,
        infer_pose_people,
        load_yolo_detection_inference,
        load_yolo_pose_inference,
    )
except ModuleNotFoundError:  # pragma: no cover - convenience for direct script execution
    from config import BehaviorDetectionConfig
    from inference import (
        PosePerson,
        YOLODetectionInference,
        YOLOPoseInference,
        infer_filtered_detections,
        infer_pose_people,
        load_yolo_detection_inference,
        load_yolo_pose_inference,
    )


NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10


@dataclass
class BehaviorEvent:
    frame_index: int
    tracker_id: int
    event_type: str
    confidence: float
    details: Dict[str, float]


@dataclass
class TrackBehaviorState:
    last_seen_frame: int = -1
    smoking_streak: int = 0
    phone_present_streak: int = 0
    phone_in_hand_streak: int = 0
    phone_usage_streak: int = 0
    last_alert_frame: Dict[str, int] = field(default_factory=dict)


@dataclass
class TrackedPersonBehavior:
    tracker_id: int
    bbox_xyxy: Tuple[int, int, int, int]
    label: str
    smoking_candidate: bool
    phone_present: bool
    phone_in_hand: bool
    phone_usage: bool
    events: List[BehaviorEvent]


def _clip_bbox(
    xyxy: np.ndarray,
    frame_shape: Tuple[int, int, int],
    padding: int,
) -> Tuple[int, int, int, int]:
    frame_height, frame_width = frame_shape[:2]
    x1, y1, x2, y2 = xyxy.astype(int)
    x1 = int(max(0, x1 - padding))
    y1 = int(max(0, y1 - padding))
    x2 = int(min(frame_width, x2 + padding))
    y2 = int(min(frame_height, y2 + padding))
    return x1, y1, x2, y2


def _bbox_center(box: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _bbox_diag(box: Tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = box
    return max(1.0, float(np.hypot(max(1, x2 - x1), max(1, y2 - y1))))


def _distance(point_a: Tuple[float, float], point_b: Tuple[float, float]) -> float:
    return float(np.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1]))


def _iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height
    if inter_area <= 0:
        return 0.0
    box_a_area = max(1, ax2 - ax1) * max(1, ay2 - ay1)
    box_b_area = max(1, bx2 - bx1) * max(1, by2 - by1)
    return float(inter_area / max(1.0, box_a_area + box_b_area - inter_area))


def _point_in_box(point: Tuple[float, float], box: Tuple[int, int, int, int]) -> bool:
    x_coord, y_coord = point
    x1, y1, x2, y2 = box
    return x1 <= x_coord <= x2 and y1 <= y_coord <= y2


def _visible_points(
    pose: PosePerson,
    indices: Sequence[int],
    threshold: float = 0.25,
) -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []
    for index in indices:
        if index >= pose.keypoints_xy.shape[0]:
            continue
        if float(pose.keypoints_conf[index]) < threshold:
            continue
        points.append(
            (
                float(pose.keypoints_xy[index][0]),
                float(pose.keypoints_xy[index][1]),
            )
        )
    return points


def _estimate_head_point(pose: Optional[PosePerson], person_box: Tuple[int, int, int, int]) -> Tuple[float, float]:
    if pose is not None:
        face_points = _visible_points(
            pose,
            [NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR],
        )
        if face_points:
            x_values = [point[0] for point in face_points]
            y_values = [point[1] for point in face_points]
            return float(sum(x_values) / len(x_values)), float(sum(y_values) / len(y_values))
    center_x, _ = _bbox_center(person_box)
    x1, y1, _, y2 = person_box
    return center_x, y1 + (y2 - y1) * 0.18


def _estimate_mouth_point(pose: Optional[PosePerson], person_box: Tuple[int, int, int, int]) -> Tuple[float, float]:
    if pose is not None:
        nose_points = _visible_points(pose, [NOSE])
        shoulder_points = _visible_points(pose, [LEFT_SHOULDER, RIGHT_SHOULDER])
        if nose_points:
            nose_x, nose_y = nose_points[0]
            if shoulder_points:
                shoulder_y = min(point[1] for point in shoulder_points)
                return nose_x, nose_y + max(4.0, (shoulder_y - nose_y) * 0.16)
            return nose_x, nose_y + 6.0
    head_x, head_y = _estimate_head_point(pose, person_box)
    return head_x, head_y + max(4.0, (person_box[3] - person_box[1]) * 0.05)


def _wrist_points(pose: Optional[PosePerson]) -> List[Tuple[float, float]]:
    if pose is None:
        return []
    return _visible_points(pose, [LEFT_WRIST, RIGHT_WRIST])


def _best_pose_for_person(
    person_box: Tuple[int, int, int, int],
    pose_people: Sequence[PosePerson],
) -> Optional[PosePerson]:
    best_pose: Optional[PosePerson] = None
    best_score = 0.0
    person_center = _bbox_center(person_box)
    for pose_person in pose_people:
        pose_box = pose_person.bbox_xyxy
        overlap_score = _iou(person_box, pose_box)
        if overlap_score <= 0:
            pose_center = _bbox_center(pose_box)
            if not _point_in_box(pose_center, person_box) and not _point_in_box(person_center, pose_box):
                continue
        if overlap_score > best_score:
            best_score = overlap_score
            best_pose = pose_person
    return best_pose


def _detections_inside_person(
    detections: sv.Detections,
    person_box: Tuple[int, int, int, int],
) -> List[Tuple[Tuple[int, int, int, int], float]]:
    if len(detections) == 0:
        return []

    confidence_array = detections.confidence
    if confidence_array is None:
        confidence_array = np.zeros(len(detections), dtype=float)

    matches: List[Tuple[Tuple[int, int, int, int], float]] = []
    for xyxy, confidence in zip(detections.xyxy, confidence_array):
        box = tuple(int(value) for value in xyxy.astype(int))
        box_center = _bbox_center(box)
        if _point_in_box(box_center, person_box) or _iou(box, person_box) >= 0.05:
            matches.append((box, float(confidence)))
    return matches


def _update_streak(previous_value: int, condition: bool) -> int:
    if condition:
        return previous_value + 1
    return 0


class BehaviorDetectionPipeline:
    def __init__(self, config: BehaviorDetectionConfig) -> None:
        self.config = config
        self.person_phone_engine: YOLODetectionInference = load_yolo_detection_inference(
            model_path=config.person_phone_model_path,
            device=config.device,
            confidence_threshold=min(config.person_confidence_threshold, config.phone_confidence_threshold),
            iou_threshold=config.iou_threshold,
            input_resize_scale=config.input_resize_scale,
        )
        self.pose_engine: YOLOPoseInference = load_yolo_pose_inference(
            model_path=config.pose_model_path,
            device=config.device,
            confidence_threshold=config.pose_confidence_threshold,
            iou_threshold=config.iou_threshold,
            input_resize_scale=config.input_resize_scale,
        )
        self.smoke_engine: YOLODetectionInference = load_yolo_detection_inference(
            model_path=config.smoke_model_path,
            device=config.device,
            confidence_threshold=config.smoke_confidence_threshold,
            iou_threshold=config.iou_threshold,
            input_resize_scale=config.input_resize_scale,
        )
        self.tracker = sv.ByteTrack()
        self.track_states: Dict[int, TrackBehaviorState] = {}
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_LEFT)

    def _should_emit_event(
        self,
        state: TrackBehaviorState,
        event_type: str,
        frame_index: int,
        streak_value: int,
        min_consecutive_frames: int,
    ) -> bool:
        if streak_value < min_consecutive_frames:
            return False
        last_alert_frame = state.last_alert_frame.get(event_type)
        if last_alert_frame is not None and (
            frame_index - last_alert_frame < self.config.alert_cooldown_frames
        ):
            return False
        state.last_alert_frame[event_type] = frame_index
        return True

    def _evict_stale_tracks(self, frame_index: int) -> None:
        stale_track_ids = [
            track_id
            for track_id, state in self.track_states.items()
            if frame_index - state.last_seen_frame > self.config.track_ttl_frames
        ]
        for track_id in stale_track_ids:
            self.track_states.pop(track_id, None)

    def _build_event(
        self,
        frame_index: int,
        tracker_id: int,
        event_type: str,
        confidence: float,
        details: Dict[str, float],
    ) -> BehaviorEvent:
        return BehaviorEvent(
            frame_index=frame_index,
            tracker_id=tracker_id,
            event_type=event_type,
            confidence=confidence,
            details=details,
        )

    def process_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
    ) -> Tuple[np.ndarray, List[TrackedPersonBehavior], List[BehaviorEvent]]:
        person_detections = infer_filtered_detections(
            self.person_phone_engine,
            frame,
            ["person"],
        )
        tracked_detections = self.tracker.update_with_detections(person_detections)
        phone_detections = infer_filtered_detections(
            self.person_phone_engine,
            frame,
            ["cell phone"],
        )
        smoke_detections = infer_filtered_detections(
            self.smoke_engine,
            frame,
            ["cigarette", "vape"],
        )
        pose_people = infer_pose_people(self.pose_engine, frame)

        behaviors: List[TrackedPersonBehavior] = []
        events: List[BehaviorEvent] = []
        labels: List[str] = []
        valid_indices: List[int] = []

        if len(tracked_detections) == 0:
            self._evict_stale_tracks(frame_index)
            return frame.copy(), behaviors, events

        confidence_array = tracked_detections.confidence
        tracker_id_array = tracked_detections.tracker_id
        if confidence_array is None:
            confidence_array = np.zeros(len(tracked_detections), dtype=float)
        if tracker_id_array is None:
            tracker_id_array = np.full(len(tracked_detections), -1, dtype=int)

        for index, (xyxy, confidence, tracker_id_raw) in enumerate(
            zip(tracked_detections.xyxy, confidence_array, tracker_id_array)
        ):
            if tracker_id_raw is None or int(tracker_id_raw) < 0:
                continue

            tracker_id = int(tracker_id_raw)
            person_box = _clip_bbox(xyxy, frame.shape, self.config.crop_padding)
            state = self.track_states.setdefault(tracker_id, TrackBehaviorState())
            state.last_seen_frame = frame_index

            pose_person = _best_pose_for_person(person_box, pose_people)
            mouth_point = _estimate_mouth_point(pose_person, person_box)
            head_point = _estimate_head_point(pose_person, person_box)
            wrist_points = _wrist_points(pose_person)
            person_diag = _bbox_diag(person_box)

            person_phone_matches = _detections_inside_person(phone_detections, person_box)
            person_smoke_matches = _detections_inside_person(smoke_detections, person_box)

            phone_present = len(person_phone_matches) > 0
            phone_in_hand = False
            phone_usage = False
            smoking_candidate = False
            phone_best_conf = 0.0
            smoking_best_conf = 0.0

            if person_phone_matches:
                best_phone_box, phone_best_conf = max(person_phone_matches, key=lambda item: item[1])
                phone_center = _bbox_center(best_phone_box)
                near_hand = any(
                    _distance(phone_center, wrist_point)
                    <= person_diag * self.config.phone_hand_distance_ratio
                    for wrist_point in wrist_points
                )
                near_face = (
                    _distance(phone_center, head_point)
                    <= person_diag * self.config.phone_face_distance_ratio
                )
                phone_in_hand = near_hand
                phone_usage = near_hand or near_face

            if person_smoke_matches:
                best_smoke_box, smoking_best_conf = max(person_smoke_matches, key=lambda item: item[1])
                smoke_center = _bbox_center(best_smoke_box)
                smoke_near_mouth = (
                    _distance(smoke_center, mouth_point)
                    <= person_diag * self.config.smoking_mouth_distance_ratio
                )
                smoke_near_hand = any(
                    _distance(smoke_center, wrist_point)
                    <= person_diag * self.config.smoking_hand_distance_ratio
                    for wrist_point in wrist_points
                )
                hand_to_mouth = any(
                    _distance(wrist_point, mouth_point)
                    <= person_diag * self.config.hand_to_mouth_distance_ratio
                    for wrist_point in wrist_points
                )
                smoking_candidate = smoke_near_mouth and (smoke_near_hand or hand_to_mouth)
            else:
                smoke_near_mouth = False
                smoke_near_hand = False
                hand_to_mouth = False

            state.smoking_streak = _update_streak(state.smoking_streak, smoking_candidate)
            state.phone_present_streak = _update_streak(state.phone_present_streak, phone_present)
            state.phone_in_hand_streak = _update_streak(state.phone_in_hand_streak, phone_in_hand)
            state.phone_usage_streak = _update_streak(state.phone_usage_streak, phone_usage)

            person_events: List[BehaviorEvent] = []

            if self._should_emit_event(
                state,
                "smoking",
                frame_index,
                state.smoking_streak,
                self.config.smoking_min_consecutive_frames,
            ):
                person_events.append(
                    self._build_event(
                        frame_index,
                        tracker_id,
                        "smoking",
                        smoking_best_conf,
                        {
                            "smoking_streak": float(state.smoking_streak),
                            "smoke_near_mouth": float(smoke_near_mouth),
                            "smoke_near_hand": float(smoke_near_hand),
                            "hand_to_mouth": float(hand_to_mouth),
                        },
                    )
                )

            if self._should_emit_event(
                state,
                "phone_present",
                frame_index,
                state.phone_present_streak,
                self.config.phone_present_min_consecutive_frames,
            ):
                person_events.append(
                    self._build_event(
                        frame_index,
                        tracker_id,
                        "phone_present",
                        phone_best_conf,
                        {"phone_present_streak": float(state.phone_present_streak)},
                    )
                )

            if self._should_emit_event(
                state,
                "phone_in_hand",
                frame_index,
                state.phone_in_hand_streak,
                self.config.phone_in_hand_min_consecutive_frames,
            ):
                person_events.append(
                    self._build_event(
                        frame_index,
                        tracker_id,
                        "phone_in_hand",
                        phone_best_conf,
                        {"phone_in_hand_streak": float(state.phone_in_hand_streak)},
                    )
                )

            if self._should_emit_event(
                state,
                "phone_usage",
                frame_index,
                state.phone_usage_streak,
                self.config.phone_usage_min_consecutive_frames,
            ):
                person_events.append(
                    self._build_event(
                        frame_index,
                        tracker_id,
                        "phone_usage",
                        phone_best_conf,
                        {"phone_usage_streak": float(state.phone_usage_streak)},
                    )
                )

            status_tokens: List[str] = []
            if smoking_candidate:
                status_tokens.append(f"SMOKE[{state.smoking_streak}]")
            if phone_usage:
                status_tokens.append(f"PHONE_USE[{state.phone_usage_streak}]")
            elif phone_in_hand:
                status_tokens.append(f"PHONE_HAND[{state.phone_in_hand_streak}]")
            elif phone_present:
                status_tokens.append(f"PHONE[{state.phone_present_streak}]")

            label = f"id:{tracker_id} {float(confidence):.2f}"
            if status_tokens:
                label = f"{label} | {' '.join(status_tokens)}"

            behaviors.append(
                TrackedPersonBehavior(
                    tracker_id=tracker_id,
                    bbox_xyxy=person_box,
                    label=label,
                    smoking_candidate=smoking_candidate,
                    phone_present=phone_present,
                    phone_in_hand=phone_in_hand,
                    phone_usage=phone_usage,
                    events=person_events,
                )
            )
            events.extend(person_events)
            labels.append(label)
            valid_indices.append(index)

        self._evict_stale_tracks(frame_index)

        valid_tracked_detections = (
            tracked_detections[np.array(valid_indices, dtype=int)]
            if valid_indices
            else tracked_detections[np.array([], dtype=int)]
        )
        annotated_frame = self.box_annotator.annotate(
            scene=frame.copy(),
            detections=valid_tracked_detections,
        )
        annotated_frame = self.label_annotator.annotate(
            scene=annotated_frame,
            detections=valid_tracked_detections,
            labels=labels,
        )
        return annotated_frame, behaviors, events

    def process_video(
        self,
        source: Union[str, int],
        output_root: Optional[Union[str, Path]] = None,
        frame_callback: Optional[
            Callable[[int, Optional[int], np.ndarray, List[TrackedPersonBehavior], List[BehaviorEvent]], None]
        ] = None,
    ) -> Dict[str, object]:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

        total_frames_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_frames = total_frames_raw if total_frames_raw > 0 else None
        destination_root = Path(output_root) if output_root else Path(self.config.output_root)
        destination_root.mkdir(parents=True, exist_ok=True)
        events_path = destination_root / "events.jsonl"
        snapshots_root = destination_root / "snapshots"
        snapshot_interval = max(1, int(self.config.snapshot_interval_frames))

        processed_frames = 0
        emitted_events = 0
        frame_index = 0
        try:
            with events_path.open("w", encoding="utf-8") as events_file:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if self.config.max_frames is not None and frame_index >= self.config.max_frames:
                        break
                    if self.config.frame_skip > 0 and frame_index % (self.config.frame_skip + 1) != 0:
                        frame_index += 1
                        continue

                    annotated_frame, behaviors, events = self.process_frame(frame, frame_index)
                    processed_frames += 1

                    for event in events:
                        emitted_events += 1
                        event_payload = {
                            "frame_index": event.frame_index,
                            "tracker_id": event.tracker_id,
                            "event_type": event.event_type,
                            "confidence": event.confidence,
                            "details": event.details,
                        }
                        events_file.write(json.dumps(event_payload, ensure_ascii=True) + "\n")

                        if frame_index % snapshot_interval == 0:
                            snapshot_dir = snapshots_root / f"track_{event.tracker_id}"
                            snapshot_dir.mkdir(parents=True, exist_ok=True)
                            snapshot_path = snapshot_dir / f"frame_{frame_index:06d}_{event.event_type}.jpg"
                            cv2.imwrite(str(snapshot_path), annotated_frame)

                    if frame_callback is not None:
                        frame_callback(frame_index, total_frames, annotated_frame, behaviors, events)

                    frame_index += 1
        finally:
            cap.release()

        return {
            "source": str(source),
            "output_root": str(destination_root),
            "events_path": str(events_path),
            "processed_frames": processed_frames,
            "emitted_events": emitted_events,
        }
