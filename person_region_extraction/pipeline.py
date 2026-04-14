from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover - import error is environment-specific
    raise ImportError(
        "Missing dependency 'ultralytics'. Install it with `pip install ultralytics`."
    ) from exc

try:
    from person_region_extraction.config import RegionExtractionConfig
except ModuleNotFoundError:  # pragma: no cover - convenience for direct script execution
    from config import RegionExtractionConfig

from runtime_support.onnx import (
    ONNXSessionInfo,
    format_runtime_backend_label,
    is_onnx_model_path,
    load_onnx_session,
    run_yolo_onnx_pose,
)


@dataclass
class RegionResult:
    head_face: np.ndarray
    hands: np.ndarray
    torso: np.ndarray
    face_bbox: Optional[Tuple[int, int, int, int]]
    hand_bboxes: List[Tuple[int, int, int, int]] = field(default_factory=list)
    hand_crop_boxes: List[Tuple[int, int, int, int]] = field(default_factory=list)


@dataclass
class PoseEstimate:
    keypoints_xy: np.ndarray
    keypoints_conf: np.ndarray


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
LEFT_HIP = 11
RIGHT_HIP = 12
YOLO_DEFAULT_IOU_THRESHOLD = 0.7


def _clip_box(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    clipped_x1 = max(0, min(x1, width - 1))
    clipped_y1 = max(0, min(y1, height - 1))
    clipped_x2 = max(clipped_x1 + 1, min(x2, width))
    clipped_y2 = max(clipped_y1 + 1, min(y2, height))
    return clipped_x1, clipped_y1, clipped_x2, clipped_y2


def _crop(image: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    return image[y1:y2, x1:x2].copy()


def _ensure_min_image_size(
    image: np.ndarray,
    min_size: int,
) -> np.ndarray:
    if image.size == 0 or min_size <= 1:
        return image

    height, width = image.shape[:2]
    if height >= min_size and width >= min_size:
        return image

    resized_width = max(width, min_size)
    resized_height = max(height, min_size)
    return cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )


def _concat_hand_regions(
    left: np.ndarray,
    right: np.ndarray,
) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
    if left.size == 0 and right.size == 0:
        return np.empty((0, 0, 3), dtype=np.uint8), []
    if left.size == 0:
        return right, [(0, 0, right.shape[1], right.shape[0])]
    if right.size == 0:
        return left, [(0, 0, left.shape[1], left.shape[0])]

    target_height = max(left.shape[0], right.shape[0])

    def resize_to_height(image: np.ndarray, height: int) -> np.ndarray:
        if image.shape[0] == height:
            return image
        width = max(1, int(round(image.shape[1] * (height / image.shape[0]))))
        return cv2.resize(image, (width, height))

    left_resized = resize_to_height(left, target_height)
    right_resized = resize_to_height(right, target_height)
    spacer = np.zeros((target_height, 8, 3), dtype=left.dtype)
    left_box = (0, 0, left_resized.shape[1], left_resized.shape[0])
    right_offset_x = left_resized.shape[1] + spacer.shape[1]
    right_box = (
        right_offset_x,
        0,
        right_offset_x + right_resized.shape[1],
        right_resized.shape[0],
    )
    return cv2.hconcat([left_resized, spacer, right_resized]), [left_box, right_box]


def _xyxy_from_points(
    points: List[Tuple[float, float]],
    width: int,
    height: int,
    padding: float = 0.0,
) -> Optional[Tuple[int, int, int, int]]:
    if not points:
        return None

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    pad_x = int(round((max(xs) - min(xs)) * padding))
    pad_y = int(round((max(ys) - min(ys)) * padding))
    return _clip_box(
        int(round(min(xs))) - pad_x,
        int(round(min(ys))) - pad_y,
        int(round(max(xs))) + pad_x,
        int(round(max(ys))) + pad_y,
        width,
        height,
    )


def _bbox_area(xyxy: np.ndarray) -> float:
    return max(1.0, float(xyxy[2] - xyxy[0])) * max(1.0, float(xyxy[3] - xyxy[1]))


def _bbox_contains_point(
    xyxy: np.ndarray,
    point: Tuple[float, float],
) -> bool:
    return (
        float(xyxy[0]) <= point[0] <= float(xyxy[2])
        and float(xyxy[1]) <= point[1] <= float(xyxy[3])
    )


def _select_primary_person_index(
    person_indices: np.ndarray,
    boxes_xyxy: np.ndarray,
    confidences: Optional[np.ndarray],
    image_shape: Tuple[int, int, int],
) -> int:
    if person_indices.size == 0:
        raise ValueError("person_indices must not be empty.")
    if person_indices.size == 1:
        return int(person_indices[0])

    height, width = image_shape[:2]
    crop_center = (width / 2.0, height / 2.0)
    crop_area = max(1.0, float(width * height))
    crop_diag = max(1.0, float(np.hypot(width, height)))

    best_index = int(person_indices[0])
    best_score: Optional[Tuple[int, float, float, float]] = None

    for person_index_raw in person_indices:
        person_index = int(person_index_raw)
        box = boxes_xyxy[person_index]
        box_center = (
            float(box[0] + box[2]) / 2.0,
            float(box[1] + box[3]) / 2.0,
        )
        center_distance = float(
            np.hypot(box_center[0] - crop_center[0], box_center[1] - crop_center[1])
        ) / crop_diag
        area_ratio = _bbox_area(box) / crop_area
        confidence_score = (
            float(confidences[person_index]) if confidences is not None else 0.0
        )
        candidate_score = (
            int(_bbox_contains_point(box, crop_center)),
            area_ratio,
            -center_distance,
            confidence_score,
        )
        if best_score is None or candidate_score > best_score:
            best_score = candidate_score
            best_index = person_index

    return best_index


class PersonRegionExtractor:
    def __init__(self, config: RegionExtractionConfig) -> None:
        self.config = config
        normalized_device = config.device.strip()
        if not normalized_device:
            raise ValueError("Device must not be empty.")
        if normalized_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested for pose extraction but is not available."
            )
        normalized_model_path = config.model_path.strip()
        if not normalized_model_path:
            raise ValueError("Model path must not be empty.")

        self.device = normalized_device
        self.backend = "torch"
        self.model = None
        self.onnx_session: Optional[ONNXSessionInfo] = None
        if is_onnx_model_path(normalized_model_path):
            self.backend = "onnx"
            self.onnx_session = load_onnx_session(normalized_model_path, self.device)
        else:
            self.model = YOLO(normalized_model_path).to(self.device)

    def runtime_backend_label(self) -> str:
        return format_runtime_backend_label(
            self.backend,
            self.device,
            self.onnx_session,
        )

    def _predict_pose(
        self,
        image: np.ndarray,
    ) -> Optional[PoseEstimate]:
        if self.backend == "onnx":
            if self.onnx_session is None:
                raise RuntimeError("ONNX pose session is not initialized.")
            boxes_xyxy, confidence_array, class_id_array, keypoints_array = run_yolo_onnx_pose(
                self.onnx_session,
                image,
                resize_scale=1.0,
                confidence_threshold=float(self.config.person_confidence_threshold),
                iou_threshold=YOLO_DEFAULT_IOU_THRESHOLD,
            )
            if boxes_xyxy.size == 0:
                return None

            person_indices = np.where(class_id_array.astype(int) == 0)[0]
            if person_indices.size == 0:
                return None

            chosen_index = _select_primary_person_index(
                person_indices,
                boxes_xyxy,
                confidence_array,
                image.shape,
            )
            keypoints_xy = keypoints_array[chosen_index, :, :2]
            if keypoints_array.shape[2] >= 3:
                keypoints_conf = keypoints_array[chosen_index, :, 2]
            else:
                keypoints_conf = np.ones(keypoints_xy.shape[0], dtype=np.float32)
            return PoseEstimate(
                keypoints_xy=keypoints_xy.astype(np.float32),
                keypoints_conf=keypoints_conf.astype(np.float32),
            )

        if self.model is None:
            raise RuntimeError("Pose extraction model is not initialized.")

        result = self.model.predict(
            source=image,
            device=self.device,
            conf=self.config.person_confidence_threshold,
            verbose=False,
        )[0]

        boxes = getattr(result, "boxes", None)
        keypoints = getattr(result, "keypoints", None)
        if boxes is None or keypoints is None or boxes.cls is None:
            return None

        cls_tensor = boxes.cls.detach().cpu().numpy()
        if cls_tensor.size == 0:
            return None

        person_indices = np.where(cls_tensor.astype(int) == 0)[0]
        if person_indices.size == 0:
            return None

        boxes_xyxy = boxes.xyxy.detach().cpu().numpy()
        conf_tensor = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else None
        chosen_index = _select_primary_person_index(
            person_indices,
            boxes_xyxy,
            conf_tensor,
            image.shape,
        )

        keypoints_xy = keypoints.xy[chosen_index].detach().cpu().numpy()
        keypoints_conf_tensor = getattr(keypoints, "conf", None)
        if keypoints_conf_tensor is None:
            keypoints_conf = np.ones(keypoints_xy.shape[0], dtype=np.float32)
        else:
            keypoints_conf = keypoints_conf_tensor[chosen_index].detach().cpu().numpy()

        return PoseEstimate(
            keypoints_xy=keypoints_xy.astype(np.float32),
            keypoints_conf=keypoints_conf.astype(np.float32),
        )

    def _get_visible_points(
        self,
        pose: PoseEstimate,
        indices: List[int],
    ) -> List[Tuple[float, float]]:
        visible_points: List[Tuple[float, float]] = []
        threshold = self.config.keypoint_confidence_threshold
        for index in indices:
            if index >= pose.keypoints_xy.shape[0]:
                continue
            if float(pose.keypoints_conf[index]) < threshold:
                continue
            x_coord = float(pose.keypoints_xy[index][0])
            y_coord = float(pose.keypoints_xy[index][1])
            visible_points.append((x_coord, y_coord))
        return visible_points

    def _estimate_head_face_bbox(
        self,
        image: np.ndarray,
        pose: Optional[PoseEstimate],
    ) -> Tuple[int, int, int, int]:
        height, width = image.shape[:2]

        if pose is not None:
            face_points = self._get_visible_points(
                pose,
                [NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR],
            )
            shoulder_points = self._get_visible_points(
                pose,
                [LEFT_SHOULDER, RIGHT_SHOULDER],
            )
            if face_points:
                base_box = _xyxy_from_points(
                    face_points,
                    width,
                    height,
                    padding=self.config.face_padding_ratio,
                )
                if base_box is not None:
                    x1, y1, x2, y2 = base_box
                    if shoulder_points:
                        shoulder_y = min(point[1] for point in shoulder_points)
                        head_height = max(y2 - y1, int(shoulder_y - y1))
                        y2 = min(height, max(y2, y1 + head_height))
                    return _clip_box(x1, y1, x2, y2, width, height)

        head_height = max(1, int(height * self.config.head_height_ratio))
        head_width = max(1, int(width * 0.56))
        x1 = (width - head_width) // 2
        x2 = x1 + head_width
        y1 = 0
        y2 = head_height
        return _clip_box(x1, y1, x2, y2, width, height)

    def _estimate_torso_bbox(
        self,
        image: np.ndarray,
        pose: Optional[PoseEstimate],
        head_face_bbox: Tuple[int, int, int, int],
    ) -> Tuple[int, int, int, int]:
        height, width = image.shape[:2]

        if pose is not None:
            torso_points = self._get_visible_points(
                pose,
                [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP],
            )
            if len(torso_points) >= 2:
                torso_box = _xyxy_from_points(
                    torso_points,
                    width,
                    height,
                    padding=0.18,
                )
                if torso_box is not None:
                    return torso_box

        _, _, _, head_bottom = head_face_bbox
        torso_top = max(head_bottom, int(height * self.config.torso_top_ratio))
        torso_bottom = int(height * self.config.torso_bottom_ratio)
        torso_width = int(width * self.config.torso_width_ratio)
        x1 = (width - torso_width) // 2
        x2 = x1 + torso_width
        return _clip_box(x1, torso_top, x2, torso_bottom, width, height)

    def _estimate_hand_box(
        self,
        image: np.ndarray,
        pose: PoseEstimate,
        wrist_index: int,
        elbow_index: int,
    ) -> Optional[Tuple[int, int, int, int]]:
        visible_points = self._get_visible_points(pose, [wrist_index, elbow_index])
        if not visible_points:
            return None

        width = image.shape[1]
        height = image.shape[0]
        wrist_point = self._get_visible_points(pose, [wrist_index])
        center_x, center_y = visible_points[0]
        if wrist_point:
            center_x, center_y = wrist_point[0]

        size = max(24.0, min(width, height) * self.config.hand_box_size_ratio)
        if len(visible_points) == 2:
            dx = visible_points[0][0] - visible_points[1][0]
            dy = visible_points[0][1] - visible_points[1][1]
            size = max(size, (dx * dx + dy * dy) ** 0.5 * 1.2)

        half_size = int(round(size / 2.0))
        return _clip_box(
            int(round(center_x)) - half_size,
            int(round(center_y)) - half_size,
            int(round(center_x)) + half_size,
            int(round(center_y)) + half_size,
            width,
            height,
        )

    def _estimate_hands_regions(
        self,
        image: np.ndarray,
        pose: Optional[PoseEstimate],
    ) -> Tuple[
        np.ndarray,
        List[Tuple[int, int, int, int]],
        List[Tuple[int, int, int, int]],
    ]:
        if pose is not None:
            left_box = self._estimate_hand_box(image, pose, LEFT_WRIST, LEFT_ELBOW)
            right_box = self._estimate_hand_box(image, pose, RIGHT_WRIST, RIGHT_ELBOW)
            if left_box is not None or right_box is not None:
                left_crop = (
                    _crop(image, left_box)
                    if left_box is not None
                    else np.empty((0, 0, 3), dtype=np.uint8)
                )
                right_crop = (
                    _crop(image, right_box)
                    if right_box is not None
                    else np.empty((0, 0, 3), dtype=np.uint8)
                )
                hand_bboxes = [
                    hand_box
                    for hand_box in (left_box, right_box)
                    if hand_box is not None
                ]
                hand_image, hand_crop_boxes = _concat_hand_regions(left_crop, right_crop)
                return hand_image, hand_bboxes, hand_crop_boxes

        height, width = image.shape[:2]
        y1 = int(height * self.config.hands_top_ratio)
        y2 = int(height * self.config.hands_bottom_ratio)
        side_width = max(1, int(width * self.config.hand_side_width_ratio))
        left_box = _clip_box(0, y1, side_width, y2, width, height)
        right_box = _clip_box(width - side_width, y1, width, y2, width, height)
        hand_image, hand_crop_boxes = _concat_hand_regions(
            _crop(image, left_box),
            _crop(image, right_box),
        )
        return hand_image, [left_box, right_box], hand_crop_boxes

    def _ensure_min_face_crop_size(self, head_face_crop: np.ndarray) -> np.ndarray:
        return _ensure_min_image_size(
            head_face_crop,
            int(self.config.min_face_size),
        )

    def split_regions(self, image: np.ndarray) -> RegionResult:
        pose = self._predict_pose(image)
        head_face_bbox = self._estimate_head_face_bbox(image, pose)
        torso_bbox = self._estimate_torso_bbox(image, pose, head_face_bbox)
        hands_crop, hand_bboxes, hand_crop_boxes = self._estimate_hands_regions(image, pose)
        head_face_crop = self._ensure_min_face_crop_size(
            _crop(image, head_face_bbox)
        )
        return RegionResult(
            head_face=head_face_crop,
            hands=hands_crop,
            torso=_crop(image, torso_bbox),
            face_bbox=head_face_bbox,
            hand_bboxes=hand_bboxes,
            hand_crop_boxes=hand_crop_boxes,
        )


def resolve_input_crops_dir(config: RegionExtractionConfig) -> Tuple[Path, str]:
    if config.input_crops_dir:
        input_dir = Path(config.input_crops_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"Input crops directory not found: {input_dir}")
        return input_dir, input_dir.parent.name

    if not config.process_latest_job:
        raise ValueError(
            "input_crops_dir must be set when process_latest_job is False."
        )

    results_root = Path(config.results_root)
    if not results_root.exists():
        raise FileNotFoundError(f"Results root not found: {results_root}")

    candidate_jobs = [
        job_dir
        for job_dir in results_root.iterdir()
        if job_dir.is_dir() and (job_dir / "crops").exists()
    ]
    if not candidate_jobs:
        raise FileNotFoundError(f"No crop jobs found inside: {results_root}")

    latest_job_dir = max(candidate_jobs, key=lambda path: path.stat().st_mtime)
    return latest_job_dir / "crops", latest_job_dir.name


def iter_crop_images(crops_dir: Path) -> Iterable[Path]:
    for image_path in sorted(crops_dir.glob("track_*/*.jpg")):
        if image_path.is_file():
            yield image_path


def process_crop_tree(config: RegionExtractionConfig) -> Dict[str, object]:
    input_crops_dir, job_name = resolve_input_crops_dir(config)
    output_root = Path(config.output_root) / job_name
    output_root.mkdir(parents=True, exist_ok=True)

    extractor = PersonRegionExtractor(config)
    runtime_backend = extractor.runtime_backend_label()
    print(
        "[Stage Runtime] "
        f"person_region_extraction: {runtime_backend} | "
        f"device={extractor.device} | model={config.model_path}"
    )
    processed_count = 0
    skipped_count = 0

    for image_path in iter_crop_images(input_crops_dir):
        image = cv2.imread(str(image_path))
        if image is None or image.size == 0:
            skipped_count += 1
            continue

        regions = extractor.split_regions(image)
        frame_output_dir = output_root / image_path.parent.name / image_path.stem
        frame_output_dir.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(frame_output_dir / "head_face.jpg"), regions.head_face)
        cv2.imwrite(str(frame_output_dir / "hands.jpg"), regions.hands)
        cv2.imwrite(str(frame_output_dir / "torso.jpg"), regions.torso)
        (frame_output_dir / "regions.json").write_text(
            json.dumps(
                {
                    "face_bbox": list(regions.face_bbox) if regions.face_bbox else None,
                    "hand_bboxes": [list(box) for box in regions.hand_bboxes],
                    "hand_crop_boxes": [list(box) for box in regions.hand_crop_boxes],
                    "head_face_shape": list(regions.head_face.shape),
                    "hands_shape": list(regions.hands.shape),
                    "torso_shape": list(regions.torso.shape),
                    "source_crop_shape": list(image.shape),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        processed_count += 1

    return {
        "job_name": job_name,
        "input_crops_dir": str(input_crops_dir),
        "output_root": str(output_root),
        "runtime_backend": runtime_backend,
        "model_path": config.model_path,
        "processed_count": processed_count,
        "skipped_count": skipped_count,
    }
