from __future__ import annotations

from person_detection.config import CONFIG
from person_detection.pipeline import PersonDetectionPipeline


def main() -> None:
    pipeline = PersonDetectionPipeline(
        model_path=CONFIG.model_path,
        device=CONFIG.device,
        confidence_threshold=CONFIG.confidence_threshold,
        iou_threshold=CONFIG.iou_threshold,
        crop_padding=CONFIG.crop_padding,
        input_resize_scale=CONFIG.input_resize_scale,
        track_ttl_frames=CONFIG.track_ttl_frames,
        alert_cooldown_frames=CONFIG.alert_cooldown_frames,
    )
    print(f"Running person detection on device: {CONFIG.device}")
    pipeline.process_video(
        source=CONFIG.source,
        output_dir=CONFIG.output_dir,
        output_video_path=CONFIG.output_video_path,
        frame_skip=CONFIG.frame_skip,
        max_frames=CONFIG.max_frames,
    )


if __name__ == "__main__":
    main()
