from __future__ import annotations

try:
    from behavior_detection.config import CONFIG
    from behavior_detection.pipeline import BehaviorDetectionPipeline
except ModuleNotFoundError:  # pragma: no cover - convenience for direct script execution
    from config import CONFIG
    from pipeline import BehaviorDetectionPipeline


def main() -> None:
    pipeline = BehaviorDetectionPipeline(CONFIG)
    summary = pipeline.process_video(
        source=CONFIG.source,
        output_root=CONFIG.output_root,
    )
    print("Behavior detection completed.")
    print(f"Source: {summary['source']}")
    print(f"Output root: {summary['output_root']}")
    print(f"Events file: {summary['events_path']}")
    print(f"Processed frames: {summary['processed_frames']}")
    print(f"Emitted events: {summary['emitted_events']}")


if __name__ == "__main__":
    main()
