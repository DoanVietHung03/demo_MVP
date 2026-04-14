from __future__ import annotations

try:
    from ppe_detection.config import CONFIG
    from ppe_detection.pipeline import PPEDetectionPipeline
except ModuleNotFoundError:  # pragma: no cover - convenience for direct script execution
    from config import CONFIG
    from pipeline import PPEDetectionPipeline


def main() -> None:
    try:
        summary = PPEDetectionPipeline(CONFIG).process_video(
            source=CONFIG.source,
            output_root=CONFIG.output_root,
        )
    except Exception as exc:
        print(f"PPE detection failed: {exc}")
        raise SystemExit(1) from exc

    print("PPE detection completed.")
    print(f"Source: {summary['source']}")
    print(f"Output root: {summary['output_root']}")
    print(f"Detections path: {summary['detections_path']}")


if __name__ == "__main__":
    main()
