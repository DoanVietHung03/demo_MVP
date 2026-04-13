from __future__ import annotations

try:
    from glove_detection.config import CONFIG
    from glove_detection.pipeline import process_hands_tree
except ModuleNotFoundError:  # pragma: no cover - convenience for direct script execution
    from config import CONFIG
    from pipeline import process_hands_tree


def main() -> None:
    try:
        summary = process_hands_tree(CONFIG)
    except FileNotFoundError as exc:
        print(f"Glove detection could not start: {exc}")
        raise SystemExit(1) from exc
    except ImportError as exc:
        print(f"Glove detection runtime is unavailable: {exc}")
        raise SystemExit(1) from exc

    print("Glove detection completed.")
    print(f"Input root: {summary['input_root']}")
    print(f"Output root: {summary['output_root']}")
    print(f"Processed images: {summary['processed_count']}")
    print(f"Skipped images: {summary['skipped_count']}")
    print(f"Glove detections: {summary['glove_count']}")
    print(f"No-glove detections: {summary['no_glove_count']}")
    print(f"Uncertain detections: {summary['uncertain_count']}")


if __name__ == "__main__":
    main()
