from __future__ import annotations

try:
    from facemask_detection.config import CONFIG
    from facemask_detection.pipeline import process_head_face_tree
except ModuleNotFoundError:  # pragma: no cover - convenience for direct script execution
    from config import CONFIG
    from pipeline import process_head_face_tree


def main() -> None:
    try:
        summary = process_head_face_tree(CONFIG)
    except FileNotFoundError as exc:
        print(f"Facemask detection could not start: {exc}")
        raise SystemExit(1) from exc
    print("Facemask detection completed.")
    print(f"Input root: {summary['input_root']}")
    print(f"Output root: {summary['output_root']}")
    print(f"Processed images: {summary['processed_count']}")
    print(f"Skipped images: {summary['skipped_count']}")
    print(f"Mask detections: {summary['mask_count']}")
    print(f"No-mask detections: {summary['no_mask_count']}")
    print(f"No detections: {summary['no_detection_count']}")


if __name__ == "__main__":
    main()
