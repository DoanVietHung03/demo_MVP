from __future__ import annotations

try:
    from person_region_extraction.config import CONFIG
    from person_region_extraction.pipeline import process_crop_tree
except ModuleNotFoundError:  # pragma: no cover - convenience for direct script execution
    from config import CONFIG
    from pipeline import process_crop_tree


def main() -> None:
    summary = process_crop_tree(CONFIG)
    print("Person region extraction completed.")
    print(f"Input crops: {summary['input_crops_dir']}")
    print(f"Output root: {summary['output_root']}")
    print(f"Processed images: {summary['processed_count']}")
    print(f"Skipped images: {summary['skipped_count']}")


if __name__ == "__main__":
    main()
