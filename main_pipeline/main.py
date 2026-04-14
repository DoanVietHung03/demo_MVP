from __future__ import annotations

import sys


if __package__ in (None, ""):  # pragma: no cover - direct script execution helper
    package_dir = sys.path[0] or "."
    normalized_package_dir = package_dir.rstrip("/\\")
    separator_index = max(
        normalized_package_dir.rfind("/"),
        normalized_package_dir.rfind("\\"),
    )
    repo_root = (
        normalized_package_dir[:separator_index]
        if separator_index >= 0
        else "."
    )
    if package_dir in sys.path:
        sys.path.remove(package_dir)
    sys.path.insert(0, repo_root or ".")

from main_pipeline.config import CONFIG
from main_pipeline.pipeline import MainPipeline


def main() -> None:
    try:
        summary = MainPipeline(CONFIG).run()
    except Exception as exc:
        print(f"Main pipeline failed: {exc}")
        raise SystemExit(1) from exc

    print("Main pipeline completed.")
    print(f"Job name: {summary['job_name']}")
    print(f"Source: {summary['source']}")
    print(f"Output root: {summary['output_root']}")
    print(f"Summary path: {summary['paths']['summary_path']}")
    if summary.get("annotated_video"):
        print(f"Annotated video: {summary['annotated_video']['path']}")


if __name__ == "__main__":
    main()
