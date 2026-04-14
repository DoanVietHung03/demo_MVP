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

from stream_pipeline.config import CONFIG
from stream_pipeline.pipeline import StreamPipeline


def main() -> None:
    try:
        summary = StreamPipeline(CONFIG).run()
    except Exception as exc:
        print(f"Stream pipeline failed: {exc}")
        raise SystemExit(1) from exc

    print("Stream pipeline completed.")
    print(f"Job name: {summary['job_name']}")
    print(f"Source: {summary['source']}")
    print(f"Output root: {summary['output_root']}")
    print(f"Summary path: {summary['paths']['summary_path']}")


if __name__ == "__main__":
    main()
