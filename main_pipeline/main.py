from __future__ import annotations

try:
    from main_pipeline.config import CONFIG
    from main_pipeline.pipeline import MainPipeline
except ModuleNotFoundError:  # pragma: no cover - convenience for direct script execution
    from config import CONFIG
    from pipeline import MainPipeline


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


if __name__ == "__main__":
    main()
