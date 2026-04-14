from __future__ import annotations

try:
    from stream_pipeline.config import CONFIG
    from stream_pipeline.pipeline import StreamPipeline
except ModuleNotFoundError:  # pragma: no cover - convenience for direct script execution
    from config import CONFIG
    from pipeline import StreamPipeline


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
