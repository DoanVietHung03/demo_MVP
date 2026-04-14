from __future__ import annotations

import shutil
import tempfile
import threading
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

import cv2
from flask import Flask, Response, abort, redirect, render_template, request, url_for

from person_detection.config import CONFIG, PersonInstance
from stream_pipeline.config import CONFIG as STREAM_PIPELINE_CONFIG, StreamPipelineConfig
from stream_pipeline.pipeline import StreamPipeline
from stream_pipeline.types import StageRunResult


APP_ROOT = Path(__file__).resolve().parent.parent
WEB_OUTPUT_DIR = APP_ROOT / "web_output"
UPLOAD_DIR = WEB_OUTPUT_DIR / "uploads"
RESULT_DIR = WEB_OUTPUT_DIR / "results"

WEB_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024

JOB_STORE: Dict[str, Dict[str, object]] = {}
JOB_STORE_LOCK = threading.Lock()
ACTIVE_JOB_STATUSES = {"ready", "initializing", "running", "streaming"}
MAX_CONCURRENT_WEB_JOBS = 1
MAX_RETAINED_TERMINAL_JOBS = 10
PIPELINE_STAGE_ORDER = [
    "person_detection",
    "person_region_extraction",
    "facemask_detection",
    "glove_detection",
    "behavior_detection",
]
PIPELINE_STAGE_LABELS = {
    "person_detection": "Person Detection",
    "person_region_extraction": "Region Extraction",
    "facemask_detection": "Facemask Detection",
    "glove_detection": "Glove Detection",
    "behavior_detection": "Behavior Detection",
}


def _save_uploaded_file(uploaded_file) -> Path:
    suffix = Path(uploaded_file.filename or "upload.mp4").suffix or ".mp4"
    temp_file = tempfile.NamedTemporaryFile(delete=False, dir=UPLOAD_DIR, suffix=suffix)
    uploaded_file.save(temp_file.name)
    return Path(temp_file.name)


def _looks_like_stream_source(value: str) -> bool:
    return "://" in value


def _resolve_source_input(source_path: str) -> Tuple[Union[str, Path], str]:
    if _looks_like_stream_source(source_path):
        parsed = urlparse(source_path)
        stream_name = Path(parsed.path).name or parsed.netloc or source_path
        return source_path, stream_name

    file_path = Path(source_path)
    return file_path, file_path.name


def _parse_optional_int(raw_value: str) -> Optional[int]:
    value = (raw_value or "").strip()
    return int(value) if value else None


def _parse_checkbox(raw_value: Optional[str]) -> bool:
    return str(raw_value or "").strip().lower() in {"1", "true", "on", "yes"}


def _normalize_stage_selection(
    run_person_detection: bool,
    run_region_extraction: bool,
    run_facemask_detection: bool,
    run_glove_detection: bool,
    run_behavior_detection: bool,
) -> Dict[str, bool]:
    normalized = {
        "run_person_detection": bool(run_person_detection),
        "run_region_extraction": bool(run_region_extraction),
        "run_facemask_detection": bool(run_facemask_detection),
        "run_glove_detection": bool(run_glove_detection),
        "run_behavior_detection": bool(run_behavior_detection),
    }

    if normalized["run_facemask_detection"] or normalized["run_glove_detection"]:
        normalized["run_region_extraction"] = True
    if normalized["run_region_extraction"]:
        normalized["run_person_detection"] = True

    if not any(normalized.values()):
        raise ValueError("Hãy bật ít nhất 1 stage để chạy job.")

    return normalized


def _initial_stage_states(stage_flags: Dict[str, bool]) -> Dict[str, Dict[str, object]]:
    return {
        "person_detection": {
            "enabled": bool(stage_flags["run_person_detection"]),
            "status": "pending" if stage_flags["run_person_detection"] else "disabled",
        },
        "person_region_extraction": {
            "enabled": bool(stage_flags["run_region_extraction"]),
            "status": "pending" if stage_flags["run_region_extraction"] else "disabled",
        },
        "facemask_detection": {
            "enabled": bool(stage_flags["run_facemask_detection"]),
            "status": "pending" if stage_flags["run_facemask_detection"] else "disabled",
        },
        "glove_detection": {
            "enabled": bool(stage_flags["run_glove_detection"]),
            "status": "pending" if stage_flags["run_glove_detection"] else "disabled",
        },
        "behavior_detection": {
            "enabled": bool(stage_flags["run_behavior_detection"]),
            "status": "pending" if stage_flags["run_behavior_detection"] else "disabled",
        },
    }


def _stage_views(job: Dict[str, object]) -> List[Dict[str, object]]:
    lock = job["lock"]
    with lock:
        raw_stage_states = job.get("stage_states")

    stage_states = raw_stage_states if isinstance(raw_stage_states, dict) else {}
    views: List[Dict[str, object]] = []
    for stage_name in PIPELINE_STAGE_ORDER:
        raw_state = stage_states.get(stage_name, {})
        enabled = bool(raw_state.get("enabled", stage_name != "behavior_detection"))
        status = str(raw_state.get("status", "pending" if enabled else "disabled"))
        views.append(
            {
                "name": stage_name,
                "label": PIPELINE_STAGE_LABELS.get(stage_name, stage_name),
                "enabled": enabled,
                "status": status,
                "output_path": raw_state.get("output_path"),
                "error": raw_state.get("error"),
            }
        )
    return views


def _defaults() -> Dict[str, object]:
    return {
        "source": CONFIG.source,
        "model_path": CONFIG.model_path,
        "device": CONFIG.device,
        "confidence_threshold": CONFIG.confidence_threshold,
        "iou_threshold": CONFIG.iou_threshold,
        "crop_padding": CONFIG.crop_padding,
        "input_resize_scale": CONFIG.input_resize_scale,
        "frame_skip": CONFIG.frame_skip,
        "crop_save_interval": CONFIG.crop_save_interval,
        "track_ttl_frames": CONFIG.track_ttl_frames,
        "alert_cooldown_frames": CONFIG.alert_cooldown_frames,
        "max_frames": CONFIG.max_frames,
        "run_person_detection": STREAM_PIPELINE_CONFIG.run_person_detection,
        "run_region_extraction": STREAM_PIPELINE_CONFIG.run_region_extraction,
        "run_facemask_detection": STREAM_PIPELINE_CONFIG.run_facemask_detection,
        "run_glove_detection": STREAM_PIPELINE_CONFIG.run_glove_detection,
        "run_behavior_detection": STREAM_PIPELINE_CONFIG.run_behavior_detection,
    }


def _defaults_from_form(form) -> Dict[str, object]:
    defaults = _defaults()
    defaults["source"] = (form.get("source_path") or defaults["source"]).strip()
    defaults["model_path"] = (form.get("model_path") or defaults["model_path"]).strip()
    defaults["device"] = (form.get("device") or defaults["device"]).strip()
    defaults["confidence_threshold"] = (
        form.get("confidence_threshold") or defaults["confidence_threshold"]
    )
    defaults["iou_threshold"] = form.get("iou_threshold") or defaults["iou_threshold"]
    defaults["crop_padding"] = form.get("crop_padding") or defaults["crop_padding"]
    defaults["input_resize_scale"] = (
        form.get("input_resize_scale") or defaults["input_resize_scale"]
    )
    defaults["frame_skip"] = form.get("frame_skip") or defaults["frame_skip"]
    defaults["track_ttl_frames"] = (
        form.get("track_ttl_frames") or defaults["track_ttl_frames"]
    )
    defaults["alert_cooldown_frames"] = (
        form.get("alert_cooldown_frames") or defaults["alert_cooldown_frames"]
    )
    defaults["max_frames"] = form.get("max_frames") or defaults["max_frames"]
    defaults["run_person_detection"] = _parse_checkbox(form.get("run_person_detection"))
    defaults["run_region_extraction"] = _parse_checkbox(form.get("run_region_extraction"))
    defaults["run_facemask_detection"] = _parse_checkbox(form.get("run_facemask_detection"))
    defaults["run_glove_detection"] = _parse_checkbox(form.get("run_glove_detection"))
    defaults["run_behavior_detection"] = _parse_checkbox(form.get("run_behavior_detection"))
    return defaults


def _snapshot_jobs() -> List[tuple[str, Dict[str, object]]]:
    with JOB_STORE_LOCK:
        return list(JOB_STORE.items())


def _get_job(job_id: str) -> Optional[Dict[str, object]]:
    with JOB_STORE_LOCK:
        return JOB_STORE.get(job_id)


def _job_created_at(job: Dict[str, object]) -> float:
    lock = job["lock"]
    with lock:
        return float(job.get("created_at", 0.0))


def _job_finished_at(job: Dict[str, object]) -> float:
    lock = job["lock"]
    with lock:
        finished_at = job.get("finished_at")
        created_at = job.get("created_at", 0.0)
    if finished_at is None:
        return float(created_at)
    return float(finished_at)


def _job_status(job: Dict[str, object]) -> str:
    lock = job["lock"]
    with lock:
        return str(job["status"])


def _display_stage_name(stage_name: Optional[object]) -> Optional[str]:
    if stage_name is None:
        return None
    normalized = str(stage_name)
    if normalized == "realtime_stream":
        return "realtime stream"
    return normalized


def _is_terminal_status(status: str) -> bool:
    return status == "completed" or status.startswith("error")


def _is_active_status(status: str) -> bool:
    return status in ACTIVE_JOB_STATUSES


def _sorted_job_items() -> List[tuple[str, Dict[str, object]]]:
    items = _snapshot_jobs()
    items.sort(key=lambda item: _job_created_at(item[1]), reverse=True)
    return items


def _cleanup_path(path_value: Optional[object], is_dir: bool) -> None:
    if not path_value:
        return

    path = Path(str(path_value))
    if is_dir:
        shutil.rmtree(path, ignore_errors=True)
        return

    try:
        path.unlink()
    except FileNotFoundError:
        return
    except IsADirectoryError:
        shutil.rmtree(path, ignore_errors=True)


def _cleanup_job_artifacts(job: Dict[str, object]) -> None:
    lock = job["lock"]
    with lock:
        job_dir = job.get("job_dir")
        source_is_upload = bool(job.get("source_is_upload"))
        uploaded_source_path = job.get("uploaded_source_path") if source_is_upload else None

    _cleanup_path(job_dir, is_dir=True)
    _cleanup_path(uploaded_source_path, is_dir=False)


def _prune_finished_jobs(
    max_terminal_jobs: int = MAX_RETAINED_TERMINAL_JOBS,
) -> None:
    terminal_jobs: List[tuple[float, str]] = []
    for job_id, job in _sorted_job_items():
        status = _job_status(job)
        if _is_terminal_status(status):
            terminal_jobs.append((_job_finished_at(job), job_id))

    terminal_jobs.sort(reverse=True)
    for _, job_id in terminal_jobs[max_terminal_jobs:]:
        with JOB_STORE_LOCK:
            job = JOB_STORE.pop(job_id, None)
        if job is not None:
            _cleanup_job_artifacts(job)


def _job_to_view(job_id: str, job: Dict[str, object]) -> Dict[str, object]:
    lock = job["lock"]
    with lock:
        view = {
            "job_id": job_id,
            "source_name": job["source_name"],
            "device": job["device"],
            "status": job["status"],
            "current_stage": _display_stage_name(job.get("current_stage")),
            "total_alerts": job["total_alerts"],
            "unique_track_count": job["unique_track_count"],
            "crop_dir": job["crop_dir"],
            "pipeline_output_root": job.get("pipeline_output_root"),
            "summary_path": job.get("summary_path"),
        }
    view["stage_views"] = _stage_views(job)
    return view


def _get_active_job_views() -> List[Dict[str, object]]:
    active_jobs: List[Dict[str, object]] = []
    for job_id, job in _sorted_job_items():
        if _is_active_status(_job_status(job)):
            active_jobs.append(_job_to_view(job_id, job))
    return active_jobs


def _resolve_current_job_view(selected_job_id: Optional[str]) -> Optional[Dict[str, object]]:
    if selected_job_id:
        job = _get_job(selected_job_id)
        if job is not None:
            return _job_to_view(selected_job_id, job)

    active_jobs = _get_active_job_views()
    if active_jobs:
        return active_jobs[0]
    return None


def _update_job_stage_state(
    job_id: str,
    stage_name: str,
    stage_status: str,
    stage_result: Optional[StageRunResult] = None,
) -> None:
    job = _get_job(job_id)
    if job is None:
        return

    with job["lock"]:
        raw_stage_states = job.get("stage_states")
        stage_states = dict(raw_stage_states) if isinstance(raw_stage_states, dict) else {}
        stage_state = dict(stage_states.get(stage_name, {}))
        realtime_mode = str(job.get("pipeline_mode") or "").strip().lower() == "realtime"
        stage_state["enabled"] = (
            stage_result.enabled
            if stage_result is not None
            else bool(stage_state.get("enabled", True))
        )
        stage_state["status"] = stage_status

        if stage_result is not None:
            stage_state["success"] = stage_result.success
            stage_state["input_path"] = stage_result.input_path
            stage_state["output_path"] = stage_result.output_path
            stage_state["summary"] = dict(stage_result.summary)
            stage_state["error"] = stage_result.error

        stage_states[stage_name] = stage_state
        job["stage_states"] = stage_states

        if stage_status == "running":
            if realtime_mode:
                job["current_stage"] = "realtime_stream"
                job["status"] = "streaming"
            else:
                job["current_stage"] = stage_name
                job["status"] = "streaming" if stage_name == "person_detection" else "running"
        elif stage_status == "failed":
            job["current_stage"] = stage_name
            error_message = stage_state.get("error") or "unknown stage failure"
            job["status"] = f"error: {error_message}"
        elif stage_status == "completed":
            if not realtime_mode:
                job["current_stage"] = stage_name
        elif stage_status == "disabled" and job.get("current_stage") == stage_name:
            job["current_stage"] = None


def _build_stream_pipeline_config(
    job_id: str,
    job: Dict[str, object],
) -> StreamPipelineConfig:
    config = replace(STREAM_PIPELINE_CONFIG)
    config.source = str(job["source"])
    config.output_root = str(RESULT_DIR)
    config.job_name = job_id
    config.run_person_detection = bool(job.get("run_person_detection"))
    config.run_region_extraction = bool(job.get("run_region_extraction"))
    config.run_facemask_detection = bool(job.get("run_facemask_detection"))
    config.run_glove_detection = bool(job.get("run_glove_detection"))
    config.run_behavior_detection = bool(job.get("run_behavior_detection"))

    person_config = replace(config.person_detection)
    person_config.source = str(job["source"])
    person_config.model_path = str(job["model_path"])
    person_config.device = str(job["device"])
    person_config.confidence_threshold = float(job["confidence_threshold"])
    person_config.iou_threshold = float(job["iou_threshold"])
    person_config.crop_padding = int(job["crop_padding"])
    person_config.input_resize_scale = float(job["input_resize_scale"])
    person_config.frame_skip = int(job["frame_skip"])
    person_config.crop_save_interval = int(job["crop_save_interval"])
    person_config.track_ttl_frames = int(job["track_ttl_frames"])
    person_config.alert_cooldown_frames = job["alert_cooldown_frames"]
    person_config.max_frames = job["max_frames"]
    config.person_detection = person_config

    region_config = replace(config.person_region_extraction)
    region_config.device = str(job["device"])
    config.person_region_extraction = region_config

    facemask_config = replace(config.facemask_detection)
    facemask_config.device = str(job["device"])
    config.facemask_detection = facemask_config

    behavior_config = replace(config.behavior_detection)
    behavior_config.source = str(job["source"])
    behavior_config.device = str(job["device"])
    behavior_config.frame_skip = int(job["frame_skip"])
    behavior_config.max_frames = job["max_frames"]
    behavior_config.iou_threshold = float(job["iou_threshold"])
    behavior_config.input_resize_scale = float(job["input_resize_scale"])
    behavior_config.crop_padding = int(job["crop_padding"])
    behavior_config.track_ttl_frames = int(job["track_ttl_frames"])
    behavior_config.alert_cooldown_frames = int(job["alert_cooldown_frames"] or 90)
    config.behavior_detection = behavior_config

    return config


@app.get("/")
def index():
    _prune_finished_jobs()
    selected_job_id = request.args.get("job_id")
    current_job = _resolve_current_job_view(selected_job_id)
    return render_template(
        "index.html",
        defaults=_defaults(),
        current_job=current_job,
        error=None,
    )


@app.post("/start")
def start_job():
    _prune_finished_jobs()
    form_defaults = _defaults_from_form(request.form)
    active_jobs = _get_active_job_views()
    if len(active_jobs) >= MAX_CONCURRENT_WEB_JOBS:
        return render_template(
            "index.html",
            defaults=form_defaults,
            current_job=active_jobs[0],
            error=(
                "Hiện chỉ cho phép 1 job đang chạy cùng lúc để tránh quá tải model/GPU. "
                "Hãy chờ job hiện tại hoàn tất rồi thử lại."
            ),
        )

    source_file = request.files.get("source_file")
    source_path = (request.form.get("source_path") or "").strip()
    is_uploaded_source = bool(source_file and source_file.filename)

    if is_uploaded_source:
        video_source = _save_uploaded_file(source_file)
        source_name = source_file.filename
    elif source_path:
        video_source, source_name = _resolve_source_input(source_path)
    else:
        return render_template(
            "index.html",
            defaults=form_defaults,
            current_job=None,
            error="Hãy nhập đường dẫn video trên server hoặc upload một file MP4.",
        )

    try:
        stage_flags = _normalize_stage_selection(
            run_person_detection=_parse_checkbox(request.form.get("run_person_detection")),
            run_region_extraction=_parse_checkbox(request.form.get("run_region_extraction")),
            run_facemask_detection=_parse_checkbox(request.form.get("run_facemask_detection")),
            run_glove_detection=_parse_checkbox(request.form.get("run_glove_detection")),
            run_behavior_detection=_parse_checkbox(request.form.get("run_behavior_detection")),
        )
    except ValueError as exc:
        return render_template(
            "index.html",
            defaults=form_defaults,
            current_job=None,
            error=str(exc),
        )

    job_id = uuid.uuid4().hex[:12]
    job_dir = RESULT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    created_at = time.time()

    job_payload: Dict[str, object] = {
        "source": str(video_source),
        "source_name": source_name,
        "model_path": (request.form.get("model_path") or CONFIG.model_path).strip(),
        "device": (request.form.get("device") or CONFIG.device).strip(),
        "confidence_threshold": float(
            request.form.get("confidence_threshold") or CONFIG.confidence_threshold
        ),
        "iou_threshold": float(request.form.get("iou_threshold") or CONFIG.iou_threshold),
        "crop_padding": int(request.form.get("crop_padding") or CONFIG.crop_padding),
        "input_resize_scale": float(
            request.form.get("input_resize_scale") or CONFIG.input_resize_scale
        ),
        "frame_skip": int(request.form.get("frame_skip") or CONFIG.frame_skip),
        "crop_save_interval": CONFIG.crop_save_interval,
        "track_ttl_frames": int(
            request.form.get("track_ttl_frames") or CONFIG.track_ttl_frames
        ),
        "alert_cooldown_frames": _parse_optional_int(
            request.form.get("alert_cooldown_frames")
        ),
        "max_frames": _parse_optional_int(request.form.get("max_frames")),
        "crop_dir": str(job_dir / "crops"),
        "job_dir": str(job_dir),
        "pipeline_output_root": str(job_dir),
        "summary_path": str(job_dir / "summary.json"),
        "run_person_detection": stage_flags["run_person_detection"],
        "run_region_extraction": stage_flags["run_region_extraction"],
        "run_facemask_detection": stage_flags["run_facemask_detection"],
        "run_glove_detection": stage_flags["run_glove_detection"],
        "run_behavior_detection": stage_flags["run_behavior_detection"],
        "source_is_upload": is_uploaded_source,
        "uploaded_source_path": str(video_source) if is_uploaded_source else None,
        "status": "ready",
        "current_stage": None,
        "total_alerts": 0,
        "unique_track_count": 0,
        "last_frame_index": -1,
        "unique_ids": set(),
        "latest_frame_jpeg": None,
        "latest_frame_seq": -1,
        "latest_summary": None,
        "stage_states": _initial_stage_states(stage_flags),
        "pipeline_mode": "realtime",
        "created_at": created_at,
        "finished_at": None,
        "lock": threading.Lock(),
    }
    with JOB_STORE_LOCK:
        JOB_STORE[job_id] = job_payload
    worker = threading.Thread(target=run_job_worker, args=(job_id,), daemon=True)
    with JOB_STORE_LOCK:
        JOB_STORE[job_id]["worker"] = worker
    worker.start()
    return redirect(url_for("index", job_id=job_id))


def _update_job_outputs(
    job_id: str,
    annotated_frame,
    persons: List[PersonInstance],
    frame_index: int,
) -> None:
    job = _get_job(job_id)
    if job is None:
        return
    seen_tracker_ids = set()
    new_alert_count = 0

    for person in persons:
        seen_tracker_ids.add(person.tracker_id)
        if person.should_alert:
            new_alert_count += 1

    latest_frame_jpeg = None
    ok_encode, buffer = cv2.imencode(
        ".jpg", annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 72]
    )
    if ok_encode:
        latest_frame_jpeg = buffer.tobytes()

    with job["lock"]:
        unique_ids = job.get("unique_ids")
        if not isinstance(unique_ids, set):
            unique_ids = set()
        unique_ids.update(seen_tracker_ids)
        job["unique_ids"] = unique_ids
        job["unique_track_count"] = len(unique_ids)
        job["total_alerts"] = int(job.get("total_alerts", 0)) + new_alert_count
        job["last_frame_index"] = frame_index
        if latest_frame_jpeg is not None:
            job["latest_frame_jpeg"] = latest_frame_jpeg
            job["latest_frame_seq"] = frame_index


def run_job_worker(job_id: str) -> None:
    job = _get_job(job_id)
    if job is None:
        return

    with job["lock"]:
        job["pipeline_mode"] = str(job.get("pipeline_mode") or "realtime")
        job["unique_ids"] = set()
        job["total_alerts"] = 0
        job["unique_track_count"] = 0
        job["last_frame_index"] = -1
        job["latest_frame_jpeg"] = None
        job["latest_frame_seq"] = -1
        job["latest_summary"] = None
        job["current_stage"] = None
        job["stage_states"] = _initial_stage_states(
            {
                "run_person_detection": bool(job.get("run_person_detection")),
                "run_region_extraction": bool(job.get("run_region_extraction")),
                "run_facemask_detection": bool(job.get("run_facemask_detection")),
                "run_glove_detection": bool(job.get("run_glove_detection")),
                "run_behavior_detection": bool(job.get("run_behavior_detection")),
            }
        )
        job["finished_at"] = None
        job["status"] = "initializing"

    try:
        pipeline_config = _build_stream_pipeline_config(job_id, job)
        pipeline = StreamPipeline(
            pipeline_config,
            stage_callback=lambda stage_name, stage_status, stage_result: _update_job_stage_state(
                job_id,
                stage_name,
                stage_status,
                stage_result,
            ),
            frame_callback=lambda frame_index, _total_frames, annotated_frame, persons: _update_job_outputs(
                job_id,
                annotated_frame,
                persons,
                frame_index,
            ),
        )
        summary = pipeline.run()
        with job["lock"]:
            job["status"] = "completed"
            job["current_stage"] = "completed"
            job["crop_dir"] = str(summary["paths"]["crops_dir"])
            job["pipeline_output_root"] = str(summary["output_root"])
            job["summary_path"] = str(summary["paths"]["summary_path"])
            job["latest_summary"] = summary
            job["finished_at"] = time.time()
    except Exception as exc:
        with job["lock"]:
            job["status"] = f"error: {exc}"
            job["finished_at"] = time.time()
    finally:
        _prune_finished_jobs()


def generate_web_stream(job_id: str):
    job = _get_job(job_id)
    if job is None:
        return

    last_frame_seq = -1
    while True:
        with job["lock"]:
            frame_bytes = job["latest_frame_jpeg"]
            frame_seq = job["latest_frame_seq"]
            status = str(job["status"])

        if frame_bytes is not None and frame_seq != last_frame_seq:
            last_frame_seq = frame_seq
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame_bytes
                + b"\r\n"
            )
            continue

        if status == "completed" or status.startswith("error"):
            break

        time.sleep(0.03)


@app.get("/video_feed/<job_id>")
def video_feed(job_id: str):
    if _get_job(job_id) is None:
        abort(404)
    return Response(
        generate_web_stream(job_id),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/history")
def history():
    _prune_finished_jobs()
    jobs = [
        _job_to_view(job_id, job)
        for job_id, job in _sorted_job_items()
    ]
    return render_template("history.html", jobs=jobs)
