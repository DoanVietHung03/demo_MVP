from __future__ import annotations

import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import cv2
from flask import Flask, Response, abort, redirect, render_template_string, request, url_for

from person_detection.config import CONFIG, PersonInstance
from person_detection.pipeline import PersonDetectionPipeline, enqueue_log_item


APP_ROOT = Path(__file__).resolve().parent
WEB_OUTPUT_DIR = APP_ROOT / "web_output"
UPLOAD_DIR = WEB_OUTPUT_DIR / "uploads"
RESULT_DIR = WEB_OUTPUT_DIR / "results"

WEB_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024

JOB_STORE: Dict[str, Dict[str, object]] = {}
JOB_STORE_LOCK = threading.Lock()
ACTIVE_JOB_STATUSES = {"ready", "initializing", "streaming"}
MAX_CONCURRENT_WEB_JOBS = 1
MAX_RETAINED_TERMINAL_JOBS = 10


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Person Detection Dashboard</title>
    <style>
        :root {
            --bg: #000000;
            --panel: #0b0b0b;
            --panel-2: #111111;
            --accent: #00ff88;
            --accent-2: #00c46a;
            --text: #eeeeee;
            --muted: #8a8a8a;
            --danger-bg: #2b0f12;
            --danger-border: #7f1d1d;
        }
        * { box-sizing: border-box; }
        body {
            font-family: "Segoe UI", Arial, sans-serif;
            background:
                radial-gradient(circle at top right, rgba(0, 255, 136, 0.12), transparent 25%),
                linear-gradient(180deg, #020202 0%, #000000 100%);
            color: var(--text);
            margin: 0;
            min-height: 100vh;
        }
        .shell {
            width: min(1380px, calc(100vw - 32px));
            margin: 0 auto;
            padding: 20px 0 40px;
        }
        .topbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 16px;
            flex-wrap: wrap;
            margin-bottom: 18px;
        }
        h1 {
            margin: 0;
            color: var(--accent);
            text-shadow: 0 0 15px rgba(0, 255, 136, 0.35);
            font-size: clamp(2rem, 4vw, 3rem);
            font-weight: 800;
        }
        .sub {
            color: var(--muted);
            margin-top: 6px;
        }
        .btn {
            display: inline-block;
            padding: 10px 18px;
            background-color: #0d0d0d;
            color: var(--accent);
            text-decoration: none;
            font-size: 15px;
            font-weight: 700;
            border: 2px solid var(--accent);
            border-radius: 10px;
            box-shadow: 0 0 12px rgba(0, 255, 136, 0.14);
            transition: all 0.2s ease;
        }
        .btn:hover {
            background-color: var(--accent);
            color: #000000;
        }
        .layout {
            display: grid;
            grid-template-columns: minmax(320px, 390px) minmax(0, 1fr);
            gap: 20px;
        }
        .panel {
            background: linear-gradient(180deg, rgba(13, 13, 13, 0.95), rgba(6, 6, 6, 0.96));
            border: 1px solid rgba(0, 255, 136, 0.18);
            border-radius: 18px;
            box-shadow: 0 0 28px rgba(0, 255, 136, 0.08);
        }
        .form-panel {
            padding: 18px;
        }
        .video-panel {
            padding: 12px;
        }
        .video-frame {
            border: 2px solid rgba(0, 255, 136, 0.55);
            border-radius: 14px;
            overflow: hidden;
            background: #000;
            min-height: 360px;
        }
        .video-frame img {
            width: 100%;
            display: block;
        }
        .placeholder {
            min-height: 360px;
            display: grid;
            place-items: center;
            color: var(--muted);
            text-align: center;
            padding: 32px;
        }
        .status-bar {
            margin-top: 14px;
            color: var(--muted);
            font-size: 14px;
        }
        form {
            display: grid;
            gap: 14px;
        }
        .grid {
            display: grid;
            gap: 12px;
        }
        .grid.two {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
        label {
            display: grid;
            gap: 7px;
            font-size: 14px;
            font-weight: 600;
            color: #d8d8d8;
        }
        input {
            width: 100%;
            border-radius: 10px;
            border: 1px solid #2a2a2a;
            background: var(--panel-2);
            color: #ffffff;
            padding: 12px 12px;
            font: inherit;
        }
        input[type=file] {
            padding: 10px;
        }
        button {
            width: 100%;
            border: none;
            border-radius: 12px;
            padding: 14px 16px;
            background: linear-gradient(135deg, var(--accent) 0%, var(--accent-2) 100%);
            color: #000000;
            font: inherit;
            font-weight: 800;
            cursor: pointer;
        }
        .hint {
            color: var(--muted);
            font-size: 13px;
            line-height: 1.45;
        }
        .error {
            margin-bottom: 16px;
            padding: 12px 14px;
            background: var(--danger-bg);
            border: 1px solid var(--danger-border);
            border-radius: 12px;
            color: #fecaca;
        }
        .meta {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 12px;
            margin-top: 16px;
        }
        .meta-card {
            padding: 12px;
            border-radius: 12px;
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.06);
        }
        .meta-card strong {
            display: block;
            color: var(--accent);
            font-size: 1.2rem;
            margin-bottom: 4px;
        }
        @media (max-width: 1000px) {
            .layout { grid-template-columns: 1fr; }
            .meta { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="shell">
        <div class="topbar">
            <div>
                <h1>PERSON DETECTION DASHBOARD</h1>
                <div class="sub">Flask MJPEG stream + YOLO + ByteTrack + crop per person</div>
            </div>
            <a href="/history" class="btn">Xem Lịch Sử Xử Lý</a>
        </div>

        {% if error %}
            <div class="error">{{ error }}</div>
        {% endif %}

        <div class="layout">
            <section class="panel form-panel">
                <form method="post" action="{{ url_for('start_job') }}" enctype="multipart/form-data">
                    <div class="grid">
                        <label>
                            Video path trên server
                            <input type="text" name="source_path" value="{{ defaults.source }}">
                        </label>
                        <label>
                            Hoặc upload MP4
                            <input type="file" name="source_file" accept=".mp4,video/mp4">
                        </label>
                    </div>
                    <div class="hint">Nếu nhập cả hai, file upload sẽ được ưu tiên. `YOLO(...)` sẽ tự tải model hợp lệ như `yolo11n.pt` nếu chưa có local. App chỉ lưu crop mỗi 10 frame để tránh phình `web_output`.</div>
                    <div class="grid two">
                        <label>
                            Model
                            <input type="text" name="model_path" value="{{ defaults.model_path }}">
                        </label>
                        <label>
                            Device
                            <input type="text" name="device" value="{{ defaults.device }}">
                        </label>
                        <label>
                            Confidence
                            <input type="number" name="confidence_threshold" min="0.05" max="0.95" step="0.05" value="{{ defaults.confidence_threshold }}">
                        </label>
                        <label>
                            IOU
                            <input type="number" name="iou_threshold" min="0.05" max="0.95" step="0.05" value="{{ defaults.iou_threshold }}">
                        </label>
                        <label>
                            Crop padding
                            <input type="number" name="crop_padding" min="0" step="1" value="{{ defaults.crop_padding }}">
                        </label>
                        <label>
                            Resize scale
                            <input type="number" name="input_resize_scale" min="0.1" max="1.0" step="0.05" value="{{ defaults.input_resize_scale }}">
                        </label>
                        <label>
                            Frame skip
                            <input type="number" name="frame_skip" min="0" step="1" value="{{ defaults.frame_skip }}">
                        </label>
                        <label>
                            Track TTL
                            <input type="number" name="track_ttl_frames" min="1" step="1" value="{{ defaults.track_ttl_frames }}">
                        </label>
                        <label>
                            Alert cooldown
                            <input type="number" name="alert_cooldown_frames" min="1" step="1" value="{{ defaults.alert_cooldown_frames or '' }}">
                        </label>
                        <label>
                            Max frames
                            <input type="number" name="max_frames" min="1" step="1" value="{{ defaults.max_frames or '' }}">
                        </label>
                    </div>
                    <button type="submit">Bắt Đầu Stream</button>
                </form>
            </section>

            <section class="panel video-panel">
                {% if current_job %}
                    <div class="video-frame">
                        <img src="{{ url_for('video_feed', job_id=current_job.job_id) }}" alt="Live Video Feed">
                    </div>
                    <div class="status-bar">● Job {{ current_job.job_id }} | Source: {{ current_job.source_name }} | Device: {{ current_job.device }}</div>
                    <div class="meta">
                        <div class="meta-card">
                            <strong>{{ current_job.status }}</strong>
                            <span>Trạng thái job</span>
                        </div>
                        <div class="meta-card">
                            <strong>{{ current_job.total_alerts }}</strong>
                            <span>Số alert mới</span>
                        </div>
                        <div class="meta-card">
                            <strong>{{ current_job.unique_track_count }}</strong>
                            <span>Tracker ID duy nhất</span>
                        </div>
                    </div>
                {% else %}
                    <div class="video-frame placeholder">
                        <div>
                            <strong style="display:block; color: var(--accent); margin-bottom:10px;">Chưa có stream đang chạy</strong>
                            Điền cấu hình bên trái rồi bấm <em>Bắt Đầu Stream</em> để xem video detect trực tiếp trên web.
                        </div>
                    </div>
                {% endif %}
            </section>
        </div>
    </div>
</body>
</html>
"""


HISTORY_TEMPLATE = """
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lịch Sử Xử Lý</title>
    <style>
        body {
            font-family: "Segoe UI", Arial, sans-serif;
            background: #050505;
            color: #ececec;
            margin: 0;
            padding: 24px;
        }
        .shell {
            width: min(1200px, calc(100vw - 24px));
            margin: 0 auto;
        }
        h1 {
            color: #00ff88;
            margin-top: 0;
        }
        .btn {
            display: inline-block;
            margin-bottom: 18px;
            padding: 10px 18px;
            border: 2px solid #00ff88;
            color: #00ff88;
            text-decoration: none;
            border-radius: 10px;
            font-weight: 700;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            background: #0d0d0d;
            border-radius: 16px;
            overflow: hidden;
        }
        th, td {
            padding: 14px 12px;
            border-bottom: 1px solid #202020;
            text-align: left;
            vertical-align: top;
        }
        th {
            color: #00ff88;
            background: #090909;
        }
        a {
            color: #7dd3fc;
            text-decoration: none;
        }
        .muted {
            color: #8a8a8a;
        }
    </style>
</head>
<body>
    <div class="shell">
        <a class="btn" href="{{ url_for('index') }}">Quay Lại Dashboard</a>
        <h1>Lịch Sử Xử Lý</h1>
        <table>
            <thead>
                <tr>
                    <th>Job ID</th>
                    <th>Source</th>
                    <th>Status</th>
                    <th>Alerts</th>
                    <th>Tracks</th>
                    <th>Crop</th>
                </tr>
            </thead>
            <tbody>
                {% for job in jobs %}
                <tr>
                    <td>{{ job.job_id }}</td>
                    <td>{{ job.source_name }}</td>
                    <td>{{ job.status }}</td>
                    <td>{{ job.total_alerts }}</td>
                    <td>{{ job.unique_track_count }}</td>
                    <td>
                        <a href="{{ url_for('index', job_id=job.job_id) }}">Mở Stream</a><br>
                        <span class="muted">{{ job.crop_dir }}</span>
                    </td>
                </tr>
                {% else %}
                <tr>
                    <td colspan="6" class="muted">Chưa có job nào được tạo.</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </div>
</body>
</html>
"""


def _save_uploaded_file(uploaded_file) -> Path:
    suffix = Path(uploaded_file.filename or "upload.mp4").suffix or ".mp4"
    temp_file = tempfile.NamedTemporaryFile(delete=False, dir=UPLOAD_DIR, suffix=suffix)
    uploaded_file.save(temp_file.name)
    return Path(temp_file.name)


def _parse_optional_int(raw_value: str) -> Optional[int]:
    value = (raw_value or "").strip()
    return int(value) if value else None


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
    }


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


def _job_to_view(job_id: str, job: Dict[str, object]) -> Dict[str, object]:
    lock = job["lock"]
    with lock:
        return {
            "job_id": job_id,
            "source_name": job["source_name"],
            "device": job["device"],
            "status": job["status"],
            "total_alerts": job["total_alerts"],
            "unique_track_count": job["unique_track_count"],
            "crop_dir": job["crop_dir"],
        }


@app.get("/")
def index():
    _prune_finished_jobs()
    selected_job_id = request.args.get("job_id")
    current_job = _resolve_current_job_view(selected_job_id)
    return render_template_string(
        HTML_TEMPLATE,
        defaults=_defaults(),
        current_job=current_job,
        error=None,
    )


@app.post("/start")
def start_job():
    _prune_finished_jobs()
    active_jobs = _get_active_job_views()
    if len(active_jobs) >= MAX_CONCURRENT_WEB_JOBS:
        return render_template_string(
            HTML_TEMPLATE,
            defaults=_defaults(),
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
        video_source = Path(source_path)
        source_name = video_source.name
    else:
        return render_template_string(
            HTML_TEMPLATE,
            defaults=_defaults(),
            current_job=None,
            error="Hãy nhập đường dẫn video trên server hoặc upload một file MP4.",
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
        "source_is_upload": is_uploaded_source,
        "uploaded_source_path": str(video_source) if is_uploaded_source else None,
        "status": "ready",
        "total_alerts": 0,
        "unique_track_count": 0,
        "last_frame_index": -1,
        "unique_ids": set(),
        "latest_frame_jpeg": None,
        "latest_frame_seq": -1,
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
    crop_dir = Path(job["crop_dir"])
    crop_dir.mkdir(parents=True, exist_ok=True)
    crop_save_interval = max(1, int(job.get("crop_save_interval", 10)))
    seen_tracker_ids = set()
    new_alert_count = 0

    for person in persons:
        seen_tracker_ids.add(person.tracker_id)
        if frame_index % crop_save_interval == 0:
            person_dir = crop_dir / f"track_{person.tracker_id}"
            person_dir.mkdir(parents=True, exist_ok=True)
            crop_path = person_dir / f"frame_{frame_index:06d}.jpg"
            enqueue_log_item("crop", crop_path, image=person.crop.copy())
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
        job["unique_ids"] = set()
        job["total_alerts"] = 0
        job["unique_track_count"] = 0
        job["last_frame_index"] = -1
        job["latest_frame_jpeg"] = None
        job["latest_frame_seq"] = -1
        job["finished_at"] = None
        job["status"] = "initializing"

    cap = cv2.VideoCapture(str(job["source"]))
    if not cap.isOpened():
        with job["lock"]:
            job["status"] = "error"
            job["finished_at"] = time.time()
        _prune_finished_jobs()
        return

    frame_index = 0

    try:
        pipeline = PersonDetectionPipeline(
            model_path=str(job["model_path"]),
            device=str(job["device"]),
            confidence_threshold=float(job["confidence_threshold"]),
            iou_threshold=float(job["iou_threshold"]),
            crop_padding=int(job["crop_padding"]),
            input_resize_scale=float(job["input_resize_scale"]),
            track_ttl_frames=int(job["track_ttl_frames"]),
            alert_cooldown_frames=job["alert_cooldown_frames"],
        )
        frame_skip = max(0, int(job["frame_skip"]))

        with job["lock"]:
            job["status"] = "streaming"

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if job["max_frames"] is not None and frame_index >= int(job["max_frames"]):
                break
            if frame_skip > 0 and frame_index % (frame_skip + 1) != 0:
                frame_index += 1
                continue

            annotated_frame, persons = pipeline.process_frame(frame, frame_index)
            _update_job_outputs(job_id, annotated_frame, persons, frame_index)

            frame_index += 1

        with job["lock"]:
            job["status"] = "completed"
            job["finished_at"] = time.time()
    except Exception as exc:
        with job["lock"]:
            job["status"] = f"error: {exc}"
            job["finished_at"] = time.time()
    finally:
        cap.release()
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
    return render_template_string(HISTORY_TEMPLATE, jobs=jobs)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=False, threaded=True)
