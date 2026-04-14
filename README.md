# Demo MVP

Pipeline thị giác máy tính gồm 6 stage chính:

1. `person_detection`: detect + track người và lưu crop theo `track_id`
2. `ppe_detection`: detect `hardhat` / `vest` trực tiếp trên full frame
3. `person_region_extraction`: tách `head_face`, `hands`, `torso` từ crop người
4. `facemask_detection`: detect `Mask` / `NoMask` trên `head_face`
5. `glove_detection`: classify `Glove` / `NoGlove` trên `hands`
6. `behavior_detection`: detect các hành vi như dùng điện thoại / hút thuốc trên video

Ngoài các stage độc lập, repo hiện có 2 lớp orchestration:

- `stream_pipeline/`: luồng realtime cho web, RTSP và xử lý theo từng frame
- `main_pipeline/`: luồng CLI/batch giữ lại để chạy offline hoặc debug artifact

`person_detection -> person_region_extraction -> facemask_detection -> glove_detection`

và có thể bật thêm `ppe_detection` hoặc `behavior_detection` như các nhánh chạy song song trên cùng video nguồn.

## Yêu cầu

- Python `3.12` hoặc mới hơn
- Khuyến nghị dùng virtual environment
- Nếu muốn chạy bằng GPU, cài bản `torch` phù hợp từ trang PyTorch trước

## Cài đặt

```bash
python3 -m venv venv
. venv/bin/activate
pip install -r requirements.txt
```

Nếu shell của bạn không có lệnh `python`, hãy dùng trực tiếp `python3` hoặc `venv/bin/python`.

## Chạy nhanh

Person detection:

```bash
venv/bin/python person_detection/main.py
```

Region extraction:

```bash
venv/bin/python person_region_extraction/main.py
```

Facemask detection:

```bash
venv/bin/python facemask_detection/main.py
```

Behavior detection:

```bash
venv/bin/python behavior_detection/main.py
```

Glove detection:

```bash
venv/bin/python glove_detection/main.py
```

PPE detection:

```bash
venv/bin/python ppe_detection/main.py
```

Web demo:

```bash
venv/bin/python run.py
```

`run.py` hiện chỉ là entrypoint mỏng cho `web_app/`, và web sẽ gọi trực tiếp `stream_pipeline` để chạy realtime trên cùng luồng frame. Các stage `person_detection`, `ppe_detection`, `person_region_extraction`, `facemask_detection`, `glove_detection` và `behavior_detection` nếu được bật sẽ chạy theo kiểu streaming thay vì đợi tuần tự theo batch.

Stream pipeline CLI:

```bash
venv/bin/python stream_pipeline/main.py
```

Main pipeline offline:

```bash
venv/bin/python main_pipeline/main.py
```

Export ONNX cho các model PyTorch trong repo:

```bash
venv/bin/python export_onnx.py
```

Script này sẽ export:
- `weights/pytorch/yolo11n.pt -> weights/onnx/yolo11n.onnx`
- `weights/pytorch/yolo11n-pose.pt -> weights/onnx/yolo11n-pose.onnx`
- `weights/pytorch/smoke.pt -> weights/onnx/smoke.onnx`
- `weights/pytorch/PPE.pt -> weights/onnx/PPE.onnx`
- `weights/pytorch/facemask.pth -> weights/onnx/facemask.onnx`

## Ghi chú cấu hình

- `person_region_extraction.config.process_latest_job`
  - `True`: tự lấy job crop mới nhất trong `results_root`
  - `False`: bắt buộc truyền `input_crops_dir`
- `facemask_detection.config.process_latest_job`
  - `True`: tự lấy job region extraction mới nhất trong `results_root`
  - `False`: bắt buộc truyền `input_head_face_dir`
- `glove_detection.config.process_latest_job`
  - `True`: tự lấy job region extraction mới nhất trong `results_root`
  - `False`: bắt buộc truyền `input_hands_dir`
- `stream_pipeline.config.StreamPipelineConfig`
  - gom cấu hình cho luồng realtime
  - cho phép bật/tắt từng stage trên web hoặc CLI realtime
  - có `attribute_refresh_interval_frames` để cache kết quả facemask/glove theo `track_id`
  - `ppe_detection` là nhánh realtime độc lập trên full frame, không phụ thuộc `person_region_extraction`
- `main_pipeline.config.MainPipelineConfig`
  - gom cấu hình cho luồng batch/offline
  - giữ lại để xử lý artifact tuần tự hoặc debug
- Mặc định nhiều config đang để `device="cuda:0"`. Nếu máy không có CUDA, đổi sang `cpu`.
- `person_detection`, `person_region_extraction`, `behavior_detection`, `facemask_detection`
  - nếu `model_path` trỏ tới file `.onnx`, code sẽ chạy qua ONNX Runtime
  - nếu `model_path` trỏ tới `.pt`/`.pth`, code sẽ giữ backend PyTorch/Ultralytics cũ
  - với `device="cuda:0"`, helper ONNX Runtime sẽ tự preload CUDA/cuDNN libs từ `venv` trước khi tạo session
- Thư mục trọng số hiện được tách như sau:
  - `weights/pytorch/`: các checkpoint `.pt` / `.pth`
  - `weights/onnx/`: các model `.onnx`
  - `weights/converted_tflite_quantized/`: model TFLite cho glove
- `glove_detection` cần thêm runtime TFLite để chạy thật:
  - cài `tflite-runtime`
  - hoặc cài `tensorflow`
- `glove_detection` hiện vẫn là nhánh TFLite/CPU riêng, chưa được chuyển sang ONNX trong repo này.
- Nếu muốn ONNX Runtime chạy bằng GPU thật, môi trường phải có đủ CUDA/cuDNN runtime mà `onnxruntime-gpu` yêu cầu. Nếu thiếu, ONNX Runtime sẽ fallback về CPU và cảnh báo rõ provider.

## Chạy kiểm tra nhanh

```bash
venv/bin/python -m compileall person_detection ppe_detection person_region_extraction facemask_detection glove_detection behavior_detection main_pipeline stream_pipeline web_app run.py
venv/bin/python -m unittest discover -s tests -v
```
