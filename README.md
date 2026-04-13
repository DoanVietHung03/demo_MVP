# Demo MVP

Pipeline thị giác máy tính gồm 4 stage chính:

1. `person_detection`: detect + track người và lưu crop theo `track_id`
2. `person_region_extraction`: tách `head_face`, `hands`, `torso` từ crop người
3. `facemask_detection`: detect `Mask` / `NoMask` trên `head_face`
4. `behavior_detection`: detect các hành vi như dùng điện thoại / hút thuốc trên video

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

Web demo:

```bash
venv/bin/python run.py
```

## Ghi chú cấu hình

- `person_region_extraction.config.process_latest_job`
  - `True`: tự lấy job crop mới nhất trong `results_root`
  - `False`: bắt buộc truyền `input_crops_dir`
- `facemask_detection.config.process_latest_job`
  - `True`: tự lấy job region extraction mới nhất trong `results_root`
  - `False`: bắt buộc truyền `input_head_face_dir`
- Mặc định nhiều config đang để `device="cuda:0"`. Nếu máy không có CUDA, đổi sang `cpu`.

## Chạy kiểm tra nhanh

```bash
venv/bin/python -m compileall person_detection person_region_extraction facemask_detection behavior_detection run.py
venv/bin/python -m unittest discover -s tests -v
```
