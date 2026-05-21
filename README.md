# yolo-parallel

Edge runtime สำหรับ Jetson Nano รับ RTSP 5 กล้อง → ทำ YOLO inference ผ่าน
TensorRT engine ตัวเดียว → ส่ง detection events เป็น JSONL

## โครงสร้าง

```
RTSP × 5  →  GStreamer hw decode (416×416)  →  drop-old buffer  →
   fair scheduler  →  ONE TensorRT loop  →  logs/detections.jsonl
                                          ↘  logs/runtime_metrics.jsonl
```

## ติดตั้ง

**บน Jetson Nano** (Python 3.6.9, JetPack 4.6.x — cv2 + tensorrt มาจาก JetPack):

```bash
pip3 install --user -r requirements.txt
pip3 install --user 'pycuda<2022.0'
```

**บน dev box** (Mac/Linux — ใช้ทดสอบ smoke ก่อน push):

```bash
pip3 install -r requirements.txt
pip3 install 'opencv-python>=4.5'
```

⚠️ **ห้าม `pip install opencv-python` บน Jetson** — จะทับ cv2 จาก JetPack
ทำให้ GStreamer + Tegra plugin หาย hardware decode พัง

⚠️ **ห้าม `pip install tensorrt` บน Jetson** — มากับ JetPack อยู่แล้ว

## ใช้งาน 4 ขั้น

### 1. สลับโมเดล (ครั้งเดียวต่อ Jetson หรือทุกครั้งที่เปลี่ยน model)

วาง ONNX ที่ไหนก็ได้ แล้วเรียก:

```bash
tools/swap_model.sh models/yolov8s.onnx 416
```

มันจะ:
- เรียก `trtexec` แปลงเป็น `models/yolov8s_fp16.engine`
- แก้ `config.yaml` ให้ชี้ engine ใหม่อัตโนมัติ (คอมเมนต์ใน YAML ไม่หาย)
- รัน golden image canary verify

### 2. เช็คสุขภาพระบบก่อนรัน

```bash
python3 tools/health_check.py --config config.yaml
```

ตรวจ Python 3.6, tensorrt/pycuda/cv2/numpy, `cv2.dnn.NMSBoxes`, `trtexec`,
engine file, labels file, deserialize ได้ — exit 0 = ผ่าน, exit 1 พร้อม
fix instructions

### 3. รัน runtime

```bash
python3 edge_pipeline.py --config config.yaml
```

events → `logs/detections.jsonl`, metrics → `logs/runtime_metrics.jsonl`
(rotate ทุก 5 วินาที) Ctrl+C ออกได้สะอาด

### 4. ดู live (เปิดอีก terminal)

```bash
python3 tools/watch.py
```

TUI dashboard 3 ส่วน:
- **system**: backend (แดง = fake fallback), total fps, inference avg, mem
- **cameras**: ต่อกล้อง — decoder / fps / age / W×H / drops / fails / reconnects
- **recent detections**: 10 events ล่าสุด

`metrics_age` แดงเมื่อ runtime ค้าง ส่วน backend แดง = engine โหลดไม่ได้

## คำสั่งที่ใช้บ่อย

| ทำอะไร | คำสั่ง |
|---|---|
| สลับโมเดล | `tools/swap_model.sh <onnx> <imgsz>` |
| เช็คสุขภาพ | `tools/health_check.py --config config.yaml` |
| รัน runtime | `python3 edge_pipeline.py --config config.yaml` |
| ดู live | `tools/watch.py` |
| ทดสอบรูปเดียว | `tools/test_detector_on_image.py --engine <e> --image <i> --labels models/coco_labels.txt` |
| ให้คะแนน run | `tools/fit_report.py` → `reports/fit_report.md` |
| Smoke test (ไม่ต้องมี Jetson) | `python3 tests/smoke_test.py` |
| ดู log เก่าผ่าน TUI | `tools/watch.py --once --metrics logs/<...>.jsonl` |

## แก้ปัญหาเร็ว

| อาการ | สาเหตุ / แก้ |
|---|---|
| TUI โชว์ `backend=fake (FALLBACK)` | engine โหลดไม่ได้ → รัน `tools/health_check.py` |
| TUI โชว์ `metrics_age` แดง > 10s | runtime ค้าง → ดู `tail -50 logs/...`, restart |
| camera decoder = `(none)` ทุกตัว | RTSP URL ผิด / firewall / Tegra plugin หาย → `gst-inspect-1.0 nvv4l2decoder` |
| Smoke test fail | คุณไม่ได้ `cd` ไปที่ workspace root |
| Memory โต continuously | engine + cv2 leak — เช็ค `tegrastats` |

## โครงสร้างไฟล์ย่อ

```
.
├── camera_reader.py            RTSP GStreamer hw decode
├── frame_buffer.py             drop-old single-slot
├── frame_scheduler.py          fair RR + per-camera FPS
├── edge_pipeline.py            ★ entrypoint
├── config.yaml                 config หลัก
├── requirements.txt
├── detector/                   TensorRT + Fake + YOLO utils
├── events/                     JSONL detection sink
├── metrics/                    per-window snapshot collector
├── tools/                      swap_model, watch, health_check, ...
├── tests/                      smoke + fit_report smoke
├── models/                     .engine + coco_labels + samples
├── logs/                       runtime output (gitignored)
├── outputs/                    annotated PNG (gitignored)
└── reports/                    fit_report output (gitignored)
```

## ข้อสำคัญ

- **Engine ไม่ portable** ข้าม Jetson / JetPack / TRT version → ต้อง
  rebuild ทุกครั้งที่ย้ายเครื่อง
- **ห้ามแก้ `detector.engine` / `detector.imgsz` ใน config.yaml ด้วยมือ** —
  ใช้ `tools/swap_model.sh` มันจะแก้ให้ + verify
- ONNX shape ที่ pipeline รองรับ: YOLOv8 raw head `[1, 84, M]` (export
  จาก Ultralytics ปกติ ไม่ใช่ wpost / NMS-baked)
- รายละเอียดเชิงลึก (design rationale, Tasks 01-07 history, gotchas) →
  ถาม Claude (CLAUDE.md ถูก gitignore ไว้)
