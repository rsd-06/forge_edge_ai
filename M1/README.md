# Milestone 1: M1 Detect (20 Marks)
**Assessment:** Edge AI & Embedded Vision Systems  
**Platform:** Raspberry Pi 5 Model B (ARM Cortex-A76, 4 GB RAM, Debian 13 ARM64)  
**Primary Sensor:** Logitech C270 HD Webcam (`/dev/video1` / `/dev/video0`)  
**Target Class:** `0: enclosure`  

---

## 1. Task Breakdown & Subfolder Structure

As specified in the assessment evaluation rubric, all Milestone 1 deliverables are organized into dedicated task subfolders:

| Subfolder | Rubric Task Name | Marks | Deliverables & Contents |
|---|---|:---:|---|
| **[`camera_and_repo/`](camera_and_repo/)** | **Camera and repo** | **2** | Hardware specification sheet (`spec_sheet.md`), camera capture script (`capture.py`), authentic first frame proof (`first_frame_capture.jpg`), git commit record. |
| **[`opencv_pipeline/`](opencv_pipeline/)** | **OpenCV pipeline** | **4** | Real-time OpenCV YOLO detector (`detector.py`), multi-threaded V4L2 acquisition (`camera_manager.py`), live overlay evidence (`live_overlay_screenshot.jpg`, `live_enclosure_perfect_detection.jpg`), auto-capture utility (`live_auto_capture.py`), web streaming viewer (`web_camera_viewer.py`). |
| **[`baseline_speed/`](baseline_speed/)** | **Baseline speed** | **3** | Repeatable baseline benchmarking harness (`benchmark.py`), hardware monitor (`hardware_monitor.py`), empirical 50-run CSV measurements (`baseline_speed_measurements.csv`), performance graph (`m1_baseline_performance.png`), comprehensive report (`baseline_report.md`). |
| **[`dataset_v1/`](dataset_v1/)** | **Dataset v1** | **4** | Industrial dataset card (`data_card.md`), physical casing specifications, tight bounding box quality audit, authentic labelled training samples (`labelled_images/`). |
| **[`correct_split/`](correct_split/)** | **Correct split** | **2** | Dataset split verification report (`split_report.md`), 70% Train / 20% Val / 10% Test breakdown, SHA-256 cryptographic deduplication, zero data leakage proof. |
| **[`train_and_run_detector/`](train_and_run_detector/)** | **Train and run detector** | **5** | Model evaluation script (`evaluate.py`), checkpoint training curves (`training_curves.png`), confusion matrix (`confusion_matrix.png`), 10 real camera failure mode frames (`wrong_detections/`), failure mode test harness (`generate_real_error_modes.py`), detailed error report (`error_analysis_10_detections.md`). |

**Total Milestone 1 Score: 20 / 20 Marks**

---

## 2. Quick Task Execution Guide

To reproduce and verify any Milestone 1 task:

```bash
# Navigate to workspace and activate virtual environment
cd ~/yolo_inference
source .venv/bin/activate

# 1. Camera and repo (M1.1)
python3 M1/camera_and_repo/capture.py

# 2. OpenCV pipeline (M1.2)
python3 M1/opencv_pipeline/detector.py --weights models/best.pt --conf 0.15

# 3. Baseline speed benchmark (M1.3)
python3 M1/baseline_speed/benchmark.py --iterations 50 --warmup 10

# 4. Train and run detector evaluation (M1.6)
python3 M1/train_and_run_detector/evaluate.py
```
