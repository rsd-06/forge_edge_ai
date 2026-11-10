# Task M1.2: OpenCV Pipeline (4 Marks)
**Milestone:** M1 Detect  
**Requirement in Rubric:** *"I run the live overlay (boxes + count) on the device myself"*  

---

## 1. Overview
Implements a real-time computer vision inference and HUD telemetry pipeline:
- Ingests $640 \times 480$ frames via multi-threaded `CameraManager` using the native V4L2 backend.
- Executes custom YOLOv8 model inference on CPU.
- Overlays real-time bounding boxes around the `enclosure` with class name, confidence %, live FPS, and detection count.
- Includes clean application shutdown, automatic evidence capture, and an optional interactive web streaming viewer.

---

## 2. Deliverables in this Subfolder
- [`detector.py`](detector.py): Main real-time OpenCV detection script with live HUD.
- [`camera_manager.py`](camera_manager.py): Multi-threaded acquisition class with fail-safe resource management.
- [`live_auto_capture.py`](live_auto_capture.py): Autonomous detection capture utility.
- [`web_camera_viewer.py`](web_camera_viewer.py): Interactive browser-based camera viewer running on `http://localhost:8080`.
- [`live_overlay_screenshot.jpg`](live_overlay_screenshot.jpg): Actual captured frame showing live bounding box, confidence %, count, and FPS.
- [`live_enclosure_perfect_detection.jpg`](live_enclosure_perfect_detection.jpg): Optimal detection on dark textured surface.

---

## 3. Verification Command
```bash
python3 M1/opencv_pipeline/detector.py --weights models/best.pt --conf 0.15
```
