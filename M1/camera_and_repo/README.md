# Task M1.1: Camera and Repo (2 Marks)
**Milestone:** M1 Detect  
**Requirement in Rubric:** *"I capture a frame on the device and make my own commit with the spec sheet and first frame"*  

---

## 1. Overview
This task establishes and verifies the primary vision acquisition pipeline on the Raspberry Pi 5 hardware:
- Probing the UVC USB webcam (`Logitech C270`, USB ID: `046d:0825`).
- Identifying the active Linux device node (`/dev/video1` / `/dev/video0`).
- Validating native YUYV capture resolution ($640 \times 480$ @ 30 FPS).
- Capturing an authentic initial frame on device and documenting hardware specifications.

---

## 2. Deliverables in this Subfolder
- [`spec_sheet.md`](spec_sheet.md): Full technical specification sheet of the host SBC, OS, camera optics, power, and thermal management.
- [`first_frame_capture.jpg`](first_frame_capture.jpg): Verified initial frame captured on the Raspberry Pi 5 device.
- [`capture.py`](capture.py): Python acquisition script with dynamic `/dev/video*` index auto-discovery.

---

## 3. Verification Command
```bash
python3 M1/camera_and_repo/capture.py
```
