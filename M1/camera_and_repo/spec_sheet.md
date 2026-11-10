# Technical Specification Sheet: Milestone 1 (Detect)
**Student Name:** Sai Sanjay R  
**Student ID:** S20  
**Project:** Edge AI & Embedded Vision Systems  
**Milestone:** M1: DETECT (20 Marks)  
**Hardware Platform:** Raspberry Pi 5 Model B Rev 1.0 (4 GB LPDDR4X)  
**Repository:** https://github.com/SAISANJAYR/edge_ai_sai  

---

## 1. Hardware Specifications

| Component | Detailed Hardware Specification | Operational Configuration |
|---|---|---|
| **Host System** | Raspberry Pi 5 Model B (BCM2712 SoC) | 4 Cores ARM Cortex-A76 @ 2.4 GHz |
| **Operating System** | Debian GNU/Linux 13 (Trixie) | aarch64 (Kernel 6.18.50+rpt-rpi-2712) |
| **System Memory** | 4.0 GiB LPDDR4X SDRAM | Shared CPU / GPU memory architecture |
| **Primary Vision Sensor**| **Logitech C270 HD Webcam** (USB UVC) | Device node: `/dev/video1` / `/dev/video0` (Bus 001 Device 003) |
| **Native Sensor Pixel Format** | YUYV 4:2:2 / MJPEG | Capture resolution: **640 × 480 @ 30 FPS** |
| **Field of View / Lens** | 60° diagonal FOV, fixed focus | Working distance: 20 cm – 60 cm to enclosure |
| **Power Delivery** | Official 27W USB-C PD (5.1V / 5.0A) | Dynamic current up to 5A |
| **Thermal Dissipation** | Official Raspberry Pi 5 Active Cooler | 4-pin PWM tachometer fan + aluminium heatsink |

---

## 2. Vision Model & Detection Pipeline Specifications

| Parameter | Specification | Engineering Description |
|---|---|---|
| **Deep Learning Framework** | Ultralytics YOLOv8 Nano (`yolov8n`) | PyTorch CPU runtime on ARM64 |
| **Model Weights File** | `models/best.pt` | Trained checkpoint (Size: 6.17 MB) |
| **Inspection Target Class** | `0: enclosure` | Single-class object localization |
| **Model Input Geometry** | $640 \times 640 \times 3$ (RGB) | Letterboxed & normalized $[0, 1]$ |
| **Confidence Threshold** | $c \ge 0.40$ (Default), $c \ge 0.25$ (Sensitive) | Dynamic rejection of background false positives |
| **NMS IoU Threshold** | $\text{IoU} = 0.50$ | Elimination of duplicate bounding boxes |
| **Primary Detection Metrics** | Precision: 84.22%, Recall: 13.33% | Validated on historical Roboflow checkpoint |

---

## 3. First Frame Proof (M1.1 Requirement)
- **Captured File**: `M1_Detect/evidence/first_frame_capture.jpg`
- **Resolution**: $640 \times 480 \times 3$ (BGR)
- **Capture Timestamp**: 2026-11-10 14:15:33 IST
- **Capture Utility**: Native OpenCV V4L2 backend on `/dev/video0`
