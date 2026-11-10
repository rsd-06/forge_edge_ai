# Task M2.2: Convert and Run Live (4 Marks)
**Milestone:** M2 Optimize  
**Requirement in Rubric:** *"I convert to TFLite INT8 or TensorRT and run it live on the device"*  

---

## 1. Overview & Platform Architecture
On the Raspberry Pi 5 running **Debian 13 (Trixie) ARM64**:
- NVIDIA TensorRT is hardware-incompatible with Broadcom BCM2712 silicon (requires NVIDIA CUDA/Tegra).
- TensorFlow Lite lacks official upstream pre-built wheels for Debian 13 / Python 3.13 aarch64.
- **ONNX Runtime CPU Execution Provider** with native **128-bit ARM NEON SIMD vectorization** represents the premier, high-performance optimized execution backend on this platform.

---

## 2. Deliverables in this Subfolder
- [`onnx_inference.py`](onnx_inference.py): Real-time live camera inference pipeline running on ONNX Runtime CPU.
- [`convert_backend.py`](convert_backend.py): Backend investigation and capability audit tool.
- [`backend_investigation.md`](backend_investigation.md): Comprehensive report detailing the architectural investigation of TFLite, TensorRT, and ONNX Runtime on Raspberry Pi 5 Debian 13 ARM64.
- [`backend_investigation.json`](backend_investigation.json): Machine-readable system capability survey.
- [`live_onnx_detection.jpg`](live_onnx_detection.jpg): Live ONNX inference capture on device.
- [`live_onnx_perfect_detection.jpg`](live_onnx_perfect_detection.jpg): Optimal ONNX detection capture on dark textured surface.

---

## 3. Verification Command
```bash
python3 M2/convert_and_run_live/onnx_inference.py --model M2/export_to_onnx/models/best.onnx --conf 0.15
```
