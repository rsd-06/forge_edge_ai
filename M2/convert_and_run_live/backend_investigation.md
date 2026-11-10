# Backend Conversion & Live Optimization Report (M2.2)
**Student Name:** Sai Sanjay R (ID: S20)  
**Milestone:** M2 Optimize (M2.2 Convert and Run Live)  
**Target Architecture:** ARM Cortex-A76 (Debian 13 Trixie aarch64, Python 3.13.5)  

---

## 1. Technical Backend Evaluation Matrix

We evaluated four optimization pathways for deployment on the Raspberry Pi 5 CPU:

| Candidate Backend | Compatibility on Debian 13 Python 3.13 ARM64 | Conversion Pathway | Live Execution Status | Engineering Decision |
|---|---|---|---|---|
| **ONNX Runtime CPU** | **Fully Compatible** (`onnxruntime==1.31.1`) | `torch.onnx.export` $\rightarrow$ Graph Optimization | **RUNNING LIVE** (185.89 ms, 5.38 FPS) | **SELECTED**: Production Edge Engine |
| **PyTorch CPU** | **Fully Compatible** (`torch==2.15.0+cpu`) | Native checkpoint | Running (231.25 ms, 4.32 FPS) | Validated M1 Baseline |
| **NCNN** | Requires C++ build from source | ONNX $\rightarrow$ `onnx2ncnn` $\rightarrow$ NCNN Param/Bin | Source compilation required | Not selected due to build overhead |
| **TFLite INT8** | `tflite-runtime` wheels missing for Python 3.13 | ONNX $\rightarrow$ TF $\rightarrow$ TFLite Quantizer | Dependencies unavailable offline | Avoided false INT8 claim without weights |
| **TensorRT** | **Incompatible** (NVIDIA CUDA only) | Requires NVIDIA GPU | Incompatible with Broadcom BCM2712 | Strictly excluded per rubric rules |

---

## 2. Production Deployment Selection: ONNX Runtime
- **SIMD Vectorization**: Automatically leverages 128-bit ARM NEON vector instructions via optimized C++ GEMM routines.
- **Latency Gain**: Drops inference latency by **19.6%** (from 231.3 ms to 185.9 ms).
- **Execution Script**: `M2_Optimize/onnx_inference.py`.
- **Live Annotated Detection Screenshot**: `M2_Optimize/evidence/live_onnx_detection.jpg`.
