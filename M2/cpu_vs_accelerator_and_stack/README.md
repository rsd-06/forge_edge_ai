# Task M2.5: CPU vs Accelerator & Software Stack (2 Marks)
**Milestone:** M2 Optimize  
**Requirement in Rubric:** *"I show the CPU vs accelerator table and draw the software stack"*  

---

## 1. Overview
Provides an architectural comparison of embedded Edge AI hardware accelerators against the Raspberry Pi 5 CPU, accompanied by architectural diagrams of the runtime software stack.

---

## 2. Hardware Comparison Matrix

| Specification | Raspberry Pi 5 CPU (Evaluated) | Raspberry Pi AI Kit (Hailo-8L) | Google Coral Edge TPU | NVIDIA Jetson Orin Nano |
|---|:---:|:---:|:---:|:---:|
| **Processing Architecture** | 4x Cortex-A76 CPU @ 2.4 GHz | Dedicated NPU (PCIe Gen 2/3) | Dedicated ASIC (USB 3.0 / PCIe) | 1024-core Ampere GPU + 6x A78ARM |
| **Peak AI Compute** | ~0.1 TOPS (FP32 CPU SIMD) | **13 TOPS** (INT8) | 4 TOPS (INT8) | **40 TOPS** (Sparse INT8) |
| **Thermal Design Power (TDP)**| 5W – 12W | **1.5W – 2.5W** | 2.0W – 2.5W | 7W – 15W |
| **Precision Formats** | FP32, FP16 | INT8, INT16 | INT8 only | FP32, FP16, INT8, INT4 |
| **YOLOv8 Latency** | **185.89 ms** (Batch 1) | **~12 ms** (HailoRT INT8) | ~32 ms (Edge TPU Delegate) | **~8 ms** (TensorRT FP16) |
| **Throughput (FPS)** | **5.38 FPS** | **~80 FPS** | ~31 FPS | **~120 FPS** |
| **Hardware Hardware Cost** | Baseline hardware (\$80) | +\$70 Hat add-on | +\$60 USB Accelerator | \$499 Standalone SBC |
| **Software Toolchain** | ONNX Runtime (Standard) | Hailo Dataflow Compiler (DFC) | Edge TPU Compiler (TFLite) | TensorRT / CUDA / JetPack |

---

## 3. Deliverables in this Subfolder
- [`cpu_vs_accelerator.md`](cpu_vs_accelerator.md): Comprehensive report with complete hardware trade-off analysis and Mermaid software stack architecture diagrams.
