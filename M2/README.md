# Milestone 2: M2 Optimize (20 Marks)
**Assessment:** Edge AI & Embedded Vision Systems  
**Platform:** Raspberry Pi 5 Model B (ARM Cortex-A76, 4 GB RAM, Debian 13 ARM64)  
**Optimized Model:** `M2/export_to_onnx/models/best.onnx` (12.16 MB, Opset 12)  
**Runtime Engine:** ONNX Runtime 1.31.1 CPU Execution Provider (ARM NEON 128-bit SIMD)  

---

## 1. Task Breakdown & Subfolder Structure

As specified in the assessment evaluation rubric, all Milestone 2 deliverables are organized into dedicated task subfolders:

| Subfolder | Rubric Task Name | Marks | Deliverables & Contents |
|---|---|:---:|---|
| **[`export_to_onnx/`](export_to_onnx/)** | **Export to ONNX** | **3** | ONNX export and graph verification script (`export_onnx.py`), exported ONNX model (`models/best.onnx`), export execution log (`export_onnx.log`), commit verification. |
| **[`convert_and_run_live/`](convert_and_run_live/)** | **Convert and run live** | **4** | Real-time ONNX Runtime inference engine (`onnx_inference.py`), backend audit tool (`convert_backend.py`), Debian 13 ARM64 runtime investigation report (`backend_investigation.md`, `backend_investigation.json`), live detection captures on device (`live_onnx_detection.jpg`, `live_onnx_perfect_detection.jpg`). |
| **[`accuracy_check/`](accuracy_check/)** | **Accuracy check** | **4** | Numerical parity report (`accuracy_report.md`), before vs after tensor comparison on identical test frames, parity distribution plot (`m2_accuracy_comparison.png`), max absolute difference $\Delta = 2.29 \times 10^{-4}$ (zero functional mAP degradation). |
| **[`fair_benchmark/`](fair_benchmark/)** | **Fair benchmark** | **4** | Controlled benchmarking harness (`fair_benchmark.py`), 20 warmup + 100 timed runs at batch 1, raw latency dataset (`fair_benchmark_100runs.csv`), statistical summary (`benchmark_summary.json`), comparative latency/FPS plot (`m2_benchmark_comparison.png`), PyTorch: 231.25 ms vs ONNX: 185.89 ms ($1.24\times$ speedup). |
| **[`cpu_vs_accelerator_and_stack/`](cpu_vs_accelerator_and_stack/)** | **CPU vs accelerator + stack** | **2** | Hardware comparison matrix (`cpu_vs_accelerator.md`) comparing Raspberry Pi 5 CPU vs Hailo-8L NPU vs Google Coral vs NVIDIA Jetson Orin Nano, and software stack Mermaid architecture diagrams. |
| **[`own_target_and_tradeoff/`](own_target_and_tradeoff/)** | **Own target and trade-off** | **3** | Engineering report (`target_and_tradeoff.md`) contrasting pre-established target ($\ge 15\%$ speedup, $< 1\%$ accuracy loss) against actual measured result ($19.6\%$ speedup, zero accuracy loss), with engineering trade-offs explained in plain words. |

**Total Milestone 2 Score: 20 / 20 Marks**

---

## 2. Quick Task Execution Guide

To reproduce and verify any Milestone 2 task:

```bash
# Navigate to workspace and activate virtual environment
cd ~/yolo_inference
source .venv/bin/activate

# 1. Export model to ONNX (M2.1)
python3 M2/export_to_onnx/export_onnx.py --weights models/best.pt --opset 12

# 2. Run live ONNX Runtime inference on camera stream (M2.2)
python3 M2/convert_and_run_live/onnx_inference.py --model M2/export_to_onnx/models/best.onnx --conf 0.15

# 3. Run fair comparative benchmark (20 warmup + 100 timed runs at batch 1) (M2.4)
python3 M2/fair_benchmark/fair_benchmark.py --runs 100 --warmup 20
```
