# Accuracy Check & Parity Report (M2.3)
**Student Name:** Sai Sanjay R (ID: S20)  
**Milestone:** M2 Optimize (M2.3 Accuracy Check)  
**Evaluated Backends:** PyTorch Baseline vs. ONNX Runtime Optimized  

---

## 1. Comparative Accuracy & Detection Parity Matrix

Evaluated across the exact same reference test set ($B=1$, $640 \times 640$ resolution):

| Metric | PyTorch Baseline (`best.pt`) | ONNX Runtime (`best.onnx`) | Delta ($\Delta$) | Functional Impact |
|---|---|---|---|---|
| **mAP@0.5** | **0.1421 (14.21%)** | **0.1421 (14.21%)** | **0.0000** | Zero degradation in detection accuracy |
| **Precision (B)** | **0.8422 (84.22%)** | **0.8422 (84.22%)** | **0.0000** | High true-positive confidence preserved |
| **Recall (B)** | **0.1333 (13.33%)** | **0.1333 (13.33%)** | **0.0000** | Identical object localization rate |
| **Max Tensor Difference** | Baseline ($0.0$) | **$2.2888 \times 10^{-4}$** | $< 10^{-3}$ | Floating-point rounding within SIMD tolerance |
| **Mean Tensor Difference**| Baseline ($0.0$) | **$1.0086 \times 10^{-5}$** | $< 10^{-5}$ | Near-perfect numerical alignment |
| **Bounding Box Overlap** | Ground Reference | **$\text{IoU} = 1.0000$** | **0.0000** | Exactly identical bounding box coordinates |
| **Confidence Delta** | Ground Reference | **$\Delta c \le 0.0002$** | $< 0.02\%$ | Negligible confidence variance |

---

## 2. Visualization Artifacts
- **Comparative Latency vs Accuracy Graph**: `M2_Optimize/evidence/m2_accuracy_comparison.png`
- **Side-by-Side Benchmark Curves**: `M2_Optimize/evidence/m2_benchmark_comparison.png`
- **Raw Export Parity Log**: `M2_Optimize/results/export_onnx.log`
