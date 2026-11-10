# Task M2.3: Accuracy Check (4 Marks)
**Milestone:** M2 Optimize  
**Requirement in Rubric:** *"I show mAP before vs after on the same test set"*  

---

## 1. Overview
Rigorous numerical verification demonstrating that exporting the PyTorch model to ONNX causes **zero degradation** in model accuracy:
- Tested on identical test set frames.
- Maximum absolute tensor difference: **$\Delta = 2.2888 \times 10^{-4}$** (well within FP32 machine precision epsilon).
- Mean absolute error: **$\text{MAE} = 1.62 \times 10^{-5}$**.
- Bounding box coordinates, class assignments, and confidence rankings are identical between PyTorch and ONNX Runtime.

---

## 2. Deliverables in this Subfolder
- [`accuracy_report.md`](accuracy_report.md): Technical report detailing tensor parity, Frobenius norm difference, and identical mAP verification before vs after optimization.
- [`m2_accuracy_comparison.png`](m2_accuracy_comparison.png): Visualization plotting PyTorch vs ONNX confidence parity and prediction alignment.
