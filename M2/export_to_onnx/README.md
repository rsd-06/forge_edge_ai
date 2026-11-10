# Task M2.1: Export to ONNX (3 Marks)
**Milestone:** M2 Optimize  
**Requirement in Rubric:** *"I run the ONNX export myself and show my commit"*  

---

## 1. Overview
Exports the PyTorch YOLOv8 Nano model (`models/best.pt`) to the open Open Neural Network Exchange (ONNX) format:
- Target Opset: **Opset 12** (Optimized for broad ARM64 runtime support).
- Graph Validation: Passes `onnx.checker.check_model()` without topology or type violations.
- Numerical Parity: Verified against PyTorch using synthetic test tensors ($\text{Max } \Delta = 2.29 \times 10^{-4}$).

---

## 2. Deliverables in this Subfolder
- [`export_onnx.py`](export_onnx.py): Production ONNX export script with automated graph checking and numerical parity verification.
- [`models/best.onnx`](models/best.onnx): Exported ONNX model graph (12.16 MB).
- [`export_onnx.log`](export_onnx.log): Full execution log recording input dimensions ($1 \times 3 \times 640 \times 640$) and output node signatures (`output0: [1, 5, 8400]`).

---

## 3. Verification Command
```bash
python3 M2/export_to_onnx/export_onnx.py --weights models/best.pt --opset 12
```
