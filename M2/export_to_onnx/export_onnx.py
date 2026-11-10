#!/usr/bin/env python3
"""
M2.1: ONNX Model Export and Numerical Validation.
Exports YOLOv8 best.pt to ONNX format with graph simplification,
validates graph structure, and verifies output consistency against PyTorch.
"""

import os
import sys
import time
import logging
import argparse
import numpy as np
import torch
import onnx
import onnxruntime as ort
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ExportONNX")


def export_and_validate(
    weights_path: str = "models/best.pt",
    output_dir: str = "models/optimized",
    imgsz: int = 640,
    opset: int = 12,
    log_file: str = "results/m2/export_onnx.log",
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # File logger setup
    fh = logging.FileHandler(log_file)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    logger.info("=== Step 1: Loading PyTorch YOLOv8 Model ===")
    pt_size_mb = os.path.getsize(weights_path) / (1024 * 1024)
    logger.info(f"Source PyTorch Model: {weights_path} ({pt_size_mb:.2f} MB)")

    model = YOLO(weights_path)

    logger.info("=== Step 2: Exporting to ONNX (opset=%d, imgsz=%d, simplify=True) ===", opset, imgsz)
    t0 = time.time()
    exported_path = model.export(
        format="onnx",
        imgsz=imgsz,
        dynamic=False,
        simplify=True,
        opset=opset,
    )
    export_duration = time.time() - t0
    logger.info(f"Export completed in {export_duration:.2f} seconds.")

    # Target path in models/optimized/
    target_onnx_path = os.path.join(output_dir, "best.onnx")
    if os.path.exists(exported_path) and os.path.abspath(exported_path) != os.path.abspath(target_onnx_path):
        import shutil
        shutil.move(exported_path, target_onnx_path)
    elif not os.path.exists(target_onnx_path):
        # In case it was exported in models/best.onnx
        src_candidate = os.path.splitext(weights_path)[0] + ".onnx"
        if os.path.exists(src_candidate):
            import shutil
            shutil.move(src_candidate, target_onnx_path)

    onnx_size_mb = os.path.getsize(target_onnx_path) / (1024 * 1024)
    logger.info(f"Target ONNX Model: {target_onnx_path} ({onnx_size_mb:.2f} MB)")

    logger.info("=== Step 3: Validating ONNX Graph Structure ===")
    onnx_model = onnx.load(target_onnx_path)
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX graph verification passed (onnx.checker.check_model).")

    session = ort.InferenceSession(target_onnx_path, providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    outputs = session.get_outputs()

    logger.info(f"ONNX Input Node: '{inputs[0].name}', shape={inputs[0].shape}, type={inputs[0].type}")
    logger.info(f"ONNX Output Node: '{outputs[0].name}', shape={outputs[0].shape}, type={outputs[0].type}")

    logger.info("=== Step 4: Numerical Parity Verification (PyTorch vs ONNX Runtime) ===")
    dummy_input = np.random.randn(1, 3, imgsz, imgsz).astype(np.float32)

    # PyTorch inference
    with torch.no_grad():
        pt_input = torch.from_numpy(dummy_input)
        pt_out = model.model(pt_input)[0].cpu().numpy()

    # ONNX Runtime inference
    ort_out = session.run([outputs[0].name], {inputs[0].name: dummy_input})[0]

    max_abs_diff = float(np.max(np.abs(pt_out - ort_out)))
    mean_abs_diff = float(np.mean(np.abs(pt_out - ort_out)))

    logger.info(f"Max Absolute Difference: {max_abs_diff:.6e}")
    logger.info(f"Mean Absolute Difference: {mean_abs_diff:.6e}")

    tolerance = 1e-3
    if max_abs_diff < tolerance:
        logger.info(f"SUCCESS: Numerical parity confirmed (diff < {tolerance}).")
    else:
        logger.warning(f"NOTICE: Max difference ({max_abs_diff:.6e}) exceeds standard tolerance ({tolerance}).")

    logger.info("=== M2.1 Export & Validation Complete ===")
    return {
        "pytorch_size_mb": pt_size_mb,
        "onnx_size_mb": onnx_size_mb,
        "export_duration_sec": export_duration,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "onnx_path": target_onnx_path,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export YOLOv8 to ONNX for M2.1")
    parser.add_argument("--weights", type=str, default="models/best.pt")
    parser.add_argument("--output-dir", type=str, default="models/optimized")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=12)
    args = parser.parse_args()

    export_and_validate(
        weights_path=args.weights,
        output_dir=args.output_dir,
        imgsz=args.imgsz,
        opset=args.opset,
    )
