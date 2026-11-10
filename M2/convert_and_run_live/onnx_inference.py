#!/usr/bin/env python3
"""
M2.2: Unified Inference Engine supporting PyTorch and ONNX Runtime backends.
Implements standardized preprocessing, inference, and NMS post-processing
for fair comparative benchmarking on Raspberry Pi 5.
"""

import os
import sys
import time
import argparse
import logging
from typing import Dict, Any, List, Tuple, Optional
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.camera.camera_manager import CameraManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("InferenceEngine")


def letterbox(
    img: np.ndarray,
    new_shape: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
    auto: bool = False,
    scaleFill: bool = False,
    scaleup: bool = True,
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """Resizes and pads image to target dimensions while preserving aspect ratio."""
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5) -> List[int]:
    """Pure NumPy Non-Maximum Suppression (NMS)."""
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


class ONNXEngine:
    """Optimized ONNX Runtime CPU Inference Engine."""

    def __init__(
        self,
        model_path: str = "models/optimized/best.onnx",
        conf_threshold: float = 0.40,
        iou_threshold: float = 0.50,
        imgsz: int = 640,
    ):
        import onnxruntime as ort

        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.class_names = {0: "enclosure"}

        logger.info(f"Initializing ONNXRuntime CPU Engine with: {model_path}")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, frame: np.ndarray) -> Dict[str, Any]:
        """Runs pre-processing, ONNX inference, and NMS post-processing."""
        t0 = time.perf_counter()

        # Pre-process
        img, ratio, (dw, dh) = letterbox(frame, new_shape=(self.imgsz, self.imgsz), auto=False)
        blob = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, axis=0)

        # Forward pass
        t_infer_start = time.perf_counter()
        outputs = self.session.run([self.output_name], {self.input_name: blob})[0]
        infer_latency_ms = (time.perf_counter() - t_infer_start) * 1000.0

        # Post-process: output shape is [1, 5, 8400] (xc, yc, w, h, score0)
        predictions = np.squeeze(outputs).T  # [8400, 5]
        scores = predictions[:, 4]

        # Filter by confidence threshold
        mask = scores >= self.conf_threshold
        filtered_preds = predictions[mask]
        filtered_scores = scores[mask]

        detections = []
        if len(filtered_preds) > 0:
            boxes = filtered_preds[:, :4]
            # Convert [xc, yc, w, h] to [x1, y1, x2, y2]
            x1 = boxes[:, 0] - boxes[:, 2] / 2
            y1 = boxes[:, 1] - boxes[:, 3] / 2
            x2 = boxes[:, 0] + boxes[:, 2] / 2
            y2 = boxes[:, 1] + boxes[:, 3] / 2
            xyxy = np.column_stack([x1, y1, x2, y2])

            keep_indices = nms(xyxy, filtered_scores, self.iou_threshold)

            for idx in keep_indices:
                bx1, by1, bx2, by2 = xyxy[idx]
                # Scale coordinates back to original frame
                bx1 = (bx1 - dw) / ratio
                by1 = (by1 - dh) / ratio
                bx2 = (bx2 - dw) / ratio
                by2 = (by2 - dh) / ratio

                # Clip to frame boundary
                bx1 = max(0, min(frame.shape[1], bx1))
                by1 = max(0, min(frame.shape[0], by1))
                bx2 = max(0, min(frame.shape[1], bx2))
                by2 = max(0, min(frame.shape[0], by2))

                detections.append({
                    "box": [int(bx1), int(by1), int(bx2), int(by2)],
                    "confidence": float(filtered_scores[idx]),
                    "class_id": 0,
                    "class_name": "enclosure",
                })

        total_latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "detections": detections,
            "count": len(detections),
            "inference_time_ms": infer_latency_ms,
            "total_time_ms": total_latency_ms,
            "backend": "ONNX Runtime (CPU)",
        }


def get_inference_engine(backend: str = "onnx", weights_path: Optional[str] = None, conf: float = 0.40):
    if backend.lower() == "onnx":
        path = weights_path or "models/optimized/best.onnx"
        return ONNXEngine(model_path=path, conf_threshold=conf)
    elif backend.lower() == "pytorch":
        from src.detection.detector import YOLODetector
        path = weights_path or "models/best.pt"
        return YOLODetector(weights_path=path, conf_threshold=conf)
    else:
        raise ValueError(f"Unsupported backend: {backend}. Choose 'onnx' or 'pytorch'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M2.2 Unified Inference Engine")
    parser.add_argument("--backend", type=str, default="onnx", choices=["onnx", "pytorch"])
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--conf", type=float, default=0.40)
    parser.add_argument("--source", type=str, default="data/test/initial_capture.jpg")
    parser.add_argument("--output", type=str, default="evidence/screenshots/m2/m2_onnx_inference.jpg")
    args = parser.parse_args()

    engine = get_inference_engine(backend=args.backend, weights_path=args.weights, conf=args.conf)

    if os.path.exists(args.source):
        frame = cv2.imread(args.source)
    else:
        cam = CameraManager(auto_fallback_simulation=True)
        _, frame = cam.read()
        cam.release()

    res = engine.predict(frame)
    logger.info(f"Inference Result: {res['count']} detected | Latency: {res['inference_time_ms']:.1f} ms | Backend: {args.backend}")

    # Annotate and save
    annotated = frame.copy()
    for det in res["detections"]:
        x1, y1, x2, y2 = det["box"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 128, 0), 2)
        cv2.putText(
            annotated,
            f"{det['class_name']} {det['confidence']:.1%}",
            (x1, max(15, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 128, 0),
            2,
        )

    cv2.putText(
        annotated,
        f"Backend: {args.backend.upper()} | Latency: {res['inference_time_ms']:.1f}ms",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    cv2.imwrite(args.output, annotated)
    logger.info(f"Saved annotated result to: {args.output}")
