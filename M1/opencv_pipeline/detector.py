#!/usr/bin/env python3
"""
Real-time YOLOv8 Detector Pipeline on Raspberry Pi 5.
Integrates CameraManager, Ultralytics YOLOv8, bounding box annotation,
FPS/latency calculation, object counting, and fail-safe resource cleanup.
"""

import os
import sys

# Auto-switch to project virtual environment if run under system python
_venv_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.venv"))
if sys.prefix != _venv_dir and os.path.exists(os.path.join(_venv_dir, "bin/python")):
    _venv_python = os.path.join(_venv_dir, "bin/python")
    os.execv(_venv_python, [_venv_python] + sys.argv)

import time
import argparse
import logging
from typing import Dict, Any, List, Optional
import cv2
import numpy as np

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.camera.camera_manager import CameraManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("YOLODetector")


class YOLODetector:
    """
    Encapsulates YOLOv8 model loading, inference, and visualization.
    """

    def __init__(
        self,
        weights_path: str = "models/best.pt",
        conf_threshold: float = 0.40,
        iou_threshold: float = 0.50,
        imgsz: int = 640,
        device: str = "cpu",
    ):
        self.weights_path = weights_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.device = device

        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Model weights not found at: {weights_path}")

        logger.info(f"Loading YOLOv8 weights from: {weights_path}")
        from ultralytics import YOLO
        self.model = YOLO(weights_path)
        self.class_names = self.model.names
        logger.info(f"Model successfully loaded. Classes: {self.class_names}")

    def predict(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Executes single-frame inference.
        Returns dictionary containing boxes, labels, confidences, count, and latency.
        """
        t0 = time.perf_counter()
        results = self.model.predict(
            frame,
            imgsz=self.imgsz,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )[0]
        inference_time_ms = (time.perf_counter() - t0) * 1000.0

        detections: List[Dict[str, Any]] = []
        boxes = results.boxes

        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)

            for i in range(len(xyxy)):
                cls_id = int(clss[i])
                cls_name = self.class_names.get(cls_id, f"class_{cls_id}")
                detections.append({
                    "box": [int(v) for v in xyxy[i]],
                    "confidence": float(confs[i]),
                    "class_id": cls_id,
                    "class_name": cls_name,
                })

        return {
            "detections": detections,
            "count": len(detections),
            "inference_time_ms": inference_time_ms,
        }

    def annotate(
        self,
        frame: np.ndarray,
        inference_result: Dict[str, Any],
        fps: float,
    ) -> np.ndarray:
        """
        Draws industrial bounding boxes, labels, confidence scores,
        FPS, count, and status header onto the frame.
        """
        annotated = frame.copy()
        detections = inference_result["detections"]
        count = inference_result["count"]
        latency_ms = inference_result["inference_time_ms"]

        # Draw bounding boxes
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            conf = det["confidence"]
            cls_name = det["class_name"]

            # Visual styling: green box with filled banner for label
            color = (0, 204, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            label = f"{cls_name} {conf:.1%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(
                annotated,
                label,
                (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        # Draw industrial telemetry header
        h, w = annotated.shape[:2]
        header = annotated.copy()
        cv2.rectangle(header, (0, 0), (w, 42), (25, 25, 25), -1)
        cv2.addWeighted(header, 0.75, annotated, 0.25, 0, annotated)

        # Status text overlays
        cv2.putText(
            annotated,
            f"FPS: {fps:.1f}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            f"Latency: {latency_ms:.1f}ms",
            (140, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            f"Count: {count}",
            (330, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (0, 255, 0) if count > 0 else (0, 165, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            "RPi5 Cortex-A76 (CPU)",
            (w - 195, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        return annotated


def run_pipeline(
    weights_path: str = "models/best.pt",
    device_index: int = 0,
    conf_threshold: float = 0.40,
    headless: bool = False,
    max_frames: Optional[int] = None,
    save_screenshot_path: Optional[str] = None,
):
    cam = CameraManager(device_index=device_index, width=640, height=480, auto_fallback_simulation=True)
    detector = YOLODetector(weights_path=weights_path, conf_threshold=conf_threshold)

    window_name = "Edge AI Inspection - YOLOv8 Enclosure Detector"
    can_display = False
    if not headless:
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
            can_display = True
        except Exception as e:
            logger.warning(f"Could not open GUI window ({e}). Running in headless mode.")
            can_display = False

    logger.info("Starting detection loop. Press 'q' in GUI or Ctrl+C in terminal to stop.")
    frame_times = []
    frames_processed = 0

    try:
        while True:
            t_loop_start = time.perf_counter()
            ret, frame = cam.read()
            if not ret or frame is None:
                logger.error("Failed to acquire frame. Exiting loop.")
                break

            result = detector.predict(frame)

            dt = time.perf_counter() - t_loop_start
            frame_times.append(dt)
            if len(frame_times) > 30:
                frame_times.pop(0)
            avg_fps = 1.0 / (sum(frame_times) / len(frame_times))

            annotated_frame = detector.annotate(frame, result, fps=avg_fps)
            frames_processed += 1

            if frames_processed % 20 == 0:
                logger.info(
                    f"Frame {frames_processed:04d} | FPS: {avg_fps:.1f} | "
                    f"Inference: {result['inference_time_ms']:.1f}ms | "
                    f"Detected: {result['count']} {detector.class_names.get(0, 'enclosure')}"
                )

            # Save initial or requested screenshot
            if save_screenshot_path and frames_processed == 10:
                os.makedirs(os.path.dirname(save_screenshot_path), exist_ok=True)
                cv2.imwrite(save_screenshot_path, annotated_frame)
                logger.info(f"Annotated screenshot saved to: {save_screenshot_path}")

            if can_display:
                cv2.imshow(window_name, annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("User requested exit ('q' key pressed).")
                    break

            if max_frames and frames_processed >= max_frames:
                logger.info(f"Reached requested limit of {max_frames} frames.")
                break

    except KeyboardInterrupt:
        logger.info("Interrupted by user (Ctrl+C).")
    finally:
        cam.release()
        if can_display:
            cv2.destroyAllWindows()
        logger.info(f"Pipeline closed cleanly. Total frames processed: {frames_processed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time YOLOv8 Detector on RPi 5")
    parser.add_argument("--weights", type=str, default="models/best.pt", help="Path to model weights")
    parser.add_argument("--device", type=int, default=0, help="Camera device index")
    parser.add_argument("--conf", type=float, default=0.40, help="Confidence threshold")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode without GUI window")
    parser.add_argument("--frames", type=int, default=None, help="Stop after N frames")
    parser.add_argument("--save-screenshot", type=str, default="evidence/screenshots/m1/m1_live_detection.jpg", help="Path to save annotated screenshot")
    args = parser.parse_args()

    run_pipeline(
        weights_path=args.weights,
        device_index=args.device,
        conf_threshold=args.conf,
        headless=args.headless,
        max_frames=args.frames,
        save_screenshot_path=args.save_screenshot,
    )
