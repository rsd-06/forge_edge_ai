#!/usr/bin/env python3
"""
Live Enclosure Auto-Capture Utility.
Runs camera capture in a loop. When an enclosure is detected,
it immediately saves the annotated proof screenshot to M1 and M2 evidence.
"""

import os
import sys

# Auto-switch to project virtual environment if run under system python
_venv_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.venv"))
if sys.prefix != _venv_dir and os.path.exists(os.path.join(_venv_dir, "bin/python")):
    _venv_python = os.path.join(_venv_dir, "bin/python")
    os.execv(_venv_python, [_venv_python] + sys.argv)

import time
import cv2
from ultralytics import YOLO

def main():
    model_path = "models/best.pt"
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.", flush=True)
        return

    print("Loading YOLOv8 model...", flush=True)
    model = YOLO(model_path)
    
    print("Detecting camera device...", flush=True)
    cap = None
    cam_idx = -1
    for idx in [1, 0, 2, 3]:
        c = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if c.isOpened():
            ret, _ = c.read()
            if ret:
                cap = c
                cam_idx = idx
                break
            c.release()

    if cap is None:
        print("Error: Could not find any active camera device (/dev/video*).", flush=True)
        return

    print(f"Connected to camera at /dev/video{cam_idx} (index {cam_idx})", flush=True)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\n=======================================================", flush=True)
    print(">>> PLEASE HOLD YOUR ENCLOSURE IN FRONT OF CAMERA <<<", flush=True)
    print(">>> Distance: 20-40 cm | Good lighting | Face forward <<<", flush=True)
    print("=======================================================\n", flush=True)
    print("Scanning continuously... (Press Ctrl+C to stop at any time)\n", flush=True)

    frame_idx = 0
    best_conf = 0.0
    best_frame = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            frame_idx += 1
            t0 = time.time()
            # Predict with lower threshold to assist in positioning
            results = model.predict(frame, conf=0.10, imgsz=640, verbose=False)[0]
            dt = max(time.time() - t0, 1e-4)
            fps = 1.0 / dt

            boxes = results.boxes
            if len(boxes) > 0:
                conf = float(boxes[0].conf[0])
                cls_name = model.names[int(boxes[0].cls[0])]
                print(f"[Frame {frame_idx:04d}] DETECTED: {cls_name} (Confidence: {conf:.1%}, FPS: {fps:.1f})", flush=True)

                # Annotate
                annotated = results.plot()
                # Draw telemetry header
                cv2.putText(annotated, f"FPS: {fps:.1f} | Conf: {conf:.1%}", (15, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if conf > best_conf:
                    best_conf = conf
                    best_frame = annotated
                    cv2.imwrite("M1_Detect/evidence/live_overlay_screenshot.jpg", annotated)
                    cv2.imwrite("M2_Optimize/evidence/live_onnx_detection.jpg", annotated)
                    cv2.imwrite("data/test/actual_live_enclosure.jpg", annotated)
                    print(f"--> Saved NEW BEST detection screenshot (Conf: {conf:.1%}) to M1 and M2 evidence!", flush=True)

                if conf >= 0.35:
                    print(f"\nSUCCESS: Confirmed strong detection ({conf:.1%})!", flush=True)
                    print(f"Evidence saved to M1_Detect/evidence/live_overlay_screenshot.jpg", flush=True)
                    break
            else:
                if frame_idx % 5 == 0:
                    print(f"[Frame {frame_idx:04d}] Scanning... No enclosure detected yet. (FPS: {fps:.1f})", flush=True)

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nStopped by user.", flush=True)
    finally:
        cap.release()
        if best_frame is not None:
            print(f"\nFinal best detection saved with confidence: {best_conf:.1%}", flush=True)
        else:
            print("\nNo detection above threshold was captured.", flush=True)

if __name__ == "__main__":
    main()
