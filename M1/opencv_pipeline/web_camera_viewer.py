#!/usr/bin/env python3
"""
Interactive Live Web Camera Viewer & YOLOv8 Visualizer for Raspberry Pi 5.
Serves a real-time MJPEG camera stream with live bounding boxes, FPS, and confidence.
Allows the user to visually position the enclosure and capture evidence with 1 click.
"""

import os
import sys

# Auto-switch to virtual environment if run under system Python
_venv_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.venv"))
if sys.prefix != _venv_dir and os.path.exists(os.path.join(_venv_dir, "bin/python")):
    _venv_python = os.path.join(_venv_dir, "bin/python")
    os.execv(_venv_python, [_venv_python] + sys.argv)

import io
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
import cv2
import numpy as np
from ultralytics import YOLO

# Global state
CONF_THRESHOLD = 0.10
LATEST_FRAME = None
LATEST_ANNOTATED = None
BEST_CONF = 0.0
FRAME_LOCK = threading.Lock()
DETECT_STATS = {"fps": 0.0, "detected": False, "conf": 0.0, "count": 0}
MODEL = None
CAP = None
RUNNING = True


def capture_and_inference_loop():
    global LATEST_FRAME, LATEST_ANNOTATED, BEST_CONF, DETECT_STATS, RUNNING, MODEL, CAP

    print("[Viewer] Loading YOLOv8 model (models/best.pt)...", flush=True)
    MODEL = YOLO("models/best.pt")

    print("[Viewer] Detecting camera device...", flush=True)
    CAP = None
    cam_idx = -1
    for idx in [1, 0, 2, 3]:
        c = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if c.isOpened():
            ret, _ = c.read()
            if ret:
                CAP = c
                cam_idx = idx
                break
            c.release()

    if CAP is None:
        print("[Viewer] ERROR: Could not find any active camera device!", flush=True)
        return

    print(f"[Viewer] Connected to camera at /dev/video{cam_idx}", flush=True)
    CAP.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    CAP.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("[Viewer] Camera stream active. Ready for live inspection.", flush=True)

    prev_time = time.time()

    while RUNNING:
        ret, frame = CAP.read()
        if not ret or frame is None:
            time.sleep(0.02)
            continue

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-4)
        prev_time = now

        # Run inference with current confidence threshold
        results = MODEL.predict(frame, conf=CONF_THRESHOLD, imgsz=640, verbose=False)[0]
        boxes = results.boxes

        annotated = results.plot()

        detected = len(boxes) > 0
        conf = float(boxes[0].conf[0]) if detected else 0.0

        # Overlay status banner
        header_color = (0, 200, 0) if detected else (0, 140, 255)
        status_text = f"ENCLOSURE DETECTED ({conf:.1%})" if detected else "SCANNING..."
        cv2.putText(annotated, f"FPS: {fps:.1f} | {status_text}", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, header_color, 2)
        cv2.putText(annotated, f"Threshold: {CONF_THRESHOLD:.0%} | Box Count: {len(boxes)}", (15, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

        # Auto-save best detection
        if detected and conf > BEST_CONF:
            BEST_CONF = conf
            cv2.imwrite("M1_Detect/evidence/live_overlay_screenshot.jpg", annotated)
            cv2.imwrite("M2_Optimize/evidence/live_onnx_detection.jpg", annotated)
            cv2.imwrite("data/test/actual_live_enclosure.jpg", annotated)
            print(f"[Viewer] Auto-saved new best screenshot: Conf {conf:.1%} | FPS {fps:.1f}", flush=True)

        with FRAME_LOCK:
            LATEST_FRAME = frame.copy()
            LATEST_ANNOTATED = annotated.copy()
            DETECT_STATS = {
                "fps": round(fps, 1),
                "detected": detected,
                "conf": round(conf * 100, 1),
                "count": len(boxes),
                "best_conf": round(BEST_CONF * 100, 1)
            }

        time.sleep(0.01)

    CAP.release()


HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
    <title>Live Camera & YOLO Inspection - Raspberry Pi 5</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #121417; color: #e1e4e8; margin: 0; padding: 20px; text-align: center; }
        .container { max-width: 780px; margin: auto; background: #1a1e24; border-radius: 12px; padding: 20px; box-shadow: 0 4px 20px rgba(0,0,0,0.5); }
        h1 { color: #58a6ff; font-size: 22px; margin-bottom: 5px; }
        p.subtitle { color: #8b949e; font-size: 14px; margin-top: 0; margin-bottom: 15px; }
        .feed-box { position: relative; border-radius: 8px; overflow: hidden; border: 2px solid #30363d; background: #000; margin-bottom: 15px; }
        img.video-feed { width: 100%; max-width: 640px; height: auto; display: block; margin: auto; }
        .metrics-bar { display: flex; justify-content: space-around; background: #0d1117; padding: 12px; border-radius: 8px; margin-bottom: 15px; }
        .metric-item { font-size: 13px; color: #8b949e; }
        .metric-value { font-size: 18px; font-weight: bold; color: #58a6ff; display: block; margin-top: 4px; }
        .btn-group { display: flex; justify-content: center; gap: 12px; margin-bottom: 15px; flex-wrap: wrap; }
        button { background: #238636; color: white; border: none; padding: 10px 20px; font-size: 15px; font-weight: 600; border-radius: 6px; cursor: pointer; transition: background 0.2s; }
        button:hover { background: #2ea043; }
        button.btn-thresh { background: #30363d; font-size: 13px; padding: 6px 12px; }
        button.btn-thresh.active { background: #1f6feb; border: 1px solid #58a6ff; }
        .alert-box { padding: 10px; border-radius: 6px; font-size: 14px; margin-top: 10px; display: none; }
        .tips { text-align: left; background: #0d1117; padding: 12px 16px; border-radius: 8px; font-size: 13px; color: #8b949e; line-height: 1.5; }
        .tips strong { color: #f0883e; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Raspberry Pi 5 Live YOLOv8 Inspection Viewer</h1>
        <p class="subtitle">Logitech C270 Webcam (/dev/video0) | Model: best.pt (enclosure)</p>
        
        <div class="metrics-bar">
            <div class="metric-item">Status: <span id="m-status" class="metric-value">SCANNING</span></div>
            <div class="metric-item">Confidence: <span id="m-conf" class="metric-value">0%</span></div>
            <div class="metric-item">Best Captured: <span id="m-best" class="metric-value">0%</span></div>
            <div class="metric-item">FPS: <span id="m-fps" class="metric-value">0.0</span></div>
        </div>

        <div class="feed-box">
            <img class="video-feed" src="/video_feed" alt="Camera Live Feed">
        </div>

        <div class="btn-group">
            <button onclick="saveSnapshot()">📸 Take Screenshot Now</button>
            <button class="btn-thresh active" onclick="setThresh(0.10, this)">Conf: 10%</button>
            <button class="btn-thresh" onclick="setThresh(0.15, this)">Conf: 15%</button>
            <button class="btn-thresh" onclick="setThresh(0.20, this)">Conf: 20%</button>
        </div>

        <div id="alert" class="alert-box"></div>

        <div class="tips">
            <strong>Tips for Best Detection on the Physical Enclosure:</strong>
            <ul>
                <li>Hold the enclosure <strong>20 to 35 cm</strong> in front of the lens.</li>
                <li>Ensure the front side (with the two groove lines and central pinhole) is facing the camera.</li>
                <li>Position against a darker/contrasting background (e.g. table, desk, notebook) rather than bright white.</li>
                <li>Screenshots are automatically saved directly to <code>M1_Detect/evidence/</code> and <code>M2_Optimize/evidence/</code>.</li>
            </ul>
        </div>
    </div>

    <script>
        function updateStats() {
            fetch('/stats')
                .then(r => r.json())
                .then(data => {
                    document.getElementById('m-fps').innerText = data.fps;
                    document.getElementById('m-conf').innerText = data.conf + '%';
                    document.getElementById('m-best').innerText = data.best_conf + '%';
                    const st = document.getElementById('m-status');
                    if (data.detected) {
                        st.innerText = 'DETECTED';
                        st.style.color = '#3fb950';
                    } else {
                        st.innerText = 'SCANNING';
                        st.style.color = '#d29922';
                    }
                })
                .catch(() => {});
        }
        setInterval(updateStats, 500);

        function saveSnapshot() {
            fetch('/save_snapshot', { method: 'POST' })
                .then(r => r.json())
                .then(data => {
                    const alert = document.getElementById('alert');
                    alert.style.display = 'block';
                    alert.style.background = '#238636';
                    alert.innerText = data.message;
                    setTimeout(() => { alert.style.display = 'none'; }, 4000);
                });
        }

        function setThresh(val, btn) {
            fetch('/set_threshold?val=' + val, { method: 'POST' })
                .then(r => r.json())
                .then(() => {
                    document.querySelectorAll('.btn-thresh').forEach(b => b.classList.remove('active'));
                    btn.classList.add('active');
                });
        }
    </script>
</body>
</html>
"""


class StreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/" or parsed.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))

        elif parsed.path == "/stats":
            import json
            with FRAME_LOCK:
                data = dict(DETECT_STATS)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode("utf-8"))

        elif parsed.path == "/video_feed":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()

            while RUNNING:
                with FRAME_LOCK:
                    frame = LATEST_ANNOTATED.copy() if LATEST_ANNOTATED is not None else None

                if frame is not None:
                    ret, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    if ret:
                        self.wfile.write(b"--frame\r\n")
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Content-Length", str(len(jpeg)))
                        self.end_headers()
                        self.wfile.write(jpeg.tobytes())
                        self.wfile.write(b"\r\n")
                time.sleep(0.04)

        else:
            self.send_error(404)

    def do_POST(self):
        global CONF_THRESHOLD
        parsed = urlparse(self.path)
        if parsed.path == "/save_snapshot":
            import json
            with FRAME_LOCK:
                frame = LATEST_ANNOTATED.copy() if LATEST_ANNOTATED is not None else None
                conf = DETECT_STATS.get("conf", 0.0)

            if frame is not None:
                cv2.imwrite("M1_Detect/evidence/live_overlay_screenshot.jpg", frame)
                cv2.imwrite("M2_Optimize/evidence/live_onnx_detection.jpg", frame)
                cv2.imwrite("data/test/actual_live_enclosure.jpg", frame)
                msg = f"Saved screenshot to M1 & M2 evidence! (Confidence: {conf}%)"
            else:
                msg = "Frame not available to save."

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "message": msg}).encode("utf-8"))

        elif parsed.path == "/set_threshold":
            import json
            qs = parse_qs(parsed.query)
            if "val" in qs:
                try:
                    CONF_THRESHOLD = float(qs["val"][0])
                    print(f"[Viewer] Confidence threshold set to {CONF_THRESHOLD:.0%}", flush=True)
                except ValueError:
                    pass
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "threshold": CONF_THRESHOLD}).encode("utf-8"))
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        # Silence standard HTTP access logging to keep terminal clean
        pass


def main():
    port = 8080
    t = threading.Thread(target=capture_and_inference_loop, daemon=True)
    t.start()

    server = HTTPServer(("0.0.0.0", port), StreamHandler)
    print("\n" + "=" * 65, flush=True)
    print(">>> LIVE WEB CAMERA STREAMER RUNNING <<<", flush=True)
    print(f"Open in your browser:  http://localhost:{port}", flush=True)
    print("Or open in VS Code:    Press Ctrl+Shift+P -> 'Simple Browser: Show'")
    print(f"                       and enter: http://localhost:{port}")
    print("=" * 65 + "\n", flush=True)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Viewer] Shutting down server...", flush=True)
    finally:
        global RUNNING
        RUNNING = False
        server.server_close()


if __name__ == "__main__":
    main()
