#!/usr/bin/env python3
"""
Baseline Performance Benchmarking Tool for M1.3.
Runs repeatable inference benchmarks on Raspberry Pi 5 CPU,
logging latency, FPS, CPU load, RAM usage, and temperature to CSV,
and generating actual performance graphs.
"""

import os
import sys
import time
import argparse
import csv
import logging
from typing import Dict, Any, List
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.camera.camera_manager import CameraManager
from src.detection.detector import YOLODetector
from src.monitoring.hardware_monitor import HardwareMonitor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("BenchmarkM1")


def run_baseline_benchmark(
    weights_path: str = "models/best.pt",
    iterations: int = 100,
    warmup: int = 20,
    csv_output_path: str = "results/m1/baseline_benchmark.csv",
    plot_output_path: str = "evidence/graphs/m1_baseline_performance.png",
) -> Dict[str, Any]:
    logger.info(f"Initializing M1 baseline benchmark ({warmup} warm-up, {iterations} iterations)...")
    os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(plot_output_path), exist_ok=True)

    cam = CameraManager(device_index=0, width=640, height=480, auto_fallback_simulation=True)
    detector = YOLODetector(weights_path=weights_path, conf_threshold=0.40)
    monitor = HardwareMonitor()

    # Pre-capture a realistic frame to eliminate I/O variance during inference benchmarking
    ret, test_frame = cam.read()
    if not ret or test_frame is None:
        raise RuntimeError("Failed to capture reference benchmark frame.")

    # 1. Warm-up Phase
    logger.info(f"Executing {warmup} warm-up iterations...")
    for w in range(warmup):
        detector.predict(test_frame)

    # 2. Timed Benchmark Iterations
    logger.info(f"Executing {iterations} timed benchmark iterations...")
    records: List[Dict[str, Any]] = []

    for i in range(1, iterations + 1):
        t0 = time.perf_counter()
        result = detector.predict(test_frame)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0

        hw = monitor.get_metrics()
        record = {
            "iteration": i,
            "inference_latency_ms": latency_ms,
            "fps": fps,
            "detections_count": result["count"],
            "cpu_percent": hw["cpu_percent"],
            "ram_used_mb": hw["ram_used_mb"],
            "ram_percent": hw["ram_percent"],
            "cpu_temperature_c": hw["cpu_temperature_c"],
        }
        records.append(record)

        if i % 20 == 0 or i == iterations:
            logger.info(
                f"Run {i:03d}/{iterations} | Latency: {latency_ms:.1f} ms | "
                f"FPS: {fps:.1f} | CPU: {hw['cpu_percent']:.1f}% | "
                f"Temp: {hw['cpu_temperature_c']:.1f}°C"
            )

    cam.release()

    # 3. Save Raw CSV
    with open(csv_output_path, "w", newline="") as f:
        fieldnames = list(records[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    logger.info(f"Raw benchmark data saved to: {csv_output_path}")

    # 4. Compute Statistical Summary
    latencies = [r["inference_latency_ms"] for r in records]
    fps_values = [r["fps"] for r in records]
    cpu_values = [r["cpu_percent"] for r in records]
    temp_values = [r["cpu_temperature_c"] for r in records]
    ram_values = [r["ram_used_mb"] for r in records]

    summary = {
        "iterations": iterations,
        "warmup": warmup,
        "latency_median_ms": float(np.median(latencies)),
        "latency_mean_ms": float(np.mean(latencies)),
        "latency_std_ms": float(np.std(latencies)),
        "latency_min_ms": float(np.min(latencies)),
        "latency_max_ms": float(np.max(latencies)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "fps_median": float(np.median(fps_values)),
        "fps_mean": float(np.mean(fps_values)),
        "cpu_mean_percent": float(np.mean(cpu_values)),
        "temp_max_c": float(np.max(temp_values)),
        "temp_mean_c": float(np.mean(temp_values)),
        "ram_mean_mb": float(np.mean(ram_values)),
    }

    # 5. Generate Empirical Graph
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("M1.3 Baseline Performance Benchmark - YOLOv8n on Raspberry Pi 5 CPU", fontsize=14, fontweight="bold")

    iters = [r["iteration"] for r in records]

    # Latency Plot
    ax1.plot(iters, latencies, color="#1f77b4", linewidth=1.5, label="Latency (ms)")
    ax1.axhline(summary["latency_median_ms"], color="red", linestyle="--", label=f"Median: {summary['latency_median_ms']:.1f} ms")
    ax1.axhline(summary["latency_p95_ms"], color="orange", linestyle=":", label=f"P95: {summary['latency_p95_ms']:.1f} ms")
    ax1.set_title("Inference Latency per Iteration")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Latency (ms)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")

    # FPS Plot
    ax2.plot(iters, fps_values, color="#2ca02c", linewidth=1.5, label="FPS")
    ax2.axhline(summary["fps_median"], color="darkgreen", linestyle="--", label=f"Median FPS: {summary['fps_median']:.1f}")
    ax2.set_title("Inference Throughput (FPS)")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Frames Per Second")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")

    # CPU Temperature Plot
    ax3.plot(iters, temp_values, color="#d62728", linewidth=1.5, label="SoC Temp (°C)")
    ax3.set_title("CPU Temperature Profile")
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("Temperature (°C)")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="lower right")

    # CPU Utilization Plot
    ax4.plot(iters, cpu_values, color="#9467bd", linewidth=1.5, label="CPU Load (%)")
    ax4.set_title("CPU Utilization Profile")
    ax4.set_xlabel("Iteration")
    ax4.set_ylabel("CPU Load (%)")
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(plot_output_path, dpi=150)
    plt.close()
    logger.info(f"Performance plot saved to: {plot_output_path}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline Performance Benchmark for M1.3")
    parser.add_argument("--weights", type=str, default="models/best.pt")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()

    results = run_baseline_benchmark(
        weights_path=args.weights,
        iterations=args.iterations,
        warmup=args.warmup,
    )
    print("\n--- BENCHMARK SUMMARY ---")
    for k, v in results.items():
        print(f"{k:25s}: {v}")
