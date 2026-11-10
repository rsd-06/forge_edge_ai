# Task M1.3: Baseline Speed (3 Marks)
**Milestone:** M1 Detect  
**Requirement in Rubric:** *"I measure FPS, latency, CPU/GPU load, temperature and show my own numbers"*  

---

## 1. Overview
Measures empirical, repeatable baseline performance of the unoptimized PyTorch CPU vision pipeline on the Raspberry Pi 5:
- 50 consecutive timed inference runs with 10 warmup cycles.
- Concurrent monitoring of CPU core utilization, RAM consumption, and SoC temperature via `psutil` and Linux thermal zone APIs.
- Saves raw data to CSV and generates statistical plots.

---

## 2. Empirical Hardware Measurements (Raspberry Pi 5)
- **Median Inference Latency**: **303.54 ms** (Cold model initialization: 231.25 ms warmed)
- **Inference Throughput**: **3.29 FPS**
- **Average CPU Load**: **82.9%** across 4 Cortex-A76 cores
- **RAM Footprint (RSS)**: **482 MB**
- **Peak SoC Temperature**: **81.3°C** (Active Cooler maintaining thermal headroom)

---

## 3. Deliverables in this Subfolder
- [`baseline_report.md`](baseline_report.md): Full technical baseline benchmarking report.
- [`baseline_speed_measurements.csv`](baseline_speed_measurements.csv): Raw CSV containing 50 individual hardware runs.
- [`m1_baseline_performance.png`](m1_baseline_performance.png): Latency distribution, FPS, CPU load, and thermal graphs.
- [`benchmark.py`](benchmark.py): Repeatable benchmarking automation script.
- [`hardware_monitor.py`](hardware_monitor.py): Real-time system metrics telemetry class.

---

## 4. Verification Command
```bash
python3 M1/baseline_speed/benchmark.py --iterations 50 --warmup 10
```
