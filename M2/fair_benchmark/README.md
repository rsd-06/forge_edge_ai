# Task M2.4: Fair Benchmark (4 Marks)
**Milestone:** M2 Optimize  
**Requirement in Rubric:** *"I run 20 warm-up and 100+ timed runs at batch 1. I show median and p95 latency, FPS and model size"*  

---

## 1. Benchmarking Methodology & Experimental Protocol
- **Hardware**: Raspberry Pi 5 Model B (BCM2712 Cortex-A76 @ 2.4 GHz).
- **Batch Size**: 1 (Real-time industrial single-frame ingestion).
- **Stabilization**: 20 unmetered warmup cycles to eliminate CPU frequency scaling and cache warm-up latency.
- **Timed Executions**: 100 consecutive timed runs measuring pure model inference latency.

---

## 2. Empirical Benchmark Results (Raspberry Pi 5)

| Metric | PyTorch Baseline (CPU) | ONNX Runtime (ARM NEON) | Optimization Impact |
|---|:---:|:---:|:---:|
| **Warmup Runs** | 20 | 20 | Identical protocol |
| **Timed Runs** | 100 | 100 | Identical protocol |
| **Batch Size** | 1 | 1 | Identical protocol |
| **Model Size** | **6.17 MB** | **12.16 MB** | $+5.99\text{ MB}$ (Explicit unrolled graph) |
| **Median Latency** | **231.25 ms** | **185.89 ms** | **$-45.36\text{ ms}$ (19.6% faster)** |
| **P95 Latency** | **241.65 ms** | **192.41 ms** | **$-49.24\text{ ms}$ (20.4% jitter reduction)** |
| **Minimum Latency**| 224.12 ms | 179.34 ms | $-44.78\text{ ms}$ |
| **Maximum Latency**| 278.43 ms | 215.80 ms | $-62.63\text{ ms}$ |
| **Throughput** | **4.32 FPS** | **5.38 FPS** | **$+1.06\text{ FPS}$ ($1.24\times$ speedup)** |

---

## 3. Deliverables in this Subfolder
- [`fair_benchmark_100runs.csv`](fair_benchmark_100runs.csv): Full raw CSV dataset containing all individual timed runs.
- [`benchmark_summary.json`](benchmark_summary.json): Structured statistical summary (mean, median, P95, std, FPS).
- [`m2_benchmark_comparison.png`](m2_benchmark_comparison.png): High-resolution violin and box plots comparing latency distributions.
- [`fair_benchmark.py`](fair_benchmark.py): Automated benchmarking harness script.

---

## 4. Verification Command
```bash
python3 M2/fair_benchmark/fair_benchmark.py --runs 100 --warmup 20
```
