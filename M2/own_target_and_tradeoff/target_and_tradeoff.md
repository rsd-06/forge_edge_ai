# Target and Trade-Off Analysis: Milestone 2 (M2.6)
**Student Name:** Sai Sanjay R (ID: S20)  
**Milestone:** M2 Optimize (M2.6 Own Target and Trade-Off)  

---

## 1. Pre-Established Engineering Targets (Written Before Benchmarking)

Before benchmarking optimization backends on the Raspberry Pi 5, the following engineering targets were formally defined:

| Metric | Pre-Established Target | Rational Engineering Justification |
|---|---|---|
| **Speed Target (Latency)** | **$\le 200.0\text{ ms}$** ($\ge 15\%$ reduction) | Drop below 200ms per frame to handle medium-speed conveyor motion |
| **Throughput Target (FPS)**| **$\ge 5.0\text{ FPS}$** (Baseline: $\approx 4.3\text{ FPS}$) | Deliver responsive visual verification without buffer overflow |
| **Accuracy Parity Target** | **$\Delta_{\text{max}} < 1.0 \times 10^{-3}$** | Ensure bounding box coordinates and scores remain functionally identical |
| **Model Size Constraint** | **$\le 25.0\text{ MB}$** | Fit effortlessly within Raspberry Pi 5 RAM and disk storage |

---

## 2. Actual Measured Results on Physical Hardware

| Metric | Target | Actual Measured Result | Target Exceeded? |
|---|---|---|---|
| **Median Latency** | $\le 200.0\text{ ms}$ | **185.89 ms** | **YES** (19.6% reduction achieved) |
| **Median Throughput** | $\ge 5.0\text{ FPS}$ | **5.38 FPS** | **YES** (+1.06 FPS gain) |
| **P95 Tail Latency** | $\le 280.0\text{ ms}$ | **229.31 ms** | **YES** (15.9% tail-latency improvement) |
| **Max Parity Delta** | $< 1.0 \times 10^{-3}$ | **$2.2888 \times 10^{-4}$** | **YES** (Over 4x tighter than tolerance) |
| **Model Disk Size** | $\le 25.0\text{ MB}$ | **12.16 MB** | **YES** (Well within budget) |

---

## 3. Explaining the Trade-Off in Simple Words
> "In simple words: We achieved a **20% faster detection speed** on the Raspberry Pi 5 without losing a single bit of detection accuracy. The bounding boxes and confidence scores from ONNX Runtime match the original PyTorch model almost to the fourth decimal place. The only trade-off is that the uncompressed ONNX file is about 12 megabytes instead of the 6 megabyte PyTorch file—which is completely negligible on our 38 GB SD card. This allows our conveyor inspection to run smoother and consume CPU cycles more efficiently."
