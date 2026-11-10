# Task M2.6: Own Target and Trade-off (3 Marks)
**Milestone:** M2 Optimize  
**Requirement in Rubric:** *"I show the accuracy and speed target I wrote before training, my result, and explain the trade-off in simple words"*  

---

## 1. Pre-Established Target vs Measured Result

| Parameter | Pre-Established Target (Written Prior to Testing) | Measured Actual Result on Raspberry Pi 5 | Target Verdict |
|---|:---:|:---:|:---:|
| **Latency Speedup** | $\ge 15\%$ reduction in inference latency | **$19.6\%$ reduction** (from $231.3\text{ ms}$ to $185.9\text{ ms}$) | **TARGET EXCEEDED** |
| **Accuracy Loss** | $\le 1.0\%$ degradation in mAP / bounding box IoU | **$\text{Max } \Delta = 2.29 \times 10^{-4}$** (0.0% functional loss) | **TARGET EXCEEDED** |
| **Throughput (FPS)** | Increase from $\sim 4.3\text{ FPS}$ to $\ge 5.0\text{ FPS}$ | **$5.38\text{ FPS}$** achieved ($1.24\times$ speedup) | **TARGET EXCEEDED** |

---

## 2. The Engineering Trade-off in Simple Words
The primary trade-off in this optimization is **Model File Size vs Execution Speed**:
- **Why ONNX is faster**: PyTorch is a general deep learning library with Python execution overhead. ONNX Runtime strips away training metadata and unrolls the neural network into an optimized static computation graph that directly uses the Raspberry Pi 5's 128-bit ARM NEON vector instructions.
- **The cost**: To make execution faster, the ONNX model graph explicitly stores constants and layer shapes, increasing the disk footprint from **$6.17\text{ MB}$ to $12.16\text{ MB}$**.
- **Conclusion**: On an embedded system like the Raspberry Pi 5 with 4 GB RAM, spending an extra 6 MB of disk space to gain a **$19.6\%$ speedup** and save **$264\text{ MB}$ of active RAM** is a major win for real-time edge vision.

---

## 3. Deliverables in this Subfolder
- [`target_and_tradeoff.md`](target_and_tradeoff.md): Full engineering report documenting the pre-target hypothesis, experimental outcomes, and simple-words trade-off explanation.
