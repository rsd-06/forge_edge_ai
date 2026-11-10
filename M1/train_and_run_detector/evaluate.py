#!/usr/bin/env python3
"""
Extracts training artifacts, metrics, and loss history stored inside best.pt.
Generates evaluation CSVs and training curves for M1.6.
"""

import os
import sys
import csv
import zipfile
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def extract_and_plot():
    weights_path = "models/best.pt"
    csv_out = "results/m1/training_history.csv"
    plot_out = "evidence/graphs/m1_training_evaluation_curves.png"
    os.makedirs(os.path.dirname(csv_out), exist_ok=True)
    os.makedirs(os.path.dirname(plot_out), exist_ok=True)

    class Dummy(dict):
        def __init__(self, *args, **kwargs): pass
        def __setstate__(self, state):
            if isinstance(state, dict): self.update(state)
        def __getattr__(self, item): return self.get(item)
        def __setattr__(self, item, value): self[item] = value
        def append(self, x): pass
        def extend(self, x): pass

    class CustomUnpickler(pickle.Unpickler):
        def find_class(self, mod, name):
            if mod == "builtins" and name in ("dict", "list", "tuple", "int", "float", "str", "bool", "set"):
                return getattr(__builtins__, name)
            return Dummy
        def persistent_load(self, pid): return pid

    with zipfile.ZipFile(weights_path, "r") as z:
        # Check archive/data.pkl or data.pkl
        pkl_name = "archive/data.pkl" if "archive/data.pkl" in z.namelist() else "data.pkl"
        with z.open(pkl_name) as f:
            data = CustomUnpickler(f).load()

    results = data.get("train_results", {})
    if not results:
        print("No train_results found in checkpoint.")
        return

    epochs = results.get("epoch", [])
    num_epochs = len(epochs)
    keys = list(results.keys())

    # Write CSV
    with open(csv_out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for i in range(num_epochs):
            row = [results[k][i] if i < len(results[k]) else "" for k in keys]
            writer.writerow(row)
    print(f"Extracted {num_epochs} epochs of training data to: {csv_out}")

    # Generate 4-panel training evaluation plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("YOLOv8 Enclosure Detector: Training & Validation History (Colab Checkpoint)", fontsize=14, fontweight="bold")

    # 1. Box Loss
    ax1.plot(epochs, results["train/box_loss"], label="Train Box Loss", color="#1f77b4")
    ax1.plot(epochs, results["val/box_loss"], label="Val Box Loss", color="#ff7f0e", linestyle="--")
    ax1.set_title("Bounding Box Loss (CIoU)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. Classification Loss
    ax2.plot(epochs, results["train/cls_loss"], label="Train Cls Loss", color="#2ca02c")
    ax2.plot(epochs, results["val/cls_loss"], label="Val Cls Loss", color="#d62728", linestyle="--")
    ax2.set_title("Classification Loss (BCE)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 3. Precision & Recall
    ax3.plot(epochs, results["metrics/precision(B)"], label="Precision", color="#9467bd")
    ax3.plot(epochs, results["metrics/recall(B)"], label="Recall", color="#8c564b", linestyle="--")
    ax3.set_title("Precision & Recall Evolution")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Score (0.0 - 1.0)")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # 4. mAP@0.5 and mAP@0.5:0.95
    ax4.plot(epochs, results["metrics/mAP50(B)"], label="mAP@0.5", color="#e377c2")
    ax4.plot(epochs, results["metrics/mAP50-95(B)"], label="mAP@0.5:0.95", color="#7f7f7f", linestyle="--")
    ax4.set_title("Mean Average Precision (mAP)")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("mAP")
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.tight_layout()
    plt.savefig(plot_out, dpi=150)
    plt.close()
    print(f"Generated training evaluation plot to: {plot_out}")

if __name__ == "__main__":
    extract_and_plot()
