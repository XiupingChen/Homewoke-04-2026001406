"""Combine the NumPy and PyTorch metrics after both training scripts finish."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def smooth(values, width=25):
    if len(values) < width:
        return values
    kernel = np.ones(width) / width
    return np.convolve(values, kernel, mode="valid")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    directory = Path(args.results_dir)

    numpy_metrics = json.loads((directory / "numpy-metrics.json").read_text())
    torch_metrics = json.loads((directory / "pytorch-metrics.json").read_text())
    numpy_losses = np.load(directory / "numpy-losses.npz")
    torch_losses = np.load(directory / "pytorch-losses.npz")

    rows = [
        ("NumPy", "TwoLayerNet", numpy_metrics["numpy_two_layer"]),
        ("NumPy", "SimpleConvNet", numpy_metrics["numpy_cnn"]),
        ("PyTorch", "TwoLayerNet", torch_metrics["pytorch_two_layer"]),
        ("PyTorch", "SimpleConvNet", torch_metrics["pytorch_cnn"]),
    ]
    with (directory / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["framework", "model", "test_accuracy", "training_seconds",
                         "updates", "device"])
        for framework, model, values in rows:
            writer.writerow([framework, model, values["test_accuracy"],
                             values["training_seconds"], values["updates"],
                             values["device"]])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].plot(smooth(numpy_losses["two_layer"]), label="NumPy")
    axes[0].plot(smooth(torch_losses["two_layer"]), label="PyTorch")
    axes[0].set(title="TwoLayerNet", xlabel="Update", ylabel="Smoothed loss")
    axes[1].plot(smooth(numpy_losses["cnn"]), label="NumPy")
    axes[1].plot(smooth(torch_losses["cnn"]), label="PyTorch")
    axes[1].set(title="SimpleConvNet", xlabel="Update", ylabel="Smoothed loss")
    for axis in axes:
        axis.legend()
    fig.savefig(directory / "all-loss-curves.png", dpi=180)
    plt.close(fig)

    print("framework, model, accuracy, seconds, updates, device")
    for framework, model, values in rows:
        print(framework, model, f"{values['test_accuracy']:.6f}",
              f"{values['training_seconds']:.3f}", values["updates"],
              values["device"], sep=", ")


if __name__ == "__main__":
    main()

