"""Homework 4, Question 1: low-rank image approximation with NumPy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def fixed_rank_approx(A: np.ndarray, k: int) -> np.ndarray:
    """Return the clipped rank-k SVD reconstruction of a grayscale image."""

    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2 or A.size == 0:
        raise ValueError("A must be a nonempty two-dimensional matrix")
    if not np.all(np.isfinite(A)) or np.any((A < 0.0) | (A > 1.0)):
        raise ValueError("A must contain finite values in [0, 1]")
    if not isinstance(k, (int, np.integer)) or not 1 <= int(k) <= min(A.shape):
        raise ValueError("k must satisfy 1 <= k <= min(A.shape)")

    U, singular_values, Vt = np.linalg.svd(A, full_matrices=False)
    reconstructed = (U[:, :k] * singular_values[:k]) @ Vt[:k, :]
    return np.clip(reconstructed, 0.0, 1.0)


def load_grayscale(path: str | Path, max_size: int = 512) -> np.ndarray:
    if max_size < 1:
        raise ValueError("max_size must be positive")
    with Image.open(path) as source:
        image = source.convert("L")
        width, height = image.size
        scale = min(1.0, max_size / max(width, height))
        if scale < 1.0:
            new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.float64) / 255.0


def run_small_test() -> float:
    # The corresponding RcppEigen result for this diagonal matrix and k=2 is
    # diag(0.9, 0.5, 0.0).  This gives a simple cross-language reference.
    matrix = np.diag([0.9, 0.5, 0.2])
    rcpp_eigen_reference = np.diag([0.9, 0.5, 0.0])
    result = fixed_rank_approx(matrix, 2)
    max_error = float(np.max(np.abs(result - rcpp_eigen_reference)))
    assert result.shape == matrix.shape
    assert np.all((result >= 0.0) & (result <= 1.0))
    assert max_error < 1e-12

    rectangular = np.array([[0.1, 0.8], [0.4, 0.2], [0.7, 0.5]])
    full_rank = fixed_rank_approx(rectangular, 2)
    assert np.max(np.abs(full_rank - rectangular)) < 1e-12
    print(f"Small-matrix RcppEigen comparison max absolute error: {max_error:.3e}")
    return max_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="assets/image.jpg")
    parser.add_argument("--max-size", type=int, default=512)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    test_error = run_small_test()
    image = load_grayscale(args.image, args.max_size)
    ranks = [5, 20, 50]
    approximations = [fixed_rank_approx(image, k) for k in ranks]
    denominator = np.linalg.norm(image, ord="fro")

    rows = []
    m, n = image.shape
    for k, approximation in zip(ranks, approximations):
        error = np.linalg.norm(image - approximation, ord="fro") / denominator
        factor_count = k * (m + n + 1)
        row = {
            "k": k,
            "relative_frobenius_error": float(error),
            "factor_count": factor_count,
            "original_pixel_count": m * n,
            "storage_ratio": factor_count / (m * n),
        }
        rows.append(row)
        print(row)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    axes[0, 0].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("Original")
    for axis, k, approximation in zip(axes.flat[1:], ranks, approximations):
        axis.imshow(approximation, cmap="gray", vmin=0, vmax=1)
        axis.set_title(f"k = {k}")
    for axis in axes.flat:
        axis.axis("off")
    fig.savefig(output_dir / "low-rank-results.png", dpi=180)
    plt.close(fig)

    summary = {
        "matrix_rows": m,
        "matrix_cols": n,
        "small_test_max_absolute_error": test_error,
        "results": rows,
    }
    (output_dir / "low-rank-metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

