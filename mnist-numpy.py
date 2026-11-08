"""Train the required MNIST networks with NumPy only."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mnist_data import load_mnist
from mnist_numpy_models import Adam, SGD, SimpleConvNet, TwoLayerNet


SEED = 20261104


def gradient_check() -> float:
    rng = np.random.default_rng(SEED)
    x = rng.normal(size=(5, 4))
    labels = np.array([0, 1, 0, 1, 1])
    network = TwoLayerNet(4, 3, 2, weight_init_std=0.1,
                          seed=SEED, dtype=np.float64)
    backprop = network.gradient(x, labels)
    numerical = network.numerical_gradient(x, labels)
    maximum = max(float(np.max(np.abs(backprop[name] - numerical[name])))
                  for name in backprop)
    if maximum >= 1e-7:
        raise AssertionError(f"Gradient check failed: {maximum}")
    print(f"TwoLayerNet gradient-check maximum absolute difference: {maximum:.3e}")
    return maximum


def smoke_test() -> None:
    difference = gradient_check()
    rng = np.random.default_rng(SEED)
    x = rng.random((2, 1, 28, 28), dtype=np.float32)
    labels = np.array([1, 8])
    network = SimpleConvNet(seed=SEED)
    loss = network.loss(x, labels)
    gradients = network.gradient(x, labels)
    assert np.isfinite(loss)
    assert all(value.shape == network.params[name].shape
               and np.all(np.isfinite(value)) for name, value in gradients.items())
    print(f"SimpleConvNet forward/backward smoke test: PASS (loss={loss:.6f})")
    print(f"Gradient check: PASS ({difference:.3e})")


def train_two_layer(x_train, y_train, steps, batch_size, learning_rate):
    network = TwoLayerNet(seed=SEED)
    optimizer = SGD(learning_rate)
    rng = np.random.default_rng(SEED + 1)
    losses = []
    start = time.perf_counter()
    for step in range(steps):
        indices = rng.choice(len(x_train), batch_size, replace=True)
        x_batch, y_batch = x_train[indices], y_train[indices]
        gradients, loss_value = network.gradient(x_batch, y_batch, return_loss=True)
        optimizer.update(network.params, gradients)
        losses.append(loss_value)
        if (step + 1) % max(1, steps // 10) == 0:
            print(f"NumPy TwoLayerNet step {step + 1}/{steps}, loss={losses[-1]:.4f}")
    elapsed = time.perf_counter() - start
    return network, np.asarray(losses), elapsed


def train_convnet(x_train, y_train, epochs, batch_size, learning_rate):
    network = SimpleConvNet(seed=SEED)
    optimizer = Adam(learning_rate)
    rng = np.random.default_rng(SEED + 2)
    losses = []
    updates = 0
    start = time.perf_counter()
    for epoch in range(epochs):
        permutation = rng.permutation(len(x_train))
        for batch_start in range(0, len(x_train) - batch_size + 1, batch_size):
            indices = permutation[batch_start:batch_start + batch_size]
            x_batch, y_batch = x_train[indices], y_train[indices]
            gradients, loss_value = network.gradient(x_batch, y_batch, return_loss=True)
            optimizer.update(network.params, gradients)
            losses.append(loss_value)
            updates += 1
        print(f"NumPy SimpleConvNet epoch {epoch + 1}/{epochs}, loss={losses[-1]:.4f}")
    elapsed = time.perf_counter() - start
    return network, np.asarray(losses), elapsed, updates


def predict_in_batches(network, x, batch_size=500):
    predictions = []
    for start in range(0, len(x), batch_size):
        logits = network.predict(x[start:start + batch_size])
        predictions.append(np.argmax(logits, axis=1))
    return np.concatenate(predictions)


def plot_losses(fc_losses, cnn_losses, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].plot(fc_losses, linewidth=1)
    axes[0].set(title="NumPy TwoLayerNet", xlabel="Update", ylabel="Training loss")
    axes[1].plot(cnn_losses, linewidth=1)
    axes[1].set(title="NumPy SimpleConvNet", xlabel="Update", ylabel="Training loss")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_errors(images, labels, predictions, title, output_path, count=10):
    wrong = np.flatnonzero(labels != predictions)[:count]
    if wrong.size == 0:
        return
    columns = 5
    rows = int(np.ceil(wrong.size / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(10, 2 * rows), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for axis, index in zip(axes.flat, wrong):
        axis.imshow(images[index, 0], cmap="gray", vmin=0, vmax=1)
        axis.set_title(f"true={labels[index]}, pred={predictions[index]}")
        axis.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/mnist")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--fc-steps", type=int, default=10000)
    parser.add_argument("--fc-learning-rate", type=float, default=0.1)
    parser.add_argument("--cnn-epochs", type=int, default=2)
    parser.add_argument("--cnn-learning-rate", type=float, default=0.001)
    parser.add_argument("--quick", action="store_true",
                        help="Use a small subset and short training run for checking code")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run no-data forward/backward and gradient checks")
    args = parser.parse_args()

    if args.smoke_test:
        smoke_test()
        return
    if min(args.batch_size, args.fc_steps, args.cnn_epochs) < 1:
        raise ValueError("batch size, steps, and epochs must be positive")

    gradient_difference = gradient_check()
    x_train, y_train, x_test, y_test = load_mnist(args.data_dir)
    if args.quick:
        x_train, y_train = x_train[:2000], y_train[:2000]
        x_test, y_test = x_test[:1000], y_test[:1000]
        args.fc_steps = min(args.fc_steps, 200)
        args.cnn_epochs = 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    x_train_flat = x_train.reshape(len(x_train), -1)
    x_test_flat = x_test.reshape(len(x_test), -1)

    fc, fc_losses, fc_seconds = train_two_layer(
        x_train_flat, y_train, args.fc_steps, args.batch_size, args.fc_learning_rate
    )
    fc_predictions = predict_in_batches(fc, x_test_flat)
    fc_accuracy = float(np.mean(fc_predictions == y_test))

    cnn, cnn_losses, cnn_seconds, cnn_updates = train_convnet(
        x_train, y_train, args.cnn_epochs, args.batch_size, args.cnn_learning_rate
    )
    cnn_predictions = predict_in_batches(cnn, x_test, batch_size=args.batch_size)
    cnn_accuracy = float(np.mean(cnn_predictions == y_test))

    plot_losses(fc_losses, cnn_losses, output_dir / "numpy-loss-curves.png")
    plot_errors(x_test, y_test, fc_predictions, "NumPy TwoLayerNet errors",
                output_dir / "numpy-two-layer-errors.png")
    plot_errors(x_test, y_test, cnn_predictions, "NumPy SimpleConvNet errors",
                output_dir / "numpy-cnn-errors.png")
    np.savez_compressed(output_dir / "numpy-losses.npz",
                        two_layer=fc_losses, cnn=cnn_losses)

    metrics = {
        "gradient_check_max_absolute_difference": gradient_difference,
        "training_examples": len(x_train),
        "test_examples": len(x_test),
        "batch_size": args.batch_size,
        "numpy_two_layer": {
            "test_accuracy": fc_accuracy,
            "training_seconds": fc_seconds,
            "updates": args.fc_steps,
            "optimizer": "SGD",
            "learning_rate": args.fc_learning_rate,
            "device": "CPU",
        },
        "numpy_cnn": {
            "test_accuracy": cnn_accuracy,
            "training_seconds": cnn_seconds,
            "updates": cnn_updates,
            "epochs": args.cnn_epochs,
            "optimizer": "Adam",
            "learning_rate": args.cnn_learning_rate,
            "device": "CPU",
        },
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
    }
    (output_dir / "numpy-metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

