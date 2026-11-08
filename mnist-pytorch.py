"""Train the required MNIST networks with PyTorch autograd."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from mnist_data import load_mnist


SEED = 20261104


class TwoLayerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 50),
            nn.Sigmoid(),
            nn.Linear(50, 10),
        )

    def forward(self, x):
        return self.layers(x)


class SimpleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 30, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(30 * 12 * 12, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def initialize_like_numpy(module):
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.normal_(module.weight, mean=0.0, std=0.01)
        nn.init.zeros_(module.bias)


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"PyTorch device: {device}")
    return device


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def train_two_layer(x_train, y_train, steps, batch_size, learning_rate, device):
    torch.manual_seed(SEED)
    model = TwoLayerNet().to(device)
    model.apply(initialize_like_numpy)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    rng = np.random.default_rng(SEED + 1)
    losses = []
    model.train()
    synchronize(device)
    start = time.perf_counter()
    for step in range(steps):
        indices = rng.choice(len(x_train), batch_size, replace=True)
        x_batch = torch.as_tensor(x_train[indices], dtype=torch.float32, device=device)
        y_batch = torch.as_tensor(y_train[indices], dtype=torch.long, device=device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_batch)
        loss = criterion(logits, y_batch)  # logits are passed directly: no Softmax here
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if (step + 1) % max(1, steps // 10) == 0:
            print(f"PyTorch TwoLayerNet step {step + 1}/{steps}, loss={losses[-1]:.4f}")
    synchronize(device)
    elapsed = time.perf_counter() - start
    return model, np.asarray(losses), elapsed


def train_convnet(x_train, y_train, epochs, batch_size, learning_rate, device):
    torch.manual_seed(SEED)
    model = SimpleConvNet().to(device)
    model.apply(initialize_like_numpy)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    rng = np.random.default_rng(SEED + 2)
    losses = []
    updates = 0
    model.train()
    synchronize(device)
    start = time.perf_counter()
    for epoch in range(epochs):
        permutation = rng.permutation(len(x_train))
        for batch_start in range(0, len(x_train) - batch_size + 1, batch_size):
            indices = permutation[batch_start:batch_start + batch_size]
            x_batch = torch.as_tensor(x_train[indices], dtype=torch.float32, device=device)
            y_batch = torch.as_tensor(y_train[indices], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            updates += 1
        print(f"PyTorch SimpleConvNet epoch {epoch + 1}/{epochs}, loss={losses[-1]:.4f}")
    synchronize(device)
    elapsed = time.perf_counter() - start
    return model, np.asarray(losses), elapsed, updates


def evaluate(model, images, labels, device, batch_size=500, flatten=False):
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            x_batch = images[start:start + batch_size]
            if flatten:
                x_batch = x_batch.reshape(len(x_batch), -1)
            tensor = torch.as_tensor(x_batch, dtype=torch.float32, device=device)
            predictions.append(model(tensor).argmax(dim=1).cpu().numpy())
    prediction = np.concatenate(predictions)
    return prediction, float(np.mean(prediction == labels))


def plot_losses(fc_losses, cnn_losses, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].plot(fc_losses, linewidth=1)
    axes[0].set(title="PyTorch TwoLayerNet", xlabel="Update", ylabel="Training loss")
    axes[1].plot(cnn_losses, linewidth=1)
    axes[1].set(title="PyTorch SimpleConvNet", xlabel="Update", ylabel="Training loss")
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
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--fc-steps", type=int, default=10000)
    parser.add_argument("--fc-learning-rate", type=float, default=0.1)
    parser.add_argument("--cnn-epochs", type=int, default=2)
    parser.add_argument("--cnn-learning-rate", type=float, default=0.001)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if min(args.batch_size, args.fc_steps, args.cnn_epochs) < 1:
        raise ValueError("batch size, steps, and epochs must be positive")
    device = select_device(args.device)
    x_train, y_train, x_test, y_test = load_mnist(args.data_dir)
    if args.quick:
        x_train, y_train = x_train[:2000], y_train[:2000]
        x_test, y_test = x_test[:1000], y_test[:1000]
        args.fc_steps = min(args.fc_steps, 200)
        args.cnn_epochs = 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    x_train_flat = x_train.reshape(len(x_train), -1)

    fc, fc_losses, fc_seconds = train_two_layer(
        x_train_flat, y_train, args.fc_steps, args.batch_size,
        args.fc_learning_rate, device
    )
    fc_predictions, fc_accuracy = evaluate(
        fc, x_test, y_test, device, flatten=True
    )

    cnn, cnn_losses, cnn_seconds, cnn_updates = train_convnet(
        x_train, y_train, args.cnn_epochs, args.batch_size,
        args.cnn_learning_rate, device
    )
    cnn_predictions, cnn_accuracy = evaluate(cnn, x_test, y_test, device)

    plot_losses(fc_losses, cnn_losses, output_dir / "pytorch-loss-curves.png")
    plot_errors(x_test, y_test, fc_predictions, "PyTorch TwoLayerNet errors",
                output_dir / "pytorch-two-layer-errors.png")
    plot_errors(x_test, y_test, cnn_predictions, "PyTorch SimpleConvNet errors",
                output_dir / "pytorch-cnn-errors.png")
    np.savez_compressed(output_dir / "pytorch-losses.npz",
                        two_layer=fc_losses, cnn=cnn_losses)

    metrics = {
        "training_examples": len(x_train),
        "test_examples": len(x_test),
        "batch_size": args.batch_size,
        "pytorch_two_layer": {
            "test_accuracy": fc_accuracy,
            "training_seconds": fc_seconds,
            "updates": args.fc_steps,
            "optimizer": "SGD",
            "learning_rate": args.fc_learning_rate,
            "device": str(device),
        },
        "pytorch_cnn": {
            "test_accuracy": cnn_accuracy,
            "training_seconds": cnn_seconds,
            "updates": cnn_updates,
            "epochs": args.cnn_epochs,
            "optimizer": "Adam",
            "learning_rate": args.cnn_learning_rate,
            "device": str(device),
        },
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
    }
    (output_dir / "pytorch-metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

