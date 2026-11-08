"""Download and read the original MNIST IDX files.

The data are cached locally under data/mnist and are not part of the submission.
"""

from __future__ import annotations

import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np


MNIST_BASE_URLS = [
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
]
MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def _download(filename: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    errors = []
    for base_url in MNIST_BASE_URLS:
        url = base_url + filename
        print(f"Downloading {url}")
        try:
            urllib.request.urlretrieve(url, temporary)
            temporary.replace(destination)
            return
        except Exception as error:  # Try the second public mirror.
            errors.append(f"{url}: {error}")
            temporary.unlink(missing_ok=True)
    raise RuntimeError("Unable to download MNIST:\n" + "\n".join(errors))


def _ensure_files(data_dir: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for key, filename in MNIST_FILES.items():
        path = data_dir / filename
        if not path.exists():
            _download(filename, path)
        paths[key] = path
    return paths


def _read_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as stream:
        magic, count, rows, cols = struct.unpack(">IIII", stream.read(16))
        if magic != 2051:
            raise ValueError(f"Unexpected image magic number in {path}: {magic}")
        values = np.frombuffer(stream.read(), dtype=np.uint8)
    expected = count * rows * cols
    if values.size != expected:
        raise ValueError(f"Incomplete MNIST image file: {path}")
    return values.reshape(count, 1, rows, cols).astype(np.float32) / 255.0


def _read_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as stream:
        magic, count = struct.unpack(">II", stream.read(8))
        if magic != 2049:
            raise ValueError(f"Unexpected label magic number in {path}: {magic}")
        values = np.frombuffer(stream.read(), dtype=np.uint8)
    if values.size != count:
        raise ValueError(f"Incomplete MNIST label file: {path}")
    return values.astype(np.int64)


def load_mnist(data_dir: str | Path = "data/mnist"):
    """Return the 60,000 training and 10,000 test examples.

    Images are float32 arrays with shape (N, 1, 28, 28) and values in [0, 1].
    Labels are integer arrays with values from 0 through 9.
    """

    paths = _ensure_files(Path(data_dir))
    x_train = _read_images(paths["train_images"])
    y_train = _read_labels(paths["train_labels"])
    x_test = _read_images(paths["test_images"])
    y_test = _read_labels(paths["test_labels"])
    if x_train.shape != (60000, 1, 28, 28) or x_test.shape != (10000, 1, 28, 28):
        raise ValueError("MNIST dimensions do not match the official data set")
    return x_train, y_train, x_test, y_test

